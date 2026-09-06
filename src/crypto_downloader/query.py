"""Query cached Parquet data through DuckDB."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
from time import perf_counter

import duckdb
import pandas as pd

from .datasets import DatasetSpec
from .models import Gap
from .request import parse_gap_policy

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
LOGGER = logging.getLogger(__name__)


def _microseconds(value: datetime) -> int:
    """Return exact microseconds since the UTC epoch.

    Args:
        value: The UTC timestamp to convert.

    Returns:
        Signed microseconds since January 1, 1970 UTC.
    """
    delta = value - EPOCH
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def _grid(start: datetime, end: datetime, dataset: DatasetSpec) -> tuple[int, int, int]:
    """Return aligned first, stop, and step microseconds for a query range.

    Args:
        start: The inclusive first requested timestamp.
        end: The exclusive final requested timestamp.
        dataset: The schema defining the stored base interval.

    Returns:
        The first grid point, exclusive stop, and interval size.
    """
    if dataset.base_interval != "1m":
        raise ValueError(
            "missing-candle handling currently requires a 1m base interval"
        )
    step = 60_000_000
    start_us, end_us = _microseconds(start), _microseconds(end)
    first = ((start_us + step - 1) // step) * step
    stop = ((end_us + step - 1) // step) * step
    return first, stop, step


def missing_ranges(
    connection: duckdb.DuckDBPyConnection,
    paths: Sequence[Path],
    dataset: DatasetSpec,
    start: datetime,
    end: datetime,
) -> list[Gap]:
    """Locate internal missing base-interval candles in cached daily files.

    Args:
        connection: The DuckDB connection used to inspect Parquet files.
        paths: The locally available daily Parquet files.
        dataset: The schema defining the time column and base interval.
        start: The inclusive first requested timestamp.
        end: The exclusive final requested timestamp.

    Returns:
        Consecutive internal candle gaps in chronological order.
    """
    _validate_range(start, end)
    first, stop, step = _grid(start, end, dataset)
    if first >= stop or not paths:
        return []
    rows = connection.execute(
        f"""
        WITH actual AS (
            SELECT epoch_us({_identifier(dataset.time_column)}) AS point,
                   CAST({_identifier(dataset.time_column)} AS DATE) AS day
            FROM read_parquet(?)
        ), day_bounds AS (
            SELECT day, min(point) AS first_point, max(point) AS last_point
            FROM actual GROUP BY day
        ), expected AS (
            SELECT point FROM (
                SELECT unnest(range(?::BIGINT, ?::BIGINT, ?::BIGINT)) AS point
            ) JOIN day_bounds
              ON CAST(to_timestamp(point / 1000000.0) AS DATE) = day
             AND point BETWEEN first_point AND last_point
        ), missing AS (
            SELECT expected.point FROM expected ANTI JOIN actual USING (point)
        ), grouped AS (
            SELECT point,
                   point - row_number() OVER (ORDER BY point) * ? AS island
            FROM missing
        )
        SELECT min(point), max(point) + ?, count(*)
        FROM grouped GROUP BY island ORDER BY min(point)
        """,
        [[str(path) for path in paths], first, stop, step, step, step],
    ).fetchall()
    return [
        Gap(
            EPOCH + timedelta(microseconds=range_start),
            EPOCH + timedelta(microseconds=range_end),
            count,
        )
        for range_start, range_end, count in rows
    ]


def empty_frame(dataset: DatasetSpec, columns: Mapping[str, str]) -> pd.DataFrame:
    """Create an empty result with the requested names and data types.

    Args:
        dataset: The schema describing available columns.
        columns: Canonical columns mapped to output labels.

    Returns:
        An empty DataFrame with stable result data types.
    """
    values: dict[str, pd.Series] = {}
    for source, label in columns.items():
        if source.endswith("time"):
            dtype = "datetime64[us, UTC]"
        elif source == "trade_count":
            dtype = "int64"
        elif source == "is_synthetic":
            dtype = "bool"
        else:
            dtype = "float64"
        values[label] = pd.Series(dtype=dtype)
    return pd.DataFrame(values)


def _identifier(value: str) -> str:
    """Quote one SQL identifier without interpreting its contents.

    Args:
        value: The column name or caller-facing label.

    Returns:
        A safely quoted DuckDB identifier.
    """
    return '"' + value.replace('"', '""') + '"'


def _usable_identifier(value: object) -> bool:
    """Return whether a column name is safe and contains visible text.

    Args:
        value: The proposed canonical column name or output label.

    Returns:
        True when the value is a usable SQL identifier.
    """
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value


def _validate_columns(dataset: DatasetSpec, columns: Mapping[str, str]) -> None:
    """Reject empty, unknown, or ambiguous column projections.

    Args:
        dataset: The schema describing available output columns.
        columns: Canonical columns mapped to output labels.
    """
    if not columns:
        raise ValueError("column selection cannot be empty")
    if any(
        not _usable_identifier(source) or not _usable_identifier(label)
        for source, label in columns.items()
    ):
        raise ValueError("column names and labels must contain usable text")
    if any(source not in dataset.output_columns for source in columns):
        raise ValueError("column selection contains an unknown column")
    if len(set(columns.values())) != len(columns):
        raise ValueError("column output labels must be unique")


def _validate_range(start: datetime, end: datetime) -> None:
    """Require a nonempty increasing UTC query range.

    Args:
        start: The proposed inclusive start timestamp.
        end: The proposed exclusive end timestamp.
    """
    if start.utcoffset() is None or end.utcoffset() is None:
        raise ValueError("query timestamps must include a timezone")
    if start.utcoffset() != timedelta(0) or end.utcoffset() != timedelta(0):
        raise ValueError("query timestamps must use UTC")
    if end <= start:
        raise ValueError("query range must end after it starts")


def _projection(columns: Mapping[str, str], *, synthetic_column: bool = False) -> str:
    """Build the selected SQL expressions in caller order.

    Args:
        columns: Canonical columns mapped to output labels.
        synthetic_column: Whether the query source contains the generated marker.

    Returns:
        A comma-separated DuckDB projection.
    """
    expressions = []
    for source, label in columns.items():
        value = (
            _identifier(source)
            if source != "is_synthetic" or synthetic_column
            else "false"
        )
        expressions.append(f"{value} AS {_identifier(label)}")
    return ", ".join(expressions)


def _filled_fields(dataset: DatasetSpec, step: int, policy: str) -> str:
    """Build canonical SQL fields for generated kline rows.

    Args:
        dataset: The schema describing stored columns.
        step: The base interval in microseconds.
        policy: The forward, backward, or nan fill policy.

    Returns:
        A comma-separated canonical SQL projection.
    """
    fields = ["to_timestamp(grid_time / 1000000.0) AS open_time"]
    reference = "previous_close" if policy == "forward" else "next_open"
    for name in dataset.stored_columns:
        if name == dataset.time_column:
            continue
        if policy == "nan":
            value = _identifier(name)
        elif name in {"open", "high", "low", "close"}:
            value = (
                f"CASE WHEN is_synthetic THEN {reference} "
                f"ELSE {_identifier(name)} END"
            )
        elif name == "close_time":
            value = (
                "CASE WHEN is_synthetic THEN "
                f"to_timestamp((grid_time + {step} - 1) / 1000000.0) "
                f"ELSE {_identifier(name)} END"
            )
        else:
            value = f"CASE WHEN is_synthetic THEN 0 ELSE {_identifier(name)} END"
        cast = "::BIGINT" if name == "trade_count" and policy != "nan" else ""
        fields.append(f"({value}){cast} AS {_identifier(name)}")
    fields.append("is_synthetic")
    return ", ".join(fields)


def _filled_query(
    dataset: DatasetSpec,
    paths: Sequence[Path],
    start: datetime,
    end: datetime,
    policy: str,
) -> tuple[str, list[object]]:
    """Build a DuckDB query that generates only internal daily candles.

    Args:
        dataset: The schema describing cached rows.
        paths: The local daily Parquet files.
        start: The inclusive first requested timestamp.
        end: The exclusive final requested timestamp.
        policy: The forward, backward, or nan fill policy.

    Returns:
        SQL text and its positional parameters.
    """
    _, _, step = _grid(start, end, dataset)
    time_column = _identifier(dataset.time_column)
    fields = _filled_fields(dataset, step, policy)
    sql = f"""
        WITH actual AS (
            SELECT epoch_us({time_column}) AS grid_time,
                   CAST({time_column} AS DATE) AS source_day,
                   * EXCLUDE ({time_column})
            FROM read_parquet(?)
        ), day_bounds AS (
            SELECT source_day, min(grid_time) AS first_time,
                   max(grid_time) AS last_time
            FROM actual GROUP BY source_day
        ), grid AS (
            SELECT source_day,
                   unnest(range(first_time, last_time + ?::BIGINT, ?::BIGINT))
                       AS grid_time
            FROM day_bounds
        ), joined AS (
            SELECT grid.source_day, grid.grid_time,
                   actual.* EXCLUDE (grid_time, source_day),
                   actual.grid_time IS NULL AS is_synthetic
            FROM grid LEFT JOIN actual USING (source_day, grid_time)
        ), neighbors AS (
            SELECT *,
                last_value(close IGNORE NULLS) OVER (
                    PARTITION BY source_day ORDER BY grid_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS previous_close,
                first_value(open IGNORE NULLS) OVER (
                    PARTITION BY source_day ORDER BY grid_time
                    ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
                ) AS next_open
            FROM joined
        ), filled AS (
            SELECT {fields} FROM neighbors
        )
        SELECT * FROM filled
        WHERE {time_column} >= ? AND {time_column} < ?
        ORDER BY {time_column}
    """
    return sql, [[str(path) for path in paths], step, step, start, end]


def _raw_query(
    dataset: DatasetSpec,
    paths: Sequence[Path],
    start: datetime,
    end: datetime,
) -> tuple[str, list[object]]:
    """Build a query exposing real stored candles with a false marker.

    Args:
        dataset: The schema describing cached rows.
        paths: The local daily Parquet files.
        start: The inclusive first requested timestamp.
        end: The exclusive final requested timestamp.

    Returns:
        SQL text and its positional parameters.
    """
    time_column = _identifier(dataset.time_column)
    sql = (
        "SELECT *, false AS is_synthetic FROM read_parquet(?) "
        f"WHERE {time_column} >= ? AND {time_column} < ?"
    )
    return sql, [[str(path) for path in paths], start, end]


def _bucket_expression(interval: str) -> str:
    """Return the UTC-aligned DuckDB expression for an output interval.

    Args:
        interval: A supported interval larger than one minute.

    Returns:
        SQL that computes the containing bucket's opening timestamp.
    """
    if interval == "1w":
        return "timezone('UTC', date_trunc('week', timezone('UTC', open_time)))"
    if interval == "1mo":
        return "timezone('UTC', date_trunc('month', timezone('UTC', open_time)))"
    number, suffix = (
        (int(interval[:-1]), interval[-1])
        if not interval.endswith("mo")
        else (int(interval[:-2]), "mo")
    )
    units = {"m": "minutes", "h": "hours", "d": "days", "mo": "months"}
    width = f"{number} {units[suffix]}"
    return (
        f"time_bucket(INTERVAL '{width}', open_time, "
        "TIMESTAMPTZ '1970-01-01 00:00:00+00')"
    )


def _aggregate(expression: str, gap_policy: str, *, integer: bool = False) -> str:
    """Optionally null an aggregate when its bucket contains a nan-policy gap.

    Args:
        expression: The DuckDB aggregate expression.
        gap_policy: The missing-candle policy applied before aggregation.
        integer: Whether the completed expression must remain a BIGINT.

    Returns:
        The aggregate with nan-gap and integer behavior applied.
    """
    value = (
        f"CASE WHEN bool_or(is_synthetic) THEN NULL ELSE {expression} END"
        if gap_policy == "nan"
        else expression
    )
    return f"({value})::BIGINT" if integer else value


def _resampled_query(
    source_sql: str,
    dataset: DatasetSpec,
    interval: str,
    gap_policy: str,
    columns: Mapping[str, str],
) -> str:
    """Wrap a canonical candle query with OHLCV aggregation and projection.

    Args:
        source_sql: SQL returning canonical base candles and a synthetic marker.
        dataset: The schema describing canonical stored columns.
        interval: The supported output interval.
        gap_policy: The missing-candle policy applied to base rows.
        columns: Canonical columns mapped to output labels.

    Returns:
        DuckDB SQL returning projected resampled candles.
    """
    fields = [
        f"{_bucket_expression(interval)} AS open_time",
        f"{_aggregate('arg_min(open, open_time)', gap_policy)} AS open",
        f"{_aggregate('max(high)', gap_policy)} AS high",
        f"{_aggregate('min(low)', gap_policy)} AS low",
        f"{_aggregate('arg_max(close, open_time)', gap_policy)} AS close",
        f"{_aggregate('sum(volume)', gap_policy)} AS volume",
        f"{_aggregate('max(close_time)', gap_policy)} AS close_time",
        f"{_aggregate('sum(quote_volume)', gap_policy)} AS quote_volume",
        f"{_aggregate('sum(trade_count)', gap_policy, integer=True)} AS trade_count",
        f"{_aggregate('sum(taker_buy_base_volume)', gap_policy)} "
        "AS taker_buy_base_volume",
        f"{_aggregate('sum(taker_buy_quote_volume)', gap_policy)} "
        "AS taker_buy_quote_volume",
        "bool_or(is_synthetic) AS is_synthetic",
    ]
    projection = _projection(columns, synthetic_column=True)
    return (
        f"WITH base AS ({source_sql}), resampled AS ("
        f"SELECT {', '.join(fields)} FROM base GROUP BY 1) "
        f"SELECT {projection} FROM resampled ORDER BY open_time"
    )


def _normalize_result_times(frame: pd.DataFrame, columns: Mapping[str, str]) -> None:
    """Restore stable UTC timestamp dtypes after a DuckDB query.

    Args:
        frame: The query result to update in place.
        columns: Canonical columns mapped to output labels.
    """
    for source, label in columns.items():
        if source.endswith("time"):
            frame[label] = pd.to_datetime(frame[label], utc=True).astype(
                "datetime64[us, UTC]"
            )


def query_parquet(
    connection: duckdb.DuckDBPyConnection,
    paths: Sequence[Path],
    dataset: DatasetSpec,
    start: datetime,
    end: datetime,
    columns: Mapping[str, str],
    *,
    gap_policy: str = "keep",
    interval: str | None = None,
) -> pd.DataFrame:
    """Query an exact timestamp range from cached Parquet files.

    Args:
        connection: The DuckDB connection used to execute the query.
        paths: The daily Parquet files available for the range.
        dataset: The schema describing the cached rows.
        start: The inclusive first UTC timestamp.
        end: The exclusive final UTC timestamp.
        columns: Canonical columns mapped to output labels.
        gap_policy: The behavior used for internal missing candles.
        interval: The requested output interval or the stored base interval.

    Returns:
        Requested rows ordered by the dataset timestamp.
    """
    started = perf_counter()
    _validate_columns(dataset, columns)
    _validate_range(start, end)
    policy = parse_gap_policy(gap_policy)
    output_interval = dataset.resolve_interval(
        dataset.base_interval if interval is None else interval
    )
    if not paths:
        LOGGER.info(
            "Parquet query skipped: product=%s dataset=%s paths=0 range=[%s, %s)",
            dataset.product,
            dataset.name,
            start,
            end,
        )
        return empty_frame(dataset, columns)

    if policy in {"forward", "backward", "nan"}:
        sql, parameters = _filled_query(dataset, paths, start, end, policy)
    else:
        sql, parameters = _raw_query(dataset, paths, start, end)
    if output_interval == dataset.base_interval:
        projection = _projection(columns, synthetic_column=True)
        sql = f"SELECT {projection} FROM ({sql}) ORDER BY open_time"
    else:
        sql = _resampled_query(sql, dataset, output_interval, policy, columns)
    frame = connection.execute(sql, parameters).df()
    _normalize_result_times(frame, columns)
    LOGGER.info(
        "Parquet query complete: product=%s dataset=%s paths=%d range=[%s, %s) "
        "interval=%s gap_policy=%s rows=%d elapsed=%.3fs",
        dataset.product,
        dataset.name,
        len(paths),
        start,
        end,
        output_interval,
        policy,
        len(frame),
        perf_counter() - started,
    )
    return frame
