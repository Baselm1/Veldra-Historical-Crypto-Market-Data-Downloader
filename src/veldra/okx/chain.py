"""Query OKX family archives while retaining their exact contract identities."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from datetime import datetime
import math
from pathlib import Path

import duckdb
import pandas as pd

from veldra.core.datasets import DatasetSpec


@dataclass(frozen=True)
class OptionChainFilter:
    """Limit an Options family query without loading its entire chain.

    Args:
        expiry: Optional exact contract expiry.
        strike_min: Optional inclusive minimum strike.
        strike_max: Optional inclusive maximum strike.
        option_type: Optional native call or put code.
    """

    expiry: date | None = None
    strike_min: float | None = None
    strike_max: float | None = None
    option_type: str | None = None

    def __post_init__(self) -> None:
        """Reject contradictory or unsupported Option filters."""
        for name, value in (
            ("strike_min", self.strike_min),
            ("strike_max", self.strike_max),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.strike_min is not None and self.strike_min < 0:
            raise ValueError("strike_min cannot be negative")
        if self.strike_max is not None and self.strike_max < 0:
            raise ValueError("strike_max cannot be negative")
        if (
            self.strike_min is not None
            and self.strike_max is not None
            and self.strike_min > self.strike_max
        ):
            raise ValueError("strike_min cannot exceed strike_max")
        if self.option_type is not None and not isinstance(self.option_type, str):
            raise TypeError("option_type must be a string")
        if self.option_type not in {None, "C", "P"}:
            raise ValueError("option_type must be C or P")


def _option_predicate(value: OptionChainFilter | None) -> tuple[str, list[object]]:
    """Return SQL and parameters for server-side Option contract filtering.

    Args:
        value: Optional validated Option chain filter.

    Returns:
        SQL clauses and their bound parameters.
    """
    if value is None:
        return "", []
    clauses: list[str] = []
    parameters: list[object] = []
    if value.expiry is not None:
        clauses.append("regexp_extract(instrument_id, '-([0-9]{6})-', 1) = ?")
        parameters.append(value.expiry.strftime("%y%m%d"))
    strike = (
        "CAST(regexp_extract(instrument_id, "
        "'-([0-9]+(?:\\.[0-9]+)?)-[CP]$', 1) AS DOUBLE)"
    )
    if value.strike_min is not None:
        clauses.append(f"{strike} >= ?")
        parameters.append(value.strike_min)
    if value.strike_max is not None:
        clauses.append(f"{strike} <= ?")
        parameters.append(value.strike_max)
    if value.option_type is not None:
        clauses.append("right(instrument_id, 1) = ?")
        parameters.append(value.option_type)
    return (" AND " + " AND ".join(clauses) if clauses else ""), parameters


def _identifier(value: str) -> str:
    """Return one safely quoted internal SQL identifier."""
    return '"' + value.replace('"', '""') + '"'


def _bucket(interval: str) -> str:
    """Return a UTC-aligned DuckDB Kline bucket expression."""
    if interval == "1w":
        return "timezone('UTC', date_trunc('week', timezone('UTC', open_time)))"
    if interval == "1mo":
        return "timezone('UTC', date_trunc('month', timezone('UTC', open_time)))"
    number, suffix = int(interval[:-1]), interval[-1]
    unit = {"m": "minutes", "h": "hours", "d": "days"}[suffix]
    return (
        f"time_bucket(INTERVAL '{number} {unit}', open_time, "
        "TIMESTAMPTZ '1970-01-01 00:00:00+00')"
    )


def _projection(columns: Mapping[str, str], *, synthetic: bool) -> str:
    """Return the caller projection with a mandatory contract identity."""
    values = ["instrument_id"]
    for source, label in columns.items():
        expression = (
            "false"
            if source == "is_synthetic" and not synthetic
            else _identifier(source)
        )
        values.append(f"{expression} AS {_identifier(label)}")
    return ", ".join(values)


def _resampled_fields(dataset: DatasetSpec, interval: str) -> list[str]:
    """Return family-grouped OHLC and additive Kline expressions."""
    fields = ["instrument_id", f"{_bucket(interval)} AS open_time"]
    for column in dataset.stored_columns:
        if column == "open_time":
            continue
        if column == "open":
            value = "arg_min(open, open_time)"
        elif column == "high":
            value = "max(high)"
        elif column == "low":
            value = "min(low)"
        elif column == "close":
            value = "arg_max(close, open_time)"
        elif column in dataset.resample_sum_columns:
            value = f"sum({_identifier(column)})"
        else:
            raise ValueError(f"cannot resample chain column {column!r}")
        fields.append(f"{value} AS {_identifier(column)}")
    fields.append("false AS is_synthetic")
    return fields


def query_chain(
    connection: duckdb.DuckDBPyConnection,
    paths: Sequence[Path],
    dataset: DatasetSpec,
    start: datetime,
    end: datetime,
    columns: Mapping[str, str],
    *,
    interval: str | None,
    option_filter: OptionChainFilter | None = None,
) -> pd.DataFrame:
    """Return exact contracts from shared family Parquet materializations.

    Args:
        connection: Open catalog connection.
        paths: Shared family materialization paths.
        dataset: Product-specific Kline or trade declaration.
        start: Inclusive UTC timestamp.
        end: Exclusive UTC timestamp.
        columns: Canonical columns mapped to output labels.
        interval: Resolved Kline output interval or ``None`` for trades.
        option_filter: Optional server-side Option contract constraints.

    Returns:
        Contract-identified rows in deterministic time order.
    """
    if "instrument_id" in columns.values():
        raise ValueError("column labels cannot replace the required instrument_id")
    if not paths:
        values: dict[str, pd.Series] = {"instrument_id": pd.Series(dtype="string")}
        values.update(
            {
                label: pd.Series(dtype=dataset.column_dtype(source))
                for source, label in columns.items()
            }
        )
        return pd.DataFrame(values)
    unique = sorted({str(path) for path in paths})
    relation = "read_parquet(?)"
    time_column = _identifier(dataset.time_column)
    predicate, filter_parameters = _option_predicate(option_filter)
    base = (
        f"SELECT * FROM {relation} WHERE {time_column} >= ? "
        f"AND {time_column} < ?{predicate}"
    )
    resolved = dataset.resolve_interval(interval)
    if resolved is not None and resolved != dataset.base_interval:
        fields = ", ".join(_resampled_fields(dataset, resolved))
        source = (
            f"SELECT {fields} FROM ({base}) "
            f"GROUP BY instrument_id, {_bucket(resolved)}"
        )
        projection = _projection(columns, synthetic=True)
    else:
        source = base
        projection = _projection(columns, synthetic=False)
    secondary = ", trade_id" if "trade_id" in dataset.stored_columns else ""
    frame = connection.execute(
        f"SELECT {projection} FROM ({source}) "
        f"ORDER BY instrument_id, {_identifier(dataset.time_column)}{secondary}",
        [unique, start, end, *filter_parameters],
    ).df()
    for source_name, label in columns.items():
        if source_name in dataset.timestamp_columns:
            frame[label] = pd.to_datetime(frame[label], utc=True).astype(
                "datetime64[us, UTC]"
            )
    frame["instrument_id"] = frame["instrument_id"].astype("string")
    return frame
