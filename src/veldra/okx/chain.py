"""Query OKX family archives while retaining their exact contract identities."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from veldra.core.datasets import DatasetSpec


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
    base = f"SELECT * FROM {relation} WHERE {time_column} >= ? AND {time_column} < ?"
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
        [unique, start, end],
    ).df()
    for source_name, label in columns.items():
        if source_name in dataset.timestamp_columns:
            frame[label] = pd.to_datetime(frame[label], utc=True).astype(
                "datetime64[us, UTC]"
            )
    frame["instrument_id"] = frame["instrument_id"].astype("string")
    return frame
