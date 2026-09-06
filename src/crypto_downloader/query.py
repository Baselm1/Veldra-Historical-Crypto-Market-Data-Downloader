"""Query cached Parquet data through DuckDB."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from .datasets import DatasetSpec


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


def _projection(columns: Mapping[str, str]) -> str:
    """Build the selected SQL expressions in caller order.

    Args:
        columns: Canonical columns mapped to output labels.

    Returns:
        A comma-separated DuckDB projection.
    """
    expressions = []
    for source, label in columns.items():
        value = "false" if source == "is_synthetic" else _identifier(source)
        expressions.append(f"{value} AS {_identifier(label)}")
    return ", ".join(expressions)


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
) -> pd.DataFrame:
    """Query an exact timestamp range from cached Parquet files.

    Args:
        connection: The DuckDB connection used to execute the query.
        paths: The daily Parquet files available for the range.
        dataset: The schema describing the cached rows.
        start: The inclusive first UTC timestamp.
        end: The exclusive final UTC timestamp.
        columns: Canonical columns mapped to output labels.

    Returns:
        Requested rows ordered by the dataset timestamp.
    """
    _validate_columns(dataset, columns)
    _validate_range(start, end)
    if not paths:
        return empty_frame(dataset, columns)

    frame = connection.execute(
        f"SELECT {_projection(columns)} FROM read_parquet(?) "
        f"WHERE {_identifier(dataset.time_column)} >= ? "
        f"AND {_identifier(dataset.time_column)} < ? "
        f"ORDER BY {_identifier(dataset.time_column)}",
        [[str(path) for path in paths], start, end],
    ).df()
    _normalize_result_times(frame, columns)
    return frame
