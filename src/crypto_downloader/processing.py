"""Normalize and validate tabular source data."""

from datetime import date
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import DatasetSpec

LOGGER = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Report malformed or inconsistent source rows."""


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of one file.

    Args:
        path: The file whose bytes should be hashed.

    Returns:
        The lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(values: pd.Series, column: str) -> pd.Series:
    """Convert one source column into finite floating-point numbers.

    Args:
        values: The source strings to convert.
        column: The source column name used in errors.

    Returns:
        The converted floating-point values.
    """
    nonfinite_literals = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"})
    )
    if nonfinite_literals.any():
        raise DataValidationError(f"{column} values must be finite")
    try:
        result = pd.to_numeric(values, errors="raise").astype("float64")
    except (TypeError, ValueError) as error:
        raise DataValidationError(f"invalid {column} value") from error
    if not np.isfinite(result.to_numpy()).all():
        raise DataValidationError(f"{column} values must be finite")
    return result


def _integer(values: pd.Series, column: str) -> pd.Series:
    """Convert one source column into exact signed integers.

    Args:
        values: The source strings to convert.
        column: The source column name used in errors.

    Returns:
        The converted integer values.
    """
    numbers = _number(values, column)
    if not np.equal(numbers, np.floor(numbers)).all():
        raise DataValidationError(f"invalid integer {column} value")
    try:
        return numbers.astype("int64")
    except (TypeError, ValueError, OverflowError) as error:
        raise DataValidationError(f"invalid integer {column} value") from error


def _epoch(values: pd.Series, column: str) -> pd.Series:
    """Convert a consistently scaled millisecond or microsecond epoch column.

    Args:
        values: The source epoch strings to convert.
        column: The timestamp column name used in errors.

    Returns:
        UTC-aware pandas timestamps.
    """
    integers = _integer(values, column)
    magnitudes = integers.abs()
    milliseconds = magnitudes.between(100_000_000_000, 99_999_999_999_999)
    microseconds = magnitudes.between(100_000_000_000_000, 99_999_999_999_999_999)
    if milliseconds.all():
        unit = "ms"
    elif microseconds.all():
        unit = "us"
    else:
        raise DataValidationError(f"invalid or mixed timestamp unit for {column}")
    try:
        return pd.to_datetime(integers, unit=unit, utc=True, errors="raise")
    except (TypeError, ValueError, OverflowError) as error:
        raise DataValidationError(f"invalid {column} value") from error


def normalize_chunk(frame: pd.DataFrame, dataset: DatasetSpec) -> pd.DataFrame:
    """Convert one source CSV chunk into its canonical stored schema.

    Args:
        frame: The raw source rows with source column names.
        dataset: The schema describing the source and stored columns.

    Returns:
        A new DataFrame containing normalized values and column names.
    """
    if tuple(frame.columns) != dataset.source_columns:
        raise DataValidationError("CSV does not match the expected source columns")
    if (dataset.product, dataset.name) != ("spot", "klines"):
        raise ValueError(f"unsupported normalizer: {dataset.product}/{dataset.name}")

    result = pd.DataFrame(index=frame.index)
    result["open_time"] = _epoch(frame["open_time"], "open_time")
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = _number(frame[column], column)
    result["close_time"] = _epoch(frame["close_time"], "close_time")
    result["quote_volume"] = _number(frame["quote_volume"], "quote_volume")
    result["trade_count"] = _integer(frame["count"], "count")
    result["taker_buy_base_volume"] = _number(
        frame["taker_buy_volume"], "taker_buy_volume"
    )
    result["taker_buy_quote_volume"] = _number(
        frame["taker_buy_quote_volume"], "taker_buy_quote_volume"
    )
    normalized = result.loc[:, dataset.stored_columns]
    LOGGER.debug(
        "Source chunk normalized: product=%s dataset=%s rows=%d columns=%d",
        dataset.product,
        dataset.name,
        len(normalized),
        len(normalized.columns),
    )
    return normalized


def _validate_timestamps(
    frame: pd.DataFrame,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: pd.Timestamp | None,
) -> pd.Timestamp:
    """Validate timestamp types, ordering, bounds, and candle duration.

    Args:
        frame: The canonical rows containing timestamp columns.
        dataset: The schema naming the primary timestamp column.
        day: The UTC source day that must contain every row.
        previous_timestamp: The final timestamp from the preceding chunk.

    Returns:
        The final timestamp in the validated chunk.
    """
    open_time = frame[dataset.time_column]
    close_time = frame["close_time"]
    _validate_timestamp_types(open_time, close_time)
    last = _validate_open_times(open_time, day, previous_timestamp)
    _validate_close_times(open_time, close_time)
    return last


def _validate_timestamp_types(open_time: pd.Series, close_time: pd.Series) -> None:
    """Validate that candle timestamps are present and UTC-aware.

    Args:
        open_time: The candle opening timestamps.
        close_time: The candle closing timestamps.
    """
    if open_time.isna().any() or close_time.isna().any():
        raise DataValidationError("timestamps cannot be null")
    _require_utc(open_time, "open_time")
    _require_utc(close_time, "close_time")


def _validate_open_times(
    open_time: pd.Series,
    day: date,
    previous_timestamp: pd.Timestamp | None,
) -> pd.Timestamp:
    """Validate opening timestamp order, daily bounds, and alignment.

    Args:
        open_time: The candle opening timestamps.
        day: The UTC source day that must contain every row.
        previous_timestamp: The final timestamp from the preceding chunk.

    Returns:
        The final timestamp in the validated chunk.
    """
    if not open_time.is_monotonic_increasing or open_time.duplicated().any():
        raise DataValidationError("open_time must be strictly increasing")
    first = open_time.iloc[0]
    last = open_time.iloc[-1]
    if previous_timestamp is not None and first <= previous_timestamp:
        raise DataValidationError("chunk does not follow the preceding chunk")
    day_start = pd.Timestamp(day, tz="UTC")
    if first < day_start or last >= day_start + pd.Timedelta(days=1):
        raise DataValidationError("timestamps fall outside the resource day")
    if (
        (open_time.dt.second != 0)
        | (open_time.dt.microsecond != 0)
        | (open_time.dt.nanosecond != 0)
    ).any():
        raise DataValidationError("open_time is not aligned to one minute")
    return last


def _validate_close_times(open_time: pd.Series, close_time: pd.Series) -> None:
    """Validate that each close timestamp belongs to its candle.

    Args:
        open_time: The candle opening timestamps.
        close_time: The candle closing timestamps.
    """
    if (
        (close_time < open_time) | (close_time >= open_time + pd.Timedelta(minutes=1))
    ).any():
        raise DataValidationError("close_time falls outside its candle")


def _require_utc(values: pd.Series, column: str) -> None:
    """Require one pandas timestamp series to carry the UTC timezone.

    Args:
        values: The timestamp values to inspect.
        column: The column name used in errors.
    """
    if not isinstance(values.dtype, pd.DatetimeTZDtype) or str(values.dt.tz) != "UTC":
        raise DataValidationError(f"{column} must use UTC timestamps")


def _validate_numbers(frame: pd.DataFrame, dataset: DatasetSpec) -> None:
    """Validate finite and nonnegative canonical numeric values.

    Args:
        frame: The canonical rows containing numeric columns.
        dataset: The schema declaring all stored columns.
    """
    numeric_columns = [
        column
        for column in dataset.stored_columns
        if column not in {"open_time", "close_time"}
    ]
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype="float64")).all():
        raise DataValidationError("numeric values must be finite")
    nonnegative = frame[
        [
            "volume",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
    ]
    if (nonnegative < 0).any().any():
        raise DataValidationError("volume and count values must be nonnegative")


def _validate_ohlc(frame: pd.DataFrame) -> None:
    """Validate positive and internally consistent OHLC prices.

    Args:
        frame: The canonical rows containing OHLC price columns.
    """
    prices = frame[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise DataValidationError("price values must be greater than zero")
    if (frame["high"] < prices[["open", "low", "close"]].max(axis=1)).any():
        raise DataValidationError("high is below another OHLC price")
    if (frame["low"] > prices[["open", "high", "close"]].min(axis=1)).any():
        raise DataValidationError("low is above another OHLC price")


def validate_chunk(
    frame: pd.DataFrame,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Validate one canonical daily data chunk.

    Args:
        frame: The normalized rows to validate.
        dataset: The schema describing the canonical rows.
        day: The UTC source day that must contain every row.
        previous_timestamp: The final timestamp from the preceding chunk.

    Returns:
        The final timestamp in the validated chunk.
    """
    if tuple(frame.columns) != dataset.stored_columns:
        raise DataValidationError("chunk does not match the expected stored columns")
    if frame.empty:
        raise DataValidationError("chunk cannot be empty")
    if (dataset.product, dataset.name) != ("spot", "klines"):
        raise ValueError(f"unsupported validator: {dataset.product}/{dataset.name}")

    last = _validate_timestamps(frame, dataset, day, previous_timestamp)
    _validate_numbers(frame, dataset)
    _validate_ohlc(frame)
    LOGGER.debug(
        "Data chunk validated: product=%s dataset=%s day=%s rows=%d last=%s",
        dataset.product,
        dataset.name,
        day,
        len(frame),
        last,
    )
    return last
