"""Normalize and validate HTX CSV archives with Arrow."""

from datetime import UTC, date, datetime, time, timedelta
import math
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc

from crypto_downloader.core.datasets import DatasetSpec
from crypto_downloader.core.models import DataValidationError

ARCHIVE_DAY_OFFSET = timedelta(hours=8)


def _reject(condition: Any, message: str) -> None:
    """Raise a data error when any Arrow Boolean value is true.

    Args:
        condition: The Arrow Boolean array to reduce.
        message: The error message used when a value is true.
    """
    if pc.any(condition).as_py():
        raise DataValidationError(message)


def _number(values: Any, column: str) -> Any:
    """Convert one source column to finite doubles.

    Args:
        values: The Arrow source values.
        column: The canonical field named in errors.

    Returns:
        An Arrow double array.
    """
    text = pc.utf8_trim_whitespace(pc.cast(values, pa.string()))
    try:
        result = pc.cast(text, pa.float64())
    except pa.ArrowException as error:
        raise DataValidationError(f"invalid {column} value") from error
    if result.null_count:
        raise DataValidationError(f"{column} values must be finite")
    _reject(pc.invert(pc.is_finite(result)), f"{column} values must be finite")
    return result


def _integer(values: Any, column: str) -> Any:
    """Convert exact source integers without passing through floats.

    Args:
        values: The Arrow source values.
        column: The canonical field named in errors.

    Returns:
        An Arrow signed 64-bit integer array.
    """
    text = pc.utf8_trim_whitespace(pc.cast(values, pa.string()))
    valid = pc.match_substring_regex(text, r"^[+-]?[0-9]+$")
    if text.null_count:
        raise DataValidationError(f"invalid integer {column} value")
    _reject(pc.invert(valid), f"invalid integer {column} value")
    return pc.cast(text, pa.int64())


def _epoch_seconds(values: Any, column: str) -> Any:
    """Convert HTX epoch-second values to UTC microsecond timestamps.

    Args:
        values: The Arrow source values.
        column: The canonical field named in errors.

    Returns:
        An Arrow UTC timestamp array.
    """
    seconds = _integer(values, column)
    if len(seconds):
        low, high = pc.min(seconds).as_py(), pc.max(seconds).as_py()
        if low is None or low < 100_000_000 or high >= 100_000_000_000:
            raise DataValidationError(f"invalid timestamp unit for {column}")
    micros = pc.multiply_checked(seconds, 1_000_000)
    return pc.cast(micros, pa.timestamp("us", "UTC"))


def normalize_chunk(
    table: Any, dataset: DatasetSpec, contract_size: float | None = None
) -> Any:
    """Convert one HTX source table into canonical Kline columns.

    Args:
        table: The raw Arrow source table.
        dataset: The HTX schema declaration.
        contract_size: Unused source context reserved by shared ingestion.

    Returns:
        A canonical Arrow table ready for validation.
    """
    del contract_size
    if dataset.product != "spot" or dataset.name != "klines":
        raise ValueError(f"unsupported normalizer: {dataset.product}/{dataset.name}")
    names = set(table.column_names)
    if "id" in names:
        mapping = {
            "open_time": "id",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "base_volume": "amount",
            "quote_volume": "vol",
        }
    elif "ts" in names:
        mapping = {
            "open_time": "ts",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "base_volume": "vol",
            "quote_volume": "volCcyQuote",
        }
    else:
        raise DataValidationError("CSV does not match an HTX Kline schema")
    result = {
        target: (
            _epoch_seconds(table[source], target)
            if target == "open_time"
            else _number(table[source], target)
        )
        for target, source in mapping.items()
    }
    return pa.table({column: result[column] for column in dataset.stored_columns})


def _ordered(values: Any) -> bool:
    """Return whether Arrow timestamps are strictly increasing.

    Args:
        values: The timestamp array to inspect.

    Returns:
        Whether every timestamp follows the preceding timestamp.
    """
    return not pc.any(
        pc.less_equal(values.slice(1), values.slice(0, len(values) - 1))
    ).as_py()


def _coverage(day: date) -> tuple[datetime, datetime]:
    """Return exact UTC coverage for one HTX UTC+8 archive day.

    Args:
        day: The source calendar date printed in the filename.

    Returns:
        Inclusive UTC start and exclusive UTC end timestamps.
    """
    start = datetime.combine(day, time.min, UTC) - ARCHIVE_DAY_OFFSET
    return start, start + timedelta(days=1)


def _validate_types(table: Any, dataset: DatasetSpec) -> None:
    """Validate canonical Kline columns and Arrow types.

    Args:
        table: The canonical Arrow table.
        dataset: The schema defining exact stored columns.
    """
    if tuple(table.column_names) != dataset.stored_columns or not table.num_rows:
        raise DataValidationError("chunk does not match the expected stored columns")
    times = table[dataset.time_column]
    if (
        times.null_count
        or not pa.types.is_timestamp(times.type)
        or times.type.tz != "UTC"
    ):
        raise DataValidationError("open_time must contain UTC timestamps")
    for column in dataset.stored_columns:
        if column in dataset.timestamp_columns:
            continue
        values = table[column]
        _reject(pc.invert(pc.is_finite(values)), "numeric values must be finite")
        if values.null_count:
            raise DataValidationError("numeric values must be finite")


def _validate_times(
    table: Any,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: datetime | None,
    end_day: date | None,
) -> datetime:
    """Validate Kline ordering, source-day coverage, and minute alignment.

    Args:
        table: The canonical Arrow table.
        dataset: The schema identifying the primary timestamp.
        day: The first HTX source calendar day.
        previous_timestamp: The preceding chunk's final timestamp.
        end_day: The optional last HTX source calendar day.

    Returns:
        The table's final UTC timestamp.
    """
    times = table[dataset.time_column]
    if not _ordered(times):
        raise DataValidationError("open_time must be increasing")
    first = cast(datetime, times[0].as_py())
    last = cast(datetime, times[-1].as_py())
    if previous_timestamp is not None and first <= previous_timestamp:
        raise DataValidationError("chunk does not follow the preceding chunk")
    start, _ = _coverage(day)
    _, end = _coverage(end_day or day)
    if first < start or last >= end:
        raise DataValidationError("timestamps fall outside the HTX source day")
    _reject(
        pc.not_equal(times, pc.floor_temporal(times, unit="minute")),
        "open_time is not aligned to one minute",
    )
    return last


def _validate_ohlc(table: Any) -> None:
    """Validate positive OHLC values and nonnegative Spot volumes.

    Args:
        table: The canonical Kline table.
    """
    for column in ("open", "high", "low", "close"):
        _reject(pc.less_equal(table[column], 0), "price values must be positive")
    for column in ("open", "low", "close"):
        _reject(
            pc.less(table["high"], table[column]),
            "high is below another OHLC price",
        )
    for column in ("open", "high", "close"):
        _reject(
            pc.greater(table["low"], table[column]),
            "low is above another OHLC price",
        )
    for column in ("base_volume", "quote_volume"):
        _reject(pc.less(table[column], 0), "volume values must be nonnegative")


def validate_chunk(
    table: Any,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: datetime | None = None,
    end_day: date | None = None,
) -> datetime:
    """Validate canonical HTX Klines and return their final timestamp.

    Args:
        table: The canonical Arrow table to validate.
        dataset: Its HTX schema declaration.
        day: The first source calendar day in the physical archive.
        previous_timestamp: The last timestamp from a preceding batch.
        end_day: The optional last source calendar day in the archive.

    Returns:
        The final UTC timestamp in the validated table.
    """
    if dataset.product != "spot" or dataset.name != "klines":
        raise ValueError(f"unsupported validator: {dataset.product}/{dataset.name}")
    _validate_types(table, dataset)
    last = _validate_times(table, dataset, day, previous_timestamp, end_day)
    _validate_ohlc(table)
    return last
