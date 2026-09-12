"""Normalize and validate KuCoin CSV archives with Arrow."""

from datetime import UTC, date, datetime, time, timedelta
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc

from veldra.core.datasets import DatasetSpec
from veldra.core.models import DataValidationError


def _reject(condition: Any, message: str) -> None:
    """Raise a source-data error when any Arrow condition is true."""
    if pc.any(condition).as_py():
        raise DataValidationError(message)


def _number(values: Any, column: str) -> Any:
    """Convert one source column to finite nonnull doubles."""
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
    """Convert one exact source integer column to signed 64-bit values."""
    text = pc.utf8_trim_whitespace(pc.cast(values, pa.string()))
    if text.null_count:
        raise DataValidationError(f"invalid integer {column} value")
    valid = pc.match_substring_regex(text, r"^[+-]?[0-9]+$")
    _reject(pc.invert(valid), f"invalid integer {column} value")
    try:
        return pc.cast(text, pa.int64())
    except pa.ArrowException as error:
        raise DataValidationError(f"invalid integer {column} value") from error


def _epoch(values: Any, column: str, unit: str) -> Any:
    """Convert consistently scaled epoch integers to UTC microseconds."""
    numbers = _integer(values, column)
    if len(numbers):
        low, high = pc.min(numbers).as_py(), pc.max(numbers).as_py()
        limits = {
            "s": (100_000_000, 100_000_000_000, 1_000_000),
            "ms": (100_000_000_000, 100_000_000_000_000, 1_000),
        }
        minimum, maximum, scale = limits[unit]
        if low is None or low < minimum or high >= maximum:
            raise DataValidationError(f"invalid timestamp unit for {column}")
        numbers = pc.multiply_checked(numbers, scale)
    return pc.cast(numbers, pa.timestamp("us", "UTC"))


def _normalize_spot_klines(table: Any, dataset: DatasetSpec) -> Any:
    """Convert KuCoin Spot Klines into canonical OHLCV columns."""
    if tuple(table.column_names) != dataset.source_columns:
        raise DataValidationError("CSV does not match a KuCoin Spot Kline schema")
    mapping = {
        "open_time": "time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "base_volume": "volume",
        "quote_volume": "turnover",
    }
    values = {
        target: (
            _epoch(table[source], target, "s")
            if target == "open_time"
            else _number(table[source], target)
        )
        for target, source in mapping.items()
    }
    return pa.table({column: values[column] for column in dataset.stored_columns})


def normalize_chunk(
    table: Any, dataset: DatasetSpec, contract_size: float | None = None
) -> Any:
    """Convert one supported KuCoin source table into canonical columns.

    Args:
        table: The raw Arrow source table.
        dataset: The KuCoin schema declaration.
        contract_size: Reserved market context for Futures trade normalization.

    Returns:
        A canonical Arrow table ready for validation.
    """
    if dataset.product == "spot" and dataset.name == "klines":
        return _normalize_spot_klines(table, dataset)
    raise ValueError(f"unsupported normalizer: {dataset.product}/{dataset.name}")


def _ordered(values: Any, *, strict: bool) -> bool:
    """Return whether Arrow values increase in the required order."""
    compare = pc.less_equal if strict else pc.less
    return not pc.any(
        compare(values.slice(1), values.slice(0, len(values) - 1))
    ).as_py()


def _validate_schema(table: Any, dataset: DatasetSpec) -> None:
    """Validate canonical columns and the primary timestamp type."""
    if tuple(table.column_names) != dataset.stored_columns or not table.num_rows:
        raise DataValidationError("chunk does not match the expected stored columns")
    values = table[dataset.time_column]
    if (
        values.null_count
        or not pa.types.is_timestamp(values.type)
        or values.type.tz != "UTC"
    ):
        raise DataValidationError(f"{dataset.time_column} must contain UTC timestamps")


def _validate_values(table: Any, dataset: DatasetSpec) -> None:
    """Reject null or non-finite canonical numeric values."""
    for column in dataset.stored_columns:
        if column in dataset.timestamp_columns:
            continue
        values = table[column]
        if values.null_count:
            raise DataValidationError("numeric values must be finite")
        _reject(pc.invert(pc.is_finite(values)), "numeric values must be finite")


def _validate_times(
    table: Any,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: datetime | None,
    end_day: date | None,
) -> datetime:
    """Validate UTC-day bounds, ordering, and one-minute Kline alignment."""
    values = table[dataset.time_column]
    if not _ordered(values, strict=dataset.supports_resampling):
        raise DataValidationError(f"{dataset.time_column} must be increasing")
    first = cast(datetime, values[0].as_py())
    last = cast(datetime, values[-1].as_py())
    if previous_timestamp is not None and first <= previous_timestamp:
        raise DataValidationError("chunk does not follow the preceding chunk")
    start = datetime.combine(day, time.min, UTC)
    end = datetime.combine((end_day or day) + timedelta(days=1), time.min, UTC)
    if first < start or last >= end:
        raise DataValidationError("timestamps fall outside the KuCoin resource day")
    _reject(
        pc.not_equal(values, pc.floor_temporal(values, unit="minute")),
        "open_time is not aligned to one minute",
    )
    return last


def _validate_ohlc(table: Any, dataset: DatasetSpec) -> None:
    """Validate positive prices and nonnegative declared Kline quantities."""
    for column in ("open", "high", "low", "close"):
        _reject(pc.less_equal(table[column], 0), "price values must be positive")
    for column in ("open", "low", "close"):
        _reject(pc.less(table["high"], table[column]), "high is below an OHLC price")
    for column in ("open", "high", "close"):
        _reject(pc.greater(table["low"], table[column]), "low is above an OHLC price")
    for column in dataset.resample_sum_columns:
        _reject(pc.less(table[column], 0), "volume values must be nonnegative")


def validate_chunk(
    table: Any,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: datetime | None = None,
    end_day: date | None = None,
) -> datetime:
    """Validate canonical KuCoin data and return its final timestamp.

    Args:
        table: The canonical Arrow table to validate.
        dataset: Its KuCoin schema declaration.
        day: The first archive day represented by the physical resource.
        previous_timestamp: The preceding chunk's final timestamp.
        end_day: The optional final archive day represented by the resource.

    Returns:
        The final UTC timestamp in the table.
    """
    if dataset.product != "spot" or dataset.name != "klines":
        raise ValueError(f"unsupported validator: {dataset.product}/{dataset.name}")
    _validate_schema(table, dataset)
    _validate_values(table, dataset)
    last = _validate_times(table, dataset, day, previous_timestamp, end_day)
    _validate_ohlc(table, dataset)
    return last
