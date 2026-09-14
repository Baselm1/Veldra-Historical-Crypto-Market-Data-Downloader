"""Normalize and validate Gate historical market rows with Arrow."""

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
    _reject(
        pc.invert(pc.match_substring_regex(text, r"^[+-]?[0-9]+$")),
        f"invalid integer {column} value",
    )
    try:
        return pc.cast(text, pa.int64())
    except pa.ArrowException as error:
        raise DataValidationError(f"invalid integer {column} value") from error


def _epoch_seconds(values: Any, column: str) -> Any:
    """Convert integer or microsecond-fraction epoch seconds exactly to UTC."""
    text = pc.utf8_trim_whitespace(pc.cast(values, pa.string()))
    parts = pc.extract_regex(
        text,
        r"^(?P<seconds>[0-9]{9,11})(?:\.(?P<fraction>[0-9]{1,6}))?$",
    )
    seconds_text = pc.struct_field(parts, "seconds")
    if seconds_text.null_count:
        raise DataValidationError(f"invalid timestamp unit for {column}")
    seconds = _integer(seconds_text, column)
    if len(seconds):
        low, high = pc.min(seconds).as_py(), pc.max(seconds).as_py()
        if low is None or low < 100_000_000 or high >= 100_000_000_000:
            raise DataValidationError(f"invalid timestamp unit for {column}")
    fraction = pc.fill_null(pc.struct_field(parts, "fraction"), "")
    micros = pc.cast(pc.utf8_rpad(fraction, 6, "0"), pa.int64())
    epoch_micros = pc.add_checked(pc.multiply_checked(seconds, 1_000_000), micros)
    return pc.cast(epoch_micros, pa.timestamp("us", "UTC"))


def _spot_side(values: Any) -> Any:
    """Map Gate's numeric Spot aggressor side to canonical text."""
    side = _integer(values, "side")
    valid = pc.is_in(side, value_set=pa.array([1, 2], type=pa.int64()))
    _reject(pc.invert(valid), "trade side must be 1 or 2")
    return pc.if_else(pc.equal(side, 2), "buy", "sell")


def _normalize_spot_klines(table: Any, dataset: DatasetSpec) -> Any:
    """Convert Gate Spot Klines into canonical OHLCV columns."""
    if tuple(table.column_names) != dataset.source_columns:
        raise DataValidationError("CSV does not match a Gate Spot Kline schema")
    values = {
        "open_time": _epoch_seconds(table["timestamp"], "open_time"),
        "open": _number(table["open"], "open"),
        "high": _number(table["high"], "high"),
        "low": _number(table["low"], "low"),
        "close": _number(table["close"], "close"),
        "base_volume": _number(table["volume"], "base_volume"),
    }
    return pa.table({column: values[column] for column in dataset.stored_columns})


def _normalize_spot_trades(table: Any, dataset: DatasetSpec) -> Any:
    """Convert Gate Spot trades into canonical price and quantity columns."""
    if tuple(table.column_names) != dataset.source_columns:
        raise DataValidationError("CSV does not match a Gate Spot trade schema")
    price = _number(table["price"], "price")
    base = _number(table["amount"], "base_quantity")
    values = {
        "event_time": _epoch_seconds(table["timestamp"], "event_time"),
        "event_number": _integer(table["deal_id"], "event_number"),
        "price": price,
        "base_quantity": base,
        "quote_quantity": pc.multiply(price, base),
        "side": _spot_side(table["side"]),
    }
    return pa.table({column: values[column] for column in dataset.stored_columns})


def normalize_chunk(
    table: Any, dataset: DatasetSpec, contract_size: float | None = None
) -> Any:
    """Convert one supported Gate source table into canonical columns.

    Args:
        table: The raw Arrow source table.
        dataset: The Gate schema declaration.
        contract_size: Reserved Futures contract context.

    Returns:
        A canonical Arrow table ready for validation.
    """
    del contract_size
    if dataset.product == "spot" and dataset.name == "klines":
        return _normalize_spot_klines(table, dataset)
    if dataset.product == "spot" and dataset.name == "trades":
        return _normalize_spot_trades(table, dataset)
    raise ValueError(f"unsupported normalizer: {dataset.product}/{dataset.name}")


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
    """Reject null, non-finite, or incorrectly typed canonical values."""
    for column in dataset.stored_columns:
        if column in dataset.timestamp_columns:
            continue
        values = table[column]
        if column in dataset.string_columns:
            if values.null_count or not pa.types.is_string(values.type):
                raise DataValidationError("text values must be nonnull strings")
        else:
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
    """Validate source coverage, ordering, and Kline alignment."""
    values = table[dataset.time_column]
    compare = pc.less_equal if dataset.supports_resampling else pc.less
    if pc.any(compare(values.slice(1), values.slice(0, len(values) - 1))).as_py():
        raise DataValidationError(f"{dataset.time_column} must be increasing")
    first = cast(datetime, values[0].as_py())
    last = cast(datetime, values[-1].as_py())
    if previous_timestamp is not None and (
        first < previous_timestamp
        or (dataset.supports_resampling and first == previous_timestamp)
    ):
        raise DataValidationError("chunk does not follow the preceding chunk")
    start = datetime.combine(day, time.min, UTC)
    end = datetime.combine((end_day or day) + timedelta(days=1), time.min, UTC)
    if first < start or last >= end:
        raise DataValidationError("timestamps fall outside the Gate resource period")
    if dataset.supports_resampling:
        unit = "second" if dataset.base_interval == "10s" else "minute"
        _reject(
            pc.not_equal(values, pc.floor_temporal(values, unit=unit)),
            f"open_time is not aligned to one {unit}",
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


def _validate_trades(table: Any) -> None:
    """Validate Gate Spot trade identifiers, prices, quantities, and sides."""
    _reject(pc.less(table["event_number"], 0), "event_number must be nonnegative")
    _reject(pc.less_equal(table["price"], 0), "trade price must be positive")
    for column in table.column_names:
        if column.endswith("quantity"):
            _reject(
                pc.less_equal(table[column], 0), "trade quantities must be positive"
            )
    _reject(
        pc.invert(pc.is_in(table["side"], value_set=pa.array(["buy", "sell"]))),
        "trade side must be buy or sell",
    )


def validate_chunk(
    table: Any,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: datetime | None = None,
    end_day: date | None = None,
) -> datetime:
    """Validate canonical Gate rows and return the final timestamp.

    Args:
        table: The canonical Arrow table to validate.
        dataset: Its Gate schema declaration.
        day: The first archive calendar day.
        previous_timestamp: The preceding chunk's final timestamp.
        end_day: The archive's optional inclusive final day.

    Returns:
        The final UTC timestamp in the table.
    """
    if dataset.product != "spot" or dataset.name not in {"klines", "trades"}:
        raise ValueError(f"unsupported validator: {dataset.product}/{dataset.name}")
    _validate_schema(table, dataset)
    _validate_values(table, dataset)
    last = _validate_times(table, dataset, day, previous_timestamp, end_day)
    if dataset.name == "klines":
        _validate_ohlc(table, dataset)
    else:
        _validate_trades(table)
    return last
