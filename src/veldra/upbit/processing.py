"""Normalize and validate Upbit historical CSV archives with Arrow."""

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


def _epoch_milliseconds(values: Any, column: str) -> Any:
    """Convert an epoch-millisecond column to UTC microseconds."""
    numbers = _integer(values, column)
    if len(numbers):
        low, high = pc.min(numbers).as_py(), pc.max(numbers).as_py()
        if low is None or low < 100_000_000_000 or high >= 100_000_000_000_000:
            raise DataValidationError(f"invalid timestamp unit for {column}")
        numbers = pc.multiply_checked(numbers, 1_000)
    return pc.cast(numbers, pa.timestamp("us", "UTC"))


def _utc_text(values: Any, column: str) -> Any:
    """Convert Upbit's timezone-free ISO text to UTC microseconds."""
    text = pc.utf8_trim_whitespace(pc.cast(values, pa.string()))
    try:
        parsed = pc.strptime(
            text,
            format="%Y-%m-%dT%H:%M:%S",
            unit="us",
            error_is_null=True,
        )
        result = pc.assume_timezone(parsed, "UTC")
    except pa.ArrowException as error:
        raise DataValidationError(f"invalid {column} value") from error
    if result.null_count:
        raise DataValidationError(f"invalid {column} value")
    return result


def _normalize_klines(table: Any, dataset: DatasetSpec) -> Any:
    """Convert Upbit candles into canonical OHLCV columns."""
    if tuple(table.column_names) != dataset.source_columns:
        raise DataValidationError("CSV does not match an Upbit Kline schema")
    mapping = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "base_volume": "acc_trade_volume",
        "quote_volume": "acc_trade_price",
    }
    values = {
        "open_time": _utc_text(table["date_time_utc"], "open_time"),
        **{
            target: _number(table[source], target) for target, source in mapping.items()
        },
    }
    return pa.table({column: values[column] for column in dataset.stored_columns})


def _trade_side(values: Any) -> Any:
    """Map Upbit BID and ASK aggressor labels to buy and sell."""
    source = pc.utf8_upper(pc.utf8_trim_whitespace(pc.cast(values, pa.string())))
    if source.null_count:
        raise DataValidationError("trade side must be BID or ASK")
    _reject(
        pc.invert(pc.is_in(source, value_set=pa.array(["BID", "ASK"]))),
        "trade side must be BID or ASK",
    )
    return pc.if_else(pc.equal(source, "BID"), "buy", "sell")


def _normalize_trades(table: Any, dataset: DatasetSpec) -> Any:
    """Convert Upbit tick trades into canonical event columns."""
    if tuple(table.column_names) != dataset.source_columns:
        raise DataValidationError("CSV does not match an Upbit trade schema")
    price = _number(table["price"], "price")
    base_quantity = _number(table["volume"], "base_quantity")
    values = {
        "event_time": _epoch_milliseconds(table["timestamp"], "event_time"),
        "event_number": _integer(table["seq"], "event_number"),
        "price": price,
        "base_quantity": base_quantity,
        "quote_quantity": pc.multiply(price, base_quantity),
        "side": _trade_side(table["ask_bid"]),
    }
    return pa.table({column: values[column] for column in dataset.stored_columns})


def normalize_chunk(
    table: Any, dataset: DatasetSpec, contract_size: float | None = None
) -> Any:
    """Convert one supported Upbit source table into canonical columns.

    Args:
        table: The raw Arrow source table.
        dataset: The Upbit schema declaration.
        contract_size: Unused market context accepted by the shared ingest hook.

    Returns:
        A canonical Arrow table ready for validation.
    """
    del contract_size
    if dataset.product == "spot" and dataset.name == "klines":
        return _normalize_klines(table, dataset)
    if dataset.product == "spot" and dataset.name == "trades":
        return _normalize_trades(table, dataset)
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
    """Validate UTC-day bounds, ordering, and physical Kline alignment."""
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
        raise DataValidationError("timestamps fall outside the Upbit resource day")
    if dataset.supports_resampling:
        unit = "second" if dataset.base_interval == "1s" else "minute"
        _reject(
            pc.not_equal(values, pc.floor_temporal(values, unit=unit)),
            f"open_time is not aligned to one {unit}",
        )
    return last


def _validate_ohlc(table: Any, dataset: DatasetSpec) -> None:
    """Validate positive prices and nonnegative candle quantities."""
    for column in ("open", "high", "low", "close"):
        _reject(pc.less_equal(table[column], 0), "price values must be positive")
    for column in ("open", "low", "close"):
        _reject(pc.less(table["high"], table[column]), "high is below an OHLC price")
    for column in ("open", "high", "close"):
        _reject(pc.greater(table["low"], table[column]), "low is above an OHLC price")
    for column in dataset.resample_sum_columns:
        _reject(pc.less(table[column], 0), "volume values must be nonnegative")


def _validate_trades(table: Any) -> None:
    """Validate Upbit trade identifiers, prices, quantities, and sides."""
    _reject(pc.less(table["event_number"], 0), "event_number must be nonnegative")
    _reject(pc.less_equal(table["price"], 0), "trade price must be positive")
    for column in ("base_quantity", "quote_quantity"):
        _reject(pc.less(table[column], 0), "trade quantities must be nonnegative")
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
    """Validate canonical Upbit rows and return the final timestamp.

    Args:
        table: The canonical Arrow table to validate.
        dataset: Its Upbit schema declaration.
        day: The first archive day represented by the resource.
        previous_timestamp: The preceding chunk's final timestamp.
        end_day: The optional final archive day represented by the resource.

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
