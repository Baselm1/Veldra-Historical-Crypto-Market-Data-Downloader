"""Normalize and validate Bitget archive rows with Arrow."""

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
    """Convert one exact integer column to signed 64-bit values."""
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
    """Convert epoch seconds to UTC microsecond timestamps."""
    seconds = _integer(values, column)
    if len(seconds):
        low, high = pc.min(seconds).as_py(), pc.max(seconds).as_py()
        if low is None or low < 100_000_000 or high >= 100_000_000_000:
            raise DataValidationError(f"invalid timestamp unit for {column}")
    return pc.cast(pc.multiply_checked(seconds, 1_000_000), pa.timestamp("us", "UTC"))


def _epoch_milliseconds(values: Any, column: str) -> Any:
    """Convert epoch milliseconds to UTC microsecond timestamps."""
    milliseconds = _integer(values, column)
    if len(milliseconds):
        low, high = pc.min(milliseconds).as_py(), pc.max(milliseconds).as_py()
        if low is None or low < 100_000_000_000 or high >= 100_000_000_000_000:
            raise DataValidationError(f"invalid timestamp unit for {column}")
    return pc.cast(pc.multiply_checked(milliseconds, 1_000), pa.timestamp("us", "UTC"))


def _source_columns(table: Any, dataset: DatasetSpec) -> set[str]:
    """Return visible source columns after validating the hidden row number."""
    names = set(table.column_names)
    if "__row_number" not in names:
        raise DataValidationError("Bitget workbook has no physical row number")
    visible = names - {"__row_number"}
    accepted = {frozenset(schema.columns) for schema in dataset.csv_schemas}
    if frozenset(visible) not in accepted:
        raise DataValidationError("workbook does not match a Bitget schema")
    return visible


def _normalize_klines(table: Any, dataset: DatasetSpec) -> Any:
    """Convert Bitget candles into canonical OHLCV columns."""
    visible = _source_columns(table, dataset)
    base_name = "basevolume" if "basevolume" in visible else "baseVolume"
    quote_name = "usdtvolume" if "usdtvolume" in visible else "usdtVolume"
    quantity = (
        "contract_volume" if dataset.product == "coin_futures" else "quote_volume"
    )
    values = {
        "open_time": _epoch_seconds(table["timestamp"], "open_time"),
        "open": _number(table["open"], "open"),
        "high": _number(table["high"], "high"),
        "low": _number(table["low"], "low"),
        "close": _number(table["close"], "close"),
        "base_volume": _number(table[base_name], "base_volume"),
        quantity: _number(table[quote_name], quantity),
    }
    return pa.table({column: values[column] for column in dataset.stored_columns})


def _side(values: Any) -> Any:
    """Normalize the Bitget aggressor side to lowercase text."""
    result = pc.utf8_lower(pc.utf8_trim_whitespace(pc.cast(values, pa.string())))
    if result.null_count:
        raise DataValidationError("trade side must be buy or sell")
    _reject(
        pc.invert(pc.is_in(result, value_set=pa.array(["buy", "sell"]))),
        "trade side must be buy or sell",
    )
    return result


def _normalize_trades(table: Any, dataset: DatasetSpec) -> Any:
    """Convert Bitget fills into canonical price and quantity columns."""
    _source_columns(table, dataset)
    values = {
        "event_time": _epoch_milliseconds(table["timestamp"], "event_time"),
        "event_number": _integer(table["trade_id"], "event_number"),
        "price": _number(table["price"], "price"),
        "base_quantity": _number(table["size(base)"], "base_quantity"),
        "quote_quantity": _number(table["volume(quote)"], "quote_quantity"),
        "side": _side(table["side"]),
    }
    return pa.table({column: values[column] for column in dataset.stored_columns})


def normalize_chunk(
    table: Any, dataset: DatasetSpec, contract_size: float | None = None
) -> Any:
    """Convert one supported Bitget source table into canonical columns.

    Args:
        table: Raw Arrow table decoded from one worksheet.
        dataset: Bitget schema declaration.
        contract_size: Reserved contract metadata.

    Returns:
        Canonical Arrow table ready for validation.
    """
    del contract_size
    if (
        dataset.product in {"spot", "usdt_futures", "usdc_futures", "coin_futures"}
        and dataset.name == "klines"
    ):
        return _normalize_klines(table, dataset)
    if (
        dataset.product in {"spot", "usdt_futures", "usdc_futures", "coin_futures"}
        and dataset.name == "trades"
    ):
        return _normalize_trades(table, dataset)
    raise ValueError(f"unsupported normalizer: {dataset.product}/{dataset.name}")


def _validate_schema(table: Any, dataset: DatasetSpec) -> None:
    """Validate canonical columns and the UTC timestamp type."""
    if tuple(table.column_names) != dataset.stored_columns or not table.num_rows:
        raise DataValidationError("chunk does not match the expected stored columns")
    values = table[dataset.time_column]
    if (
        values.null_count
        or not pa.types.is_timestamp(values.type)
        or values.type.tz != "UTC"
    ):
        raise DataValidationError("open_time must contain UTC timestamps")


def _validate_times(
    table: Any,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: datetime | None,
    end_day: date | None,
) -> datetime:
    """Validate UTC+8 coverage, uniqueness, ordering, and minute alignment."""
    values = table[dataset.time_column]
    comparison = pc.less_equal if dataset.name == "klines" else pc.less
    if pc.any(comparison(values.slice(1), values.slice(0, len(values) - 1))).as_py():
        qualifier = "strictly " if dataset.name == "klines" else ""
        raise DataValidationError(
            f"{dataset.time_column} must be {qualifier}increasing"
        )
    first = cast(datetime, values[0].as_py())
    last = cast(datetime, values[-1].as_py())
    if previous_timestamp is not None and (
        first < previous_timestamp
        or (dataset.name == "klines" and first == previous_timestamp)
    ):
        raise DataValidationError("chunk does not follow the preceding chunk")
    start = datetime.combine(day, time.min, UTC) - dataset.archive_day_offset
    end = (
        datetime.combine((end_day or day) + timedelta(days=1), time.min, UTC)
        - dataset.archive_day_offset
    )
    if first < start or last >= end:
        raise DataValidationError("timestamps fall outside the Bitget resource period")
    if dataset.name == "klines":
        _reject(
            pc.not_equal(values, pc.floor_temporal(values, multiple=1, unit="minute")),
            "open_time is not aligned to 1m",
        )
    return last


def _validate_values(table: Any) -> None:
    """Validate Kline price and quantity invariants."""
    for column in table.column_names[1:]:
        values = table[column]
        if values.null_count:
            raise DataValidationError("numeric values must be finite")
        _reject(pc.invert(pc.is_finite(values)), "numeric values must be finite")
    for column in ("open", "high", "low", "close"):
        _reject(pc.less_equal(table[column], 0), "price values must be positive")
    for column in ("open", "low", "close"):
        _reject(pc.less(table["high"], table[column]), "high is below an OHLC price")
    for column in ("open", "high", "close"):
        _reject(pc.greater(table["low"], table[column]), "low is above an OHLC price")
    for column in table.column_names:
        if column.endswith("volume"):
            _reject(pc.less(table[column], 0), "volume values must be nonnegative")


def _validate_trades(table: Any) -> None:
    """Validate trade identifiers, prices, quantities, and sides."""
    _reject(pc.less(table["event_number"], 0), "event_number must be nonnegative")
    _reject(pc.less_equal(table["price"], 0), "trade price must be positive")
    for column in ("base_quantity", "quote_quantity"):
        _reject(pc.less_equal(table[column], 0), "trade quantities must be positive")
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
    """Validate canonical Bitget rows and return the final timestamp.

    Args:
        table: Canonical Arrow table.
        dataset: Bitget schema declaration.
        day: First UTC+8 archive day.
        previous_timestamp: Previous chunk's final timestamp.
        end_day: Optional inclusive last archive day.

    Returns:
        Final UTC timestamp in the table.
    """
    if dataset.name not in {"klines", "trades"} or dataset.product not in {
        "spot",
        "usdt_futures",
        "usdc_futures",
        "coin_futures",
    }:
        raise ValueError(f"unsupported validator: {dataset.product}/{dataset.name}")
    _validate_schema(table, dataset)
    if dataset.name == "klines":
        _validate_values(table)
    else:
        for column in dataset.stored_columns:
            if table[column].null_count:
                raise DataValidationError("trade values must be nonnull")
        _validate_trades(table)
    return _validate_times(table, dataset, day, previous_timestamp, end_day)
