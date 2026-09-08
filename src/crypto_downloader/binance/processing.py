"""Normalize Binance archive batches with Arrow and validate source values."""

from datetime import UTC, date, datetime, timedelta
import math
import logging
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc

from crypto_downloader.core.datasets import DatasetSpec


from crypto_downloader.core.models import DataValidationError

LOGGER = logging.getLogger(__name__)


def _reject(condition: Any, message: str) -> None:
    """Raise message if any value in an Arrow boolean array is true."""
    if pc.any(condition).as_py():
        raise DataValidationError(message)


def _number(values: Any, column: str, *, nullable: bool = False) -> Any:
    """Convert source values to finite doubles, retaining allowed nulls."""
    text = pc.utf8_trim_whitespace(pc.cast(values, pa.string()))
    if nullable:
        text = pc.if_else(pc.equal(text, ""), None, text)
    try:
        result = pc.cast(text, pa.float64())
    except pa.ArrowException as error:
        raise DataValidationError(f"invalid {column} value") from error
    _reject(pc.invert(pc.is_finite(result)), f"{column} values must be finite")
    if result.null_count and not nullable:
        raise DataValidationError(f"{column} values must be finite")
    return result


def _integer(values: Any, column: str) -> Any:
    """Parse exact signed integers without passing identifiers through floats."""
    text = pc.utf8_trim_whitespace(pc.cast(values, pa.string()))
    valid = pc.match_substring_regex(text, r"^[+-]?[0-9]+(?:\.0+)?$")
    _reject(pc.invert(valid), f"invalid integer {column} value")
    if text.null_count:
        raise DataValidationError(f"invalid integer {column} value")
    try:
        return pc.cast(pc.replace_substring_regex(text, r"\.0+$", ""), pa.int64())
    except pa.ArrowException as error:
        raise DataValidationError(f"invalid integer {column} value") from error


def _boolean(values: Any, column: str) -> Any:
    """Convert only true and false literals to Arrow booleans."""
    text = pc.utf8_lower(pc.utf8_trim_whitespace(pc.cast(values, pa.string())))
    _reject(
        pc.invert(pc.is_in(text, value_set=pa.array(["true", "false"]))),
        f"invalid boolean {column} value",
    )
    if text.null_count:
        raise DataValidationError(f"invalid boolean {column} value")
    return pc.equal(text, "true")


def _epoch(values: Any, column: str) -> Any:
    """Convert consistently scaled millisecond or microsecond epoch integers."""
    numbers = _integer(values, column)
    magnitude = pc.abs(numbers)
    lo, hi = pc.min(magnitude).as_py(), pc.max(magnitude).as_py()
    if lo is None:
        return pc.cast(numbers, pa.timestamp("us", "UTC"))
    if 100_000_000_000 <= lo <= hi < 100_000_000_000_000:
        numbers = pc.multiply_checked(numbers, 1000)
    elif not 100_000_000_000_000 <= lo <= hi < 100_000_000_000_000_000:
        raise DataValidationError(f"invalid or mixed timestamp unit for {column}")
    return pc.cast(numbers, pa.timestamp("us", "UTC"))


def _timestamp(values: Any, column: str) -> Any:
    """Parse Binance UTC text timestamps at microsecond resolution."""
    try:
        parsed = pc.strptime(values, format="%Y-%m-%d %H:%M:%S", unit="us")
        return pc.assume_timezone(parsed, "UTC")
    except pa.ArrowException as error:
        raise DataValidationError(f"invalid {column} value") from error


def _mapping(dataset: DatasetSpec) -> dict[str, str]:
    """Map canonical output names to Binance source columns for a dataset."""
    name, product = dataset.name, dataset.product
    if dataset.supports_resampling:
        columns = dict(zip(dataset.stored_columns, dataset.source_columns))
        if name != "klines":
            columns = {
                c: c
                for c in ("open_time", "open", "high", "low", "close", "close_time")
            }
            columns["sample_count"] = "count"
        return columns
    if name == "metrics":
        return dict(
            zip(dataset.stored_columns, ("create_time", *dataset.source_columns[2:]))
        )
    if name == "book_depth":
        return dict(zip(dataset.stored_columns, dataset.source_columns))
    columns = {c: c for c in dataset.stored_columns if c in dataset.source_columns}
    columns["buyer_is_maker"] = "is_buyer_maker"
    if product != "spot":
        columns["event_time"] = "time" if name == "trades" else "transact_time"
        columns["contract_quantity" if product == "cm" else "base_quantity"] = (
            "qty" if name == "trades" else "quantity"
        )
        if name == "trades":
            columns["trade_id"] = "id"
            columns["base_quantity" if product == "cm" else "quote_quantity"] = (
                "base_qty" if product == "cm" else "quote_qty"
            )
    return columns


def normalize_chunk(
    table: Any, dataset: DatasetSpec, contract_size: float | None = None
) -> Any:
    """Return canonical Arrow columns from a Binance source table and schema.

    Args:
        table: Raw Arrow table with source fields in their declared order.
        dataset: Binance column, unit and capability declaration.
        contract_size: Required USD face value for a COIN-M trade contract.

    Returns:
        An Arrow table with normalized columns and UTC timestamps.
    """
    if tuple(table.column_names) != dataset.source_columns:
        raise DataValidationError("CSV does not match the expected source columns")
    _supported(dataset, "normalizer")
    result: dict[str, Any] = {}
    for target, source in _mapping(dataset).items():
        values = table[source]
        if target in dataset.timestamp_columns:
            converter = (
                _timestamp if dataset.name in {"metrics", "book_depth"} else _epoch
            )
        elif target in dataset.integer_columns:
            converter = _integer
        elif target in dataset.boolean_columns:
            converter = _boolean
        else:
            converter = _number
        if target.endswith("ratio"):
            result[target] = _number(values, source, nullable=True)
        else:
            result[target] = converter(values, source)
    if dataset.supports_resampling and dataset.name != "klines":
        for column in (
            "volume",
            "quote_volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ):
            _reject(
                pc.not_equal(_number(table[column], column), 0),
                f"mark-price structural field '{column}' is not zero",
            )
    _derive_quantities(result, dataset, contract_size)
    if dataset.supports_resampling:
        _repair_close_times(result)
    return pa.table({column: result[column] for column in dataset.stored_columns})


def _repair_close_times(columns: dict[str, Any]) -> None:
    """Repair out-of-candle source close times while retaining valid source precision."""
    opens, closes = columns["open_time"], columns["close_time"]
    stop = pc.add(opens, pa.scalar(timedelta(minutes=1), pa.duration("us")))
    invalid = pc.or_(pc.less(closes, opens), pc.greater_equal(closes, stop))
    count = pc.sum(pc.cast(invalid, pa.int64())).as_py()
    if count:
        canonical = pc.subtract(
            stop, pa.scalar(timedelta(microseconds=1), pa.duration("us"))
        )
        columns["close_time"] = pc.if_else(invalid, canonical, closes)
        LOGGER.warning(
            "Repaired %d source close_time value(s) outside their one-minute candles",
            count,
        )


def _derive_quantities(
    result: dict[str, Any], dataset: DatasetSpec, contract_size: float | None
) -> None:
    """Add declared quote/base quantities to normalized event columns."""
    if dataset.requires_contract_size:
        if isinstance(contract_size, bool) or not isinstance(
            contract_size, (int, float)
        ):
            raise DataValidationError("COIN-M contract size is unavailable")
        if not math.isfinite(contract_size) or contract_size <= 0:
            raise DataValidationError(
                "COIN-M contract size must be positive and finite"
            )
        result["quote_notional"] = pc.multiply(
            result["contract_quantity"], float(contract_size)
        )
        if dataset.name == "agg_trades":
            result["base_quantity"] = pc.divide(
                result["quote_notional"], result["price"]
            )
    elif dataset.name == "agg_trades":
        result["quote_quantity"] = pc.multiply(result["base_quantity"], result["price"])


def _supported(dataset: DatasetSpec, operation: str) -> None:
    """Reject unsupported Binance normalizer or validator requests."""
    if dataset.product not in {"spot", "um", "cm"} or dataset.name not in {
        "klines",
        "trades",
        "agg_trades",
        "metrics",
        "book_depth",
        "mark_price_klines",
        "index_price_klines",
        "premium_index_klines",
    }:
        raise ValueError(f"unsupported {operation}: {dataset.product}/{dataset.name}")


def _ordered(values: Any, *, strict: bool) -> bool:
    """Return whether Arrow values are increasing, optionally strictly."""
    compare = pc.less_equal if strict else pc.less
    return not pc.any(
        compare(values.slice(1), values.slice(0, len(values) - 1))
    ).as_py()


def _times(
    table: Any,
    dataset: DatasetSpec,
    day: date,
    previous: datetime | None,
    end_day: date | None,
) -> datetime:
    """Validate UTC timestamp columns, archive bounds and chunk ordering."""
    for column in dataset.timestamp_columns:
        values = table[column]
        if values.null_count:
            raise DataValidationError("timestamps cannot be null")
        if not pa.types.is_timestamp(values.type) or values.type.tz != "UTC":
            raise DataValidationError(f"{column} must use UTC timestamps")
    values = table[dataset.time_column]
    if not _ordered(values, strict=dataset.supports_resampling):
        raise DataValidationError(f"{dataset.time_column} must be increasing")
    first, last = values[0].as_py(), values[-1].as_py()
    if previous is not None and (
        first < previous or (dataset.supports_resampling and first == previous)
    ):
        raise DataValidationError("chunk does not follow the preceding chunk")
    if first.date() < day or last.date() > (end_day or day):
        raise DataValidationError("timestamps fall outside the resource day range")
    if dataset.supports_resampling:
        _reject(
            pc.not_equal(values, pc.floor_temporal(values, unit="minute")),
            "open_time is not aligned to one minute",
        )
        closes = table["close_time"]
        _reject(
            pc.or_(
                pc.less(closes, values),
                pc.greater_equal(
                    closes,
                    pc.add(values, pa.scalar(timedelta(minutes=1), pa.duration("us"))),
                ),
            ),
            "close_time falls outside its candle",
        )
    return cast(datetime, last)


def _ohlc(table: Any, dataset: DatasetSpec) -> None:
    """Validate candle prices and declared additive volume/count fields."""
    if dataset.name != "premium_index_klines":
        for column in ("open", "high", "low", "close"):
            _reject(
                pc.less_equal(table[column], 0),
                "price values must be greater than zero",
            )
    for column in ("open", "low", "close"):
        _reject(
            pc.less(table["high"], table[column]), "high is below another OHLC price"
        )
    for column in ("open", "high", "close"):
        _reject(
            pc.greater(table["low"], table[column]), "low is above another OHLC price"
        )
    for column in dataset.resample_sum_columns:
        _reject(
            pc.less(table[column], 0), "volume and count values must be nonnegative"
        )


def _events(table: Any, dataset: DatasetSpec) -> None:
    """Validate trade IDs, positive prices and nonnegative quantities."""
    identifier = dataset.ordering_columns[-1]
    values = table[identifier]
    if pc.any(pc.less(values, 0)).as_py() or not _ordered(values, strict=False):
        raise DataValidationError(f"{identifier} must be nonnegative and increasing")
    if dataset.name == "agg_trades":
        _reject(
            pc.or_(
                pc.less(table["first_trade_id"], 0),
                pc.less(table["last_trade_id"], table["first_trade_id"]),
            ),
            "aggregate trade IDs are invalid",
        )
    _reject(pc.less_equal(table["price"], 0), "trade price must be greater than zero")
    for column in dataset.stored_columns:
        if column.endswith(("quantity", "notional")):
            _reject(pc.less(table[column], 0), "trade quantities must be nonnegative")
    if not pa.types.is_boolean(table["buyer_is_maker"].type):
        raise DataValidationError("buyer_is_maker must be boolean")


def validate_chunk(
    table: Any,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: datetime | None = None,
    end_day: date | None = None,
) -> datetime:
    """Validate a canonical Arrow batch and return its final event timestamp.

    Args:
        table: Canonical Arrow table to validate.
        dataset: Its Binance dataset declaration.
        day: First inclusive archive day.
        previous_timestamp: Last timestamp from the preceding batch.
        end_day: Last inclusive archive day, defaulting to day.

    Returns:
        Last UTC timestamp in this batch.
    """
    if tuple(table.column_names) != dataset.stored_columns:
        raise DataValidationError("chunk does not match the expected stored columns")
    if not table.num_rows:
        raise DataValidationError("chunk cannot be empty")
    _supported(dataset, "validator")
    last = _times(table, dataset, day, previous_timestamp, end_day)
    for column in dataset.stored_columns:
        if column in dataset.timestamp_columns or column in dataset.boolean_columns:
            continue
        values = table[column]
        _reject(pc.invert(pc.is_finite(values)), "numeric values must be finite")
        if values.null_count and not column.endswith("ratio"):
            raise DataValidationError("numeric values must be finite")
    if dataset.supports_resampling:
        _ohlc(table, dataset)
    elif dataset.name in {"trades", "agg_trades"}:
        _events(table, dataset)
    elif dataset.name == "metrics":
        for column in dataset.stored_columns[1:]:
            if column.endswith("ratio"):
                _reject(
                    pc.less_equal(table[column], 0),
                    "ratio must be positive when supplied",
                )
            else:
                _reject(
                    pc.less(table[column], 0),
                    "open interest values must be nonnegative",
                )
    else:
        _reject(
            pc.equal(table["percentage_bucket"], 0), "percentage bucket cannot be zero"
        )
        for column in dataset.stored_columns[2:]:
            _reject(pc.less(table[column], 0), "depth values must be nonnegative")
    return last
