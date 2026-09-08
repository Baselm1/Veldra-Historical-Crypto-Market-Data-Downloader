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


def _epoch_milliseconds(values: Any, column: str) -> Any:
    """Convert HTX epoch-millisecond values to UTC timestamps.

    Args:
        values: The Arrow source values.
        column: The canonical field named in errors.

    Returns:
        An Arrow UTC timestamp array.
    """
    milliseconds = _integer(values, column)
    if len(milliseconds):
        low, high = pc.min(milliseconds).as_py(), pc.max(milliseconds).as_py()
        if low is None or low < 100_000_000_000 or high >= 100_000_000_000_000:
            raise DataValidationError(f"invalid timestamp unit for {column}")
    return pc.cast(pc.multiply_checked(milliseconds, 1_000), pa.timestamp("us", "UTC"))


def _text(values: Any, column: str) -> Any:
    """Normalize one required source text column.

    Args:
        values: The Arrow source values.
        column: The canonical field named in errors.

    Returns:
        A lowercase Arrow string array without outer whitespace.
    """
    result = pc.utf8_lower(pc.utf8_trim_whitespace(pc.cast(values, pa.string())))
    if result.null_count:
        raise DataValidationError(f"invalid {column} value")
    _reject(pc.equal(result, ""), f"invalid {column} value")
    return result


def _kline_mapping(table: Any, dataset: DatasetSpec) -> dict[str, str]:
    """Return the canonical mapping for one HTX Kline source variant.

    Args:
        table: The raw Arrow source table.
        dataset: The product-specific Kline declaration.

    Returns:
        Canonical column names mapped to source fields.
    """
    names = set(table.column_names)
    volumes = (
        {"base_volume": "amount", "quote_volume": "vol"}
        if dataset.product == "spot"
        else {"contract_volume": "vol", "base_volume": "amount"}
    )
    if "id" in names:
        return {
            "open_time": "id",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            **volumes,
        }
    if "instId" in names and "open" in names:
        volumes = (
            {"base_volume": "vol", "quote_volume": "volCcyQuote"}
            if dataset.product == "spot"
            else {"contract_volume": "vol", "base_volume": "volCcy"}
        )
        return {
            "open_time": "ts",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            **volumes,
        }
    raise DataValidationError("CSV does not match an HTX Kline schema")


def _normalize_klines(table: Any, dataset: DatasetSpec) -> Any:
    """Normalize an old or new HTX Kline source table.

    Args:
        table: The raw Arrow source table.
        dataset: The canonical Kline declaration.

    Returns:
        Canonical Kline columns.
    """
    result = {
        target: (
            _epoch_seconds(table[source], target)
            if target == "open_time"
            else _number(table[source], target)
        )
        for target, source in _kline_mapping(table, dataset).items()
    }
    return pa.table({column: result[column] for column in dataset.stored_columns})


def _valid_contract_size(value: float | None) -> float:
    """Return a positive finite market contract size.

    Args:
        value: The contract size supplied by market metadata.

    Returns:
        The validated floating-point contract size.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise DataValidationError("perpetual contract size is unavailable")
    return float(value)


def _trade_mapping(table: Any, dataset: DatasetSpec) -> dict[str, str]:
    """Return canonical fields mapped to one trade source variant.

    Args:
        table: The raw Arrow source table.
        dataset: The product-specific trade declaration.

    Returns:
        Canonical column names mapped to source fields.
    """
    names = set(table.column_names)
    if "tradeId" in names:
        quantity = "base_quantity" if dataset.product == "spot" else "contract_quantity"
        return {
            "trade_id": "tradeId",
            "price": "px",
            quantity: "size",
            "side": "side",
            "event_time": "ts",
        }
    if "id" not in names or "direction" not in names:
        raise DataValidationError("CSV does not match an HTX trade schema")
    result = {
        "trade_id": "id",
        "price": "price",
        "side": "direction",
        "event_time": "ts",
    }
    if dataset.product == "spot":
        result["base_quantity"] = "amount"
    else:
        result.update({"contract_quantity": "amount", "base_quantity": "quantity"})
        if dataset.product == "linear_swap":
            result["quote_quantity"] = "trade_turnover"
    return result


def _derive_trade_quantities(
    result: dict[str, Any], dataset: DatasetSpec, contract_size: float | None
) -> None:
    """Derive quantities omitted by an HTX trade source variant.

    Args:
        result: The partially normalized canonical columns.
        dataset: The product-specific trade declaration.
        contract_size: Market contract size used by perpetual products.
    """
    if dataset.product == "spot":
        result["quote_quantity"] = pc.multiply(result["base_quantity"], result["price"])
        return
    size = _valid_contract_size(contract_size)
    if dataset.product == "linear_swap":
        if "base_quantity" not in result:
            result["base_quantity"] = pc.multiply(result["contract_quantity"], size)
        if "quote_quantity" not in result:
            result["quote_quantity"] = pc.multiply(
                result["base_quantity"], result["price"]
            )
        return
    result["quote_notional"] = pc.multiply(result["contract_quantity"], size)
    if "base_quantity" not in result:
        result["base_quantity"] = pc.divide(result["quote_notional"], result["price"])


def _normalize_trades(
    table: Any, dataset: DatasetSpec, contract_size: float | None
) -> Any:
    """Normalize an old or new HTX trade source table.

    Args:
        table: The raw Arrow source table.
        dataset: The canonical trade declaration.
        contract_size: Required perpetual contract size, when applicable.

    Returns:
        Canonical trade columns with derived quote quantity.
    """
    result: dict[str, Any] = {}
    for target, source in _trade_mapping(table, dataset).items():
        if target in dataset.integer_columns:
            result[target] = _integer(table[source], target)
        elif target in dataset.timestamp_columns:
            result[target] = _epoch_milliseconds(table[source], target)
        elif target in dataset.string_columns:
            result[target] = _text(table[source], target)
        else:
            result[target] = _number(table[source], target)
    _derive_trade_quantities(result, dataset, contract_size)
    return pa.table({column: result[column] for column in dataset.stored_columns})


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
    if (
        dataset.product in {"spot", "linear_swap", "coin_swap"}
        and dataset.name == "klines"
    ):
        return _normalize_klines(table, dataset)
    if (
        dataset.product in {"spot", "linear_swap", "coin_swap"}
        and dataset.name == "trades"
    ):
        return _normalize_trades(table, dataset, contract_size)
    raise ValueError(f"unsupported normalizer: {dataset.product}/{dataset.name}")


def _ordered(values: Any, *, strict: bool) -> bool:
    """Return whether Arrow values are increasing.

    Args:
        values: The timestamp array to inspect.
        strict: Whether equal adjacent values are invalid.

    Returns:
        Whether every timestamp follows the preceding timestamp.
    """
    compare = pc.less_equal if strict else pc.less
    return not pc.any(
        compare(values.slice(1), values.slice(0, len(values) - 1))
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


def _validate_schema(table: Any, dataset: DatasetSpec) -> None:
    """Validate canonical columns and primary timestamp type.

    Args:
        table: The canonical HTX table.
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
        raise DataValidationError(f"{dataset.time_column} must contain UTC timestamps")


def _validate_values(table: Any, dataset: DatasetSpec) -> None:
    """Validate nonnull canonical numeric and text values.

    Args:
        table: The canonical HTX table.
        dataset: The schema declaring typed columns.
    """
    for column in dataset.stored_columns:
        if column in dataset.timestamp_columns:
            continue
        values = table[column]
        if column in dataset.string_columns:
            if values.null_count or not pa.types.is_string(values.type):
                raise DataValidationError("text values must be nonnull strings")
            continue
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
    if not _ordered(times, strict=dataset.supports_resampling):
        raise DataValidationError(f"{dataset.time_column} must be increasing")
    first = cast(datetime, times[0].as_py())
    last = cast(datetime, times[-1].as_py())
    if previous_timestamp is not None and first <= previous_timestamp:
        raise DataValidationError("chunk does not follow the preceding chunk")
    start, _ = _coverage(day)
    _, end = _coverage(end_day or day)
    if first < start or last >= end:
        raise DataValidationError("timestamps fall outside the HTX source day")
    if dataset.supports_resampling:
        _reject(
            pc.not_equal(times, pc.floor_temporal(times, unit="minute")),
            "open_time is not aligned to one minute",
        )
    return last


def _validate_ohlc(table: Any, dataset: DatasetSpec) -> None:
    """Validate positive OHLC values and nonnegative declared volumes.

    Args:
        table: The canonical Kline table.
        dataset: The product-specific Kline declaration.
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
    for column in dataset.resample_sum_columns:
        _reject(pc.less(table[column], 0), "volume values must be nonnegative")


def _validate_trades(table: Any) -> None:
    """Validate trade IDs, quantities, prices, and sides.

    Args:
        table: The canonical trade table.
    """
    identifiers = table["trade_id"]
    _reject(pc.less(identifiers, 0), "trade_id must be nonnegative")
    same_time = pc.equal(
        table["event_time"].slice(1), table["event_time"].slice(0, len(table) - 1)
    )
    decreasing_id = pc.less(identifiers.slice(1), identifiers.slice(0, len(table) - 1))
    _reject(
        pc.and_(same_time, decreasing_id),
        "trade_id must not decrease within a timestamp",
    )
    _reject(pc.less_equal(table["price"], 0), "trade price must be positive")
    for column in table.column_names:
        if column.endswith(("quantity", "notional")):
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
    if dataset.product not in {
        "spot",
        "linear_swap",
        "coin_swap",
    } or dataset.name not in {"klines", "trades"}:
        raise ValueError(f"unsupported validator: {dataset.product}/{dataset.name}")
    _validate_schema(table, dataset)
    _validate_values(table, dataset)
    last = _validate_times(table, dataset, day, previous_timestamp, end_day)
    if dataset.name == "klines":
        _validate_ohlc(table, dataset)
    else:
        _validate_trades(table)
    return last
