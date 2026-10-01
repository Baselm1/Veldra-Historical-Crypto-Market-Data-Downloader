"""Normalize and validate Bybit public trade archives with Arrow."""

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pyarrow as pa
import pyarrow.compute as pc

from veldra.core.datasets import DatasetSpec
from veldra.core.ingest import ingest_archive, ingest_gzip_archive
from veldra.core.models import DataValidationError, IngestedResource, Resource


def _reject(condition: Any, message: str) -> None:
    """Raise a source-data error when any Arrow condition is true."""
    if pc.any(condition).as_py():
        raise DataValidationError(message)


def _text(values: Any, column: str) -> Any:
    """Return trimmed, nonempty source strings."""
    result = pc.utf8_trim_whitespace(pc.cast(values, pa.string()))
    if result.null_count:
        raise DataValidationError(f"{column} values must be nonempty")
    _reject(pc.equal(result, ""), f"{column} values must be nonempty")
    return result


def _number(values: Any, column: str) -> Any:
    """Convert one source column to finite doubles."""
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
    text = _text(values, column)
    _reject(
        pc.invert(pc.match_substring_regex(text, r"^[+-]?[0-9]+$")),
        f"invalid integer {column} value",
    )
    try:
        return pc.cast(text, pa.int64())
    except pa.ArrowException as error:
        raise DataValidationError(f"invalid integer {column} value") from error


def _epoch_milliseconds(values: Any, column: str) -> Any:
    """Convert epoch milliseconds to UTC microsecond timestamps."""
    milliseconds = _integer(values, column)
    if len(milliseconds):
        low, high = pc.min(milliseconds).as_py(), pc.max(milliseconds).as_py()
        if low is None or low < 100_000_000_000 or high >= 100_000_000_000_000:
            raise DataValidationError(f"invalid timestamp unit for {column}")
    return pc.cast(pc.multiply_checked(milliseconds, 1_000), pa.timestamp("us", "UTC"))


def _epoch_seconds(values: Any, column: str) -> Any:
    """Convert fractional epoch seconds to UTC microsecond timestamps."""
    seconds = _number(values, column)
    if len(seconds):
        low, high = pc.min(seconds).as_py(), pc.max(seconds).as_py()
        if low is None or low < 100_000_000 or high >= 100_000_000_000:
            raise DataValidationError(f"invalid timestamp unit for {column}")
    micros = pc.cast(pc.round(pc.multiply(seconds, 1_000_000)), pa.int64())
    return pc.cast(micros, pa.timestamp("us", "UTC"))


def _side(values: Any) -> Any:
    """Normalize aggressor direction to lowercase buy or sell."""
    result = pc.utf8_lower(_text(values, "side"))
    _reject(
        pc.invert(pc.is_in(result, value_set=pa.array(["buy", "sell"]))),
        "trade side must be buy or sell",
    )
    return result


def _boolean(values: Any, column: str) -> Any:
    """Convert a source Boolean column while accepting zero and one."""
    text = pc.utf8_lower(
        pc.fill_null(pc.utf8_trim_whitespace(pc.cast(values, pa.string())), "false")
    )
    truthy = pc.is_in(text, value_set=pa.array(["true", "1"]))
    valid = pc.is_in(text, value_set=pa.array(["true", "false", "1", "0"]))
    _reject(pc.invert(valid), f"invalid Boolean {column} value")
    return truthy


def _spot(table: Any, dataset: DatasetSpec) -> Any:
    """Normalize Spot trade rows."""
    volume = _number(table["volume"], "base_quantity")
    price = _number(table["price"], "price")
    rpi = (
        _boolean(table["rpi"], "is_rpi")
        if "rpi" in table.column_names
        else pa.array([False] * table.num_rows, type=pa.bool_())
    )
    values = {
        "event_time": _epoch_milliseconds(table["timestamp"], "event_time"),
        "trade_id": _text(table["id"], "trade_id"),
        "price": price,
        "base_quantity": volume,
        "quote_quantity": pc.multiply(price, volume),
        "side": _side(table["side"]),
        "is_rpi": rpi,
    }
    return pa.table({column: values[column] for column in dataset.stored_columns})


def _derivative(table: Any, dataset: DatasetSpec) -> Any:
    """Normalize linear or inverse derivative trade rows."""
    price = _number(table["price"], "price")
    size = _number(table["size"], "contract_quantity")
    home = _number(table["homeNotional"], "home_notional")
    foreign = _number(table["foreignNotional"], "foreign_notional")
    rpi = (
        _boolean(table["RPI"], "is_rpi")
        if "RPI" in table.column_names
        else pa.array([False] * table.num_rows, type=pa.bool_())
    )
    values: dict[str, Any] = {
        "event_time": _epoch_seconds(table["timestamp"], "event_time"),
        "trade_id": _text(table["trdMatchID"], "trade_id"),
        "price": price,
        "side": _side(table["side"]),
        "tick_direction": _text(table["tickDirection"], "tick_direction"),
        "is_rpi": rpi,
    }
    if dataset.product == "linear":
        values.update(base_quantity=home, quote_quantity=foreign)
    else:
        values.update(
            contract_quantity=size,
            base_quantity=foreign,
            quote_notional=home,
        )
    return pa.table({column: values[column] for column in dataset.stored_columns})


def _options(table: Any, dataset: DatasetSpec) -> Any:
    """Normalize shared Option-family trade rows."""
    values = {
        "event_time": _epoch_milliseconds(table["timestamp"], "event_time"),
        "trade_id": _text(table["trade_id"], "trade_id"),
        "trade_sequence": _integer(table["trade_seq"], "trade_sequence"),
        "instrument": _text(table["instrument_name"], "instrument"),
        "side": _side(table["direction"]),
        "price": _number(table["price"], "price"),
        "contract_quantity": _number(table["amount"], "contract_quantity"),
        "implied_volatility": _number(table["iv"], "implied_volatility"),
        "index_price": _number(table["index_price"], "index_price"),
        "mark_price": _number(table["mark_price"], "mark_price"),
        "mark_implied_volatility": _number(table["mark_iv"], "mark_implied_volatility"),
    }
    return pa.table({column: values[column] for column in dataset.stored_columns})


def normalize_chunk(
    table: Any, dataset: DatasetSpec, contract_size: float | None = None
) -> Any:
    """Normalize one Bybit trade chunk into its declared schema."""
    del contract_size
    if dataset.name != "trades":
        raise ValueError(f"unsupported normalizer: {dataset.product}/{dataset.name}")
    if dataset.product == "spot":
        return _spot(table, dataset)
    if dataset.product in {"linear", "inverse"}:
        return _derivative(table, dataset)
    if dataset.product == "options":
        return _options(table, dataset)
    raise ValueError(f"unsupported normalizer: {dataset.product}/{dataset.name}")


def _validate_schema(table: Any, dataset: DatasetSpec) -> None:
    """Require canonical columns and UTC timestamp values."""
    if tuple(table.column_names) != dataset.stored_columns or not table.num_rows:
        raise DataValidationError("chunk does not match the expected stored columns")
    timestamps = table[dataset.time_column]
    if (
        timestamps.null_count
        or not pa.types.is_timestamp(timestamps.type)
        or timestamps.type.tz != "UTC"
    ):
        raise DataValidationError("event_time must contain UTC timestamps")


def _validate_times(
    table: Any,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: datetime | None,
    end_day: date | None,
) -> datetime:
    """Validate nondecreasing timestamps inside the UTC archive day."""
    values = table[dataset.time_column]
    if pc.any(pc.less(values.slice(1), values.slice(0, len(values) - 1))).as_py():
        raise DataValidationError("event_time must be nondecreasing")
    first = cast(datetime, values[0].as_py())
    last = cast(datetime, values[-1].as_py())
    if previous_timestamp is not None and first < previous_timestamp:
        raise DataValidationError("chunk does not follow the preceding chunk")
    start = datetime.combine(day, time.min, UTC)
    end = datetime.combine((end_day or day) + timedelta(days=1), time.min, UTC)
    if first < start or last >= end:
        raise DataValidationError("timestamps fall outside the Bybit resource period")
    return last


def _validate_values(table: Any, dataset: DatasetSpec) -> None:
    """Validate trade identifiers, quantities, prices, and directions."""
    for column in dataset.stored_columns:
        if table[column].null_count:
            raise DataValidationError("trade values must be nonnull")
    _reject(pc.less_equal(table["price"], 0), "trade price must be positive")
    for column in (
        "base_quantity",
        "quote_quantity",
        "contract_quantity",
        "quote_notional",
    ):
        if column in table.column_names:
            _reject(pc.less_equal(table[column], 0), f"{column} must be positive")
    _reject(
        pc.invert(pc.is_in(table["side"], value_set=pa.array(["buy", "sell"]))),
        "trade side must be buy or sell",
    )
    if pc.count_distinct(table["trade_id"]).as_py() != table.num_rows:
        raise DataValidationError("trade_id values must be unique within one archive")


def validate_chunk(
    table: Any,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: datetime | None = None,
    end_day: date | None = None,
) -> datetime:
    """Validate canonical Bybit trades and return their final timestamp."""
    if dataset.name != "trades" or dataset.product not in {
        "spot",
        "linear",
        "inverse",
        "options",
    }:
        raise ValueError(f"unsupported validator: {dataset.product}/{dataset.name}")
    _validate_schema(table, dataset)
    _validate_values(table, dataset)
    return _validate_times(table, dataset, day, previous_timestamp, end_day)


def ingest_trades(
    client: httpx.Client,
    resource: Resource,
    dataset: DatasetSpec,
    destination: Path,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.5,
) -> IngestedResource:
    """Download, validate, and materialize one Bybit trade archive."""
    if dataset.product == "options" and not resource.archive_symbol:
        raise ValueError("Option archive requires a target instrument")

    def normalize_resource(
        table: Any, spec: DatasetSpec, contract_size: float | None
    ) -> Any:
        """Normalize a chunk and retain only its requested Option instrument."""
        normalized = normalize_chunk(table, spec, contract_size)
        if spec.product != "options":
            return normalized
        target = resource.archive_symbol
        assert target is not None
        return normalized.filter(pc.equal(normalized["instrument"], target))

    if resource.url.endswith(".zip"):
        return ingest_archive(
            client,
            resource,
            dataset,
            destination,
            normalizer=normalize_resource,
            validator=validate_chunk,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            allow_member_name_prefix=False,
        )
    return ingest_gzip_archive(
        client,
        resource,
        dataset,
        destination,
        normalizer=normalize_resource,
        validator=validate_chunk,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
    )
