"""Normalize and validate tabular source data."""

from collections.abc import Callable
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


def _boolean(values: pd.Series, column: str) -> pd.Series:
    """Convert one source column into strict true-or-false values.

    Args:
        values: The source strings to convert.
        column: The source column name used in errors.

    Returns:
        The converted boolean values.
    """
    result = values.astype(str).str.strip().str.lower()
    if not result.isin({"true", "false"}).all():
        raise DataValidationError(f"invalid boolean {column} value")
    return result.eq("true")


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
        return pd.to_datetime(integers, unit=unit, utc=True, errors="raise").astype(
            "datetime64[us, UTC]"
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise DataValidationError(f"invalid {column} value") from error


def _normalize_kline_chunk(
    frame: pd.DataFrame,
    dataset: DatasetSpec,
    contract_size: float | None = None,
    *,
    volume_column: str,
    quote_volume_column: str,
    taker_volume_column: str,
    taker_quote_volume_column: str,
) -> pd.DataFrame:
    """Convert one product-specific Kline CSV chunk into canonical columns.

    Args:
        frame: The raw source rows with source column names.
        dataset: The schema describing the source and stored columns.
        contract_size: The optional COIN-M contract size, unused by Klines.
        volume_column: The canonical field receiving Binance's ``volume``.
        quote_volume_column: The canonical field receiving ``quote_volume``.
        taker_volume_column: The canonical field receiving ``taker_buy_volume``.
        taker_quote_volume_column: The canonical field receiving Binance's
            ``taker_buy_quote_volume``.

    Returns:
        A new DataFrame containing normalized values and column names.
    """
    result = pd.DataFrame(index=frame.index)
    result["open_time"] = _epoch(frame["open_time"], "open_time")
    for column in ("open", "high", "low", "close"):
        result[column] = _number(frame[column], column)
    result[volume_column] = _number(frame["volume"], "volume")
    result["close_time"] = _epoch(frame["close_time"], "close_time")
    result[quote_volume_column] = _number(frame["quote_volume"], "quote_volume")
    result["trade_count"] = _integer(frame["count"], "count")
    result[taker_volume_column] = _number(frame["taker_buy_volume"], "taker_buy_volume")
    result[taker_quote_volume_column] = _number(
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


def _normalize_spot_klines(
    frame: pd.DataFrame, dataset: DatasetSpec, contract_size: float | None = None
) -> pd.DataFrame:
    """Normalize one Spot Kline source chunk.

    Args:
        frame: Headerless Spot CSV rows with declared source names.
        dataset: The Spot Kline schema declaration.
        contract_size: The optional COIN-M contract size, unused by Spot data.

    Returns:
        Canonical Spot candle rows with base and quote volume fields.
    """
    return _normalize_kline_chunk(
        frame,
        dataset,
        volume_column="volume",
        quote_volume_column="quote_volume",
        taker_volume_column="taker_buy_base_volume",
        taker_quote_volume_column="taker_buy_quote_volume",
    )


def _normalize_um_klines(
    frame: pd.DataFrame, dataset: DatasetSpec, contract_size: float | None = None
) -> pd.DataFrame:
    """Normalize one USD-M perpetual Kline source chunk.

    Args:
        frame: Header-bearing USD-M CSV rows with declared source names.
        dataset: The USD-M Kline schema declaration.
        contract_size: The optional COIN-M contract size, unused by USD-M data.

    Returns:
        Canonical USD-M candles with explicit base and quote volumes.
    """
    return _normalize_kline_chunk(
        frame,
        dataset,
        volume_column="base_volume",
        quote_volume_column="quote_volume",
        taker_volume_column="taker_buy_base_volume",
        taker_quote_volume_column="taker_buy_quote_volume",
    )


def _normalize_cm_klines(
    frame: pd.DataFrame, dataset: DatasetSpec, contract_size: float | None = None
) -> pd.DataFrame:
    """Normalize one COIN-M perpetual Kline source chunk.

    Args:
        frame: Header-bearing COIN-M CSV rows with declared source names.
        dataset: The COIN-M Kline schema declaration.
        contract_size: The cataloged contract size, unused by Kline data.

    Returns:
        Canonical COIN-M candles with explicit contract and base volumes.
    """
    return _normalize_kline_chunk(
        frame,
        dataset,
        volume_column="contract_volume",
        quote_volume_column="base_volume",
        taker_volume_column="taker_buy_contract_volume",
        taker_quote_volume_column="taker_buy_base_volume",
    )


def _normalize_price_klines(
    frame: pd.DataFrame, dataset: DatasetSpec, contract_size: float | None = None
) -> pd.DataFrame:
    """Normalize one perpetual mark-price candle source chunk.

    Args:
        frame: Header-bearing Binance mark-price Kline source rows.
        dataset: The USD-M or COIN-M mark-price schema declaration.
        contract_size: The optional COIN-M contract size, unused by price data.

    Returns:
        Canonical price candles and their source sample counts.

    Raises:
        DataValidationError: If a structural source field is no longer zero.
    """
    structural_columns = (
        "volume",
        "quote_volume",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    )
    for column in structural_columns:
        if (_number(frame[column], column) != 0).any():
            raise DataValidationError(
                f"mark-price structural field '{column}' is not zero"
            )
    result = pd.DataFrame(index=frame.index)
    result["open_time"] = _epoch(frame["open_time"], "open_time")
    for column in ("open", "high", "low", "close"):
        result[column] = _number(frame[column], column)
    result["close_time"] = _epoch(frame["close_time"], "close_time")
    result["sample_count"] = _integer(frame["count"], "count")
    normalized = result.loc[:, dataset.stored_columns]
    LOGGER.debug(
        "Mark-price Kline chunk normalized: product=%s rows=%d",
        dataset.product,
        len(normalized),
    )
    return normalized


def _normalize_spot_trades(
    frame: pd.DataFrame, dataset: DatasetSpec, contract_size: float | None = None
) -> pd.DataFrame:
    """Normalize one Spot trade or aggregate-trade source chunk.

    Args:
        frame: Headerless Binance rows with the declared source columns.
        dataset: The Spot trades or aggregate-trades declaration.
        contract_size: The optional COIN-M contract size, unused by Spot data.

    Returns:
        Canonical event rows with exact IDs, UTC timestamps, and maker side.
    """
    result = pd.DataFrame(index=frame.index)
    id_column = "trade_id" if dataset.name == "trades" else "agg_trade_id"
    result[id_column] = _integer(frame[id_column], id_column)
    if dataset.name == "agg_trades":
        result["first_trade_id"] = _integer(frame["first_trade_id"], "first_trade_id")
        result["last_trade_id"] = _integer(frame["last_trade_id"], "last_trade_id")
    result["price"] = _number(frame["price"], "price")
    result["base_quantity"] = _number(frame["base_quantity"], "base_quantity")
    if dataset.name == "trades":
        result["quote_quantity"] = _number(frame["quote_quantity"], "quote_quantity")
    else:
        result["quote_quantity"] = result["price"] * result["base_quantity"]
    result["event_time"] = _epoch(frame["event_time"], "event_time")
    result["buyer_is_maker"] = _boolean(frame["is_buyer_maker"], "is_buyer_maker")
    normalized = result.loc[:, dataset.stored_columns]
    LOGGER.debug(
        "Spot event chunk normalized: dataset=%s rows=%d", dataset.name, len(normalized)
    )
    return normalized


def _contract_size(value: float | None) -> float:
    """Return one positive finite COIN-M contract size.

    Args:
        value: The market contract size supplied through the catalog context.

    Returns:
        The validated contract size as a float.

    Raises:
        DataValidationError: If the required contract size is missing or invalid.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError("COIN-M contract size is unavailable")
    size = float(value)
    if not np.isfinite(size) or size <= 0:
        raise DataValidationError("COIN-M contract size must be positive and finite")
    return size


def _normalize_um_trades(
    frame: pd.DataFrame, dataset: DatasetSpec, contract_size: float | None = None
) -> pd.DataFrame:
    """Normalize one USD-M perpetual trade CSV chunk.

    Args:
        frame: Header-bearing USD-M source trade rows.
        dataset: The USD-M trade schema declaration.
        contract_size: The optional COIN-M contract size, unused by USD-M data.

    Returns:
        Canonical USD-M trades with base and quote quantities.
    """
    result = pd.DataFrame(index=frame.index)
    result["trade_id"] = _integer(frame["id"], "id")
    result["price"] = _number(frame["price"], "price")
    result["base_quantity"] = _number(frame["qty"], "qty")
    result["quote_quantity"] = _number(frame["quote_qty"], "quote_qty")
    result["event_time"] = _epoch(frame["time"], "time")
    result["buyer_is_maker"] = _boolean(frame["is_buyer_maker"], "is_buyer_maker")
    return result.loc[:, dataset.stored_columns]


def _normalize_cm_trades(
    frame: pd.DataFrame, dataset: DatasetSpec, contract_size: float | None = None
) -> pd.DataFrame:
    """Normalize one COIN-M perpetual trade CSV chunk.

    Args:
        frame: Header-bearing COIN-M source trade rows.
        dataset: The COIN-M trade schema declaration.
        contract_size: The cataloged USD value of one contract.

    Returns:
        Canonical COIN-M trades with source quantities and derived quote notional.
    """
    size = _contract_size(contract_size)
    result = pd.DataFrame(index=frame.index)
    result["trade_id"] = _integer(frame["id"], "id")
    result["price"] = _number(frame["price"], "price")
    result["contract_quantity"] = _number(frame["qty"], "qty")
    result["base_quantity"] = _number(frame["base_qty"], "base_qty")
    result["quote_notional"] = result["contract_quantity"] * size
    result["event_time"] = _epoch(frame["time"], "time")
    result["buyer_is_maker"] = _boolean(frame["is_buyer_maker"], "is_buyer_maker")
    return result.loc[:, dataset.stored_columns]


def _normalize_um_agg_trades(
    frame: pd.DataFrame, dataset: DatasetSpec, contract_size: float | None = None
) -> pd.DataFrame:
    """Normalize one USD-M perpetual aggregate-trade CSV chunk.

    Args:
        frame: Header-bearing USD-M aggregate-trade source rows.
        dataset: The USD-M aggregate-trade schema declaration.
        contract_size: The optional COIN-M contract size, unused by USD-M data.

    Returns:
        Canonical USD-M aggregates with base quantity and derived quote quantity.
    """
    result = pd.DataFrame(index=frame.index)
    result["agg_trade_id"] = _integer(frame["agg_trade_id"], "agg_trade_id")
    result["first_trade_id"] = _integer(frame["first_trade_id"], "first_trade_id")
    result["last_trade_id"] = _integer(frame["last_trade_id"], "last_trade_id")
    result["price"] = _number(frame["price"], "price")
    result["base_quantity"] = _number(frame["quantity"], "quantity")
    result["quote_quantity"] = result["price"] * result["base_quantity"]
    result["event_time"] = _epoch(frame["transact_time"], "transact_time")
    result["buyer_is_maker"] = _boolean(frame["is_buyer_maker"], "is_buyer_maker")
    return result.loc[:, dataset.stored_columns]


def _normalize_cm_agg_trades(
    frame: pd.DataFrame, dataset: DatasetSpec, contract_size: float | None = None
) -> pd.DataFrame:
    """Normalize one COIN-M perpetual aggregate-trade CSV chunk.

    Args:
        frame: Header-bearing COIN-M aggregate-trade source rows.
        dataset: The COIN-M aggregate-trade schema declaration.
        contract_size: The cataloged USD value of one COIN-M contract.

    Returns:
        Canonical COIN-M aggregates with preserved contracts and derived values.
    """
    size = _contract_size(contract_size)
    result = pd.DataFrame(index=frame.index)
    result["agg_trade_id"] = _integer(frame["agg_trade_id"], "agg_trade_id")
    result["first_trade_id"] = _integer(frame["first_trade_id"], "first_trade_id")
    result["last_trade_id"] = _integer(frame["last_trade_id"], "last_trade_id")
    result["price"] = _number(frame["price"], "price")
    if (result["price"] <= 0).any():
        raise DataValidationError("trade price must be greater than zero")
    result["contract_quantity"] = _number(frame["quantity"], "quantity")
    result["quote_notional"] = result["contract_quantity"] * size
    result["base_quantity"] = result["quote_notional"] / result["price"]
    result["event_time"] = _epoch(frame["transact_time"], "transact_time")
    result["buyer_is_maker"] = _boolean(frame["is_buyer_maker"], "is_buyer_maker")
    return result.loc[:, dataset.stored_columns]


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
        if column not in dataset.timestamp_columns
    ]
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype="float64")).all():
        raise DataValidationError("numeric values must be finite")
    nonnegative = frame.loc[:, dataset.resample_sum_columns]
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


def _validate_kline_chunk(
    frame: pd.DataFrame,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Validate one canonical daily Kline chunk.

    Args:
        frame: The normalized rows to validate.
        dataset: The Spot, USD-M, or COIN-M Kline schema.
        day: The UTC source day that must contain every row.
        previous_timestamp: The final timestamp from the preceding chunk.

    Returns:
        The final timestamp in the validated chunk.
    """
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


def _validate_event_timestamps(
    values: pd.Series, day: date, previous_timestamp: pd.Timestamp | None
) -> pd.Timestamp:
    """Validate sorted UTC event timestamps inside one source day.

    Args:
        values: The event timestamps in their source order.
        day: The UTC archive day that must contain all events.
        previous_timestamp: The final timestamp from the preceding CSV chunk.

    Returns:
        The final event timestamp in the chunk.
    """
    if values.isna().any():
        raise DataValidationError("event_time cannot be null")
    _require_utc(values, "event_time")
    if not values.is_monotonic_increasing:
        raise DataValidationError("event_time must be increasing")
    first = values.iloc[0]
    last = values.iloc[-1]
    if previous_timestamp is not None and first < previous_timestamp:
        raise DataValidationError("chunk does not follow the preceding chunk")
    day_start = pd.Timestamp(day, tz="UTC")
    if first < day_start or last >= day_start + pd.Timedelta(days=1):
        raise DataValidationError("timestamps fall outside the resource day")
    return last


def _validate_spot_trade_rows(frame: pd.DataFrame, dataset: DatasetSpec) -> None:
    """Validate numeric, ID, and maker-side rules for Spot event rows.

    Args:
        frame: Canonical Spot trade or aggregate-trade rows.
        dataset: The matching Spot event dataset declaration.
    """
    id_column = dataset.ordering_columns[-1]
    identifiers = frame[id_column]
    if (identifiers < 0).any() or not identifiers.is_monotonic_increasing:
        raise DataValidationError(f"{id_column} must be nonnegative and increasing")
    if dataset.name == "agg_trades" and (
        (frame["first_trade_id"] < 0).any()
        or (frame["last_trade_id"] < frame["first_trade_id"]).any()
    ):
        raise DataValidationError("aggregate trade IDs are invalid")
    values = frame[["price", "base_quantity", "quote_quantity"]]
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise DataValidationError("trade values must be finite")
    if (frame["price"] <= 0).any():
        raise DataValidationError("trade price must be greater than zero")
    if (frame[["base_quantity", "quote_quantity"]] < 0).any().any():
        raise DataValidationError("trade quantities must be nonnegative")
    if not pd.api.types.is_bool_dtype(frame["buyer_is_maker"]):
        raise DataValidationError("buyer_is_maker must be boolean")


def _validate_spot_event_chunk(
    frame: pd.DataFrame,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: pd.Timestamp | None,
) -> pd.Timestamp:
    """Validate one canonical Spot trade-family chunk.

    Args:
        frame: Canonical event rows to validate.
        dataset: The Spot trade or aggregate-trade declaration.
        day: The UTC archive day that must contain all events.
        previous_timestamp: The final timestamp from the preceding chunk.

    Returns:
        The final event timestamp in the validated chunk.
    """
    last = _validate_event_timestamps(frame["event_time"], day, previous_timestamp)
    _validate_spot_trade_rows(frame, dataset)
    return last


def _validate_futures_event_ids(frame: pd.DataFrame, dataset: DatasetSpec) -> None:
    """Validate ordered primary and aggregate component identifiers.

    Args:
        frame: Canonical USD-M or COIN-M trade-family rows.
        dataset: The matching Futures event schema declaration.
    """
    id_column = dataset.ordering_columns[-1]
    identifiers = frame[id_column]
    if (identifiers < 0).any() or not identifiers.is_monotonic_increasing:
        raise DataValidationError(f"{id_column} must be nonnegative and increasing")
    if dataset.name == "agg_trades" and (
        (frame["first_trade_id"] < 0).any()
        or (frame["last_trade_id"] < frame["first_trade_id"]).any()
    ):
        raise DataValidationError("aggregate trade IDs are invalid")


def _futures_quantity_columns(dataset: DatasetSpec) -> list[str]:
    """Return canonical Futures quantity fields without IDs or event metadata.

    Args:
        dataset: The Futures trade-family schema declaration.

    Returns:
        Canonical source or derived quantity column names.
    """
    excluded = {
        dataset.ordering_columns[-1],
        "first_trade_id",
        "last_trade_id",
        "price",
        "event_time",
        "buyer_is_maker",
    }
    return [column for column in dataset.stored_columns if column not in excluded]


def _validate_futures_event_values(frame: pd.DataFrame, quantities: list[str]) -> None:
    """Validate finite price, quantity, and maker-side values for Futures rows.

    Args:
        frame: Canonical USD-M or COIN-M trade-family rows.
        quantities: Canonical quantity columns that must remain nonnegative.
    """
    values = frame[["price", *quantities]]
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise DataValidationError("trade values must be finite")
    if (frame["price"] <= 0).any():
        raise DataValidationError("trade price must be greater than zero")
    if (frame[quantities] < 0).any().any():
        raise DataValidationError("trade quantities must be nonnegative")
    if not pd.api.types.is_bool_dtype(frame["buyer_is_maker"]):
        raise DataValidationError("buyer_is_maker must be boolean")


def _validate_futures_event_rows(frame: pd.DataFrame, dataset: DatasetSpec) -> None:
    """Validate numeric, ID, and maker-side rules for Futures event rows.

    Args:
        frame: Canonical USD-M or COIN-M trade-family rows.
        dataset: The matching Futures event schema declaration.
    """
    _validate_futures_event_ids(frame, dataset)
    _validate_futures_event_values(frame, _futures_quantity_columns(dataset))


def _validate_futures_event_chunk(
    frame: pd.DataFrame,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: pd.Timestamp | None,
) -> pd.Timestamp:
    """Validate one canonical USD-M or COIN-M trade-family chunk.

    Args:
        frame: Canonical Futures event rows to validate.
        dataset: The Futures trade or aggregate-trade declaration.
        day: The UTC archive day that must contain all events.
        previous_timestamp: The final timestamp from the preceding CSV chunk.

    Returns:
        The final event timestamp in the chunk.
    """
    last = _validate_event_timestamps(frame["event_time"], day, previous_timestamp)
    _validate_futures_event_rows(frame, dataset)
    return last


type Normalizer = Callable[[pd.DataFrame, DatasetSpec, float | None], pd.DataFrame]
type Validator = Callable[
    [pd.DataFrame, DatasetSpec, date, pd.Timestamp | None], pd.Timestamp
]

_NORMALIZERS: dict[tuple[str, str], Normalizer] = {
    ("spot", "klines"): _normalize_spot_klines,
    ("um", "klines"): _normalize_um_klines,
    ("cm", "klines"): _normalize_cm_klines,
    ("um", "mark_price_klines"): _normalize_price_klines,
    ("cm", "mark_price_klines"): _normalize_price_klines,
    ("spot", "trades"): _normalize_spot_trades,
    ("spot", "agg_trades"): _normalize_spot_trades,
    ("um", "trades"): _normalize_um_trades,
    ("cm", "trades"): _normalize_cm_trades,
    ("um", "agg_trades"): _normalize_um_agg_trades,
    ("cm", "agg_trades"): _normalize_cm_agg_trades,
}
_VALIDATORS: dict[tuple[str, str], Validator] = {
    ("spot", "klines"): _validate_kline_chunk,
    ("um", "klines"): _validate_kline_chunk,
    ("cm", "klines"): _validate_kline_chunk,
    ("um", "mark_price_klines"): _validate_kline_chunk,
    ("cm", "mark_price_klines"): _validate_kline_chunk,
    ("spot", "trades"): _validate_spot_event_chunk,
    ("spot", "agg_trades"): _validate_spot_event_chunk,
    ("um", "trades"): _validate_futures_event_chunk,
    ("cm", "trades"): _validate_futures_event_chunk,
    ("um", "agg_trades"): _validate_futures_event_chunk,
    ("cm", "agg_trades"): _validate_futures_event_chunk,
}


def normalize_chunk(
    frame: pd.DataFrame,
    dataset: DatasetSpec,
    *,
    contract_size: float | None = None,
) -> pd.DataFrame:
    """Convert one declared source CSV chunk into its stored schema.

    Args:
        frame: The raw source rows with declared source column names.
        dataset: The dataset declaration selecting a normalizer.
        contract_size: The cataloged COIN-M contract size when required.

    Returns:
        A new DataFrame containing canonical stored columns.

    Raises:
        DataValidationError: If the source columns do not match the declaration.
        ValueError: If the dataset has no registered normalizer.
    """
    if tuple(frame.columns) != dataset.source_columns:
        raise DataValidationError("CSV does not match the expected source columns")
    try:
        normalizer = _NORMALIZERS[(dataset.product, dataset.name)]
    except KeyError as error:
        raise ValueError(
            f"unsupported normalizer: {dataset.product}/{dataset.name}"
        ) from error
    return normalizer(frame, dataset, contract_size)


def validate_chunk(
    frame: pd.DataFrame,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Validate one canonical chunk with its dataset-specific rules.

    Args:
        frame: The normalized rows to validate.
        dataset: The dataset declaration selecting a validator.
        day: The UTC source day that must contain every row.
        previous_timestamp: The final timestamp from the preceding chunk.

    Returns:
        The final timestamp in the validated chunk.

    Raises:
        DataValidationError: If the canonical schema is invalid or empty.
        ValueError: If the dataset has no registered validator.
    """
    if tuple(frame.columns) != dataset.stored_columns:
        raise DataValidationError("chunk does not match the expected stored columns")
    if frame.empty:
        raise DataValidationError("chunk cannot be empty")
    try:
        validator = _VALIDATORS[(dataset.product, dataset.name)]
    except KeyError as error:
        raise ValueError(
            f"unsupported validator: {dataset.product}/{dataset.name}"
        ) from error
    return validator(frame, dataset, day, previous_timestamp)
