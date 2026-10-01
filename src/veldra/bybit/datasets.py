"""Declare Bybit products, historical datasets, and canonical schemas."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Literal

from veldra.core.datasets import DatasetSpec

type BybitProduct = Literal["spot", "linear", "inverse", "options"]
type BybitDataset = Literal[
    "klines",
    "trades",
    "order_book_updates",
    "mark_price_klines",
    "index_price_klines",
    "premium_index_klines",
    "funding_rates",
    "open_interest",
    "long_short_ratios",
    "historical_volatility",
    "delivery_prices",
]

PRODUCTS: tuple[BybitProduct, ...] = ("spot", "linear", "inverse", "options")
KLINE_INTERVALS: tuple[str, ...] = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
    "1w",
    "1mo",
)

_COMMON_ARCHIVES = frozenset({"trades", "order_book_updates"})
SUPPORTED_DATASETS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "spot": _COMMON_ARCHIVES | {"klines"},
        "linear": _COMMON_ARCHIVES
        | {
            "klines",
            "mark_price_klines",
            "index_price_klines",
            "premium_index_klines",
            "funding_rates",
            "open_interest",
            "long_short_ratios",
            "delivery_prices",
        },
        "inverse": _COMMON_ARCHIVES
        | {
            "klines",
            "mark_price_klines",
            "index_price_klines",
            "funding_rates",
            "open_interest",
            "long_short_ratios",
            "delivery_prices",
        },
        "options": _COMMON_ARCHIVES
        | {
            "mark_price_klines",
            "historical_volatility",
            "delivery_prices",
        },
    }
)

_SPOT_TRADE_SOURCE = ("id", "timestamp", "price", "volume", "side", "rpi")
_DERIVATIVE_TRADE_SOURCE = (
    "timestamp",
    "symbol",
    "side",
    "size",
    "price",
    "tickDirection",
    "trdMatchID",
    "grossValue",
    "homeNotional",
    "foreignNotional",
    "RPI",
)
_OPTION_TRADE_SOURCE = (
    "trade_id",
    "trade_seq",
    "timestamp",
    "instrument_name",
    "direction",
    "price",
    "amount",
    "iv",
    "index_price",
    "mark_price",
    "mark_iv",
)
_ORDER_BOOK_SOURCE = ("topic", "type", "ts", "data", "cts")
_KLINE_SOURCE = (
    "startTime",
    "openPrice",
    "highPrice",
    "lowPrice",
    "closePrice",
    "volume",
    "turnover",
)
_REFERENCE_KLINE_SOURCE = (
    "startTime",
    "openPrice",
    "highPrice",
    "lowPrice",
    "closePrice",
)


def _native_kline(product: str, name: str, *, reference: bool = False) -> DatasetSpec:
    """Return one native-interval market or reference Kline declaration."""
    if reference:
        source: tuple[str, ...] = _REFERENCE_KLINE_SOURCE
        stored: tuple[str, ...] = ("open_time", "open", "high", "low", "close")
    else:
        source = _KLINE_SOURCE
        stored = (
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "base_volume",
            "quote_volume",
        )
    return DatasetSpec(
        product=product,
        name=name,
        remote_name="kline" if name == "klines" else name.removesuffix("_klines"),
        source_columns=source,
        stored_columns=stored,
        time_column="open_time",
        base_interval="1m",
        output_intervals=("1m",),
        aliases=MappingProxyType({"volume": "base_volume"} if not reference else {}),
        max_concurrency=16,
        supports_gap_policy=True,
        gap_semantics="sparse",
        timestamp_columns=("open_time",),
        sort_source_rows=True,
    )


def _trades(product: str) -> DatasetSpec:
    """Return the product-specific public-trade declaration."""
    if product == "spot":
        source: tuple[str, ...] = _SPOT_TRADE_SOURCE
        stored: tuple[str, ...] = (
            "event_time",
            "trade_id",
            "price",
            "base_quantity",
            "quote_quantity",
            "side",
            "is_rpi",
        )
        integers: tuple[str, ...] = ()
        strings: tuple[str, ...] = ("trade_id", "side")
        booleans: tuple[str, ...] = ("is_rpi",)
    elif product == "linear":
        source = _DERIVATIVE_TRADE_SOURCE
        stored = (
            "event_time",
            "trade_id",
            "price",
            "base_quantity",
            "quote_quantity",
            "side",
            "tick_direction",
            "is_rpi",
        )
        integers = ()
        strings = ("trade_id", "side", "tick_direction")
        booleans = ("is_rpi",)
    elif product == "inverse":
        source = _DERIVATIVE_TRADE_SOURCE
        stored = (
            "event_time",
            "trade_id",
            "price",
            "contract_quantity",
            "base_quantity",
            "quote_notional",
            "side",
            "tick_direction",
            "is_rpi",
        )
        integers = ()
        strings = ("trade_id", "side", "tick_direction")
        booleans = ("is_rpi",)
    else:
        source = _OPTION_TRADE_SOURCE
        stored = (
            "event_time",
            "trade_id",
            "trade_sequence",
            "instrument",
            "side",
            "price",
            "contract_quantity",
            "implied_volatility",
            "index_price",
            "mark_price",
            "mark_implied_volatility",
        )
        integers = ("trade_sequence",)
        strings = ("trade_id", "instrument", "side")
        booleans = ()
    return DatasetSpec(
        product=product,
        name="trades",
        remote_name="trade",
        source_columns=source,
        stored_columns=stored,
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=16 if product != "options" else 4,
        csv_header="present",
        ordering_columns=("event_time", "trade_id"),
        timestamp_columns=("event_time",),
        integer_columns=integers,
        boolean_columns=booleans,
        string_columns=strings,
        sort_source_rows=True,
    )


def _order_book(product: str) -> DatasetSpec:
    """Return one lossless snapshot-and-delta order-book declaration."""
    workers = {"spot": 4, "linear": 2, "inverse": 2, "options": 1}[product]
    return DatasetSpec(
        product=product,
        name="order_book_updates",
        remote_name="orderbook",
        source_columns=_ORDER_BOOK_SOURCE,
        stored_columns=(
            "event_time",
            "engine_time",
            "event_number",
            "update_id",
            "cross_sequence",
            "instrument",
            "action",
            "source_depth",
            "bids",
            "asks",
        ),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=workers,
        ordering_columns=("event_time", "event_number"),
        timestamp_columns=("event_time", "engine_time"),
        integer_columns=(
            "event_number",
            "update_id",
            "cross_sequence",
            "source_depth",
        ),
        string_columns=("instrument", "action"),
        object_columns=("bids", "asks"),
    )


def _raw_dataset(
    product: str,
    name: str,
    source: tuple[str, ...],
    stored: tuple[str, ...],
    time_column: str,
    *,
    integers: tuple[str, ...] = (),
) -> DatasetSpec:
    """Return one interval-less REST history declaration."""
    return DatasetSpec(
        product=product,
        name=name,
        remote_name=name,
        source_columns=source,
        stored_columns=stored,
        time_column=time_column,
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=16,
        ordering_columns=(time_column,),
        timestamp_columns=(time_column,),
        integer_columns=integers,
        sort_source_rows=True,
    )


DATASETS: Mapping[tuple[str, str], DatasetSpec] = MappingProxyType(
    {
        **{(product, "trades"): _trades(product) for product in PRODUCTS},
        **{
            (product, "order_book_updates"): _order_book(product)
            for product in PRODUCTS
        },
        **{
            (product, "klines"): _native_kline(product, "klines")
            for product in ("spot", "linear", "inverse")
        },
        **{
            (product, name): _native_kline(product, name, reference=True)
            for product, names in {
                "linear": (
                    "mark_price_klines",
                    "index_price_klines",
                    "premium_index_klines",
                ),
                "inverse": ("mark_price_klines", "index_price_klines"),
                "options": ("mark_price_klines",),
            }.items()
            for name in names
        },
        **{
            (product, "funding_rates"): _raw_dataset(
                product,
                "funding_rates",
                ("symbol", "fundingRate", "fundingRateTimestamp"),
                ("funding_time", "funding_rate"),
                "funding_time",
            )
            for product in ("linear", "inverse")
        },
        **{
            (product, "open_interest"): _raw_dataset(
                product,
                "open_interest",
                ("symbol", "openInterest", "singleOpenInterest", "timestamp"),
                ("event_time", "open_interest", "single_open_interest"),
                "event_time",
            )
            for product in ("linear", "inverse")
        },
        **{
            (product, "long_short_ratios"): _raw_dataset(
                product,
                "long_short_ratios",
                ("symbol", "buyRatio", "sellRatio", "timestamp"),
                ("event_time", "buy_ratio", "sell_ratio"),
                "event_time",
            )
            for product in ("linear", "inverse")
        },
        ("options", "historical_volatility"): _raw_dataset(
            "options",
            "historical_volatility",
            ("period", "value", "time"),
            ("event_time", "period", "volatility"),
            "event_time",
            integers=("period",),
        ),
        **{
            (product, "delivery_prices"): _raw_dataset(
                product,
                "delivery_prices",
                ("symbol", "deliveryPrice", "deliveryTime"),
                ("delivery_time", "delivery_price"),
                "delivery_time",
            )
            for product in ("linear", "inverse", "options")
        },
    }
)


def supports(product: object, dataset: object) -> bool:
    """Return whether Bybit publishes one product and dataset combination."""
    return (
        isinstance(product, str)
        and isinstance(dataset, str)
        and dataset in (SUPPORTED_DATASETS.get(product, frozenset()))
    )


def _interval(value: object) -> str:
    """Return one supported native Bybit Kline interval."""
    if not isinstance(value, str):
        raise TypeError("interval must be a string")
    if value not in KLINE_INTERVALS:
        choices = ", ".join(KLINE_INTERVALS)
        raise ValueError(f"unsupported Bybit interval {value!r}; choose from {choices}")
    return value


def get_dataset(
    product: object,
    dataset: object,
    *,
    kline_base_interval: object = None,
    requested_interval: object = None,
) -> DatasetSpec:
    """Return the exact Bybit schema for a validated request."""
    del kline_base_interval
    if not isinstance(product, str):
        raise TypeError("product must be a string")
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")
    if not supports(product, dataset):
        raise ValueError(f"unsupported dataset '{product}/{dataset}'")
    spec = DATASETS[(product, dataset)]
    if not spec.needs_interval:
        if requested_interval is not None:
            raise ValueError(f"dataset '{dataset}' does not accept an interval")
        return spec
    interval = _interval("1m" if requested_interval is None else requested_interval)
    return replace(spec, base_interval=interval, output_intervals=(interval,))
