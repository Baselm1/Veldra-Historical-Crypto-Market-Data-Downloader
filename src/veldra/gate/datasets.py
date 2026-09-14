"""Declare Gate products, datasets, schemas, and physical intervals."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from veldra.core.datasets import DatasetSpec

type GateProduct = Literal["spot", "um", "cm"]
type GateDataset = Literal[
    "klines",
    "trades",
    "order_book_updates",
    "order_book_snapshots",
    "mark_prices",
    "funding_rates",
    "funding_rate_updates",
]
type GateReferenceDataset = Literal[
    "mark_prices", "funding_rates", "funding_rate_updates"
]

PRODUCTS: tuple[GateProduct, ...] = ("spot", "um", "cm")
OUTPUT_INTERVALS: tuple[str, ...] = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1mo",
)
SUPPORTED_DATASETS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "spot": frozenset(
            {
                "klines",
                "trades",
                "order_book_updates",
                "order_book_snapshots",
            }
        ),
        "um": frozenset(
            {
                "klines",
                "trades",
                "order_book_updates",
                "order_book_snapshots",
                "mark_prices",
                "funding_rates",
                "funding_rate_updates",
            }
        ),
        "cm": frozenset(
            {
                "klines",
                "trades",
                "order_book_updates",
                "order_book_snapshots",
                "mark_prices",
                "funding_rates",
                "funding_rate_updates",
            }
        ),
    }
)

SPOT_KLINE_SOURCE = ("timestamp", "volume", "close", "high", "low", "open")
FUTURES_KLINE_SOURCE = ("timestamp", "size", "close", "high", "low", "open")
SPOT_TRADE_SOURCE = ("timestamp", "deal_id", "price", "amount", "side")
FUTURES_TRADE_SOURCE = ("timestamp", "deal_id", "price", "size")
SPOT_UPDATE_SOURCE = (
    "timestamp",
    "side",
    "action",
    "price",
    "amount",
    "begin_id",
    "merged_count",
)
FUTURES_UPDATE_SOURCE = (
    "timestamp",
    "action",
    "price",
    "size",
    "begin_id",
    "merged_count",
)
SNAPSHOT_SOURCE = ("timestamp", "update", "id", "bids", "asks")
MARK_PRICE_SOURCE = ("timestamp", "index_price", "mark_price", "last_price")
FUNDING_SOURCE = ("timestamp", "funding_rate")
FUNDING_UPDATE_SOURCE = (
    "timestamp",
    "funding_rate",
    "interest_rate",
    "bid_diff",
    "ask_diff",
    "mark_price",
    "index_price",
    "update_count",
)


def _klines(product: GateProduct, base_interval: Literal["10s", "1m"]) -> DatasetSpec:
    """Return one Gate Kline schema.

    Args:
        product: The Spot or perpetual Futures product.
        base_interval: The physical archive interval stored locally.

    Returns:
        The immutable Kline capability declaration.
    """
    spot = product == "spot"
    source = SPOT_KLINE_SOURCE if spot else FUTURES_KLINE_SOURCE
    quantity = "base_volume" if spot else "contract_volume"
    return DatasetSpec(
        product=product,
        name="klines",
        remote_name=f"candlesticks_{base_interval}",
        source_columns=source,
        stored_columns=("open_time", "open", "high", "low", "close", quantity),
        time_column="open_time",
        base_interval=base_interval,
        output_intervals=("10s",) if base_interval == "10s" else OUTPUT_INTERVALS,
        aliases=MappingProxyType({"volume": quantity}),
        max_concurrency=32,
        supports_resampling=base_interval == "1m",
        supports_gap_policy=True,
        gap_semantics="sparse",
        resample_sum_columns=(quantity,) if base_interval == "1m" else (),
        timestamp_columns=("open_time",),
    )


def _trades(product: GateProduct) -> DatasetSpec:
    """Return one Gate trade schema.

    Args:
        product: The Spot or perpetual Futures product.

    Returns:
        The immutable trade capability declaration.
    """
    spot = product == "spot"
    source = SPOT_TRADE_SOURCE if spot else FUTURES_TRADE_SOURCE
    quantities = ("base_quantity", "quote_quantity") if spot else ("contract_quantity",)
    return DatasetSpec(
        product=product,
        name="trades",
        remote_name="deals" if spot else "trades",
        source_columns=source,
        stored_columns=("event_time", "event_number", "price", *quantities, "side"),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=8,
        ordering_columns=("event_time", "event_number"),
        timestamp_columns=("event_time",),
        integer_columns=("event_number",),
        string_columns=("side",),
        sort_source_rows=True,
    )


def _updates(product: GateProduct) -> DatasetSpec:
    """Return one Gate order-book update schema.

    Args:
        product: The Spot or perpetual Futures product.

    Returns:
        The immutable update capability declaration.
    """
    spot = product == "spot"
    quantity = "base_quantity" if spot else "contract_quantity"
    return DatasetSpec(
        product=product,
        name="order_book_updates",
        remote_name="orderbooks",
        source_columns=SPOT_UPDATE_SOURCE if spot else FUTURES_UPDATE_SOURCE,
        stored_columns=(
            "event_time",
            "event_number",
            "update_id",
            "side",
            "action",
            "price",
            quantity,
            "merged_count",
        ),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=4,
        ordering_columns=("event_time", "update_id", "event_number"),
        timestamp_columns=("event_time",),
        integer_columns=("event_number", "update_id", "merged_count"),
        string_columns=("side", "action"),
        sort_source_rows=True,
    )


def _snapshots(product: GateProduct) -> DatasetSpec:
    """Return one Gate order-book snapshot schema.

    Args:
        product: The Spot or perpetual Futures product.

    Returns:
        The immutable snapshot capability declaration.
    """
    return DatasetSpec(
        product=product,
        name="order_book_snapshots",
        remote_name="orderbooks_slice",
        source_columns=SNAPSHOT_SOURCE,
        stored_columns=("event_time", "event_number", "update_id", "bids", "asks"),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=4,
        ordering_columns=("event_time", "event_number"),
        timestamp_columns=("event_time",),
        integer_columns=("event_number", "update_id"),
        object_columns=("bids", "asks"),
        sort_source_rows=True,
    )


def _reference(product: Literal["um", "cm"], name: GateReferenceDataset) -> DatasetSpec:
    """Return one Futures mark-price or funding schema.

    Args:
        product: The USDT- or BTC-margined Futures product.
        name: The requested reference dataset.

    Returns:
        The immutable reference-data declaration.
    """
    declarations = {
        "mark_prices": (
            "mark_prices",
            MARK_PRICE_SOURCE,
            ("event_time", "index_price", "mark_price", "last_price"),
            (),
        ),
        "funding_rates": (
            "funding_applies",
            FUNDING_SOURCE,
            ("event_time", "funding_rate"),
            (),
        ),
        "funding_rate_updates": (
            "funding_updates",
            FUNDING_UPDATE_SOURCE,
            (
                "event_time",
                "funding_rate",
                "interest_rate",
                "bid_diff",
                "ask_diff",
                "mark_price",
                "index_price",
                "update_count",
            ),
            ("update_count",),
        ),
    }
    remote, source, stored, integers = declarations[name]
    return DatasetSpec(
        product=product,
        name=name,
        remote_name=remote,
        source_columns=source,
        stored_columns=stored,
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=32,
        ordering_columns=("event_time",),
        timestamp_columns=("event_time",),
        integer_columns=integers,
        sort_source_rows=True,
    )


DATASETS: Mapping[tuple[str, str, str], DatasetSpec] = MappingProxyType(
    {
        ("spot", "klines", "1m"): _klines("spot", "1m"),
        ("um", "klines", "10s"): _klines("um", "10s"),
        ("um", "klines", "1m"): _klines("um", "1m"),
        ("cm", "klines", "10s"): _klines("cm", "10s"),
        ("cm", "klines", "1m"): _klines("cm", "1m"),
        ("spot", "trades", "raw"): _trades("spot"),
        ("um", "trades", "raw"): _trades("um"),
        ("cm", "trades", "raw"): _trades("cm"),
        ("spot", "order_book_updates", "raw"): _updates("spot"),
        ("um", "order_book_updates", "raw"): _updates("um"),
        ("cm", "order_book_updates", "raw"): _updates("cm"),
        ("spot", "order_book_snapshots", "raw"): _snapshots("spot"),
        ("um", "order_book_snapshots", "raw"): _snapshots("um"),
        ("cm", "order_book_snapshots", "raw"): _snapshots("cm"),
        ("um", "mark_prices", "raw"): _reference("um", "mark_prices"),
        ("cm", "mark_prices", "raw"): _reference("cm", "mark_prices"),
        ("um", "funding_rates", "raw"): _reference("um", "funding_rates"),
        ("cm", "funding_rates", "raw"): _reference("cm", "funding_rates"),
        ("um", "funding_rate_updates", "raw"): _reference("um", "funding_rate_updates"),
        ("cm", "funding_rate_updates", "raw"): _reference("cm", "funding_rate_updates"),
    }
)


def supports(product: str, dataset: str) -> bool:
    """Return whether Gate publishes one product and dataset combination.

    Args:
        product: The public Gate product identifier.
        dataset: The canonical dataset name.

    Returns:
        Whether the combination is supported.
    """
    return dataset in SUPPORTED_DATASETS.get(product, frozenset())


def get_dataset(
    product: object,
    dataset: object,
    *,
    kline_base_interval: object = "1m",
    requested_interval: object = None,
) -> DatasetSpec:
    """Return the physical Gate schema needed by one request.

    Args:
        product: The requested Gate product.
        dataset: The requested canonical dataset.
        kline_base_interval: The configured ordinary Kline storage interval.
        requested_interval: The optional requested output interval.

    Returns:
        The matching immutable dataset declaration.
    """
    if not isinstance(product, str):
        raise TypeError("product must be a string")
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")
    if not supports(product, dataset):
        raise ValueError(f"unsupported dataset '{product}/{dataset}'")
    storage = "raw"
    if dataset == "klines":
        if requested_interval == "10s":
            if product == "spot":
                raise ValueError("Gate Spot does not publish 10s Klines")
            storage = "10s"
        else:
            if kline_base_interval != "1m":
                raise ValueError("Gate stores ordinary Klines at 1m")
            storage = "1m"
    return DATASETS[(product, dataset, storage)]
