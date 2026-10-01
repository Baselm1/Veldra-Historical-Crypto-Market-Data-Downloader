"""Declare Bitget products, datasets, schemas, and physical intervals."""

from collections.abc import Mapping
from datetime import timedelta
from types import MappingProxyType
from typing import Literal

from veldra.core.datasets import CsvSchema, DatasetSpec

type BitgetProduct = Literal["spot", "usdt_futures", "usdc_futures", "coin_futures"]
type BitgetDataset = Literal[
    "klines",
    "trades",
    "best_book_snapshots",
    "order_book_snapshots",
    "mark_price_klines",
    "index_price_klines",
    "premium_index_klines",
    "funding_rates",
]

PRODUCTS: tuple[BitgetProduct, ...] = (
    "spot",
    "usdt_futures",
    "usdc_futures",
    "coin_futures",
)
FUTURES_PRODUCTS: tuple[BitgetProduct, ...] = PRODUCTS[1:]
ARCHIVE_DAY_OFFSET = timedelta(hours=8)
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

ARCHIVE_DATASETS = frozenset(
    {"klines", "trades", "best_book_snapshots", "order_book_snapshots"}
)
REFERENCE_DATASETS = frozenset(
    {
        "mark_price_klines",
        "index_price_klines",
        "premium_index_klines",
        "funding_rates",
    }
)
SUPPORTED_DATASETS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "spot": ARCHIVE_DATASETS,
        "usdt_futures": ARCHIVE_DATASETS | REFERENCE_DATASETS,
        "usdc_futures": ARCHIVE_DATASETS | REFERENCE_DATASETS,
        "coin_futures": ARCHIVE_DATASETS | REFERENCE_DATASETS,
    }
)

KLINE_SOURCE = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "basevolume",
    "usdtvolume",
)
KLINE_SOURCE_LEGACY = CsvSchema(
    (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "baseVolume",
        "usdtVolume",
    ),
    "present",
)
TRADE_SOURCE = (
    "trade_id",
    "timestamp",
    "price",
    "side",
    "volume(quote)",
    "size(base)",
)
BEST_BOOK_SOURCE = (
    "timestamp",
    "ask_price",
    "bid_price",
    "ask_volume",
    "bid_volume",
)
ORDER_BOOK_SOURCE = ("timestamp", "asks", "bids")
REFERENCE_KLINE_SOURCE = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "quote_volume",
)


def _klines(product: BitgetProduct) -> DatasetSpec:
    """Return one Bitget market-Kline declaration.

    Args:
        product: Spot or one Futures settlement product.

    Returns:
        The immutable market-Kline declaration.
    """
    quantity = "contract_volume" if product == "coin_futures" else "quote_volume"
    return DatasetSpec(
        product=product,
        name="klines",
        remote_name="kline",
        source_columns=KLINE_SOURCE,
        stored_columns=(
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "base_volume",
            quantity,
        ),
        time_column="open_time",
        base_interval="1m",
        output_intervals=OUTPUT_INTERVALS,
        aliases=MappingProxyType({"volume": "base_volume"}),
        max_concurrency=32,
        csv_header="present",
        supports_resampling=True,
        supports_gap_policy=True,
        gap_semantics="sparse",
        resample_sum_columns=("base_volume", quantity),
        timestamp_columns=("open_time",),
        source_schemas=(KLINE_SOURCE_LEGACY,),
        sort_source_rows=True,
        archive_day_offset=ARCHIVE_DAY_OFFSET,
    )


def _trades(product: BitgetProduct) -> DatasetSpec:
    """Return one Bitget public-trade declaration.

    Args:
        product: Spot or one Futures settlement product.

    Returns:
        The immutable trade declaration.
    """
    return DatasetSpec(
        product=product,
        name="trades",
        remote_name="trade",
        source_columns=TRADE_SOURCE,
        stored_columns=(
            "event_time",
            "event_number",
            "price",
            "base_quantity",
            "quote_quantity",
            "side",
        ),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=16,
        csv_header="present",
        ordering_columns=("event_time", "event_number"),
        timestamp_columns=("event_time",),
        integer_columns=("event_number",),
        string_columns=("side",),
        sort_source_rows=True,
        archive_day_offset=ARCHIVE_DAY_OFFSET,
    )


def _best_book(product: BitgetProduct) -> DatasetSpec:
    """Return one Bitget best-book snapshot declaration.

    Args:
        product: Spot or one Futures settlement product.

    Returns:
        The immutable best-book declaration.
    """
    return DatasetSpec(
        product=product,
        name="best_book_snapshots",
        remote_name="depth_1",
        source_columns=BEST_BOOK_SOURCE,
        stored_columns=(
            "event_time",
            "event_number",
            "ask_price",
            "bid_price",
            "ask_quantity",
            "bid_quantity",
        ),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=8,
        csv_header="present",
        ordering_columns=("event_time", "event_number"),
        timestamp_columns=("event_time",),
        integer_columns=("event_number",),
        sort_source_rows=True,
        archive_day_offset=ARCHIVE_DAY_OFFSET,
    )


def _order_book(product: BitgetProduct) -> DatasetSpec:
    """Return one Bitget level-500 snapshot declaration.

    Args:
        product: Spot or one Futures settlement product.

    Returns:
        The immutable level-500 declaration.
    """
    return DatasetSpec(
        product=product,
        name="order_book_snapshots",
        remote_name="depth_500",
        source_columns=ORDER_BOOK_SOURCE,
        stored_columns=("event_time", "event_number", "bids", "asks"),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=4,
        csv_header="present",
        ordering_columns=("event_time", "event_number"),
        timestamp_columns=("event_time",),
        integer_columns=("event_number",),
        object_columns=("bids", "asks"),
        sort_source_rows=True,
        archive_day_offset=ARCHIVE_DAY_OFFSET,
    )


def _reference_klines(product: BitgetProduct, name: str) -> DatasetSpec:
    """Return one REST-backed Futures reference-Kline declaration.

    Args:
        product: One Futures settlement product.
        name: Canonical reference-Kline dataset name.

    Returns:
        The immutable reference-Kline declaration.
    """
    return DatasetSpec(
        product=product,
        name=name,
        remote_name=name.removesuffix("_klines"),
        source_columns=REFERENCE_KLINE_SOURCE,
        stored_columns=(
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "base_volume",
            "quote_volume",
        ),
        time_column="open_time",
        base_interval="1m",
        output_intervals=OUTPUT_INTERVALS,
        aliases=MappingProxyType({"volume": "base_volume"}),
        max_concurrency=8,
        supports_resampling=True,
        supports_gap_policy=True,
        gap_semantics="sparse",
        resample_sum_columns=("base_volume", "quote_volume"),
        timestamp_columns=("open_time",),
        sort_source_rows=True,
    )


def _funding(product: BitgetProduct) -> DatasetSpec:
    """Return one REST-backed Futures funding-rate declaration.

    Args:
        product: One Futures settlement product.

    Returns:
        The immutable funding-rate declaration.
    """
    return DatasetSpec(
        product=product,
        name="funding_rates",
        remote_name="history-fund-rate",
        source_columns=("symbol", "fundingRate", "fundingRateTimestamp"),
        stored_columns=("funding_time", "funding_rate"),
        time_column="funding_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=8,
        ordering_columns=("funding_time",),
        timestamp_columns=("funding_time",),
        sort_source_rows=True,
    )


DATASETS: Mapping[tuple[str, str], DatasetSpec] = MappingProxyType(
    {
        **{(product, "klines"): _klines(product) for product in PRODUCTS},
        **{(product, "trades"): _trades(product) for product in PRODUCTS},
        **{
            (product, "best_book_snapshots"): _best_book(product)
            for product in PRODUCTS
        },
        **{
            (product, "order_book_snapshots"): _order_book(product)
            for product in PRODUCTS
        },
        **{
            (product, name): _reference_klines(product, name)
            for product in FUTURES_PRODUCTS
            for name in (
                "mark_price_klines",
                "index_price_klines",
                "premium_index_klines",
            )
        },
        **{
            (product, "funding_rates"): _funding(product)
            for product in FUTURES_PRODUCTS
        },
    }
)


def supports(product: str, dataset: str) -> bool:
    """Return whether Bitget publishes one product and dataset combination.

    Args:
        product: Public Bitget product identifier.
        dataset: Canonical dataset name.

    Returns:
        Whether the combination is supported.
    """
    return dataset in SUPPORTED_DATASETS.get(product, frozenset())


def get_dataset(product: object, dataset: object) -> DatasetSpec:
    """Return the physical Bitget schema needed by one request.

    Args:
        product: Requested Bitget product.
        dataset: Requested canonical dataset.

    Returns:
        The matching immutable dataset declaration.
    """
    if not isinstance(product, str):
        raise TypeError("product must be a string")
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")
    if not supports(product, dataset):
        raise ValueError(f"unsupported dataset '{product}/{dataset}'")
    return DATASETS[(product, dataset)]
