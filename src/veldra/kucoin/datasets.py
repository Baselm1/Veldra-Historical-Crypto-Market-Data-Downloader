"""Declare KuCoin products, datasets, and canonical schemas."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from veldra.core.datasets import DatasetSpec

type KuCoinProduct = Literal["spot", "linear_futures", "inverse_futures"]
type KuCoinDataset = Literal[
    "klines",
    "trades",
    "index_price_klines",
    "mark_price_klines",
    "funding_rates",
    "order_book_snapshots",
]

PRODUCTS: tuple[str, ...] = ("spot", "linear_futures", "inverse_futures")
ARCHIVE_KLINE_INTERVALS: tuple[str, ...] = (
    "1m",
    "5m",
    "15m",
    "1h",
    "8h",
    "12h",
    "1d",
)
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
        "spot": frozenset({"klines", "trades", "order_book_snapshots"}),
        "linear_futures": frozenset(
            {
                "klines",
                "trades",
                "index_price_klines",
                "mark_price_klines",
                "funding_rates",
                "order_book_snapshots",
            }
        ),
        "inverse_futures": frozenset(
            {
                "klines",
                "trades",
                "index_price_klines",
                "mark_price_klines",
                "funding_rates",
                "order_book_snapshots",
            }
        ),
    }
)

KLINE_COLUMNS = ("time", "open", "close", "high", "low", "volume", "turnover")
FUTURES_KLINE_COLUMNS = ("time", "open", "high", "low", "close", "volume")
REFERENCE_KLINE_COLUMNS = ("time", "open", "high", "low", "close")
TRADE_COLUMNS = ("trade_id", "trade_time", "price", "size", "side")
FUNDING_COLUMNS = ("symbol", "time", "fundingRate")


def _spot_klines() -> DatasetSpec:
    """Return KuCoin's Spot Kline declaration."""
    return DatasetSpec(
        product="spot",
        name="klines",
        remote_name="klines",
        source_columns=KLINE_COLUMNS,
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
        max_concurrency=64,
        csv_header="present",
        supports_resampling=True,
        supports_gap_policy=True,
        resample_sum_columns=("base_volume", "quote_volume"),
        timestamp_columns=("open_time",),
        archive_symbol_attribute="pair",
        sort_source_rows=True,
    )


def _spot_trades() -> DatasetSpec:
    """Return KuCoin's Spot trade declaration."""
    return DatasetSpec(
        product="spot",
        name="trades",
        remote_name="trades",
        source_columns=TRADE_COLUMNS,
        stored_columns=(
            "event_time",
            "trade_id",
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
        ordering_columns=("event_time", "trade_id"),
        timestamp_columns=("event_time",),
        string_columns=("trade_id", "side"),
        archive_symbol_attribute="pair",
    )


def _futures_klines(product: str) -> DatasetSpec:
    """Return one KuCoin perpetual trade-price Kline declaration."""
    return DatasetSpec(
        product=product,
        name="klines",
        remote_name="klines",
        source_columns=FUTURES_KLINE_COLUMNS,
        stored_columns=(
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "contract_volume",
        ),
        time_column="open_time",
        base_interval="1m",
        output_intervals=OUTPUT_INTERVALS,
        aliases=MappingProxyType({"volume": "contract_volume"}),
        max_concurrency=64,
        csv_header="present",
        supports_resampling=True,
        supports_gap_policy=True,
        resample_sum_columns=("contract_volume",),
        timestamp_columns=("open_time",),
        archive_symbol_attribute="pair",
        sort_source_rows=True,
    )


def _reference_klines(product: str, name: str) -> DatasetSpec:
    """Return one KuCoin perpetual reference-price Kline declaration."""
    return DatasetSpec(
        product=product,
        name=name,
        remote_name=name,
        source_columns=REFERENCE_KLINE_COLUMNS,
        stored_columns=("open_time", "open", "high", "low", "close", "sample_count"),
        time_column="open_time",
        base_interval="1m",
        output_intervals=OUTPUT_INTERVALS,
        aliases=MappingProxyType({"count": "sample_count"}),
        max_concurrency=64,
        csv_header="present",
        supports_resampling=True,
        supports_gap_policy=True,
        resample_sum_columns=("sample_count",),
        timestamp_columns=("open_time",),
        integer_columns=("sample_count",),
        archive_symbol_attribute="pair",
        sort_source_rows=True,
    )


def _futures_trades(product: str) -> DatasetSpec:
    """Return one KuCoin perpetual trade declaration."""
    inverse = product == "inverse_futures"
    final_quantity = "quote_notional" if inverse else "quote_quantity"
    return DatasetSpec(
        product=product,
        name="trades",
        remote_name="trades",
        source_columns=TRADE_COLUMNS,
        stored_columns=(
            "event_time",
            "trade_id",
            "price",
            "contract_quantity",
            "base_quantity",
            final_quantity,
            "side",
        ),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=16,
        csv_header="present",
        requires_contract_size=True,
        ordering_columns=("event_time", "trade_id"),
        timestamp_columns=("event_time",),
        string_columns=("trade_id", "side"),
        archive_symbol_attribute="pair",
    )


def _funding_rates(product: str) -> DatasetSpec:
    """Return one KuCoin perpetual funding-rate declaration."""
    return DatasetSpec(
        product=product,
        name="funding_rates",
        remote_name="fundingRates",
        source_columns=FUNDING_COLUMNS,
        stored_columns=("funding_time", "funding_rate"),
        time_column="funding_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=16,
        csv_header="present",
        ordering_columns=("funding_time",),
        timestamp_columns=("funding_time",),
        archive_symbol_attribute="pair",
        sort_source_rows=True,
    )


def _order_book_snapshots(product: str) -> DatasetSpec:
    """Return one nested KuCoin level-50 order-book declaration."""
    futures = product != "spot"
    columns = (
        ("event_time", "sequence", "bids", "asks")
        if futures
        else ("event_time", "bids", "asks")
    )
    return DatasetSpec(
        product=product,
        name="order_book_snapshots",
        remote_name="orderbooklv50",
        source_columns=columns,
        stored_columns=columns,
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=4,
        schema_version=1,
        ordering_columns=("event_time", "sequence") if futures else ("event_time",),
        timestamp_columns=("event_time",),
        integer_columns=("sequence",) if futures else (),
        object_columns=("bids", "asks"),
        archive_symbol_attribute="pair",
        discovery_lookahead_days=1,
    )


SPOT_KLINES = _spot_klines()
SPOT_TRADES = _spot_trades()
LINEAR_KLINES = _futures_klines("linear_futures")
INVERSE_KLINES = _futures_klines("inverse_futures")
LINEAR_TRADES = _futures_trades("linear_futures")
INVERSE_TRADES = _futures_trades("inverse_futures")
LINEAR_INDEX_PRICE_KLINES = _reference_klines("linear_futures", "index_price_klines")
INVERSE_INDEX_PRICE_KLINES = _reference_klines("inverse_futures", "index_price_klines")
LINEAR_MARK_PRICE_KLINES = _reference_klines("linear_futures", "mark_price_klines")
INVERSE_MARK_PRICE_KLINES = _reference_klines("inverse_futures", "mark_price_klines")
LINEAR_FUNDING_RATES = _funding_rates("linear_futures")
INVERSE_FUNDING_RATES = _funding_rates("inverse_futures")
SPOT_ORDER_BOOK_SNAPSHOTS = _order_book_snapshots("spot")
LINEAR_ORDER_BOOK_SNAPSHOTS = _order_book_snapshots("linear_futures")
INVERSE_ORDER_BOOK_SNAPSHOTS = _order_book_snapshots("inverse_futures")

DATASETS: Mapping[tuple[str, str], DatasetSpec] = MappingProxyType(
    {
        ("spot", "klines"): SPOT_KLINES,
        ("spot", "trades"): SPOT_TRADES,
        ("linear_futures", "klines"): LINEAR_KLINES,
        ("inverse_futures", "klines"): INVERSE_KLINES,
        ("linear_futures", "trades"): LINEAR_TRADES,
        ("inverse_futures", "trades"): INVERSE_TRADES,
        ("linear_futures", "index_price_klines"): LINEAR_INDEX_PRICE_KLINES,
        ("inverse_futures", "index_price_klines"): INVERSE_INDEX_PRICE_KLINES,
        ("linear_futures", "mark_price_klines"): LINEAR_MARK_PRICE_KLINES,
        ("inverse_futures", "mark_price_klines"): INVERSE_MARK_PRICE_KLINES,
        ("linear_futures", "funding_rates"): LINEAR_FUNDING_RATES,
        ("inverse_futures", "funding_rates"): INVERSE_FUNDING_RATES,
        ("spot", "order_book_snapshots"): SPOT_ORDER_BOOK_SNAPSHOTS,
        ("linear_futures", "order_book_snapshots"): LINEAR_ORDER_BOOK_SNAPSHOTS,
        ("inverse_futures", "order_book_snapshots"): INVERSE_ORDER_BOOK_SNAPSHOTS,
    }
)


@dataclass(frozen=True)
class ArchiveRoute:
    """Describe one KuCoin archive folder and filename prefix."""

    prefix: str
    stem: str


def supports(product: str, dataset: str) -> bool:
    """Return whether KuCoin publishes a product and dataset combination."""
    return dataset in SUPPORTED_DATASETS.get(product, frozenset())


def get_dataset(
    product: object, dataset: object, *, kline_base_interval: object = "1m"
) -> DatasetSpec:
    """Return one implemented KuCoin dataset declaration.

    Args:
        product: The public KuCoin product name.
        dataset: The canonical dataset name.
        kline_base_interval: The configured Kline storage interval.

    Returns:
        The matching immutable dataset declaration.
    """
    if not isinstance(product, str):
        raise TypeError("product must be a string")
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")
    try:
        specification = DATASETS[(product, dataset)]
    except KeyError as error:
        raise ValueError(f"unsupported dataset '{product}/{dataset}'") from error
    if specification.needs_interval and kline_base_interval != "1m":
        raise ValueError("KuCoin currently stores Klines at the 1m archive interval")
    return specification
