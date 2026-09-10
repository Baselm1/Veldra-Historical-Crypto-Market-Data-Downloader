"""Declare HTX products, datasets, and archive interval spellings."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from veldra.core.datasets import CsvSchema, DatasetSpec

type HTXProduct = Literal["spot", "linear_swap", "coin_swap"]
type HTXDataset = Literal[
    "klines",
    "trades",
    "index_price_klines",
    "mark_price_klines",
    "funding_rates",
    "order_book_updates",
]

PRODUCTS: tuple[str, ...] = ("spot", "linear_swap", "coin_swap")
KLINE_INTERVALS: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")
OLD_INTERVALS: Mapping[str, str] = MappingProxyType(
    {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "60min",
        "4h": "4hour",
        "1d": "1day",
    }
)
NEW_INTERVALS: Mapping[str, str] = MappingProxyType(
    {interval: interval for interval in KLINE_INTERVALS}
)
SUPPORTED_DATASETS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "spot": frozenset({"klines", "trades", "order_book_updates"}),
        "linear_swap": frozenset(
            {
                "klines",
                "trades",
                "index_price_klines",
                "mark_price_klines",
                "funding_rates",
                "order_book_updates",
            }
        ),
        "coin_swap": frozenset(
            {
                "klines",
                "trades",
                "index_price_klines",
                "mark_price_klines",
                "order_book_updates",
            }
        ),
    }
)

KLINE_OUTPUT_INTERVALS: tuple[str, ...] = (
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
OLD_KLINE_COLUMNS = ("id", "open", "close", "high", "low", "vol", "amount")
NEW_KLINE_COLUMNS = (
    "instId",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "volCcy",
    "volCcyQuote",
    "ts",
)
OLD_TRADE_COLUMNS = ("id", "ts", "price", "amount", "direction")
NEW_TRADE_COLUMNS = ("instId", "tradeId", "px", "side", "size", "ts")
NEW_REFERENCE_KLINE_COLUMNS = ("instId", "open", "high", "low", "close", "ts")
NEW_FUNDING_COLUMNS = ("instId", "fundingRate", "fundingTime")
OLD_LINEAR_TRADE_COLUMNS = (
    "id",
    "ts",
    "price",
    "amount",
    "quantity",
    "trade_turnover",
    "direction",
)
OLD_COIN_TRADE_COLUMNS = (
    "id",
    "ts",
    "price",
    "amount",
    "quantity",
    "direction",
)
SPOT_KLINES = DatasetSpec(
    product="spot",
    name="klines",
    remote_name="klines",
    source_columns=OLD_KLINE_COLUMNS,
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
    output_intervals=KLINE_OUTPUT_INTERVALS,
    aliases=MappingProxyType({"volume": "base_volume"}),
    max_concurrency=32,
    schema_version=1,
    supports_resampling=True,
    supports_gap_policy=True,
    resample_sum_columns=("base_volume", "quote_volume"),
    ordering_columns=("open_time",),
    timestamp_columns=("open_time",),
    archive_symbol_attribute="pair",
    source_schemas=(
        CsvSchema(OLD_KLINE_COLUMNS, "present"),
        CsvSchema(NEW_KLINE_COLUMNS, "present"),
    ),
    sort_source_rows=True,
)
SPOT_TRADES = DatasetSpec(
    product="spot",
    name="trades",
    remote_name="trades",
    source_columns=OLD_TRADE_COLUMNS,
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
    ordering_columns=("event_time", "trade_id"),
    timestamp_columns=("event_time",),
    integer_columns=("trade_id",),
    string_columns=("side",),
    archive_symbol_attribute="pair",
    source_schemas=(
        CsvSchema(OLD_TRADE_COLUMNS, "present"),
        CsvSchema(NEW_TRADE_COLUMNS, "present"),
    ),
    sort_source_rows=True,
)


def _perpetual_klines(product: str) -> DatasetSpec:
    """Build one perpetual Kline declaration.

    Args:
        product: The linear- or coin-margined product.

    Returns:
        A canonical Kline schema shared by both perpetual products.
    """
    return DatasetSpec(
        product=product,
        name="klines",
        remote_name="klines",
        source_columns=OLD_KLINE_COLUMNS,
        stored_columns=(
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "contract_volume",
            "base_volume",
        ),
        time_column="open_time",
        base_interval="1m",
        output_intervals=KLINE_OUTPUT_INTERVALS,
        aliases=MappingProxyType({"volume": "contract_volume"}),
        max_concurrency=32,
        supports_resampling=True,
        supports_gap_policy=True,
        resample_sum_columns=("contract_volume", "base_volume"),
        ordering_columns=("open_time",),
        timestamp_columns=("open_time",),
        archive_symbol_attribute="pair",
        source_schemas=(
            CsvSchema(OLD_KLINE_COLUMNS, "present"),
            CsvSchema(NEW_KLINE_COLUMNS, "present"),
        ),
        sort_source_rows=True,
    )


def _perpetual_trades(product: str) -> DatasetSpec:
    """Build one perpetual trade declaration with explicit quantity units.

    Args:
        product: The linear- or coin-margined product.

    Returns:
        The product-specific canonical trade schema.
    """
    linear = product == "linear_swap"
    old_columns = OLD_LINEAR_TRADE_COLUMNS if linear else OLD_COIN_TRADE_COLUMNS
    quote_column = "quote_quantity" if linear else "quote_notional"
    return DatasetSpec(
        product=product,
        name="trades",
        remote_name="trades",
        source_columns=old_columns,
        stored_columns=(
            "event_time",
            "trade_id",
            "price",
            "contract_quantity",
            "base_quantity",
            quote_column,
            "side",
        ),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=16,
        requires_contract_size=True,
        ordering_columns=("event_time", "trade_id"),
        timestamp_columns=("event_time",),
        integer_columns=("trade_id",),
        string_columns=("side",),
        archive_symbol_attribute="pair",
        source_schemas=(
            CsvSchema(old_columns, "present"),
            CsvSchema(NEW_TRADE_COLUMNS, "present"),
        ),
        sort_source_rows=True,
    )


def _reference_klines(product: str, name: str) -> DatasetSpec:
    """Build one perpetual reference-price Kline declaration.

    Args:
        product: The linear- or coin-margined product.
        name: The index- or mark-price dataset name.

    Returns:
        A resampleable OHLC schema for HTX's new archive tree.
    """
    return DatasetSpec(
        product=product,
        name=name,
        remote_name=name,
        source_columns=NEW_REFERENCE_KLINE_COLUMNS,
        stored_columns=(
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "sample_count",
        ),
        time_column="open_time",
        base_interval="1m",
        output_intervals=KLINE_OUTPUT_INTERVALS,
        aliases=MappingProxyType({"count": "sample_count"}),
        max_concurrency=32,
        csv_header="present",
        supports_resampling=True,
        supports_gap_policy=True,
        resample_sum_columns=("sample_count",),
        ordering_columns=("open_time",),
        timestamp_columns=("open_time",),
        integer_columns=("sample_count",),
        archive_symbol_attribute="pair",
        sort_source_rows=True,
    )


def _funding_rates() -> DatasetSpec:
    """Build the linear-swap funding-rate declaration.

    Returns:
        The signed point-in-time funding-rate schema.
    """
    return DatasetSpec(
        product="linear_swap",
        name="funding_rates",
        remote_name="funding_rates",
        source_columns=NEW_FUNDING_COLUMNS,
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


def _order_book_updates(product: str) -> DatasetSpec:
    """Build one flattened snapshot-and-update declaration.

    Args:
        product: The Spot or perpetual product.

    Returns:
        A point-in-time order-book event schema.
    """
    columns = (
        "event_time",
        "event_number",
        "action",
        "side",
        "level_number",
        "price",
        "quantity",
    )
    return DatasetSpec(
        product=product,
        name="order_book_updates",
        remote_name="order_book_updates",
        source_columns=columns,
        stored_columns=columns,
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=8,
        ordering_columns=("event_time", "event_number", "side", "level_number"),
        timestamp_columns=("event_time",),
        integer_columns=("event_number", "level_number"),
        string_columns=("action", "side"),
        archive_symbol_attribute="pair",
    )


LINEAR_KLINES = _perpetual_klines("linear_swap")
COIN_KLINES = _perpetual_klines("coin_swap")
LINEAR_TRADES = _perpetual_trades("linear_swap")
COIN_TRADES = _perpetual_trades("coin_swap")
LINEAR_INDEX_PRICE_KLINES = _reference_klines("linear_swap", "index_price_klines")
COIN_INDEX_PRICE_KLINES = _reference_klines("coin_swap", "index_price_klines")
LINEAR_MARK_PRICE_KLINES = _reference_klines("linear_swap", "mark_price_klines")
COIN_MARK_PRICE_KLINES = _reference_klines("coin_swap", "mark_price_klines")
LINEAR_FUNDING_RATES = _funding_rates()
SPOT_ORDER_BOOK_UPDATES = _order_book_updates("spot")
LINEAR_ORDER_BOOK_UPDATES = _order_book_updates("linear_swap")
COIN_ORDER_BOOK_UPDATES = _order_book_updates("coin_swap")

DATASETS: Mapping[tuple[str, str], DatasetSpec] = MappingProxyType(
    {
        ("spot", "klines"): SPOT_KLINES,
        ("spot", "trades"): SPOT_TRADES,
        ("linear_swap", "klines"): LINEAR_KLINES,
        ("linear_swap", "trades"): LINEAR_TRADES,
        ("coin_swap", "klines"): COIN_KLINES,
        ("coin_swap", "trades"): COIN_TRADES,
        ("linear_swap", "index_price_klines"): LINEAR_INDEX_PRICE_KLINES,
        ("coin_swap", "index_price_klines"): COIN_INDEX_PRICE_KLINES,
        ("linear_swap", "mark_price_klines"): LINEAR_MARK_PRICE_KLINES,
        ("coin_swap", "mark_price_klines"): COIN_MARK_PRICE_KLINES,
        ("linear_swap", "funding_rates"): LINEAR_FUNDING_RATES,
        ("spot", "order_book_updates"): SPOT_ORDER_BOOK_UPDATES,
        ("linear_swap", "order_book_updates"): LINEAR_ORDER_BOOK_UPDATES,
        ("coin_swap", "order_book_updates"): COIN_ORDER_BOOK_UPDATES,
    }
)


@dataclass(frozen=True)
class ArchiveRoute:
    """Describe one HTX archive tree path and filename convention."""

    generation: Literal["old", "new"]
    prefix: str
    stem: str
    suffix: Literal[".zip", ".tar.gz"] = ".zip"


def supports(product: str, dataset: str) -> bool:
    """Return whether HTX publishes a dataset for a product.

    Args:
        product: The public HTX product name.
        dataset: The canonical dataset name.

    Returns:
        Whether the product and dataset combination is supported.
    """
    return dataset in SUPPORTED_DATASETS.get(product, frozenset())


def get_dataset(
    product: object, dataset: object, *, kline_base_interval: object = "1m"
) -> DatasetSpec:
    """Return an implemented HTX dataset declaration.

    Args:
        product: The public HTX product name.
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
    if (
        specification.needs_interval
        and kline_base_interval != specification.base_interval
    ):
        raise ValueError("HTX currently stores Klines at the 1m archive interval")
    return specification
