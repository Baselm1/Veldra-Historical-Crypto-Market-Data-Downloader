"""Declare OKX historical archive modules and capabilities."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import Literal

from veldra.core.subjects import SubjectKind
from veldra.core.datasets import DatasetSpec
from veldra.okx.identities import OKXProduct

type OKXDataset = Literal[
    "klines",
    "trades",
    "funding_rates",
    "order_book_400",
    "order_book_5000",
    "borrow_rates",
    "legacy_order_book_50",
]
type Cadence = Literal["daily", "monthly"]

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
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1mo",
)
KLINE_SOURCE_COLUMNS: tuple[str, ...] = (
    "instrument_name",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "vol_ccy",
    "vol_quote",
    "open_time",
    "confirm",
)
TRADE_SOURCE_COLUMNS: tuple[str, ...] = (
    "instrument_name",
    "trade_id",
    "side",
    "price",
    "size",
    "created_time",
)
FUNDING_SOURCE_COLUMNS: tuple[str, ...] = (
    "instrument_name",
    "funding_rate",
    "funding_time",
)
ORDER_BOOK_SOURCE_COLUMNS: tuple[str, ...] = (
    "instId",
    "action",
    "ts",
    "bids",
    "asks",
)
BORROW_RATE_SOURCE_COLUMNS: tuple[str, ...] = (
    "currency_name",
    "borrow_rate",
    "time",
)

SPOT_KLINES = DatasetSpec(
    product="spot",
    name="klines",
    remote_name="module_2",
    source_columns=KLINE_SOURCE_COLUMNS,
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
    output_intervals=KLINE_INTERVALS,
    aliases=MappingProxyType({"volume": "base_volume"}),
    max_concurrency=32,
    csv_header="present",
    supports_resampling=True,
    supports_gap_policy=True,
    resample_sum_columns=("base_volume", "quote_volume"),
    ordering_columns=("open_time",),
    timestamp_columns=("open_time",),
    sort_source_rows=True,
    archive_day_offset=timedelta(hours=8),
    publication_delay_days=2,
)

SPOT_TRADES = DatasetSpec(
    product="spot",
    name="trades",
    remote_name="module_1",
    source_columns=TRADE_SOURCE_COLUMNS,
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
    integer_columns=("trade_id",),
    string_columns=("side",),
    sort_source_rows=True,
    archive_day_offset=timedelta(hours=8),
    publication_delay_days=2,
)


def _perpetual_klines(product: str) -> DatasetSpec:
    """Build one perpetual Kline declaration with explicit volume units.

    Args:
        product: Linear- or inverse-margined swap product.

    Returns:
        Product-specific one-minute Kline schema.
    """
    return DatasetSpec(
        product=product,
        name="klines",
        remote_name="module_2",
        source_columns=KLINE_SOURCE_COLUMNS,
        stored_columns=(
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "contract_volume",
            "base_volume",
            "quote_volume",
        ),
        time_column="open_time",
        base_interval="1m",
        output_intervals=KLINE_INTERVALS,
        aliases=MappingProxyType({"volume": "contract_volume"}),
        max_concurrency=32,
        csv_header="present",
        supports_resampling=True,
        supports_gap_policy=True,
        resample_sum_columns=("contract_volume", "base_volume", "quote_volume"),
        ordering_columns=("open_time",),
        timestamp_columns=("open_time",),
        sort_source_rows=True,
        archive_day_offset=timedelta(hours=8),
        publication_delay_days=2,
    )


def _perpetual_trades(product: str) -> DatasetSpec:
    """Build one perpetual trade declaration with explicit contract units.

    Args:
        product: Linear- or inverse-margined swap product.

    Returns:
        Product-specific raw trade schema.
    """
    quote = "quote_quantity" if product == "linear_swap" else "quote_notional"
    return DatasetSpec(
        product=product,
        name="trades",
        remote_name="module_1",
        source_columns=TRADE_SOURCE_COLUMNS,
        stored_columns=(
            "event_time",
            "trade_id",
            "price",
            "contract_quantity",
            "base_quantity",
            quote,
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
        integer_columns=("trade_id",),
        string_columns=("side",),
        sort_source_rows=True,
        archive_day_offset=timedelta(hours=8),
        publication_delay_days=2,
    )


def _chain_trades(product: str) -> DatasetSpec:
    """Build one Futures or Options trade declaration without guessed units.

    Args:
        product: Dated Futures or Options product.

    Returns:
        Native contract-quantity trade schema.
    """
    return DatasetSpec(
        product=product,
        name="trades",
        remote_name="module_1",
        source_columns=TRADE_SOURCE_COLUMNS,
        stored_columns=(
            "event_time",
            "trade_id",
            "price",
            "contract_quantity",
            "side",
        ),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=8,
        csv_header="present",
        ordering_columns=("event_time", "trade_id"),
        timestamp_columns=("event_time",),
        integer_columns=("trade_id",),
        string_columns=("side",),
        sort_source_rows=True,
        archive_day_offset=timedelta(hours=8),
        publication_delay_days=2,
    )


LINEAR_SWAP_KLINES = _perpetual_klines("linear_swap")
INVERSE_SWAP_KLINES = _perpetual_klines("inverse_swap")
LINEAR_SWAP_TRADES = _perpetual_trades("linear_swap")
INVERSE_SWAP_TRADES = _perpetual_trades("inverse_swap")
LINEAR_FUTURES_KLINES = _perpetual_klines("linear_futures")
INVERSE_FUTURES_KLINES = _perpetual_klines("inverse_futures")
LINEAR_FUTURES_TRADES = _chain_trades("linear_futures")
INVERSE_FUTURES_TRADES = _chain_trades("inverse_futures")
OPTIONS_KLINES = _perpetual_klines("options")
OPTIONS_TRADES = _chain_trades("options")


def _funding_rates(product: str) -> DatasetSpec:
    """Build one perpetual funding observation declaration.

    Args:
        product: Linear- or inverse-margined swap product.

    Returns:
        Raw signed funding-rate schema.
    """
    return DatasetSpec(
        product=product,
        name="funding_rates",
        remote_name="module_3",
        source_columns=FUNDING_SOURCE_COLUMNS,
        stored_columns=("funding_time", "funding_rate"),
        time_column="funding_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=32,
        csv_header="present",
        ordering_columns=("funding_time",),
        timestamp_columns=("funding_time",),
        archive_day_offset=timedelta(hours=8),
        publication_delay_days=2,
    )


LINEAR_SWAP_FUNDING = _funding_rates("linear_swap")
INVERSE_SWAP_FUNDING = _funding_rates("inverse_swap")

MARGIN_BORROW_RATES = DatasetSpec(
    product="margin",
    name="borrow_rates",
    remote_name="module_11",
    source_columns=BORROW_RATE_SOURCE_COLUMNS,
    stored_columns=("event_time", "borrow_rate"),
    time_column="event_time",
    base_interval=None,
    output_intervals=(),
    aliases=MappingProxyType({}),
    max_concurrency=32,
    csv_header="present",
    ordering_columns=("event_time",),
    timestamp_columns=("event_time",),
    archive_day_offset=timedelta(hours=8),
    publication_delay_days=2,
)


def _order_book(product: str, depth: int) -> DatasetSpec:
    """Build one native OKX order-book update declaration.

    Args:
        product: Spot or perpetual product.
        depth: Maximum source-book depth.

    Returns:
        Nested event schema with native level quantities.
    """
    return DatasetSpec(
        product=product,
        name=f"order_book_{depth}",
        remote_name=f"module_{4 if depth == 400 else 5}",
        source_columns=ORDER_BOOK_SOURCE_COLUMNS,
        stored_columns=("event_time", "event_number", "action", "bids", "asks"),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases=MappingProxyType({}),
        max_concurrency=4 if depth == 400 else 2,
        ordering_columns=("event_time", "event_number"),
        timestamp_columns=("event_time",),
        integer_columns=("event_number",),
        string_columns=("action",),
        object_columns=("bids", "asks"),
        archive_day_offset=timedelta(0),
        publication_delay_days=3,
    )


SPOT_ORDER_BOOK_400 = _order_book("spot", 400)
SPOT_ORDER_BOOK_5000 = _order_book("spot", 5000)
LINEAR_SWAP_ORDER_BOOK_400 = _order_book("linear_swap", 400)
LINEAR_SWAP_ORDER_BOOK_5000 = _order_book("linear_swap", 5000)
INVERSE_SWAP_ORDER_BOOK_400 = _order_book("inverse_swap", 400)
INVERSE_SWAP_ORDER_BOOK_5000 = _order_book("inverse_swap", 5000)
OPTIONS_ORDER_BOOK_400 = _order_book("options", 400)
OPTIONS_ORDER_BOOK_5000 = _order_book("options", 5000)

DATASETS: Mapping[tuple[str, str], DatasetSpec] = MappingProxyType(
    {
        ("spot", "klines"): SPOT_KLINES,
        ("spot", "trades"): SPOT_TRADES,
        ("linear_swap", "klines"): LINEAR_SWAP_KLINES,
        ("linear_swap", "trades"): LINEAR_SWAP_TRADES,
        ("inverse_swap", "klines"): INVERSE_SWAP_KLINES,
        ("inverse_swap", "trades"): INVERSE_SWAP_TRADES,
        ("linear_swap", "funding_rates"): LINEAR_SWAP_FUNDING,
        ("inverse_swap", "funding_rates"): INVERSE_SWAP_FUNDING,
        ("margin", "borrow_rates"): MARGIN_BORROW_RATES,
        ("linear_futures", "klines"): LINEAR_FUTURES_KLINES,
        ("linear_futures", "trades"): LINEAR_FUTURES_TRADES,
        ("inverse_futures", "klines"): INVERSE_FUTURES_KLINES,
        ("inverse_futures", "trades"): INVERSE_FUTURES_TRADES,
        ("options", "klines"): OPTIONS_KLINES,
        ("options", "trades"): OPTIONS_TRADES,
        ("spot", "order_book_400"): SPOT_ORDER_BOOK_400,
        ("spot", "order_book_5000"): SPOT_ORDER_BOOK_5000,
        ("linear_swap", "order_book_400"): LINEAR_SWAP_ORDER_BOOK_400,
        ("linear_swap", "order_book_5000"): LINEAR_SWAP_ORDER_BOOK_5000,
        ("inverse_swap", "order_book_400"): INVERSE_SWAP_ORDER_BOOK_400,
        ("inverse_swap", "order_book_5000"): INVERSE_SWAP_ORDER_BOOK_5000,
        ("options", "order_book_400"): OPTIONS_ORDER_BOOK_400,
        ("options", "order_book_5000"): OPTIONS_ORDER_BOOK_5000,
    }
)


@dataclass(frozen=True)
class ManifestSpec:
    """Declare one OKX manifest module and physical archive behavior."""

    module: int
    archive_day_offset: int
    publication_delay_days: int
    supports_monthly: bool
    supports_any_daily: bool
    subject_kind: SubjectKind
    max_download_workers: int


_MODULES: dict[OKXDataset, tuple[int, int, int, bool, bool, int]] = {
    "klines": (2, 8 * 3600, 2, True, True, 32),
    "trades": (1, 8 * 3600, 2, True, True, 16),
    "funding_rates": (3, 8 * 3600, 2, True, True, 32),
    "order_book_400": (4, 0, 3, False, False, 4),
    "order_book_5000": (5, 0, 3, False, False, 2),
    "borrow_rates": (11, 8 * 3600, 2, True, True, 32),
    "legacy_order_book_50": (6, 0, 3, False, False, 1),
}


def _subject_kind(product: OKXProduct, dataset: OKXDataset) -> SubjectKind:
    """Return the specific archive scope required by one combination.

    Args:
        product: Veldra OKX product.
        dataset: Historical dataset.

    Returns:
        Instrument, family, or currency scope.
    """
    if dataset == "borrow_rates":
        return "currency"
    if product in {"spot", "margin"}:
        return "instrument"
    return "instrument_family"


def manifest_spec(product: str, dataset: str) -> ManifestSpec:
    """Return the archive declaration for one supported combination.

    Args:
        product: Veldra OKX product.
        dataset: Public historical dataset.

    Returns:
        Immutable manifest capabilities.
    """
    if product not in {
        "spot",
        "margin",
        "linear_swap",
        "inverse_swap",
        "linear_futures",
        "inverse_futures",
        "options",
    }:
        raise ValueError(f"unsupported OKX product {product!r}")
    if dataset not in _MODULES:
        raise ValueError(f"unsupported OKX dataset {dataset!r}")
    typed_product: OKXProduct = product  # type: ignore[assignment]
    typed_dataset: OKXDataset = dataset
    if typed_dataset == "funding_rates" and typed_product not in {
        "linear_swap",
        "inverse_swap",
        "linear_futures",
        "inverse_futures",
    }:
        raise ValueError(f"unsupported OKX dataset {product}/{dataset}")
    if typed_dataset == "borrow_rates" and typed_product != "margin":
        raise ValueError(f"unsupported OKX dataset {product}/{dataset}")
    if (
        typed_dataset.startswith("order_book")
        or typed_dataset == "legacy_order_book_50"
    ):
        pass
    module, offset, delay, monthly, any_daily, workers = _MODULES[typed_dataset]
    return ManifestSpec(
        module,
        offset,
        delay,
        monthly,
        any_daily,
        _subject_kind(typed_product, typed_dataset),
        workers,
    )


def get_dataset(
    product: object, dataset: object, *, kline_base_interval: object = "1m"
) -> DatasetSpec:
    """Return one implemented OKX canonical dataset declaration.

    Args:
        product: Public Veldra OKX product.
        dataset: Public historical dataset.
        kline_base_interval: Configured cached Kline resolution.

    Returns:
        Matching immutable dataset declaration.
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
        raise ValueError("OKX currently stores Klines at the 1m archive interval")
    return specification
