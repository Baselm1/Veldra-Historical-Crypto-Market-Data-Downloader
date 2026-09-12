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

DATASETS: Mapping[tuple[str, str], DatasetSpec] = MappingProxyType(
    {("spot", "klines"): SPOT_KLINES, ("spot", "trades"): SPOT_TRADES}
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
