"""Declare OKX historical archive modules and capabilities."""

from dataclasses import dataclass
from typing import Literal

from veldra.core.subjects import SubjectKind
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
