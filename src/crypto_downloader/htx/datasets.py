"""Declare HTX products, datasets, and archive interval spellings."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

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
