"""Declare HTX products, datasets, and archive interval spellings."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from crypto_downloader.core.datasets import CsvSchema, DatasetSpec

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

DATASETS: Mapping[tuple[str, str], DatasetSpec] = MappingProxyType(
    {("spot", "klines"): SPOT_KLINES}
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
