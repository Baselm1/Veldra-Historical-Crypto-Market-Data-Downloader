"""Declare Upbit Spot datasets and physical archive intervals."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from veldra.core.datasets import DatasetSpec

type UpbitProduct = Literal["spot"]
type UpbitDataset = Literal["klines", "trades"]

PRODUCTS: tuple[str, ...] = ("spot",)
ARCHIVE_KLINE_INTERVALS: tuple[str, ...] = (
    "1s",
    "1m",
    "3m",
    "5m",
    "10m",
    "15m",
    "30m",
    "60m",
    "240m",
    "day",
    "week",
)
OUTPUT_INTERVALS: tuple[str, ...] = (
    "1m",
    "3m",
    "5m",
    "10m",
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
    {"spot": frozenset({"klines", "trades"})}
)

KLINE_COLUMNS = (
    "date_time_utc",
    "open",
    "high",
    "low",
    "close",
    "acc_trade_price",
    "acc_trade_volume",
)
TRADE_COLUMNS = ("seq", "timestamp", "volume", "price", "ask_bid")
STORED_KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "quote_volume",
)


def _klines(base_interval: Literal["1s", "1m"]) -> DatasetSpec:
    """Return an Upbit Kline declaration for one physical resolution."""
    return DatasetSpec(
        product="spot",
        name="klines",
        remote_name="candle",
        source_columns=KLINE_COLUMNS,
        stored_columns=STORED_KLINE_COLUMNS,
        time_column="open_time",
        base_interval=base_interval,
        output_intervals=("1s",) if base_interval == "1s" else OUTPUT_INTERVALS,
        aliases=MappingProxyType({"volume": "base_volume"}),
        max_concurrency=32,
        csv_header="present",
        supports_resampling=True,
        supports_gap_policy=True,
        gap_semantics="sparse",
        resample_sum_columns=("base_volume", "quote_volume"),
        timestamp_columns=("open_time",),
        archive_symbol_attribute="pair",
        sort_source_rows=True,
    )


SECOND_KLINES = _klines("1s")
MINUTE_KLINES = _klines("1m")
SPOT_TRADES = DatasetSpec(
    product="spot",
    name="trades",
    remote_name="trade",
    source_columns=TRADE_COLUMNS,
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
    archive_symbol_attribute="pair",
    sort_source_rows=True,
)


def supports(product: str, dataset: str) -> bool:
    """Return whether Upbit publishes one product and dataset combination."""
    return dataset in SUPPORTED_DATASETS.get(product, frozenset())


def get_dataset(
    product: object,
    dataset: object,
    *,
    kline_base_interval: object = "1m",
    requested_interval: object = None,
) -> DatasetSpec:
    """Return the physical Upbit schema needed by one public request.

    Args:
        product: The requested Upbit product.
        dataset: The requested canonical dataset.
        kline_base_interval: The configured coarse Kline storage interval.
        requested_interval: The caller's requested output interval.

    Returns:
        The matching immutable physical dataset declaration.
    """
    if not isinstance(product, str):
        raise TypeError("product must be a string")
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")
    if not supports(product, dataset):
        raise ValueError(f"unsupported dataset '{product}/{dataset}'")
    if dataset == "trades":
        return SPOT_TRADES
    if requested_interval == "1s":
        return SECOND_KLINES
    if kline_base_interval != "1m":
        raise ValueError(
            "Upbit currently stores coarse Klines at the 1m archive interval"
        )
    return MINUTE_KLINES
