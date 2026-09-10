"""Declare Binance archive schemas and dataset capabilities."""

from collections.abc import Mapping
from dataclasses import replace
import logging
from types import MappingProxyType
from veldra.core.datasets import DatasetSpec, Columns

LOGGER = logging.getLogger(__name__)

SPOT_KLINE_SOURCE_COLUMNS: Columns = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)

SPOT_KLINE_STORED_COLUMNS: Columns = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)

SPOT_KLINE_OUTPUT_INTERVALS: Columns = (
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

SPOT_TRADE_SOURCE_COLUMNS: Columns = (
    "trade_id",
    "price",
    "base_quantity",
    "quote_quantity",
    "event_time",
    "is_buyer_maker",
    "is_best_match",
)

SPOT_TRADE_STORED_COLUMNS: Columns = (
    "trade_id",
    "price",
    "base_quantity",
    "quote_quantity",
    "event_time",
    "buyer_is_maker",
)

SPOT_AGG_TRADE_SOURCE_COLUMNS: Columns = (
    "agg_trade_id",
    "price",
    "base_quantity",
    "first_trade_id",
    "last_trade_id",
    "event_time",
    "is_buyer_maker",
    "is_best_match",
)

SPOT_AGG_TRADE_STORED_COLUMNS: Columns = (
    "agg_trade_id",
    "first_trade_id",
    "last_trade_id",
    "price",
    "base_quantity",
    "quote_quantity",
    "event_time",
    "buyer_is_maker",
)


SPOT_KLINES = DatasetSpec(
    product="spot",
    name="klines",
    remote_name="klines",
    source_columns=SPOT_KLINE_SOURCE_COLUMNS,
    stored_columns=SPOT_KLINE_STORED_COLUMNS,
    time_column="open_time",
    base_interval="1m",
    output_intervals=SPOT_KLINE_OUTPUT_INTERVALS,
    aliases=MappingProxyType({"base_volume": "volume"}),
    max_concurrency=64,
    csv_header="absent",
    schema_version=1,
    supports_resampling=True,
    supports_gap_policy=True,
    resample_sum_columns=(
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ),
    ordering_columns=("open_time",),
    timestamp_columns=("open_time", "close_time"),
    integer_columns=("trade_count",),
)

UM_KLINES = replace(
    SPOT_KLINES,
    product="um",
    stored_columns=(
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ),
    aliases=MappingProxyType({}),
    csv_header="present",
    resample_sum_columns=(
        "base_volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ),
)

CM_KLINES = replace(
    SPOT_KLINES,
    product="cm",
    stored_columns=(
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "contract_volume",
        "close_time",
        "base_volume",
        "trade_count",
        "taker_buy_contract_volume",
        "taker_buy_base_volume",
    ),
    aliases=MappingProxyType({}),
    csv_header="present",
    resample_sum_columns=(
        "contract_volume",
        "base_volume",
        "trade_count",
        "taker_buy_contract_volume",
        "taker_buy_base_volume",
    ),
)

UM_MARK_PRICE_KLINES = DatasetSpec(
    product="um",
    name="mark_price_klines",
    remote_name="markPriceKlines",
    source_columns=SPOT_KLINE_SOURCE_COLUMNS,
    stored_columns=(
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "close_time",
        "sample_count",
    ),
    time_column="open_time",
    base_interval="1m",
    output_intervals=SPOT_KLINE_OUTPUT_INTERVALS,
    aliases=MappingProxyType({"count": "sample_count"}),
    max_concurrency=64,
    csv_header="present",
    schema_version=1,
    supports_resampling=True,
    supports_gap_policy=True,
    resample_sum_columns=("sample_count",),
    ordering_columns=("open_time",),
    timestamp_columns=("open_time", "close_time"),
    integer_columns=("sample_count",),
)

CM_MARK_PRICE_KLINES = replace(
    UM_MARK_PRICE_KLINES,
    product="cm",
)

UM_INDEX_PRICE_KLINES = replace(
    UM_MARK_PRICE_KLINES,
    name="index_price_klines",
    remote_name="indexPriceKlines",
)

CM_INDEX_PRICE_KLINES = replace(
    UM_INDEX_PRICE_KLINES,
    product="cm",
    archive_symbol_attribute="pair",
)

UM_PREMIUM_INDEX_KLINES = replace(
    UM_MARK_PRICE_KLINES,
    name="premium_index_klines",
    remote_name="premiumIndexKlines",
)

CM_PREMIUM_INDEX_KLINES = replace(
    UM_PREMIUM_INDEX_KLINES,
    product="cm",
)

METRICS_SOURCE_COLUMNS: Columns = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

METRICS_RATIO_COLUMNS: Columns = (
    "top_trader_account_long_short_ratio",
    "top_trader_position_long_short_ratio",
    "account_long_short_ratio",
    "taker_long_short_volume_ratio",
)

UM_METRICS = DatasetSpec(
    product="um",
    name="metrics",
    remote_name="metrics",
    source_columns=METRICS_SOURCE_COLUMNS,
    stored_columns=(
        "event_time",
        "open_interest_base_quantity",
        "open_interest_quote_value",
        *METRICS_RATIO_COLUMNS,
    ),
    time_column="event_time",
    base_interval=None,
    output_intervals=(),
    aliases=MappingProxyType({}),
    max_concurrency=16,
    csv_header="present",
    schema_version=1,
    ordering_columns=("event_time",),
    timestamp_columns=("event_time",),
)

CM_METRICS = replace(
    UM_METRICS,
    product="cm",
    stored_columns=(
        "event_time",
        "open_interest_contract_quantity",
        "open_interest_base_quantity",
        *METRICS_RATIO_COLUMNS,
    ),
)

BOOK_DEPTH_SOURCE_COLUMNS: Columns = (
    "timestamp",
    "percentage",
    "depth",
    "notional",
)

UM_BOOK_DEPTH = DatasetSpec(
    product="um",
    name="book_depth",
    remote_name="bookDepth",
    source_columns=BOOK_DEPTH_SOURCE_COLUMNS,
    stored_columns=(
        "event_time",
        "percentage_bucket",
        "base_depth",
        "quote_notional",
    ),
    time_column="event_time",
    base_interval=None,
    output_intervals=(),
    aliases=MappingProxyType({"percentage": "percentage_bucket"}),
    max_concurrency=16,
    csv_header="present",
    schema_version=1,
    ordering_columns=("event_time", "percentage_bucket"),
    timestamp_columns=("event_time",),
    integer_columns=("percentage_bucket",),
)

CM_BOOK_DEPTH = replace(
    UM_BOOK_DEPTH,
    product="cm",
    stored_columns=(
        "event_time",
        "percentage_bucket",
        "contract_depth",
        "base_notional",
    ),
)

UM_TRADES = DatasetSpec(
    product="um",
    name="trades",
    remote_name="trades",
    source_columns=("id", "price", "qty", "quote_qty", "time", "is_buyer_maker"),
    stored_columns=(
        "trade_id",
        "price",
        "base_quantity",
        "quote_quantity",
        "event_time",
        "buyer_is_maker",
    ),
    time_column="event_time",
    base_interval=None,
    output_intervals=(),
    aliases=MappingProxyType({"id": "trade_id", "quantity": "base_quantity"}),
    max_concurrency=8,
    csv_header="present",
    schema_version=1,
    ordering_columns=("event_time", "trade_id"),
    timestamp_columns=("event_time",),
    integer_columns=("trade_id",),
    boolean_columns=("buyer_is_maker",),
)

CM_TRADES = replace(
    UM_TRADES,
    product="cm",
    source_columns=("id", "price", "qty", "base_qty", "time", "is_buyer_maker"),
    stored_columns=(
        "trade_id",
        "price",
        "contract_quantity",
        "base_quantity",
        "quote_notional",
        "event_time",
        "buyer_is_maker",
    ),
    aliases=MappingProxyType({"id": "trade_id", "quantity": "contract_quantity"}),
    requires_contract_size=True,
)

UM_AGG_TRADES = DatasetSpec(
    product="um",
    name="agg_trades",
    remote_name="aggTrades",
    source_columns=(
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
        "is_buyer_maker",
    ),
    stored_columns=(
        "agg_trade_id",
        "first_trade_id",
        "last_trade_id",
        "price",
        "base_quantity",
        "quote_quantity",
        "event_time",
        "buyer_is_maker",
    ),
    time_column="event_time",
    base_interval=None,
    output_intervals=(),
    aliases=MappingProxyType({"id": "agg_trade_id", "quantity": "base_quantity"}),
    max_concurrency=8,
    csv_header="present",
    schema_version=1,
    ordering_columns=("event_time", "agg_trade_id"),
    timestamp_columns=("event_time",),
    integer_columns=("agg_trade_id", "first_trade_id", "last_trade_id"),
    boolean_columns=("buyer_is_maker",),
)

CM_AGG_TRADES = replace(
    UM_AGG_TRADES,
    product="cm",
    stored_columns=(
        "agg_trade_id",
        "first_trade_id",
        "last_trade_id",
        "price",
        "contract_quantity",
        "base_quantity",
        "quote_notional",
        "event_time",
        "buyer_is_maker",
    ),
    aliases=MappingProxyType({"id": "agg_trade_id", "quantity": "contract_quantity"}),
    requires_contract_size=True,
)

SPOT_TRADES = replace(
    UM_TRADES,
    product="spot",
    source_columns=SPOT_TRADE_SOURCE_COLUMNS,
    csv_header="absent",
)

SPOT_AGG_TRADES = replace(
    UM_AGG_TRADES,
    product="spot",
    source_columns=SPOT_AGG_TRADE_SOURCE_COLUMNS,
    csv_header="absent",
)

DATASETS: Mapping[tuple[str, str], DatasetSpec] = MappingProxyType(
    {
        ("spot", "klines"): SPOT_KLINES,
        ("spot", "trades"): SPOT_TRADES,
        ("spot", "agg_trades"): SPOT_AGG_TRADES,
        ("um", "klines"): UM_KLINES,
        ("cm", "klines"): CM_KLINES,
        ("um", "mark_price_klines"): UM_MARK_PRICE_KLINES,
        ("cm", "mark_price_klines"): CM_MARK_PRICE_KLINES,
        ("um", "index_price_klines"): UM_INDEX_PRICE_KLINES,
        ("cm", "index_price_klines"): CM_INDEX_PRICE_KLINES,
        ("um", "premium_index_klines"): UM_PREMIUM_INDEX_KLINES,
        ("cm", "premium_index_klines"): CM_PREMIUM_INDEX_KLINES,
        ("um", "metrics"): UM_METRICS,
        ("cm", "metrics"): CM_METRICS,
        ("um", "book_depth"): UM_BOOK_DEPTH,
        ("cm", "book_depth"): CM_BOOK_DEPTH,
        ("um", "trades"): UM_TRADES,
        ("cm", "trades"): CM_TRADES,
        ("um", "agg_trades"): UM_AGG_TRADES,
        ("cm", "agg_trades"): CM_AGG_TRADES,
    }
)


def get_dataset(
    product: object, dataset: object, *, kline_base_interval: object = "1m"
) -> DatasetSpec:
    """Return the specification for a supported product and dataset.

    Args:
        product: The parsed source product identifier.
        dataset: The parsed dataset identifier.
        kline_base_interval: The configured Spot Kline archive resolution.

    Returns:
        The matching dataset specification.
    """
    if not isinstance(product, str):
        raise TypeError("product must be a string")
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")

    try:
        specification = DATASETS[(product, dataset)]
    except KeyError as error:
        raise ValueError(f"unsupported dataset '{product}/{dataset}'") from error
    if specification is SPOT_KLINES and kline_base_interval != "1m":
        raise ValueError("configured Spot Kline base interval must be '1m'")
    LOGGER.debug(
        "Dataset resolved: product=%s dataset=%s base_interval=%s",
        product,
        dataset,
        specification.base_interval,
    )
    return specification
