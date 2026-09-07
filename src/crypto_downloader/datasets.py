"""Describe the datasets and columns supported by the downloader."""

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from types import MappingProxyType
from typing import Literal

from .request import ColumnSelection

type Columns = tuple[str, ...]
type CsvHeader = Literal["absent", "present"]
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


def _validate_header(value: str) -> None:
    """Reject an unsupported CSV header declaration.

    Args:
        value: The declared CSV header behavior.

    Raises:
        ValueError: If the declaration is not known.
    """
    if value not in {"absent", "present"}:
        raise ValueError("dataset csv_header must be absent or present")


def _validate_columns(
    source_columns: Columns, stored_columns: Columns, time_column: str
) -> None:
    """Reject empty schemas or a primary timestamp that is not stored.

    Args:
        source_columns: The source archive columns.
        stored_columns: The normalized Parquet columns.
        time_column: The canonical primary timestamp column.

    Raises:
        ValueError: If the declared schema is inconsistent.
    """
    if not source_columns or not stored_columns:
        raise ValueError("dataset columns cannot be empty")
    if time_column not in stored_columns:
        raise ValueError("dataset time_column must be stored")


def _validate_intervals(base_interval: str | None, output_intervals: Columns) -> None:
    """Reject contradictory interval declarations.

    Args:
        base_interval: The stored archive interval or ``None`` for raw data.
        output_intervals: The caller-facing intervals supported by the dataset.

    Raises:
        ValueError: If interval capabilities do not agree.
    """
    if base_interval == "":
        raise ValueError("dataset base_interval cannot be empty")
    if base_interval is not None and not output_intervals:
        raise ValueError("interval datasets must declare output intervals")
    if base_interval is None and output_intervals:
        raise ValueError("interval-less datasets cannot declare output intervals")


def _validate_kline_capabilities(
    base_interval: str | None,
    supports_resampling: bool,
    supports_gap_policy: bool,
) -> None:
    """Reject kline-only features on an interval-less dataset.

    Args:
        base_interval: The stored archive interval or ``None`` for raw data.
        supports_resampling: Whether higher interval aggregation is supported.
        supports_gap_policy: Whether synthetic-candle behavior is supported.

    Raises:
        ValueError: If a raw dataset declares a kline-only feature.
    """
    if base_interval is None and (supports_resampling or supports_gap_policy):
        raise ValueError("raw datasets cannot support kline-only capabilities")


def _validate_resample_columns(
    stored_columns: Columns,
    time_column: str,
    supports_resampling: bool,
    resample_sum_columns: Columns,
) -> None:
    """Require resampled candle schemas to declare every additive field.

    Args:
        stored_columns: The canonical fields retained in Parquet.
        time_column: The candle opening timestamp field.
        supports_resampling: Whether the dataset can produce larger candles.
        resample_sum_columns: The canonical numeric fields summed per bucket.

    Raises:
        ValueError: If additive fields are missing or invalid for the dataset.
    """
    if not supports_resampling:
        if resample_sum_columns:
            raise ValueError("non-resampled datasets cannot declare resample columns")
        return
    if not resample_sum_columns:
        raise ValueError("resampled datasets must declare resample columns")
    if len(set(resample_sum_columns)) != len(resample_sum_columns):
        raise ValueError("resample columns cannot contain duplicates")
    if any(column not in stored_columns for column in resample_sum_columns):
        raise ValueError("resample columns must be stored")
    structural = {time_column, "open", "high", "low", "close", "close_time"}
    unhandled = set(stored_columns) - structural - set(resample_sum_columns)
    if unhandled:
        raise ValueError("resample columns must cover every additive candle field")


def _validate_ordering(stored_columns: Columns, ordering_columns: Columns) -> None:
    """Reject ordering columns that do not exist in the stored schema.

    Args:
        stored_columns: The normalized Parquet columns.
        ordering_columns: The deterministic query ordering columns.

    Raises:
        ValueError: If an ordering column is not stored.
    """
    if any(column not in stored_columns for column in ordering_columns):
        raise ValueError("dataset ordering columns must be stored")


def _validate_typed_columns(
    stored_columns: Columns,
    time_column: str,
    timestamp_columns: Columns,
    integer_columns: Columns,
    boolean_columns: Columns,
) -> None:
    """Reject typed columns that are absent, ambiguous, or miss the primary time.

    Args:
        stored_columns: The canonical columns retained in Parquet.
        time_column: The primary timestamp used for range filtering.
        timestamp_columns: Canonical columns stored as UTC timestamps.
        integer_columns: Canonical columns stored as signed integers.
        boolean_columns: Canonical columns stored as booleans.

    Raises:
        ValueError: If a type declaration is not compatible with the schema.
    """
    for name, columns in (
        ("timestamp", timestamp_columns),
        ("integer", integer_columns),
        ("boolean", boolean_columns),
    ):
        if any(column not in stored_columns for column in columns):
            raise ValueError(f"dataset {name} columns must be stored")
    declared = (*timestamp_columns, *integer_columns, *boolean_columns)
    if time_column not in timestamp_columns:
        raise ValueError("dataset time_column must be a timestamp column")
    if len(set(declared)) != len(declared):
        raise ValueError("dataset typed columns cannot overlap")


def _validate_schema_version(value: int) -> None:
    """Reject an unusable dataset schema version.

    Args:
        value: The declared integer schema version.

    Raises:
        ValueError: If the version is not a positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("dataset schema_version must be positive")


@dataclass(frozen=True)
class DatasetSpec:
    """Describe one product and dataset combination."""

    product: str
    name: str
    remote_name: str
    source_columns: Columns
    stored_columns: Columns
    time_column: str
    base_interval: str | None
    output_intervals: Columns
    aliases: Mapping[str, str]
    max_concurrency: int = 16
    csv_header: CsvHeader = "absent"
    schema_version: int = 1
    supports_resampling: bool = False
    supports_gap_policy: bool = False
    requires_contract_size: bool = False
    resample_sum_columns: Columns = ()
    ordering_columns: Columns = ()
    timestamp_columns: Columns = ()
    integer_columns: Columns = ()
    boolean_columns: Columns = ()

    def __post_init__(self) -> None:
        """Validate the immutable capability declaration.

        Raises:
            ValueError: If declared columns or capabilities contradict each other.
        """
        _validate_header(self.csv_header)
        _validate_columns(self.source_columns, self.stored_columns, self.time_column)
        _validate_intervals(self.base_interval, self.output_intervals)
        _validate_kline_capabilities(
            self.base_interval,
            self.supports_resampling,
            self.supports_gap_policy,
        )
        _validate_resample_columns(
            self.stored_columns,
            self.time_column,
            self.supports_resampling,
            self.resample_sum_columns,
        )
        if not self.ordering_columns:
            object.__setattr__(self, "ordering_columns", (self.time_column,))
        if not self.timestamp_columns:
            object.__setattr__(self, "timestamp_columns", (self.time_column,))
        _validate_ordering(self.stored_columns, self.ordering_columns)
        _validate_typed_columns(
            self.stored_columns,
            self.time_column,
            self.timestamp_columns,
            self.integer_columns,
            self.boolean_columns,
        )
        _validate_schema_version(self.schema_version)

    @property
    def needs_interval(self) -> bool:
        """Return whether the archive layout and request require an interval.

        Returns:
            True when this dataset has a stored base interval.
        """
        return self.base_interval is not None

    @property
    def csv_header_row(self) -> int | None:
        """Return the pandas CSV header row required by the source archive.

        Returns:
            Zero for a source header or ``None`` for a headerless archive.
        """
        return 0 if self.csv_header == "present" else None

    @property
    def storage_interval(self) -> str:
        """Return a safe local path label for interval and raw datasets.

        Returns:
            The base interval or ``raw`` for interval-less datasets.
        """
        return self.base_interval if self.base_interval is not None else "raw"

    @property
    def output_columns(self) -> Columns:
        """Return the columns callers may request.

        Returns:
            The canonical stored columns plus generated result columns.
        """
        generated = ("is_synthetic",) if self.supports_gap_policy else ()
        return (*self.stored_columns, *generated)

    def column_dtype(self, column: str) -> str:
        """Return the stable pandas dtype for one output column.

        Args:
            column: A stored or generated canonical output column.

        Returns:
            The pandas dtype string used for empty result frames.

        Raises:
            ValueError: If the column is not exposed by this dataset.
        """
        if column not in self.output_columns:
            raise ValueError(f"unknown dataset column '{column}'")
        if column in self.timestamp_columns:
            return "datetime64[us, UTC]"
        if column in self.integer_columns:
            return "int64"
        if column in self.boolean_columns or column == "is_synthetic":
            return "bool"
        return "float64"

    def resolve_interval(self, value: object) -> str | None:
        """Validate an output interval against this dataset.

        Args:
            value: The parsed interval requested by the caller.

        Returns:
            The supported interval spelling.
        """
        if not self.needs_interval:
            if value is None:
                return None
            raise ValueError(f"{self.product}/{self.name} does not accept an interval")
        if value is None:
            assert self.base_interval is not None
            return self.base_interval
        if not isinstance(value, str):
            raise TypeError("interval must be a string")
        if value.endswith("s") and value[:-1].isdigit():
            raise ValueError(
                f"interval '{value}' is finer than stored {self.base_interval} data"
            )
        if value not in self.output_intervals:
            supported = ", ".join(self.output_intervals)
            raise ValueError(
                f"unsupported interval '{value}'; supported intervals: {supported}"
            )
        return value

    def resolve_gap_policy(self, value: str | None) -> str | None:
        """Apply dataset-specific missing-candle policy support.

        Args:
            value: The validated caller policy, or ``None`` when omitted.

        Returns:
            The effective kline policy, or ``None`` for raw datasets.

        Raises:
            ValueError: If a raw dataset receives a candle-only policy.
        """
        if self.supports_gap_policy:
            return "forward" if value is None else value
        if value is not None:
            raise ValueError(f"{self.product}/{self.name} does not accept gap_policy")
        return None

    def resolve_columns(self, value: ColumnSelection) -> dict[str, str]:
        """Resolve requested canonical columns and aliases.

        Args:
            value: Parsed source columns and their requested output labels.

        Returns:
            Canonical stored columns mapped to output labels.
        """
        if value is None:
            return {column: column for column in self.output_columns}

        resolved: dict[str, str] = {}
        for requested_column, output_label in value.items():
            column = self.aliases.get(requested_column, requested_column)
            if column not in self.output_columns:
                raise ValueError(f"unknown column '{requested_column}'")
            if column in resolved:
                raise ValueError(
                    f"duplicate column '{requested_column}' resolves to '{column}'"
                )
            resolved[column] = output_label
        return resolved


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
    max_concurrency=32,
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

UM_KLINES = DatasetSpec(
    product="um",
    name="klines",
    remote_name="klines",
    source_columns=SPOT_KLINE_SOURCE_COLUMNS,
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
    time_column="open_time",
    base_interval="1m",
    output_intervals=SPOT_KLINE_OUTPUT_INTERVALS,
    aliases=MappingProxyType({}),
    max_concurrency=32,
    csv_header="present",
    schema_version=1,
    supports_resampling=True,
    supports_gap_policy=True,
    resample_sum_columns=(
        "base_volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ),
    ordering_columns=("open_time",),
    timestamp_columns=("open_time", "close_time"),
    integer_columns=("trade_count",),
)

CM_KLINES = DatasetSpec(
    product="cm",
    name="klines",
    remote_name="klines",
    source_columns=SPOT_KLINE_SOURCE_COLUMNS,
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
    time_column="open_time",
    base_interval="1m",
    output_intervals=SPOT_KLINE_OUTPUT_INTERVALS,
    aliases=MappingProxyType({}),
    max_concurrency=32,
    csv_header="present",
    schema_version=1,
    supports_resampling=True,
    supports_gap_policy=True,
    resample_sum_columns=(
        "contract_volume",
        "base_volume",
        "trade_count",
        "taker_buy_contract_volume",
        "taker_buy_base_volume",
    ),
    ordering_columns=("open_time",),
    timestamp_columns=("open_time", "close_time"),
    integer_columns=("trade_count",),
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
    max_concurrency=32,
    csv_header="present",
    schema_version=1,
    supports_resampling=True,
    supports_gap_policy=True,
    resample_sum_columns=("sample_count",),
    ordering_columns=("open_time",),
    timestamp_columns=("open_time", "close_time"),
    integer_columns=("sample_count",),
)

CM_MARK_PRICE_KLINES = DatasetSpec(
    product="cm",
    name="mark_price_klines",
    remote_name="markPriceKlines",
    source_columns=SPOT_KLINE_SOURCE_COLUMNS,
    stored_columns=UM_MARK_PRICE_KLINES.stored_columns,
    time_column="open_time",
    base_interval="1m",
    output_intervals=SPOT_KLINE_OUTPUT_INTERVALS,
    aliases=MappingProxyType({"count": "sample_count"}),
    max_concurrency=32,
    csv_header="present",
    schema_version=1,
    supports_resampling=True,
    supports_gap_policy=True,
    resample_sum_columns=("sample_count",),
    ordering_columns=("open_time",),
    timestamp_columns=("open_time", "close_time"),
    integer_columns=("sample_count",),
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

CM_TRADES = DatasetSpec(
    product="cm",
    name="trades",
    remote_name="trades",
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
    time_column="event_time",
    base_interval=None,
    output_intervals=(),
    aliases=MappingProxyType({"id": "trade_id", "quantity": "contract_quantity"}),
    max_concurrency=8,
    csv_header="present",
    schema_version=1,
    requires_contract_size=True,
    ordering_columns=("event_time", "trade_id"),
    timestamp_columns=("event_time",),
    integer_columns=("trade_id",),
    boolean_columns=("buyer_is_maker",),
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

CM_AGG_TRADES = DatasetSpec(
    product="cm",
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
        "contract_quantity",
        "base_quantity",
        "quote_notional",
        "event_time",
        "buyer_is_maker",
    ),
    time_column="event_time",
    base_interval=None,
    output_intervals=(),
    aliases=MappingProxyType({"id": "agg_trade_id", "quantity": "contract_quantity"}),
    max_concurrency=8,
    csv_header="present",
    schema_version=1,
    requires_contract_size=True,
    ordering_columns=("event_time", "agg_trade_id"),
    timestamp_columns=("event_time",),
    integer_columns=("agg_trade_id", "first_trade_id", "last_trade_id"),
    boolean_columns=("buyer_is_maker",),
)

SPOT_TRADES = DatasetSpec(
    product="spot",
    name="trades",
    remote_name="trades",
    source_columns=SPOT_TRADE_SOURCE_COLUMNS,
    stored_columns=SPOT_TRADE_STORED_COLUMNS,
    time_column="event_time",
    base_interval=None,
    output_intervals=(),
    aliases=MappingProxyType({"id": "trade_id", "quantity": "base_quantity"}),
    max_concurrency=8,
    csv_header="absent",
    schema_version=1,
    ordering_columns=("event_time", "trade_id"),
    timestamp_columns=("event_time",),
    integer_columns=("trade_id",),
    boolean_columns=("buyer_is_maker",),
)

SPOT_AGG_TRADES = DatasetSpec(
    product="spot",
    name="agg_trades",
    remote_name="aggTrades",
    source_columns=SPOT_AGG_TRADE_SOURCE_COLUMNS,
    stored_columns=SPOT_AGG_TRADE_STORED_COLUMNS,
    time_column="event_time",
    base_interval=None,
    output_intervals=(),
    aliases=MappingProxyType({"id": "agg_trade_id", "quantity": "base_quantity"}),
    max_concurrency=8,
    csv_header="absent",
    schema_version=1,
    ordering_columns=("event_time", "agg_trade_id"),
    timestamp_columns=("event_time",),
    integer_columns=("agg_trade_id", "first_trade_id", "last_trade_id"),
    boolean_columns=("buyer_is_maker",),
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
