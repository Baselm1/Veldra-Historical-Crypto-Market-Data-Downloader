"""Describe the datasets and columns supported by the downloader."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .request import ColumnSelection

type Columns = tuple[str, ...]

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


@dataclass(frozen=True)
class DatasetSpec:
    """Describe one product and dataset combination."""

    product: str
    name: str
    remote_name: str
    source_columns: Columns
    stored_columns: Columns
    time_column: str
    base_interval: str
    output_intervals: Columns
    aliases: Mapping[str, str]
    max_concurrency: int = 16

    @property
    def output_columns(self) -> Columns:
        """Return the columns callers may request.

        Returns:
            The canonical stored columns plus generated result columns.
        """
        return (*self.stored_columns, "is_synthetic")

    def resolve_interval(self, value: object) -> str:
        """Validate an output interval against this dataset.

        Args:
            value: The parsed interval requested by the caller.

        Returns:
            The supported interval spelling.
        """
        if not isinstance(value, str):
            raise TypeError("interval must be a string")
        if value not in self.output_intervals:
            supported = ", ".join(self.output_intervals)
            raise ValueError(
                f"unsupported interval '{value}'; supported intervals: {supported}"
            )
        return value

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
)

DATASETS: Mapping[tuple[str, str], DatasetSpec] = MappingProxyType(
    {("spot", "klines"): SPOT_KLINES}
)


def get_dataset(product: object, dataset: object) -> DatasetSpec:
    """Return the specification for a supported product and dataset.

    Args:
        product: The parsed source product identifier.
        dataset: The parsed dataset identifier.

    Returns:
        The matching dataset specification.
    """
    if not isinstance(product, str):
        raise TypeError("product must be a string")
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")

    try:
        return DATASETS[(product, dataset)]
    except KeyError as error:
        raise ValueError(f"unsupported dataset '{product}/{dataset}'") from error
