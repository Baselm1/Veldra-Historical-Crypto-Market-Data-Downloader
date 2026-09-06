"""Define the public results and messages returned by the downloader."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
import logging
from pathlib import Path

import pandas as pd

TimeRange = tuple[datetime, datetime]
type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Market:
    """Describe one market reported by a source."""

    symbol: str
    normalized_symbol: str
    base_asset: str | None = None
    quote_asset: str | None = None
    status: str | None = None
    pair: str | None = None
    contract_type: str | None = None
    contract_size: float | None = None
    onboard_time: datetime | None = None
    delivery_time: datetime | None = None


@dataclass(frozen=True)
class ResourceKey:
    """Identify one requested dataset and optional source archive symbol."""

    source: str
    product: str
    dataset: str
    symbol: str
    interval: str | None
    archive_symbol: str | None = None


@dataclass(frozen=True)
class Resource:
    """Describe one discovered daily archive and its local cache state."""

    day: date
    url: str
    checksum_url: str
    status: str = "discovered"
    archive_sha256: str | None = None
    parquet_path: Path | None = None
    parquet_sha256: str | None = None
    parquet_size: int | None = None
    parquet_mtime_ns: int | None = None
    row_count: int | None = None
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    archive_symbol: str | None = None
    timestamp_column: str | None = None
    schema_version: int = 1
    error: str | None = None
    last_attempt_at: datetime | None = None


@dataclass(frozen=True)
class IngestedResource:
    """Describe a verified Parquet file produced from one archive."""

    archive_sha256: str
    parquet_sha256: str
    parquet_size: int
    parquet_mtime_ns: int
    row_count: int
    first_timestamp: datetime
    last_timestamp: datetime
    timestamp_column: str | None = None
    schema_version: int = 1


@dataclass(frozen=True)
class Message:
    """Describe one warning, problem, or error for a requested pair."""

    code: str
    message: str
    date: date | None = None
    suggestions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Gap:
    """Describe a consecutive range of missing candles with an exclusive end."""

    start: datetime
    end: datetime
    count: int


class MissingCandlesError(RuntimeError):
    """Report missing source candles when strict gap handling is requested."""

    pair: str
    gaps: tuple[Gap, ...]

    def __init__(self, pair: str, gaps: Sequence[Gap]) -> None:
        """Create an error for a pair and its missing candle ranges.

        Args:
            pair: The exchange symbol that has missing candles.
            gaps: The missing candle ranges found in the requested data.
        """
        self.pair = pair
        self.gaps = tuple(gaps)
        missing = sum(gap.count for gap in gaps)
        LOGGER.error(
            "Strict missing-candle policy failed: pair=%s gaps=%d candles=%d",
            pair,
            len(gaps),
            missing,
        )
        super().__init__(
            f"{pair} is missing {missing} source candles across {len(gaps)} gaps"
        )


@dataclass
class Result:
    """Hold one pair's data and the report describing how it was obtained."""

    pair: str
    data: pd.DataFrame
    requested_range: TimeRange
    used_range: TimeRange | None = None
    available_range: TimeRange | None = None
    warnings: list[Message] = field(default_factory=list)
    problems: list[Message] = field(default_factory=list)
    errors: list[Message] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    gap_policy: str = "forward"
    source: str = "binance"
    product: str = "spot"
    dataset: str = "klines"

    @property
    def complete(self) -> bool:
        """Return whether the used range has no known problems or errors.

        Returns:
            True when a used range exists and no problem or error was recorded.
        """
        return self.used_range is not None and not self.problems and not self.errors

    def frame(self) -> pd.DataFrame:
        """Attach a serializable download report and return the DataFrame.

        Returns:
            The result's DataFrame with its report in ``attrs["download"]``.
        """
        self.data.attrs["download"] = result_report(self)
        LOGGER.debug(
            "Result report attached: pair=%s rows=%d complete=%s warnings=%d "
            "problems=%d errors=%d gaps=%d",
            self.pair,
            len(self.data),
            self.complete,
            len(self.warnings),
            len(self.problems),
            len(self.errors),
            len(self.gaps),
        )
        return self.data


def _range_value(value: TimeRange | None) -> list[JsonValue] | None:
    """Convert a timestamp range into ISO strings.

    Args:
        value: The optional start and exclusive end timestamps.

    Returns:
        Two ISO timestamp strings, or ``None`` when no range is available.
    """
    return [item.isoformat() for item in value] if value is not None else None


def _message_value(value: Message) -> dict[str, JsonValue]:
    """Convert one result message into JSON-safe values.

    Args:
        value: The warning, problem, or error to convert.

    Returns:
        A dictionary containing the message details.
    """
    return {
        "code": value.code,
        "message": value.message,
        "date": value.date.isoformat() if value.date is not None else None,
        "suggestions": list(value.suggestions),
    }


def _gap_value(value: Gap) -> dict[str, JsonValue]:
    """Convert one missing candle range into JSON-safe values.

    Args:
        value: The missing candle range to convert.

    Returns:
        A dictionary containing the range and missing candle count.
    """
    return {
        "start": value.start.isoformat(),
        "end": value.end.isoformat(),
        "count": value.count,
    }


def result_report(result: Result) -> dict[str, JsonValue]:
    """Convert a result report into values that JSON can serialize.

    Args:
        result: The completed or failed result to describe.

    Returns:
        A dictionary containing ranges, messages, gaps, and completion state.
    """
    return {
        "pair": result.pair,
        "source": result.source,
        "product": result.product,
        "dataset": result.dataset,
        "requested_range": _range_value(result.requested_range),
        "used_range": _range_value(result.used_range),
        "available_range": _range_value(result.available_range),
        "complete": result.complete,
        "gap_policy": result.gap_policy,
        "gaps": [_gap_value(gap) for gap in result.gaps],
        "warnings": [_message_value(message) for message in result.warnings],
        "problems": [_message_value(message) for message in result.problems],
        "errors": [_message_value(message) for message in result.errors],
    }
