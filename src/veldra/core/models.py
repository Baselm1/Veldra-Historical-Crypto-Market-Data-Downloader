"""Define the public results and messages returned by the downloader."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
import hashlib
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from veldra.core.subjects import DataSubject, SubjectKind

TimeRange = tuple[datetime, datetime]
type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)
type ChecksumAlgorithm = Literal["md5", "sha256"]
type IntegrityMode = Literal["sidecar", "response_header", "archive_only"]
type ArchiveStatus = Literal["discovered", "ready", "failed", "missing"]
LOGGER = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Report malformed source rows or archive structure."""


@dataclass(frozen=True)
class IntegritySpec:
    """Describe how downloaded source bytes prove their integrity."""

    mode: IntegrityMode
    algorithm: ChecksumAlgorithm | None = None
    expected: str | None = None
    sidecar_url: str | None = None

    def __post_init__(self) -> None:
        """Reject contradictory or malformed integrity metadata."""
        if self.mode == "sidecar":
            if self.algorithm not in {"md5", "sha256"}:
                raise ValueError("sidecar integrity requires a checksum algorithm")
            if not isinstance(self.sidecar_url, str) or not self.sidecar_url:
                raise ValueError("sidecar integrity requires sidecar_url")
        elif self.mode == "response_header":
            if self.algorithm != "md5":
                raise ValueError("response-header integrity requires MD5")
            if self.sidecar_url is not None:
                raise ValueError("response-header integrity cannot use a sidecar")
        elif self.mode == "archive_only":
            if self.algorithm is not None:
                raise ValueError("archive-only integrity cannot declare an algorithm")
            if self.expected is not None:
                raise ValueError(
                    "archive-only integrity cannot declare an expected digest"
                )
            if self.sidecar_url is not None:
                raise ValueError("archive-only integrity cannot use a sidecar")
            return
        else:
            raise ValueError("unsupported archive integrity mode")
        if self.expected is not None:
            length = 32 if self.algorithm == "md5" else 64
            if len(self.expected) != length or any(
                character not in "0123456789abcdefABCDEF" for character in self.expected
            ):
                raise ValueError("integrity expected digest is malformed")
            object.__setattr__(self, "expected", self.expected.lower())


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
    source: str | None = None
    product: str | None = None
    quote_volume_24h: float | None = None
    active: bool = False


@dataclass(frozen=True)
class Availability:
    """Summarize known remote and local coverage for one dataset."""

    source: str
    product: str
    dataset: str
    symbol: str
    interval: str | None
    storage_interval: str | None
    remote_range: tuple[date, date] | None
    configured_range: tuple[date, date] | None
    cached_range: tuple[date, date] | None
    scanned_ranges: tuple[tuple[date, date], ...]
    scanned_days: int
    available_days: int
    cached_days: int
    missing_days: int
    unavailable_days: int
    failed_days: int
    row_count: int
    local_bytes: int


@dataclass(frozen=True)
class ResourceKey:
    """Identify one requested dataset and optional source archive symbol."""

    source: str
    product: str
    dataset: str
    symbol: str
    interval: str | None
    archive_symbol: str | None = None
    cadence: str = "daily"
    subject: DataSubject | None = None

    @property
    def data_subject(self) -> DataSubject:
        """Return explicit scope or adapt a legacy symbol to an instrument."""
        return self.subject or DataSubject("instrument", self.symbol)


def _stable_id(*values: object) -> str:
    """Return a deterministic identifier for immutable catalog key values.

    Args:
        values: Values forming one stable catalog identity.

    Returns:
        A lowercase SHA-256 hexadecimal identifier.
    """
    encoded = "\x1f".join("" if value is None else str(value) for value in values)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArchiveKey:
    """Identify one remote archive independently from its temporary URL."""

    source: str
    product: str
    dataset: str
    provider: str
    remote_scope_kind: SubjectKind
    remote_scope_value: str
    cadence: str
    period_start: date
    period_end: date
    remote_name: str

    def __post_init__(self) -> None:
        """Reject invalid remote scope and period declarations."""
        DataSubject(self.remote_scope_kind, self.remote_scope_value)
        if self.period_end < self.period_start:
            raise ValueError("archive period ends before it starts")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.source,
                self.product,
                self.dataset,
                self.provider,
                self.cadence,
                self.remote_name,
            )
        ):
            raise ValueError("archive identity fields must be non-empty strings")

    @property
    def archive_id(self) -> str:
        """Return the stable ID that excludes mutable signed URLs."""
        return _stable_id(
            self.source,
            self.product,
            self.dataset,
            self.provider,
            self.remote_scope_kind,
            self.remote_scope_value,
            self.cadence,
            self.period_start,
            self.period_end,
            self.remote_name,
        )

    @property
    def subject(self) -> DataSubject:
        """Return the native scope represented by this physical archive."""
        return DataSubject(self.remote_scope_kind, self.remote_scope_value)


@dataclass(frozen=True)
class ArchiveObject:
    """Describe a discovered remote archive and its mutable retrieval state."""

    key: ArchiveKey
    url: str
    url_expires_at: datetime | None = None
    remote_size: int | None = None
    integrity: IntegritySpec | None = None
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: ArchiveStatus = "discovered"
    revision_id: str | None = None
    last_attempt_at: datetime | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        """Reject unsafe sizes, times, status values, and empty URLs."""
        if not isinstance(self.url, str) or not self.url:
            raise ValueError("archive URL must be a non-empty string")
        if self.remote_size is not None and self.remote_size < 0:
            raise ValueError("archive size cannot be negative")
        if self.status not in {"discovered", "ready", "failed", "missing"}:
            raise ValueError("archive status is unsupported")
        for value in (self.url_expires_at, self.discovered_at, self.last_attempt_at):
            if value is not None and value.tzinfo is None:
                raise ValueError("archive timestamps must include a timezone")


@dataclass(frozen=True)
class Materialization:
    """Describe one local Parquet file produced from a physical archive."""

    archive_key: ArchiveKey
    local_path: Path
    schema_version: int
    row_count: int
    first_timestamp: datetime
    last_timestamp: datetime
    local_size: int
    local_mtime_ns: int | None = None
    file_format: str = "parquet"
    layout_version: int = 1
    archive_revision: str | None = None
    ready_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    superseded_at: datetime | None = None

    def __post_init__(self) -> None:
        """Reject malformed local file metadata and timestamp bounds."""
        if self.schema_version < 1 or self.layout_version < 1:
            raise ValueError("materialization versions must be positive")
        if self.row_count < 0 or self.local_size < 0:
            raise ValueError("materialization sizes and rows cannot be negative")
        if not self.file_format:
            raise ValueError("materialization file format must not be empty")
        for value in (self.first_timestamp, self.last_timestamp, self.ready_at):
            if value.tzinfo is None:
                raise ValueError("materialization timestamps must include a timezone")
        if self.superseded_at is not None and self.superseded_at.tzinfo is None:
            raise ValueError("materialization timestamps must include a timezone")
        if self.last_timestamp < self.first_timestamp:
            raise ValueError("materialization timestamps are reversed")

    @property
    def materialization_id(self) -> str:
        """Return the stable archive-revision and layout identifier."""
        return _stable_id(
            self.archive_key.archive_id,
            self.archive_revision,
            self.schema_version,
            self.layout_version,
        )


@dataclass(frozen=True)
class LogicalPartition:
    """Map one logical subject and time range to a local materialization."""

    source: str
    product: str
    dataset: str
    subject: DataSubject
    interval: str | None
    coverage_start: datetime
    coverage_end: datetime
    materialization_path: Path
    predicate_column: str | None
    predicate_value: str | None
    row_count: int
    source_day: date | None = None

    def __post_init__(self) -> None:
        """Reject invalid coverage, predicates, and row counts."""
        if self.coverage_start.tzinfo is None or self.coverage_end.tzinfo is None:
            raise ValueError("partition timestamps must include a timezone")
        if self.coverage_start >= self.coverage_end:
            raise ValueError("partition coverage must end after it starts")
        if self.row_count < 0:
            raise ValueError("partition row count cannot be negative")
        if (self.predicate_column is None) != (self.predicate_value is None):
            raise ValueError("partition predicates require both column and value")

    def partition_id(self, materialization_id: str) -> str:
        """Return a stable identity within one materialization.

        Args:
            materialization_id: The local materialization containing the rows.

        Returns:
            A deterministic logical partition identifier.
        """
        return _stable_id(
            materialization_id,
            self.source,
            self.product,
            self.dataset,
            self.subject.kind,
            self.subject.value,
            self.interval,
            self.coverage_start.isoformat(),
            self.coverage_end.isoformat(),
            self.predicate_column,
            self.predicate_value,
        )


@dataclass(frozen=True)
class Resource:
    """Describe one discovered daily archive and its local cache state."""

    day: date
    url: str
    checksum_url: str | None
    status: str = "discovered"
    archive_checksum: str | None = None
    checksum_algorithm: ChecksumAlgorithm = "sha256"
    parquet_path: Path | None = None
    parquet_size: int | None = None
    parquet_mtime_ns: int | None = None
    row_count: int | None = None
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    archive_symbol: str | None = None
    contract_size: float | None = None
    timestamp_column: str | None = None
    schema_version: int = 1
    error: str | None = None
    last_attempt_at: datetime | None = None
    end_day: date | None = None
    cadence: str = "daily"
    coverage_start: datetime | None = None
    coverage_end: datetime | None = None
    integrity: IntegritySpec | None = None

    def __post_init__(self) -> None:
        """Reject unsupported source checksum and integrity declarations."""
        if self.checksum_algorithm not in {"md5", "sha256"}:
            raise ValueError("unsupported archive checksum algorithm")
        if self.integrity is None and not self.checksum_url:
            raise ValueError("resource requires an archive integrity policy")

    @property
    def integrity_spec(self) -> IntegritySpec:
        """Return explicit integrity metadata for new and legacy resources.

        Returns:
            The declared policy or a sidecar policy adapted from legacy fields.
        """
        if self.integrity is not None:
            return self.integrity
        assert self.checksum_url is not None
        return IntegritySpec(
            "sidecar",
            algorithm=self.checksum_algorithm,
            sidecar_url=self.checksum_url,
        )

    @property
    def last_day(self) -> date:
        """Return the inclusive last day covered by this physical archive."""
        return self.end_day or self.day

    @property
    def coverage(self) -> TimeRange:
        """Return the archive's exact inclusive-start, exclusive-end UTC coverage.

        Returns:
            Explicit source coverage or UTC calendar coverage for legacy rows.
        """
        start = self.coverage_start or datetime.combine(self.day, time.min, UTC)
        end = self.coverage_end or datetime.combine(
            self.last_day + timedelta(days=1), time.min, UTC
        )
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("resource coverage timestamps must include a timezone")
        start = start.astimezone(UTC)
        end = end.astimezone(UTC)
        if start >= end:
            raise ValueError("resource coverage must end after it starts")
        return start, end


@dataclass(frozen=True)
class IngestedResource:
    """Describe a verified Parquet file produced from one archive."""

    archive_checksum: str
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
    gap_policy: str | None = "forward"
    source: str = ""
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
