"""Describe explicit OKX all-market cache operations."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CacheReport:
    """Summarize one cache-only operation without returning market rows."""

    source: str
    product: str
    dataset: str
    requested_range: tuple[datetime, datetime]
    transport: str
    explanation: str
    cached_files: int
    remote_files: int
    remote_bytes: int
    downloaded_files: int
    failed_files: int
    logical_subjects: int
    cached_rows: int
    local_bytes: int
    dry_run: bool
    offline: bool

    def __post_init__(self) -> None:
        """Reject malformed ranges and negative operation counters."""
        start, end = self.requested_range
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("cache report requires a valid aware time range")
        values = (
            self.cached_files,
            self.remote_files,
            self.remote_bytes,
            self.downloaded_files,
            self.failed_files,
            self.logical_subjects,
            self.cached_rows,
            self.local_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("cache report counters cannot be negative")

    @property
    def complete(self) -> bool:
        """Return whether every attempted physical archive succeeded."""
        return (
            not self.dry_run
            and self.failed_files == 0
            and self.remote_files == 0
            and self.cached_files + self.downloaded_files > 0
        )
