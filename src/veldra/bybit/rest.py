"""Persist normalized Bybit REST histories as queryable Parquet ranges."""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path

import duckdb
import pandas as pd

from veldra.bybit.datasets import get_dataset
from veldra.core.catalog import Catalog
from veldra.core.models import (
    ArchiveKey,
    ArchiveObject,
    IntegritySpec,
    LogicalPartition,
    Materialization,
)
from veldra.core.subjects import DataSubject

type FrameFetcher = Callable[[], pd.DataFrame]


class BybitRESTCache:
    """Cache bounded public REST responses and reuse enclosing ranges."""

    def __init__(
        self, catalog: Catalog, root: Path, *, mutable_hours: float = 6
    ) -> None:
        """Retain local storage and the lifetime of recent response ranges."""
        if mutable_hours <= 0:
            raise ValueError("mutable_hours must be positive")
        self.catalog = catalog
        self.root = root
        self.mutable_hours = mutable_hours

    def get(
        self,
        dataset: str,
        subject: DataSubject,
        start: datetime,
        end: datetime,
        *,
        product: str,
        fetch: FrameFetcher,
        interval: str | None = None,
        variant: str | None = None,
        offline: bool = False,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return one exact range from a reusable local REST materialization."""
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("REST history requires a valid aware time range")
        spec = get_dataset(product, dataset, requested_interval=interval)
        cache_interval = variant or interval
        candidates = self.catalog.partitions_between(
            "bybit", product, dataset, subject, cache_interval, start, end
        )
        cached = self._covering_partition(candidates, start, end)
        if cached is not None and (
            offline or (not refresh and self._fresh(cached, end))
        ):
            frame = self._read_cached(
                cached,
                spec.time_column,
                spec.ordering_columns,
                start,
                end,
                offline=offline,
            )
            if frame is not None:
                return frame
        if offline:
            raise RuntimeError("offline mode requires a cached covering REST range")
        frame = fetch()
        self._validate_frame(frame, spec.stored_columns, spec.time_column, start, end)
        partition = self._publish(
            dataset,
            subject,
            product,
            cache_interval,
            start,
            end,
            frame,
            spec.time_column,
        )
        return self._query(
            partition, spec.time_column, spec.ordering_columns, start, end
        )

    @classmethod
    def _read_cached(
        cls,
        partition: LogicalPartition,
        time_column: str,
        ordering: tuple[str, ...],
        start: datetime,
        end: datetime,
        *,
        offline: bool,
    ) -> pd.DataFrame | None:
        """Return a readable cached range or permit one online rebuild."""
        try:
            return cls._query(partition, time_column, ordering, start, end)
        except duckdb.Error, OSError:
            if offline:
                raise RuntimeError("offline REST cache is unreadable") from None
            return None

    def _fresh(self, partition: LogicalPartition, end: datetime) -> bool:
        """Return whether one historical or recent local range can be reused."""
        now = datetime.now(UTC)
        if end <= now - timedelta(days=3):
            return True
        try:
            modified = datetime.fromtimestamp(
                partition.materialization_path.stat().st_mtime, UTC
            )
        except OSError:
            return False
        return modified >= now - timedelta(hours=self.mutable_hours)

    @staticmethod
    def _covering_partition(
        partitions: Sequence[LogicalPartition], start: datetime, end: datetime
    ) -> LogicalPartition | None:
        """Return the smallest cached range enclosing the request."""
        return min(
            (
                item
                for item in partitions
                if item.coverage_start <= start and item.coverage_end >= end
            ),
            key=lambda item: (
                item.coverage_end - item.coverage_start,
                item.coverage_start,
            ),
            default=None,
        )

    @staticmethod
    def _validate_frame(
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        time_column: str,
        start: datetime,
        end: datetime,
    ) -> None:
        """Reject source responses that cannot safely enter the local cache."""
        if tuple(frame.columns) != columns:
            raise ValueError("Bybit REST response columns do not match the dataset")
        if frame.empty:
            return
        timestamps = pd.to_datetime(frame[time_column], utc=True)
        if (
            timestamps.isna().any()
            or timestamps.lt(start).any()
            or timestamps.ge(end).any()
        ):
            raise ValueError("Bybit REST response lies outside the requested range")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("Bybit REST response is not sorted by time")

    @staticmethod
    def _key(
        dataset: str,
        subject: DataSubject,
        product: str,
        interval: str | None,
        start: datetime,
        end: datetime,
    ) -> ArchiveKey:
        """Build a stable physical identity for one bounded REST request."""
        identity = json.dumps(
            [
                product,
                dataset,
                subject.kind,
                subject.value,
                interval,
                start.isoformat(),
                end.isoformat(),
            ],
            separators=(",", ":"),
        )
        digest = sha256(identity.encode()).hexdigest()
        return ArchiveKey(
            "bybit",
            product,
            dataset,
            "rest_api",
            subject.kind,
            subject.value,
            "request",
            start.date(),
            (end - timedelta(microseconds=1)).date(),
            f"{digest}.json",
        )

    def _publish(
        self,
        dataset: str,
        subject: DataSubject,
        product: str,
        interval: str | None,
        start: datetime,
        end: datetime,
        frame: pd.DataFrame,
        time_column: str,
    ) -> LogicalPartition:
        """Atomically publish one normalized REST response and its metadata."""
        key = self._key(dataset, subject, product, interval, start, end)
        self.catalog.save_archives(
            [
                ArchiveObject(
                    key,
                    f"https://api.bybit.com/{dataset}",
                    integrity=IntegritySpec("archive_only"),
                )
            ]
        )
        destination = (
            self.root
            / "bybit"
            / "rest"
            / product
            / dataset
            / f"{key.archive_id}.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.name}.part")
        frame.to_parquet(partial, compression="zstd", index=False)
        partial.replace(destination)
        stat = destination.stat()
        first = start if frame.empty else frame[time_column].iloc[0].to_pydatetime()
        last = (
            end - timedelta(microseconds=1)
            if frame.empty
            else frame[time_column].iloc[-1].to_pydatetime()
        )
        materialization = Materialization(
            key,
            destination,
            1,
            len(frame),
            first,
            last,
            stat.st_size,
            local_mtime_ns=stat.st_mtime_ns,
            archive_revision=sha256(destination.read_bytes()).hexdigest(),
        )
        partition = LogicalPartition(
            "bybit",
            product,
            dataset,
            subject,
            interval,
            start,
            end,
            destination,
            None,
            None,
            len(frame),
            source_day=start.date(),
        )
        self.catalog.publish_materialization(materialization, [partition])
        return partition

    @staticmethod
    def _query(
        partition: LogicalPartition,
        time_column: str,
        ordering: tuple[str, ...],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Use DuckDB to filter an exact half-open range from Parquet."""
        order = ordering or (time_column,)
        expression = ", ".join(f'"{column}"' for column in order)
        with duckdb.connect() as connection:
            frame = connection.execute(
                f'SELECT * FROM read_parquet(?) WHERE "{time_column}" >= ? '
                f'AND "{time_column}" < ? ORDER BY {expression}',
                [str(partition.materialization_path), start, end],
            ).df()
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True).astype(
            "datetime64[us, UTC]"
        )
        return frame
