"""Persist normalized Bitget REST histories as queryable Parquet ranges."""

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path

import duckdb
import pandas as pd

from veldra.bitget.datasets import get_dataset
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


class BitgetRESTCache:
    """Store each fetched Bitget REST range once and reuse its Parquet rows."""

    def __init__(self, catalog: Catalog, root: Path) -> None:
        """Retain the catalog and local storage root.

        Args:
            catalog: Open metadata catalog used to find and publish ranges.
            root: Root directory containing Veldra's local data.
        """
        self.catalog = catalog
        self.root = root

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
    ) -> pd.DataFrame:
        """Return an exact range from local Parquet or fetch and store it.

        Args:
            dataset: Canonical Bitget dataset name.
            subject: Native instrument whose rows are requested.
            start: Inclusive UTC request boundary.
            end: Exclusive UTC request boundary.
            product: Bitget Futures settlement product.
            fetch: Source request to call only when local data is absent.
            interval: Stored Kline interval, or ``None`` for event data.

        Returns:
            Canonical rows restricted to the exact requested range.
        """
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("Bitget REST history requires a valid aware time range")
        spec = get_dataset(product, dataset)
        candidates = self.catalog.partitions_between(
            "bitget", product, dataset, subject, interval, start, end
        )
        cached = self._covering_partition(candidates, start, end)
        if cached is not None:
            frame = self._read_cached(
                cached, spec.time_column, spec.ordering_columns, start, end
            )
            if frame is not None:
                return frame
        frame = fetch()
        self._validate_frame(frame, spec.stored_columns, spec.time_column, start, end)
        partition = self._publish(
            dataset,
            subject,
            product,
            interval,
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
    ) -> pd.DataFrame | None:
        """Return readable local rows or permit one source rebuild.

        Args:
            partition: Cataloged local range that should cover the request.
            time_column: Timestamp column used for exact filtering.
            ordering: Deterministic result ordering columns.
            start: Inclusive UTC request boundary.
            end: Exclusive UTC request boundary.

        Returns:
            Local rows, or ``None`` when the file must be rebuilt.
        """
        try:
            return cls._query(partition, time_column, ordering, start, end)
        except duckdb.Error, OSError:
            return None

    @staticmethod
    def _covering_partition(
        partitions: Sequence[LogicalPartition], start: datetime, end: datetime
    ) -> LogicalPartition | None:
        """Return the smallest local range enclosing the request.

        Args:
            partitions: Candidate catalog partitions overlapping the request.
            start: Inclusive UTC request boundary.
            end: Exclusive UTC request boundary.

        Returns:
            The narrowest enclosing partition, if one exists.
        """
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
        """Reject source rows that cannot safely enter local storage.

        Args:
            frame: Normalized REST response proposed for storage.
            columns: Exact canonical dataset columns.
            time_column: Timestamp column used for range validation.
            start: Inclusive UTC request boundary.
            end: Exclusive UTC request boundary.
        """
        if tuple(frame.columns) != columns:
            raise ValueError("Bitget REST response columns do not match the dataset")
        if frame.empty:
            return
        timestamps = pd.to_datetime(frame[time_column], utc=True)
        if (
            timestamps.isna().any()
            or timestamps.lt(start).any()
            or timestamps.ge(end).any()
        ):
            raise ValueError("Bitget REST response lies outside the requested range")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("Bitget REST response is not sorted by time")

    @staticmethod
    def _key(
        dataset: str,
        subject: DataSubject,
        product: str,
        interval: str | None,
        start: datetime,
        end: datetime,
    ) -> ArchiveKey:
        """Build a stable catalog identity for one bounded REST request.

        Args:
            dataset: Canonical Bitget dataset name.
            subject: Native instrument whose rows are stored.
            product: Bitget Futures settlement product.
            interval: Stored Kline interval, or ``None`` for event data.
            start: Inclusive UTC request boundary.
            end: Exclusive UTC request boundary.

        Returns:
            Deterministic physical archive identity for the response.
        """
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
            "bitget",
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
        """Atomically publish one normalized REST response and its metadata.

        Args:
            dataset: Canonical Bitget dataset name.
            subject: Native instrument whose rows are stored.
            product: Bitget Futures settlement product.
            interval: Stored Kline interval, or ``None`` for event data.
            start: Inclusive UTC request boundary.
            end: Exclusive UTC request boundary.
            frame: Validated canonical rows.
            time_column: Timestamp column used to describe the rows.

        Returns:
            Catalog partition representing the stored range.
        """
        key = self._key(dataset, subject, product, interval, start, end)
        self.catalog.save_archives(
            [
                ArchiveObject(
                    key,
                    f"https://api.bitget.com/{dataset}",
                    integrity=IntegritySpec("archive_only"),
                )
            ]
        )
        destination = (
            self.root
            / "parquet"
            / "bitget"
            / product
            / dataset
            / subject.value
            / (interval or "raw")
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
        )
        partition = LogicalPartition(
            "bitget",
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
        """Use DuckDB to filter an exact range from local Parquet.

        Args:
            partition: Local Parquet range to query.
            time_column: Timestamp column used for exact filtering.
            ordering: Deterministic result ordering columns.
            start: Inclusive UTC request boundary.
            end: Exclusive UTC request boundary.

        Returns:
            Exact ordered rows from the local materialization.
        """
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
