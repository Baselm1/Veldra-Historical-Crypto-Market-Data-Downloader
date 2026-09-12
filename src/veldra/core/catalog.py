"""Store market and daily resource metadata in DuckDB."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
import logging
import math
from pathlib import Path
from threading import RLock
from typing import Any, cast
from urllib.parse import urlsplit

import duckdb
import pyarrow as pa

from veldra.core.models import (
    ArchiveKey,
    ArchiveObject,
    ArchiveStatus,
    IngestedResource,
    IntegrityMode,
    IntegritySpec,
    LogicalPartition,
    Market,
    Materialization,
    Resource,
    ResourceKey,
)
from veldra.core.subjects import DataSubject, SubjectKind

_CATALOG_LOCKS = tuple(RLock() for _ in range(64))
LOGGER = logging.getLogger(__name__)

type ReadyResourceOutcome = tuple[date, Path, IngestedResource]
type FailedResourceOutcome = tuple[date, str]
type DiscoveryCheckpoint = tuple[date, date, datetime]
type ResourceOutcomeRow = tuple[
    date,
    str,
    str | None,
    str | None,
    int | None,
    int | None,
    int | None,
    datetime | None,
    datetime | None,
    str | None,
    int | None,
    str | None,
]


def _arrow_rows(rows: Sequence[Sequence[object]], columns: Sequence[str]) -> Any:
    """Build an Arrow table from metadata rows, preserving nullable integers.

    Args:
        rows: Values in the same order as the column names.
        columns: Names used when DuckDB reads the table.

    Returns:
        An Arrow table ready to register with DuckDB.
    """
    return pa.table({name: [row[i] for row in rows] for i, name in enumerate(columns)})


def _key_values(key: ResourceKey) -> tuple[str, str, str, str, str, str]:
    """Return a resource key as database parameter values.

    Args:
        key: The resource identity to convert.

    Returns:
        The six values forming the physical archive identity.
    """
    return (
        key.source,
        key.product,
        key.dataset,
        key.symbol,
        key.interval or "",
        key.cadence,
    )


def _validate_market_snapshot(markets: Sequence[Market]) -> None:
    """Reject a market snapshot that is empty or contains duplicate symbols.

    Args:
        markets: The complete market snapshot to validate.
    """
    symbols = [market.symbol for market in markets]
    if not symbols:
        raise ValueError("market snapshot cannot be empty")
    if len(symbols) != len(set(symbols)):
        raise ValueError("market snapshot contains duplicate symbols")


def _quote_volume_row(symbol: object, volume: object) -> tuple[str, float]:
    """Validate one cached market activity value.

    Args:
        symbol: The proposed native market symbol.
        volume: The proposed rolling quote volume.

    Returns:
        The validated symbol and floating-point volume.
    """
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("quote volumes must contain valid symbols and values")
    if isinstance(volume, bool) or not isinstance(volume, (int, float)):
        raise ValueError("quote volumes must contain valid symbols and values")
    parsed = float(volume)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("quote volumes must contain valid symbols and values")
    return symbol, parsed


def _validate_range(start_day: date, end_day: date) -> None:
    """Reject an inclusive day range whose end precedes its start.

    Args:
        start_day: The first day in the range.
        end_day: The last day in the range.
    """
    if end_day < start_day:
        raise ValueError("day range ends before it starts")


def _validate_discovery(
    start_day: date, end_day: date, resources: Sequence[Resource]
) -> None:
    """Reject duplicate resources and resources outside the searched range.

    Args:
        start_day: The first day searched.
        end_day: The last day searched.
        resources: The resources found during the search.
    """
    _validate_range(start_day, end_day)
    days = [resource.day for resource in resources]
    if len(days) != len(set(days)):
        raise ValueError("discovery contains a duplicate resource day")
    if any(day < start_day or day > end_day for day in days):
        raise ValueError("discovery resource falls outside the searched range")
    if any(resource.last_day < resource.day for resource in resources):
        raise ValueError("resource range ends before it starts")
    for resource in resources:
        resource.coverage


def _database_timestamp(value: datetime) -> datetime:
    """Convert an aware timestamp to naive UTC for DuckDB storage.

    Args:
        value: The aware timestamp to store.

    Returns:
        The same instant represented as naive UTC.
    """
    if value.tzinfo is None:
        raise ValueError("catalog timestamps must include a timezone")
    return value.astimezone(UTC).replace(tzinfo=None)


def _utc_timestamp(value: datetime | None) -> datetime | None:
    """Restore UTC awareness to a timestamp read from DuckDB.

    Args:
        value: A naive UTC timestamp or ``None``.

    Returns:
        The UTC-aware timestamp or ``None``.
    """
    return value.replace(tzinfo=UTC) if value is not None else None


@contextmanager
def open_catalog(path: Path, *, initialize: bool = True) -> Iterator[Catalog]:
    """Open a catalog database and close it after use.

    Args:
        path: The DuckDB file to create or open.
        initialize: Whether to create and migrate the catalog schema.

    Yields:
        A catalog connected to the requested database.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Opening DuckDB catalog: path=%s", path)
    connection = duckdb.connect(str(path))
    try:
        yield Catalog(connection, initialize=initialize)
    finally:
        connection.close()
        LOGGER.debug("Closed DuckDB catalog: path=%s", path)


@contextmanager
def catalog_lock(path: Path) -> Iterator[None]:
    """Prevent overlapping in-process work against one catalog path.

    Args:
        path: The catalog database path identifying the shared pipeline.

    Yields:
        Control after the matching process-local lock is acquired.
    """
    lock = _CATALOG_LOCKS[hash(path.resolve()) % len(_CATALOG_LOCKS)]
    with lock:
        yield


class Catalog:
    """Read and write downloader metadata in one DuckDB connection."""

    def __init__(
        self, connection: duckdb.DuckDBPyConnection, *, initialize: bool = True
    ) -> None:
        """Prepare a catalog around an open DuckDB connection.

        Args:
            connection: The DuckDB connection used for catalog operations.
            initialize: Whether to create and migrate the catalog schema.
        """
        self.connection = connection
        self.connection.execute("SET TimeZone = 'UTC'")
        if initialize:
            self._create_schema()

    def _create_schema(self) -> None:
        """Create metadata tables and migrate older catalogs when needed."""
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                source VARCHAR NOT NULL,
                product VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                normalized_symbol VARCHAR NOT NULL,
                base_asset VARCHAR,
                quote_asset VARCHAR,
                status VARCHAR,
                active BOOLEAN NOT NULL DEFAULT false,
                pair VARCHAR,
                contract_type VARCHAR,
                contract_size DOUBLE,
                onboard_time TIMESTAMP,
                delivery_time TIMESTAMP,
                refreshed_at TIMESTAMP,
                quote_volume_24h DOUBLE,
                volume_refreshed_at TIMESTAMP,
                PRIMARY KEY (source, product, symbol)
            );

            CREATE TABLE IF NOT EXISTS resources (
                source VARCHAR NOT NULL,
                product VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                cadence VARCHAR NOT NULL DEFAULT 'daily',
                day DATE NOT NULL,
                end_day DATE,
                coverage_start TIMESTAMP,
                coverage_end TIMESTAMP,
                archive_symbol VARCHAR,
                url VARCHAR NOT NULL,
                checksum_url VARCHAR NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'discovered',
                archive_checksum VARCHAR,
                checksum_algorithm VARCHAR NOT NULL DEFAULT 'sha256',
                parquet_path VARCHAR,
                parquet_size BIGINT,
                parquet_mtime_ns BIGINT,
                row_count BIGINT,
                first_timestamp TIMESTAMP,
                last_timestamp TIMESTAMP,
                timestamp_column VARCHAR,
                schema_version INTEGER NOT NULL DEFAULT 1,
                error VARCHAR,
                last_attempt_at TIMESTAMP,
                PRIMARY KEY (source, product, dataset, symbol, interval, cadence, day)
            );

            CREATE TABLE IF NOT EXISTS discovery_segments (
                source VARCHAR NOT NULL,
                product VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                cadence VARCHAR NOT NULL DEFAULT 'daily',
                start_day DATE NOT NULL,
                end_day DATE NOT NULL,
                scanned_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (
                    source, product, dataset, symbol, interval, cadence,
                    start_day, end_day
                )
            );

            CREATE TABLE IF NOT EXISTS source_bounds (
                source VARCHAR NOT NULL,
                product VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                cadence VARCHAR NOT NULL DEFAULT 'daily',
                first_day DATE NOT NULL,
                last_day DATE,
                checked_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (source, product, dataset, symbol, interval, cadence)
            );

            CREATE TABLE IF NOT EXISTS archive_objects (
                archive_id VARCHAR PRIMARY KEY,
                source VARCHAR NOT NULL,
                product VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                provider VARCHAR NOT NULL,
                remote_scope_kind VARCHAR NOT NULL,
                remote_scope_value VARCHAR NOT NULL,
                cadence VARCHAR NOT NULL,
                period_start DATE NOT NULL,
                period_end DATE NOT NULL,
                remote_name VARCHAR NOT NULL,
                url VARCHAR NOT NULL,
                url_expires_at TIMESTAMP,
                remote_size BIGINT,
                integrity_mode VARCHAR,
                integrity_algorithm VARCHAR,
                integrity_expected VARCHAR,
                integrity_sidecar_url VARCHAR,
                discovered_at TIMESTAMP NOT NULL,
                status VARCHAR NOT NULL,
                revision_id VARCHAR,
                last_attempt_at TIMESTAMP,
                error VARCHAR,
                UNIQUE (
                    source, product, dataset, provider, remote_scope_kind,
                    remote_scope_value, cadence, period_start, period_end, remote_name
                )
            );

            CREATE TABLE IF NOT EXISTS materializations (
                materialization_id VARCHAR PRIMARY KEY,
                archive_id VARCHAR NOT NULL,
                archive_revision VARCHAR,
                local_path VARCHAR NOT NULL,
                file_format VARCHAR NOT NULL,
                schema_version INTEGER NOT NULL,
                layout_version INTEGER NOT NULL,
                local_size BIGINT NOT NULL,
                local_mtime_ns BIGINT,
                row_count BIGINT NOT NULL,
                first_timestamp TIMESTAMP NOT NULL,
                last_timestamp TIMESTAMP NOT NULL,
                ready_at TIMESTAMP NOT NULL,
                superseded_at TIMESTAMP,
                UNIQUE (archive_id, archive_revision, schema_version, layout_version)
            );

            CREATE TABLE IF NOT EXISTS logical_partitions (
                partition_id VARCHAR PRIMARY KEY,
                materialization_id VARCHAR NOT NULL,
                source VARCHAR NOT NULL,
                product VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                subject_kind VARCHAR NOT NULL,
                subject_value VARCHAR NOT NULL,
                coverage_start TIMESTAMP NOT NULL,
                coverage_end TIMESTAMP NOT NULL,
                predicate_column VARCHAR,
                predicate_value VARCHAR,
                row_count BIGINT NOT NULL,
                source_day DATE
            );

            CREATE TABLE IF NOT EXISTS request_metrics (
                provider VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                host VARCHAR NOT NULL,
                download_bytes BIGINT NOT NULL,
                download_seconds DOUBLE NOT NULL,
                normalization_seconds DOUBLE NOT NULL,
                observed_at TIMESTAMP NOT NULL DEFAULT current_timestamp
            );
            """)
        self.connection.execute(
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS refreshed_at TIMESTAMP"
        )
        for statement in (
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS pair VARCHAR",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS active BOOLEAN",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS contract_type VARCHAR",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS contract_size DOUBLE",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS onboard_time TIMESTAMP",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS delivery_time TIMESTAMP",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS quote_volume_24h DOUBLE",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS volume_refreshed_at TIMESTAMP",
            "ALTER TABLE resources ADD COLUMN IF NOT EXISTS archive_symbol VARCHAR",
            "ALTER TABLE resources ADD COLUMN IF NOT EXISTS end_day DATE",
            "ALTER TABLE resources ADD COLUMN IF NOT EXISTS coverage_start TIMESTAMP",
            "ALTER TABLE resources ADD COLUMN IF NOT EXISTS coverage_end TIMESTAMP",
            "ALTER TABLE resources ADD COLUMN IF NOT EXISTS timestamp_column VARCHAR",
            "ALTER TABLE resources ADD COLUMN IF NOT EXISTS schema_version INTEGER",
            "ALTER TABLE resources ADD COLUMN IF NOT EXISTS archive_checksum VARCHAR",
            "ALTER TABLE resources ADD COLUMN IF NOT EXISTS checksum_algorithm VARCHAR",
        ):
            self.connection.execute(statement)
        resource_columns = {
            row[1]
            for row in self.connection.execute(
                "PRAGMA table_info('resources')"
            ).fetchall()
        }
        if "archive_sha256" in resource_columns:
            self.connection.execute(
                "UPDATE resources SET archive_checksum = archive_sha256 "
                "WHERE archive_checksum IS NULL"
            )
            self.connection.execute("ALTER TABLE resources DROP COLUMN archive_sha256")
        self.connection.execute(
            "UPDATE resources SET checksum_algorithm = 'sha256' "
            "WHERE checksum_algorithm IS NULL"
        )
        self.connection.execute(
            "ALTER TABLE resources ALTER COLUMN checksum_algorithm "
            "SET DEFAULT 'sha256'"
        )
        self.connection.execute(
            "ALTER TABLE resources ALTER COLUMN checksum_algorithm SET NOT NULL"
        )
        self.connection.execute(
            "UPDATE markets SET active = (status = 'TRADING') "
            "WHERE active IS NULL AND source = 'binance'"
        )
        self.connection.execute(
            "UPDATE markets SET active = false WHERE active IS NULL"
        )
        self.connection.execute(
            "ALTER TABLE markets ALTER COLUMN active SET DEFAULT false"
        )
        self.connection.execute("ALTER TABLE markets ALTER COLUMN active SET NOT NULL")
        self.connection.execute(
            "UPDATE resources SET schema_version = 1 WHERE schema_version IS NULL"
        )
        self.connection.execute(
            "UPDATE resources SET coverage_start = CAST(day AS TIMESTAMP) "
            "WHERE coverage_start IS NULL"
        )
        self.connection.execute(
            "UPDATE resources SET coverage_end = "
            "CAST(COALESCE(end_day, day) AS TIMESTAMP) + INTERVAL 1 DAY "
            "WHERE coverage_end IS NULL"
        )
        self.connection.execute(
            "ALTER TABLE resources DROP COLUMN IF EXISTS parquet_sha256"
        )
        self._migrate_archive_keys()
        if "discoveries" in self._tables():
            self.connection.execute("""
                INSERT INTO discovery_segments
                    (source, product, dataset, symbol, interval, cadence,
                     start_day, end_day, scanned_at)
                SELECT d.source, d.product, d.dataset, d.symbol, d.interval, d.cadence,
                       d.start_day, d.end_day, d.scanned_at
                FROM discoveries AS d
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM discovery_segments AS s
                    WHERE s.source = d.source
                      AND s.product = d.product
                      AND s.dataset = d.dataset
                      AND s.symbol = d.symbol
                      AND s.interval = d.interval
                      AND s.cadence = d.cadence
                )
                ON CONFLICT DO NOTHING
                """)
            self.connection.execute("DROP TABLE discoveries")
        self._backfill_logical_catalog()

    def _tables(self) -> set[str]:
        """Return the names of tables already present in this catalog."""
        return {row[0] for row in self.connection.execute("SHOW TABLES").fetchall()}

    def _migrate_archive_keys(self) -> None:
        """Retain legacy daily cache rows while adding cadence to archive identities."""
        keys = {
            "resources": "source, product, dataset, symbol, interval, cadence, day",
            "discoveries": "source, product, dataset, symbol, interval, cadence",
            "discovery_segments": "source, product, dataset, symbol, interval, cadence, start_day, end_day",
            "source_bounds": "source, product, dataset, symbol, interval, cadence",
        }
        with self._transaction():
            tables = self._tables()
            for table, key_columns in keys.items():
                if table not in tables:
                    continue
                columns = {
                    row[1]
                    for row in self.connection.execute(
                        f"PRAGMA table_info('{table}')"
                    ).fetchall()
                }
                if table == "resources" and "end_day" not in columns:
                    self.connection.execute(
                        "ALTER TABLE resources ADD COLUMN end_day DATE"
                    )
                if "cadence" in columns:
                    continue
                self.connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN cadence VARCHAR DEFAULT 'daily'"
                )
                self.connection.execute(
                    f"CREATE TABLE {table}_migrated AS SELECT * FROM {table}"
                )
                self.connection.execute(
                    f"ALTER TABLE {table}_migrated ADD PRIMARY KEY ({key_columns})"
                )
                self.connection.execute(f"DROP TABLE {table}")
                self.connection.execute(
                    f"ALTER TABLE {table}_migrated RENAME TO {table}"
                )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Commit all enclosed catalog writes together or roll them back."""
        self.connection.execute("BEGIN TRANSACTION")
        try:
            yield
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    @staticmethod
    def _archive_row(archive: ArchiveObject) -> tuple[object, ...]:
        """Convert one physical archive into database values.

        Args:
            archive: The physical archive to store.

        Returns:
            Values matching the archive-object table order.
        """
        integrity = archive.integrity
        return (
            archive.key.archive_id,
            archive.key.source,
            archive.key.product,
            archive.key.dataset,
            archive.key.provider,
            archive.key.remote_scope_kind,
            archive.key.remote_scope_value,
            archive.key.cadence,
            archive.key.period_start,
            archive.key.period_end,
            archive.key.remote_name,
            archive.url,
            (
                _database_timestamp(archive.url_expires_at)
                if archive.url_expires_at is not None
                else None
            ),
            archive.remote_size,
            integrity.mode if integrity is not None else None,
            integrity.algorithm if integrity is not None else None,
            integrity.expected if integrity is not None else None,
            integrity.sidecar_url if integrity is not None else None,
            _database_timestamp(archive.discovered_at),
            archive.status,
            archive.revision_id,
            (
                _database_timestamp(archive.last_attempt_at)
                if archive.last_attempt_at is not None
                else None
            ),
            archive.error,
        )

    def _write_archive(self, archive: ArchiveObject) -> None:
        """Insert or refresh one archive inside the caller's transaction.

        Args:
            archive: The physical archive metadata to write.
        """
        self.connection.execute(
            """
            INSERT INTO archive_objects VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT (archive_id) DO UPDATE SET
                url = excluded.url,
                url_expires_at = excluded.url_expires_at,
                remote_size = excluded.remote_size,
                integrity_mode = excluded.integrity_mode,
                integrity_algorithm = excluded.integrity_algorithm,
                integrity_expected = excluded.integrity_expected,
                integrity_sidecar_url = excluded.integrity_sidecar_url,
                discovered_at = excluded.discovered_at,
                status = excluded.status,
                revision_id = excluded.revision_id,
                last_attempt_at = excluded.last_attempt_at,
                error = excluded.error
            """,
            self._archive_row(archive),
        )

    @staticmethod
    def _archive(row: Sequence[Any]) -> ArchiveObject:
        """Rebuild one physical archive from catalog values.

        Args:
            row: Values selected in physical archive column order.

        Returns:
            The reconstructed archive object.
        """
        mode = cast(IntegrityMode | None, row[14])
        integrity = (
            IntegritySpec(
                mode,
                algorithm=row[15],
                expected=row[16],
                sidecar_url=row[17],
            )
            if mode is not None
            else None
        )
        return ArchiveObject(
            ArchiveKey(
                source=row[1],
                product=row[2],
                dataset=row[3],
                provider=row[4],
                remote_scope_kind=cast(SubjectKind, row[5]),
                remote_scope_value=row[6],
                cadence=row[7],
                period_start=row[8],
                period_end=row[9],
                remote_name=row[10],
            ),
            url=row[11],
            url_expires_at=_utc_timestamp(row[12]),
            remote_size=row[13],
            integrity=integrity,
            discovered_at=cast(datetime, _utc_timestamp(row[18])),
            status=cast(ArchiveStatus, row[19]),
            revision_id=row[20],
            last_attempt_at=_utc_timestamp(row[21]),
            error=row[22],
        )

    def save_archives(self, archives: Sequence[ArchiveObject]) -> None:
        """Store discovered physical archives in one transaction.

        Args:
            archives: The physical archives to insert or refresh.
        """
        identifiers = [archive.key.archive_id for archive in archives]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("archive batch contains a duplicate identity")
        if not archives:
            return
        with self._transaction():
            for archive in archives:
                self._write_archive(archive)

    def archive(self, key: ArchiveKey) -> ArchiveObject | None:
        """Return one physical archive by its stable identity.

        Args:
            key: The stable physical archive key.

        Returns:
            The archive metadata, or ``None`` when it is unknown.
        """
        row = self.connection.execute(
            "SELECT * FROM archive_objects WHERE archive_id = ?",
            [key.archive_id],
        ).fetchone()
        return self._archive(row) if row is not None else None

    def ready_archives_between(
        self,
        source: str,
        product: str,
        dataset: str,
        start_day: date,
        end_day: date,
    ) -> list[ArchiveObject]:
        """Return materialized physical archives overlapping source dates.

        Args:
            source: Historical source identifier.
            product: Source product identifier.
            dataset: Historical dataset identifier.
            start_day: First inclusive source date.
            end_day: Last inclusive source date.

        Returns:
            Ready physical objects in deterministic source-period order.
        """
        _validate_range(start_day, end_day)
        rows = self.connection.execute(
            """
            SELECT DISTINCT a.*
            FROM archive_objects AS a
            JOIN materializations AS m USING (archive_id)
            WHERE a.source = ? AND a.product = ? AND a.dataset = ?
              AND a.status = 'ready' AND m.superseded_at IS NULL
              AND a.period_start <= ? AND a.period_end >= ?
            ORDER BY a.period_start, a.period_end, a.remote_name
            """,
            [source, product, dataset, end_day, start_day],
        ).fetchall()
        return [self._archive(row) for row in rows]

    def mark_archive_failed(self, key: ArchiveKey, error: str) -> None:
        """Record a failed attempt against a known physical archive.

        Args:
            key: The physical archive that failed.
            error: The failure description.
        """
        row = self.connection.execute(
            """
            UPDATE archive_objects SET
                status = 'failed', error = ?, last_attempt_at = current_timestamp
            WHERE archive_id = ?
            RETURNING archive_id
            """,
            [error, key.archive_id],
        ).fetchone()
        if row is None:
            raise KeyError(f"archive {key.archive_id} was not discovered")

    @staticmethod
    def _materialization_row(materialization: Materialization) -> tuple[object, ...]:
        """Convert one local materialization into database values.

        Args:
            materialization: The local file metadata to store.

        Returns:
            Values matching the materialization table order.
        """
        return (
            materialization.materialization_id,
            materialization.archive_key.archive_id,
            materialization.archive_revision,
            str(materialization.local_path),
            materialization.file_format,
            materialization.schema_version,
            materialization.layout_version,
            materialization.local_size,
            materialization.local_mtime_ns,
            materialization.row_count,
            _database_timestamp(materialization.first_timestamp),
            _database_timestamp(materialization.last_timestamp),
            _database_timestamp(materialization.ready_at),
            (
                _database_timestamp(materialization.superseded_at)
                if materialization.superseded_at is not None
                else None
            ),
        )

    def _write_materialization(self, materialization: Materialization) -> None:
        """Publish local file metadata inside the caller's transaction.

        Args:
            materialization: The local materialization to publish.
        """
        self.connection.execute(
            """
            INSERT INTO materializations VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT (materialization_id) DO UPDATE SET
                local_path = excluded.local_path,
                local_size = excluded.local_size,
                local_mtime_ns = excluded.local_mtime_ns,
                row_count = excluded.row_count,
                first_timestamp = excluded.first_timestamp,
                last_timestamp = excluded.last_timestamp,
                ready_at = excluded.ready_at,
                superseded_at = excluded.superseded_at
            """,
            self._materialization_row(materialization),
        )

    @staticmethod
    def _partition_row(
        materialization_id: str, partition: LogicalPartition
    ) -> tuple[object, ...]:
        """Convert one logical partition into database values.

        Args:
            materialization_id: The physical file containing the partition.
            partition: The logical rows exposed from the file.

        Returns:
            Values matching the logical partition table order.
        """
        return (
            partition.partition_id(materialization_id),
            materialization_id,
            partition.source,
            partition.product,
            partition.dataset,
            partition.interval or "",
            partition.subject.kind,
            partition.subject.value,
            _database_timestamp(partition.coverage_start),
            _database_timestamp(partition.coverage_end),
            partition.predicate_column,
            partition.predicate_value,
            partition.row_count,
            partition.source_day,
        )

    def _write_partitions(
        self,
        materialization: Materialization,
        partitions: Sequence[LogicalPartition],
    ) -> None:
        """Replace logical rows for one materialization in a transaction.

        Args:
            materialization: The containing local file.
            partitions: The logical subject ranges to expose.
        """
        identifier = materialization.materialization_id
        self.connection.execute(
            "DELETE FROM logical_partitions WHERE materialization_id = ?",
            [identifier],
        )
        for partition in partitions:
            self.connection.execute(
                "INSERT INTO logical_partitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._partition_row(identifier, partition),
            )

    def publish_materialization(
        self,
        materialization: Materialization,
        partitions: Sequence[LogicalPartition],
    ) -> None:
        """Atomically publish one local file and all of its logical partitions.

        Args:
            materialization: The verified local file metadata.
            partitions: Every logical range provided by the file.
        """
        archive = self.archive(materialization.archive_key)
        if archive is None:
            raise KeyError(
                f"archive {materialization.archive_key.archive_id} was not discovered"
            )
        if not partitions:
            raise ValueError("materialization must contain a logical partition")
        expected = (
            materialization.archive_key.source,
            materialization.archive_key.product,
            materialization.archive_key.dataset,
        )
        for partition in partitions:
            if partition.materialization_path != materialization.local_path:
                raise ValueError(
                    "logical partition path does not match materialization"
                )
            if (partition.source, partition.product, partition.dataset) != expected:
                raise ValueError("logical partition dataset does not match archive")
        identifiers = [
            partition.partition_id(materialization.materialization_id)
            for partition in partitions
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("materialization contains duplicate logical partitions")
        with self._transaction():
            self._write_materialization(materialization)
            self._write_partitions(materialization, partitions)
            self.connection.execute(
                """
                UPDATE archive_objects SET
                    status = 'ready', error = NULL, last_attempt_at = current_timestamp,
                    revision_id = COALESCE(?, revision_id)
                WHERE archive_id = ?
                """,
                [materialization.archive_revision, archive.key.archive_id],
            )

    @staticmethod
    def _logical_partition(row: Sequence[Any]) -> LogicalPartition:
        """Rebuild one logical partition from catalog values.

        Args:
            row: Values selected in logical partition column order.

        Returns:
            The reconstructed partition.
        """
        start = cast(datetime, _utc_timestamp(row[6]))
        end = cast(datetime, _utc_timestamp(row[7]))
        return LogicalPartition(
            source=row[0],
            product=row[1],
            dataset=row[2],
            subject=DataSubject(cast(SubjectKind, row[3]), row[4]),
            interval=row[5] or None,
            coverage_start=start,
            coverage_end=end,
            materialization_path=Path(row[8]),
            predicate_column=row[9],
            predicate_value=row[10],
            row_count=row[11],
            source_day=row[12],
        )

    def partitions_between(
        self,
        source: str,
        product: str,
        dataset: str,
        subject: DataSubject,
        interval: str | None,
        start: datetime,
        end: datetime,
    ) -> list[LogicalPartition]:
        """Return logical partitions overlapping an exact requested range.

        Args:
            source: The source identifier.
            product: The product identifier.
            dataset: The dataset identifier.
            subject: The native logical subject.
            interval: The stored interval or ``None`` for raw events.
            start: The inclusive UTC request boundary.
            end: The exclusive UTC request boundary.

        Returns:
            Matching logical partitions ordered by coverage.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("partition query timestamps must include a timezone")
        if start >= end:
            raise ValueError("partition range must end after it starts")
        rows = self.connection.execute(
            """
            SELECT p.source, p.product, p.dataset, p.subject_kind, p.subject_value,
                   p.interval, p.coverage_start, p.coverage_end, m.local_path,
                   p.predicate_column, p.predicate_value, p.row_count, p.source_day
            FROM logical_partitions AS p
            JOIN materializations AS m USING (materialization_id)
            WHERE p.source = ? AND p.product = ? AND p.dataset = ?
              AND p.subject_kind = ? AND p.subject_value = ? AND p.interval = ?
              AND p.coverage_start < ? AND p.coverage_end > ?
              AND m.superseded_at IS NULL
            ORDER BY p.coverage_start, p.coverage_end, m.local_path
            """,
            [
                source,
                product,
                dataset,
                subject.kind,
                subject.value,
                interval or "",
                _database_timestamp(end),
                _database_timestamp(start),
            ],
        ).fetchall()
        return [self._logical_partition(row) for row in rows]

    def delete_partitions(
        self,
        source: str,
        product: str,
        dataset: str,
        subject: DataSubject,
        interval: str | None,
        start: datetime,
        end: datetime,
    ) -> int:
        """Remove logical references without deleting shared physical files.

        Args:
            source: The source identifier.
            product: The product identifier.
            dataset: The dataset identifier.
            subject: The logical subject to remove.
            interval: The stored interval or ``None`` for raw events.
            start: The inclusive removal boundary.
            end: The exclusive removal boundary.

        Returns:
            The number of logical partition rows removed.
        """
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("partition removal requires a valid aware time range")
        rows = self.connection.execute(
            """
            DELETE FROM logical_partitions
            WHERE source = ? AND product = ? AND dataset = ?
              AND subject_kind = ? AND subject_value = ? AND interval = ?
              AND coverage_start < ? AND coverage_end > ?
            RETURNING partition_id
            """,
            [
                source,
                product,
                dataset,
                subject.kind,
                subject.value,
                interval or "",
                _database_timestamp(end),
                _database_timestamp(start),
            ],
        ).fetchall()
        return len(rows)

    @staticmethod
    def _stored_materialization(row: Sequence[Any]) -> Materialization:
        """Rebuild one materialization joined to its archive key.

        Args:
            row: Materialization values followed by archive identity values.

        Returns:
            The reconstructed local materialization.
        """
        key = ArchiveKey(
            source=row[14],
            product=row[15],
            dataset=row[16],
            provider=row[17],
            remote_scope_kind=cast(SubjectKind, row[18]),
            remote_scope_value=row[19],
            cadence=row[20],
            period_start=row[21],
            period_end=row[22],
            remote_name=row[23],
        )
        return Materialization(
            archive_key=key,
            local_path=Path(row[3]),
            schema_version=row[5],
            row_count=row[9],
            first_timestamp=cast(datetime, _utc_timestamp(row[10])),
            last_timestamp=cast(datetime, _utc_timestamp(row[11])),
            local_size=row[7],
            local_mtime_ns=row[8],
            file_format=row[4],
            layout_version=row[6],
            archive_revision=row[2],
            ready_at=cast(datetime, _utc_timestamp(row[12])),
            superseded_at=_utc_timestamp(row[13]),
        )

    def unreferenced_materializations(self) -> list[Materialization]:
        """Return local files that no logical partition still references."""
        rows = self.connection.execute("""
            SELECT m.*, a.source, a.product, a.dataset, a.provider,
                   a.remote_scope_kind, a.remote_scope_value, a.cadence,
                   a.period_start, a.period_end, a.remote_name
            FROM materializations AS m
            JOIN archive_objects AS a USING (archive_id)
            WHERE NOT EXISTS (
                SELECT 1 FROM logical_partitions AS p
                WHERE p.materialization_id = m.materialization_id
            )
            ORDER BY m.local_path
            """).fetchall()
        return [self._stored_materialization(row) for row in rows]

    def delete_materialization(self, materialization_id: str) -> Path:
        """Delete unreferenced local metadata and return its filesystem path.

        Args:
            materialization_id: The local materialization identity to remove.

        Returns:
            The path that the caller may safely delete from the filesystem.
        """
        referenced = self.connection.execute(
            "SELECT count(*) FROM logical_partitions WHERE materialization_id = ?",
            [materialization_id],
        ).fetchone()
        if referenced is not None and referenced[0]:
            raise RuntimeError("materialization is still referenced")
        row = self.connection.execute(
            "DELETE FROM materializations WHERE materialization_id = ? RETURNING local_path",
            [materialization_id],
        ).fetchone()
        if row is None:
            raise KeyError(f"materialization {materialization_id} was not found")
        return Path(row[0])

    def _backfill_logical_catalog(self) -> None:
        """Adapt existing one-file resources into physical and logical metadata."""
        rows = self.connection.execute("""
            SELECT source, product, dataset, symbol, interval, cadence,
                   day, url, checksum_url, status, archive_checksum,
                   parquet_path, parquet_size, parquet_mtime_ns, row_count,
                   first_timestamp, last_timestamp, archive_symbol, timestamp_column,
                   schema_version, error, last_attempt_at, end_day, cadence,
                   coverage_start, coverage_end, checksum_algorithm
            FROM resources
            ORDER BY source, product, dataset, symbol, interval, cadence, day
            """).fetchall()
        if not rows:
            return
        now = datetime.now(UTC)
        with self._transaction():
            for row in rows:
                resource = self._resource(row[6:])
                remote_name = Path(urlsplit(resource.url).path).name or resource.url
                key = ArchiveKey(
                    source=row[0],
                    product=row[1],
                    dataset=row[2],
                    provider="legacy_archive",
                    remote_scope_kind="instrument",
                    remote_scope_value=row[3],
                    cadence=row[5],
                    period_start=resource.day,
                    period_end=resource.last_day,
                    remote_name=remote_name,
                )
                integrity = resource.integrity_spec
                status = (
                    resource.status
                    if resource.status in {"discovered", "ready", "failed", "missing"}
                    else "discovered"
                )
                archive = ArchiveObject(
                    key=key,
                    url=resource.url,
                    integrity=integrity,
                    discovered_at=resource.last_attempt_at or now,
                    status=cast(ArchiveStatus, status),
                    revision_id=resource.archive_checksum,
                    last_attempt_at=resource.last_attempt_at,
                    error=resource.error,
                )
                self._write_archive(archive)
                if resource.status != "ready" or resource.parquet_path is None:
                    continue
                coverage_start, coverage_end = resource.coverage
                materialization = Materialization(
                    archive_key=key,
                    local_path=resource.parquet_path,
                    schema_version=resource.schema_version,
                    row_count=resource.row_count or 0,
                    first_timestamp=resource.first_timestamp or coverage_start,
                    last_timestamp=(
                        resource.last_timestamp or coverage_end - datetime.resolution
                    ),
                    local_size=resource.parquet_size or 0,
                    local_mtime_ns=resource.parquet_mtime_ns,
                    archive_revision=resource.archive_checksum,
                    ready_at=resource.last_attempt_at or now,
                )
                partition = LogicalPartition(
                    source=key.source,
                    product=key.product,
                    dataset=key.dataset,
                    subject=DataSubject("instrument", row[3]),
                    interval=row[4] or None,
                    coverage_start=coverage_start,
                    coverage_end=coverage_end,
                    materialization_path=resource.parquet_path,
                    predicate_column=None,
                    predicate_value=None,
                    row_count=resource.row_count or 0,
                    source_day=resource.day,
                )
                self._write_materialization(materialization)
                self._write_partitions(materialization, [partition])

    def markets(self, source: str, product: str) -> list[Market]:
        """Return markets stored for one source product.

        Args:
            source: The source identifier.
            product: The product identifier.

        Returns:
            Markets ordered by their native symbols.
        """
        rows = self.connection.execute(
            """
            SELECT symbol, normalized_symbol, base_asset, quote_asset, status, active,
                   pair, contract_type, contract_size, onboard_time, delivery_time,
                   quote_volume_24h
            FROM markets
            WHERE source = ? AND product = ?
            ORDER BY symbol
            """,
            [source, product],
        ).fetchall()
        return [
            Market(
                symbol=row[0],
                normalized_symbol=row[1],
                base_asset=row[2],
                quote_asset=row[3],
                status=row[4],
                active=row[5],
                pair=row[6],
                contract_type=row[7],
                contract_size=row[8],
                onboard_time=_utc_timestamp(row[9]),
                delivery_time=_utc_timestamp(row[10]),
                quote_volume_24h=row[11],
            )
            for row in rows
        ]

    def market_snapshot_at(self, source: str, product: str) -> datetime | None:
        """Return when one complete market snapshot was last stored.

        Args:
            source: The source identifier.
            product: The product identifier.

        Returns:
            The UTC refresh timestamp, or ``None`` without a current snapshot.
        """
        row = self.connection.execute(
            """
            SELECT max(refreshed_at)
            FROM markets
            WHERE source = ? AND product = ?
            """,
            [source, product],
        ).fetchone()
        return _utc_timestamp(row[0]) if row is not None else None

    def quote_volume_snapshot_at(
        self,
        source: str,
        product: str,
    ) -> datetime | None:
        """Return when rolling market volumes were last stored.

        Args:
            source: The source identifier.
            product: The product identifier.

        Returns:
            The UTC refresh timestamp, or ``None`` without cached volumes.
        """
        row = self.connection.execute(
            """
            SELECT max(volume_refreshed_at)
            FROM markets
            WHERE source = ? AND product = ?
            """,
            [source, product],
        ).fetchone()
        return _utc_timestamp(row[0]) if row is not None else None

    def save_quote_volumes(
        self,
        source: str,
        product: str,
        volumes: dict[str, float],
    ) -> None:
        """Store one complete rolling quote-volume snapshot.

        Args:
            source: The source identifier.
            product: The product identifier.
            volumes: Nonnegative quote volumes indexed by native symbol.
        """
        rows = [_quote_volume_row(symbol, volume) for symbol, volume in volumes.items()]
        frame = _arrow_rows(
            rows,
            columns=("symbol", "quote_volume_24h"),
        )
        if volumes:
            self.connection.register("incoming_quote_volumes", frame)
        try:
            with self._transaction():
                self.connection.execute(
                    """
                    UPDATE markets SET
                        quote_volume_24h = NULL,
                        volume_refreshed_at = now()
                    WHERE source = ? AND product = ?
                    """,
                    [source, product],
                )
                if volumes:
                    self.connection.execute(
                        """
                        UPDATE markets AS stored SET
                            quote_volume_24h = incoming.quote_volume_24h
                        FROM incoming_quote_volumes AS incoming
                        WHERE stored.source = ? AND stored.product = ?
                          AND stored.symbol = incoming.symbol
                        """,
                        [source, product],
                    )
        finally:
            if volumes:
                self.connection.unregister("incoming_quote_volumes")

    def save_markets(
        self, source: str, product: str, markets: Sequence[Market]
    ) -> None:
        """Replace the current status snapshot for one source product.

        Args:
            source: The source identifier.
            product: The product identifier.
            markets: The complete current market snapshot.
        """
        _validate_market_snapshot(markets)
        rows = [
            (
                source,
                product,
                market.symbol,
                market.normalized_symbol,
                market.base_asset,
                market.quote_asset,
                market.status,
                market.active,
                market.pair,
                market.contract_type,
                market.contract_size,
                (
                    _database_timestamp(market.onboard_time)
                    if market.onboard_time is not None
                    else None
                ),
                (
                    _database_timestamp(market.delivery_time)
                    if market.delivery_time is not None
                    else None
                ),
            )
            for market in markets
        ]
        frame = _arrow_rows(
            rows,
            columns=(
                "source",
                "product",
                "symbol",
                "normalized_symbol",
                "base_asset",
                "quote_asset",
                "status",
                "active",
                "pair",
                "contract_type",
                "contract_size",
                "onboard_time",
                "delivery_time",
            ),
        )
        self.connection.register("incoming_markets", frame)
        try:
            with self._transaction():
                self.connection.execute(
                    "UPDATE markets SET status = NULL, active = false "
                    "WHERE source = ? AND product = ?",
                    [source, product],
                )
                self.connection.execute("""
                    INSERT INTO markets (
                        source, product, symbol, normalized_symbol,
                        base_asset, quote_asset, status, active, pair, contract_type,
                        contract_size, onboard_time, delivery_time, refreshed_at
                    )
                    SELECT source, product, symbol, normalized_symbol,
                           base_asset, quote_asset, status, active, pair, contract_type,
                           contract_size, onboard_time, delivery_time, current_timestamp
                    FROM incoming_markets
                    ON CONFLICT (source, product, symbol) DO UPDATE SET
                        normalized_symbol = excluded.normalized_symbol,
                        base_asset = excluded.base_asset,
                        quote_asset = excluded.quote_asset,
                        status = excluded.status,
                        active = excluded.active,
                        pair = excluded.pair,
                        contract_type = excluded.contract_type,
                        contract_size = excluded.contract_size,
                        onboard_time = excluded.onboard_time,
                        delivery_time = excluded.delivery_time,
                        refreshed_at = excluded.refreshed_at
                    """)
        finally:
            self.connection.unregister("incoming_markets")
        LOGGER.debug(
            "Market snapshot stored: source=%s product=%s markets=%d",
            source,
            product,
            len(markets),
        )

    def discovery_range(self, key: ResourceKey) -> tuple[date, date] | None:
        """Return the inclusive days already searched for one resource key.

        Args:
            key: The dataset identity whose discovery range is needed.

        Returns:
            The inclusive first and last searched days, or ``None``.
        """
        row = self.connection.execute(
            """
            SELECT min(start_day), max(end_day)
            FROM discovery_segments
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ? AND cadence = ?
            """,
            _key_values(key),
        ).fetchone()
        return (row[0], row[1]) if row is not None and row[0] is not None else None

    def discovery_ranges(self, key: ResourceKey) -> list[tuple[date, date]]:
        """Return every distinct day range already searched for a resource key.

        Args:
            key: The dataset identity whose discovery coverage is needed.

        Returns:
            Ordered inclusive ranges that were actually searched.
        """
        rows = self.connection.execute(
            """
            SELECT start_day, end_day
            FROM discovery_segments
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ? AND cadence = ?
            ORDER BY start_day, end_day
            """,
            _key_values(key),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def discovery_checkpoints(self, key: ResourceKey) -> list[DiscoveryCheckpoint]:
        """Return searched day ranges together with their latest scan times.

        Args:
            key: The dataset identity whose discovery checkpoints are needed.

        Returns:
            Ordered inclusive ranges and their UTC-aware scan timestamps.
        """
        rows = self.connection.execute(
            """
            SELECT start_day, end_day, scanned_at
            FROM discovery_segments
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ? AND cadence = ?
            ORDER BY start_day, end_day
            """,
            _key_values(key),
        ).fetchall()
        return [
            (row[0], row[1], timestamp)
            for row in rows
            if (timestamp := _utc_timestamp(row[2])) is not None
        ]

    def resource_bounds(self, key: ResourceKey) -> tuple[date, date] | None:
        """Return the earliest and latest discovered resource days.

        Args:
            key: The dataset identity whose availability is needed.

        Returns:
            The inclusive resource bounds, or ``None`` without known files.
        """
        row = self.connection.execute(
            """
            SELECT min(day), max(COALESCE(end_day, day))
            FROM resources
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ? AND cadence = ?
            """,
            _key_values(key),
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return None
        return row[0], row[1]

    def resource_coverage_bounds(
        self, key: ResourceKey
    ) -> tuple[datetime, datetime] | None:
        """Return the exact UTC bounds of all discovered physical archives.

        Args:
            key: The dataset identity whose timestamp coverage is needed.

        Returns:
            The earliest inclusive start and latest exclusive end, or ``None``.
        """
        row = self.connection.execute(
            """
            SELECT min(coverage_start), max(coverage_end)
            FROM resources
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ? AND cadence = ?
            """,
            _key_values(key),
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return None
        start = _utc_timestamp(row[0])
        end = _utc_timestamp(row[1])
        assert start is not None and end is not None
        return start, end

    def source_bounds(self, key: ResourceKey) -> tuple[date, date | None] | None:
        """Return separately verified source archive boundaries.

        Args:
            key: The dataset identity whose source boundaries are needed.

        Returns:
            The first and optional final source days, or ``None`` when unknown.
        """
        row = self.connection.execute(
            """
            SELECT first_day, last_day
            FROM source_bounds
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ? AND cadence = ?
            """,
            _key_values(key),
        ).fetchone()
        return (row[0], row[1]) if row is not None else None

    def save_source_bounds(
        self,
        key: ResourceKey,
        first_day: date,
        last_day: date | None,
    ) -> None:
        """Store verified source boundaries independently from bounded scans.

        Args:
            key: The dataset identity whose boundaries were checked.
            first_day: The first archive day found at the source.
            last_day: The final archive day when it is known.
        """
        if last_day is not None and last_day < first_day:
            raise ValueError("source boundary ends before it starts")
        self.connection.execute(
            """
            INSERT INTO source_bounds (
                source, product, dataset, symbol, interval, cadence, first_day, last_day
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source, product, dataset, symbol, interval, cadence)
            DO UPDATE SET
                first_day = LEAST(source_bounds.first_day, excluded.first_day),
                last_day = CASE
                    WHEN source_bounds.last_day IS NULL THEN excluded.last_day
                    WHEN excluded.last_day IS NULL THEN source_bounds.last_day
                    ELSE GREATEST(source_bounds.last_day, excluded.last_day)
                END,
                checked_at = now()
            """,
            [*_key_values(key), first_day, last_day],
        )
        LOGGER.debug(
            "Connector boundaries stored: key=%s first=%s last=%s",
            key,
            first_day,
            last_day,
        )

    def save_discovery(
        self,
        key: ResourceKey,
        start_day: date,
        end_day: date,
        resources: Sequence[Resource],
    ) -> None:
        """Store discovered resources and their inclusive search range.

        Args:
            key: The dataset identity that was searched.
            start_day: The first day searched.
            end_day: The last day searched.
            resources: The daily resources found during the search.
        """
        _validate_discovery(start_day, end_day, resources)
        if any(resource.cadence != key.cadence for resource in resources):
            raise ValueError("resource cadence does not match its catalog key")
        key_values = _key_values(key)
        scanned_at = _database_timestamp(datetime.now(UTC))
        rows = [
            (
                *key_values,
                resource.day,
                resource.end_day,
                _database_timestamp(resource.coverage[0]),
                _database_timestamp(resource.coverage[1]),
                resource.archive_symbol,
                resource.url,
                resource.checksum_url,
                resource.checksum_algorithm,
                resource.timestamp_column,
                resource.schema_version,
            )
            for resource in resources
        ]
        frame = _arrow_rows(
            rows,
            columns=(
                "source",
                "product",
                "dataset",
                "symbol",
                "interval",
                "cadence",
                "day",
                "end_day",
                "coverage_start",
                "coverage_end",
                "archive_symbol",
                "url",
                "checksum_url",
                "checksum_algorithm",
                "timestamp_column",
                "schema_version",
            ),
        )
        if rows:
            self.connection.register("incoming_resources", frame)
        try:
            with self._transaction():
                if rows:
                    self.connection.execute("""
                    UPDATE resources AS stored SET
                        status = 'discovered',
                        archive_checksum = NULL,
                        parquet_path = NULL,
                        parquet_size = NULL,
                        parquet_mtime_ns = NULL,
                        row_count = NULL,
                        first_timestamp = NULL,
                        last_timestamp = NULL,
                        error = NULL,
                        last_attempt_at = NULL
                    FROM incoming_resources AS incoming
                    WHERE stored.source = incoming.source
                      AND stored.product = incoming.product
                      AND stored.dataset = incoming.dataset
                      AND stored.symbol = incoming.symbol
                      AND stored.interval = incoming.interval
                      AND stored.cadence = incoming.cadence
                      AND stored.day = incoming.day
                      AND (
                          stored.schema_version IS DISTINCT FROM
                              incoming.schema_version
                          OR stored.checksum_algorithm IS DISTINCT FROM
                              incoming.checksum_algorithm
                          OR stored.timestamp_column IS DISTINCT FROM
                              incoming.timestamp_column
                      )
                    """)
                    self.connection.execute("""
                    INSERT INTO resources (
                        source, product, dataset, symbol, interval, cadence, day, end_day,
                        coverage_start, coverage_end, archive_symbol, url, checksum_url,
                        checksum_algorithm, timestamp_column, schema_version
                    )
                    SELECT source, product, dataset, symbol, interval, cadence, day, end_day,
                           coverage_start, coverage_end, archive_symbol, url, checksum_url,
                           checksum_algorithm, timestamp_column, schema_version
                    FROM incoming_resources
                    ON CONFLICT (
                        source, product, dataset, symbol, interval, cadence, day
                    ) DO UPDATE SET
                        end_day = excluded.end_day,
                        coverage_start = excluded.coverage_start,
                        coverage_end = excluded.coverage_end,
                        url = excluded.url,
                        checksum_url = excluded.checksum_url,
                        checksum_algorithm = excluded.checksum_algorithm,
                        archive_symbol = excluded.archive_symbol,
                        timestamp_column = excluded.timestamp_column,
                        schema_version = excluded.schema_version
                    """)
                self.connection.execute(
                    """
                    INSERT INTO discovery_segments (
                        source, product, dataset, symbol, interval, cadence,
                        start_day, end_day, scanned_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (
                        source, product, dataset, symbol, interval, cadence,
                        start_day, end_day
                    ) DO UPDATE SET scanned_at = excluded.scanned_at
                    """,
                    [*key_values, start_day, end_day, scanned_at],
                )
        finally:
            if rows:
                self.connection.unregister("incoming_resources")
        LOGGER.debug(
            "Discovery stored: key=%s range=[%s, %s] resources=%d",
            key,
            start_day,
            end_day,
            len(resources),
        )

    def resources(
        self, key: ResourceKey, start_day: date, end_day: date
    ) -> list[Resource]:
        """Return discovered resources in an inclusive day range.

        Args:
            key: The dataset identity to query.
            start_day: The first day to include.
            end_day: The last day to include.

        Returns:
            Matching resources ordered by day.
        """
        _validate_range(start_day, end_day)
        rows = self.connection.execute(
            """
            SELECT day, url, checksum_url, status, archive_checksum,
                   parquet_path, parquet_size, parquet_mtime_ns, row_count,
                   first_timestamp, last_timestamp, archive_symbol, timestamp_column,
                   schema_version, error, last_attempt_at, end_day, cadence,
                   coverage_start, coverage_end, checksum_algorithm
            FROM resources
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ? AND cadence = ?
              AND COALESCE(end_day, day) >= ? AND day <= ?
            ORDER BY day
            """,
            [*_key_values(key), start_day, end_day],
        ).fetchall()
        return [self._resource(row) for row in rows]

    def resources_between(
        self, key: ResourceKey, start: datetime, end: datetime
    ) -> list[Resource]:
        """Return physical archives overlapping an exact UTC range.

        Args:
            key: The dataset identity to query.
            start: The inclusive UTC timestamp.
            end: The exclusive UTC timestamp.

        Returns:
            Matching resources ordered by exact coverage and source day.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("resource query timestamps must include a timezone")
        if start >= end:
            raise ValueError("resource range must end after it starts")
        rows = self.connection.execute(
            """
            SELECT day, url, checksum_url, status, archive_checksum,
                   parquet_path, parquet_size, parquet_mtime_ns, row_count,
                   first_timestamp, last_timestamp, archive_symbol, timestamp_column,
                   schema_version, error, last_attempt_at, end_day, cadence,
                   coverage_start, coverage_end, checksum_algorithm
            FROM resources
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ? AND cadence = ?
              AND coverage_start < ? AND coverage_end > ?
            ORDER BY coverage_start, day
            """,
            [
                *_key_values(key),
                _database_timestamp(end),
                _database_timestamp(start),
            ],
        ).fetchall()
        return [self._resource(row) for row in rows]

    @staticmethod
    def _resource(row: Sequence[Any]) -> Resource:
        """Build one resource from a catalog result row.

        Args:
            row: Values selected in the catalog's resource column order.

        Returns:
            The reconstructed physical archive metadata.
        """
        day = row[0]
        end_day = row[16]
        coverage_start = _utc_timestamp(row[18])
        coverage_end = _utc_timestamp(row[19])
        default_start = datetime.combine(day, datetime.min.time(), UTC)
        default_end = datetime.combine(
            (end_day or day) + date.resolution,
            datetime.min.time(),
            UTC,
        )
        return Resource(
            day=row[0],
            url=row[1],
            checksum_url=row[2],
            status=row[3],
            archive_checksum=row[4],
            parquet_path=Path(row[5]) if row[5] is not None else None,
            parquet_size=row[6],
            parquet_mtime_ns=row[7],
            row_count=row[8],
            first_timestamp=_utc_timestamp(row[9]),
            last_timestamp=_utc_timestamp(row[10]),
            archive_symbol=row[11],
            timestamp_column=row[12],
            schema_version=row[13],
            error=row[14],
            last_attempt_at=_utc_timestamp(row[15]),
            end_day=end_day,
            cadence=row[17],
            coverage_start=(
                None if coverage_start == default_start else coverage_start
            ),
            coverage_end=None if coverage_end == default_end else coverage_end,
            checksum_algorithm=row[20],
        )

    def mark_ready(
        self,
        key: ResourceKey,
        day: date,
        parquet_path: Path,
        metadata: IngestedResource,
    ) -> None:
        """Record a successfully cached daily resource.

        Args:
            key: The dataset identity containing the resource.
            day: The resource day that finished caching.
            parquet_path: The path of the verified Parquet file.
            metadata: The file hashes, size, rows, and timestamp bounds.
        """
        self.mark_outcomes(key, [(day, parquet_path, metadata)], [])
        LOGGER.debug("Resource marked ready: key=%s day=%s", key, day)

    def mark_failed(self, key: ResourceKey, day: date, error: str) -> None:
        """Record a failed daily resource attempt.

        Args:
            key: The dataset identity containing the resource.
            day: The resource day that failed.
            error: The failure reported by the downloader.
        """
        self.mark_outcomes(key, [], [(day, error)])
        LOGGER.debug("Resource marked failed: key=%s day=%s error=%s", key, day, error)

    def mark_outcomes(
        self,
        key: ResourceKey,
        ready: Sequence[ReadyResourceOutcome],
        failed: Sequence[FailedResourceOutcome],
    ) -> None:
        """Record successful and failed resource attempts in one transaction.

        Args:
            key: The dataset identity containing every resource.
            ready: Successful days with their Parquet paths and metadata.
            failed: Failed days with their error messages.
        """
        days = [day for day, _path, _metadata in ready]
        days.extend(day for day, _error in failed)
        if len(days) != len(set(days)):
            raise ValueError("resource outcomes contain a duplicate day")
        if not days:
            return
        columns = (
            "day",
            "status",
            "archive_checksum",
            "parquet_path",
            "parquet_size",
            "parquet_mtime_ns",
            "row_count",
            "first_timestamp",
            "last_timestamp",
            "timestamp_column",
            "schema_version",
            "error",
        )
        rows: list[ResourceOutcomeRow] = [
            (
                day,
                "ready",
                metadata.archive_checksum,
                str(path),
                metadata.parquet_size,
                metadata.parquet_mtime_ns,
                metadata.row_count,
                _database_timestamp(metadata.first_timestamp),
                _database_timestamp(metadata.last_timestamp),
                metadata.timestamp_column,
                metadata.schema_version,
                None,
            )
            for day, path, metadata in ready
        ]
        rows.extend(
            (day, "failed", None, None, None, None, None, None, None, None, None, error)
            for day, error in failed
        )
        frame = _arrow_rows(rows, columns=columns)
        self.connection.register("incoming_resource_outcomes", frame)
        try:
            with self._transaction():
                updated = self.connection.execute(
                    """
                    UPDATE resources AS stored SET
                        status = incoming.status,
                        archive_checksum = incoming.archive_checksum,
                        parquet_path = incoming.parquet_path,
                        parquet_size = incoming.parquet_size,
                        parquet_mtime_ns = incoming.parquet_mtime_ns,
                        row_count = incoming.row_count,
                        first_timestamp = incoming.first_timestamp,
                        last_timestamp = incoming.last_timestamp,
                        timestamp_column = CASE WHEN incoming.status = 'ready'
                            THEN incoming.timestamp_column
                            ELSE stored.timestamp_column END,
                        schema_version = CASE WHEN incoming.status = 'ready'
                            THEN incoming.schema_version
                            ELSE stored.schema_version END,
                        error = incoming.error,
                        last_attempt_at = current_timestamp
                    FROM incoming_resource_outcomes AS incoming
                    WHERE stored.source = ? AND stored.product = ?
                      AND stored.dataset = ? AND stored.symbol = ?
                      AND stored.interval = ? AND cadence = ? AND stored.day = incoming.day
                    RETURNING stored.day
                    """,
                    _key_values(key),
                ).fetchall()
                updated_days = {row[0] for row in updated}
                if updated_days != set(days):
                    missing = min(set(days) - updated_days)
                    raise KeyError(f"resource {missing.isoformat()} was not discovered")
        finally:
            self.connection.unregister("incoming_resource_outcomes")
        LOGGER.debug(
            "Resource outcomes stored: key=%s ready=%d failed=%d",
            key,
            len(ready),
            len(failed),
        )
