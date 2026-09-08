"""Store market and daily resource metadata in DuckDB."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
import logging
import math
from pathlib import Path
from threading import RLock

import duckdb
import pandas as pd

from crypto_downloader._core.models import (
    IngestedResource,
    Market,
    Resource,
    ResourceKey,
)

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


def _key_values(key: ResourceKey) -> tuple[str, str, str, str, str]:
    """Return a resource key as database parameter values.

    Args:
        key: The resource identity to convert.

    Returns:
        The five values forming the resource key.
    """
    return key.source, key.product, key.dataset, key.symbol, key.interval or ""


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
                day DATE NOT NULL,
                archive_symbol VARCHAR,
                url VARCHAR NOT NULL,
                checksum_url VARCHAR NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'discovered',
                archive_sha256 VARCHAR,
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
                PRIMARY KEY (source, product, dataset, symbol, interval, day)
            );

            CREATE TABLE IF NOT EXISTS discoveries (
                source VARCHAR NOT NULL,
                product VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                start_day DATE NOT NULL,
                end_day DATE NOT NULL,
                scanned_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (source, product, dataset, symbol, interval)
            );

            CREATE TABLE IF NOT EXISTS discovery_segments (
                source VARCHAR NOT NULL,
                product VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                start_day DATE NOT NULL,
                end_day DATE NOT NULL,
                scanned_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (
                    source, product, dataset, symbol, interval,
                    start_day, end_day
                )
            );

            CREATE TABLE IF NOT EXISTS source_bounds (
                source VARCHAR NOT NULL,
                product VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                first_day DATE NOT NULL,
                last_day DATE,
                checked_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (source, product, dataset, symbol, interval)
            );
            """)
        self.connection.execute(
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS refreshed_at TIMESTAMP"
        )
        for statement in (
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS pair VARCHAR",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS contract_type VARCHAR",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS contract_size DOUBLE",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS onboard_time TIMESTAMP",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS delivery_time TIMESTAMP",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS quote_volume_24h DOUBLE",
            "ALTER TABLE markets ADD COLUMN IF NOT EXISTS volume_refreshed_at TIMESTAMP",
            "ALTER TABLE resources ADD COLUMN IF NOT EXISTS archive_symbol VARCHAR",
            "ALTER TABLE resources ADD COLUMN IF NOT EXISTS timestamp_column VARCHAR",
            "ALTER TABLE resources ADD COLUMN IF NOT EXISTS schema_version INTEGER",
        ):
            self.connection.execute(statement)
        self.connection.execute(
            "UPDATE resources SET schema_version = 1 WHERE schema_version IS NULL"
        )
        self.connection.execute(
            "ALTER TABLE resources DROP COLUMN IF EXISTS parquet_sha256"
        )
        self.connection.execute("""
            INSERT INTO discovery_segments
            SELECT d.source, d.product, d.dataset, d.symbol, d.interval,
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
            )
            ON CONFLICT DO NOTHING
            """)

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
            SELECT symbol, normalized_symbol, base_asset, quote_asset, status,
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
                pair=row[5],
                contract_type=row[6],
                contract_size=row[7],
                onboard_time=_utc_timestamp(row[8]),
                delivery_time=_utc_timestamp(row[9]),
                quote_volume_24h=row[10],
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
        frame = pd.DataFrame.from_records(
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
        frame = pd.DataFrame.from_records(
            rows,
            columns=(
                "source",
                "product",
                "symbol",
                "normalized_symbol",
                "base_asset",
                "quote_asset",
                "status",
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
                    "UPDATE markets SET status = NULL "
                    "WHERE source = ? AND product = ?",
                    [source, product],
                )
                self.connection.execute("""
                    INSERT INTO markets (
                        source, product, symbol, normalized_symbol,
                        base_asset, quote_asset, status, pair, contract_type,
                        contract_size, onboard_time, delivery_time, refreshed_at
                    )
                    SELECT source, product, symbol, normalized_symbol,
                           base_asset, quote_asset, status, pair, contract_type,
                           contract_size, onboard_time, delivery_time, current_timestamp
                    FROM incoming_markets
                    ON CONFLICT (source, product, symbol) DO UPDATE SET
                        normalized_symbol = excluded.normalized_symbol,
                        base_asset = excluded.base_asset,
                        quote_asset = excluded.quote_asset,
                        status = excluded.status,
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
            SELECT start_day, end_day
            FROM discoveries
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ?
            """,
            _key_values(key),
        ).fetchone()
        return (row[0], row[1]) if row is not None else None

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
              AND symbol = ? AND interval = ?
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
              AND symbol = ? AND interval = ?
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
            SELECT min(day), max(day)
            FROM resources
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ?
            """,
            _key_values(key),
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return None
        return row[0], row[1]

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
              AND symbol = ? AND interval = ?
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
                source, product, dataset, symbol, interval, first_day, last_day
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source, product, dataset, symbol, interval)
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
            "Source boundaries stored: key=%s first=%s last=%s",
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
        key_values = _key_values(key)
        scanned_at = _database_timestamp(datetime.now(UTC))
        rows = [
            (
                *key_values,
                resource.day,
                resource.archive_symbol,
                resource.url,
                resource.checksum_url,
                resource.timestamp_column,
                resource.schema_version,
            )
            for resource in resources
        ]
        frame = pd.DataFrame.from_records(
            rows,
            columns=(
                "source",
                "product",
                "dataset",
                "symbol",
                "interval",
                "day",
                "archive_symbol",
                "url",
                "checksum_url",
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
                        archive_sha256 = NULL,
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
                      AND stored.day = incoming.day
                      AND (
                          stored.schema_version IS DISTINCT FROM
                              incoming.schema_version
                          OR stored.timestamp_column IS DISTINCT FROM
                              incoming.timestamp_column
                      )
                    """)
                    self.connection.execute("""
                    INSERT INTO resources (
                        source, product, dataset, symbol, interval, day,
                        archive_symbol, url, checksum_url, timestamp_column,
                        schema_version
                    )
                    SELECT source, product, dataset, symbol, interval, day,
                           archive_symbol, url, checksum_url, timestamp_column,
                           schema_version
                    FROM incoming_resources
                    ON CONFLICT (
                        source, product, dataset, symbol, interval, day
                    ) DO UPDATE SET
                        url = excluded.url,
                        checksum_url = excluded.checksum_url,
                        archive_symbol = excluded.archive_symbol,
                        timestamp_column = excluded.timestamp_column,
                        schema_version = excluded.schema_version
                    """)
                self.connection.execute(
                    """
                INSERT INTO discoveries (
                    source, product, dataset, symbol, interval,
                    start_day, end_day, scanned_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    source, product, dataset, symbol, interval
                ) DO UPDATE SET
                    start_day = LEAST(discoveries.start_day, excluded.start_day),
                    end_day = GREATEST(discoveries.end_day, excluded.end_day),
                    scanned_at = excluded.scanned_at
                    """,
                    [*key_values, start_day, end_day, scanned_at],
                )
                self.connection.execute(
                    """
                    INSERT INTO discovery_segments (
                        source, product, dataset, symbol, interval,
                        start_day, end_day, scanned_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (
                        source, product, dataset, symbol, interval,
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
            SELECT day, url, checksum_url, status, archive_sha256,
                   parquet_path, parquet_size, parquet_mtime_ns, row_count,
                   first_timestamp, last_timestamp, archive_symbol, timestamp_column,
                   schema_version, error, last_attempt_at
            FROM resources
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ?
              AND day BETWEEN ? AND ?
            ORDER BY day
            """,
            [*_key_values(key), start_day, end_day],
        ).fetchall()
        return [
            Resource(
                day=row[0],
                url=row[1],
                checksum_url=row[2],
                status=row[3],
                archive_sha256=row[4],
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
            )
            for row in rows
        ]

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
            "archive_sha256",
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
                metadata.archive_sha256,
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
        frame = pd.DataFrame.from_records(rows, columns=columns)
        for index, column in (
            (4, "parquet_size"),
            (5, "parquet_mtime_ns"),
            (6, "row_count"),
            (10, "schema_version"),
        ):
            frame[column] = pd.array([row[index] for row in rows], dtype="Int64")
        self.connection.register("incoming_resource_outcomes", frame)
        try:
            with self._transaction():
                updated = self.connection.execute(
                    """
                    UPDATE resources AS stored SET
                        status = incoming.status,
                        archive_sha256 = incoming.archive_sha256,
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
                      AND stored.interval = ? AND stored.day = incoming.day
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
