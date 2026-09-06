"""Store market and daily resource metadata in DuckDB."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

from .models import IngestedResource, Market, Resource, ResourceKey


def _key_values(key: ResourceKey) -> tuple[str, str, str, str, str]:
    """Return a resource key as database parameter values.

    Args:
        key: The resource identity to convert.

    Returns:
        The five values forming the resource key.
    """
    return key.source, key.product, key.dataset, key.symbol, key.interval


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
def open_catalog(path: Path) -> Iterator[Catalog]:
    """Open a catalog database and close it after use.

    Args:
        path: The DuckDB file to create or open.

    Yields:
        A catalog connected to the requested database.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    try:
        yield Catalog(connection)
    finally:
        connection.close()


class Catalog:
    """Read and write downloader metadata in one DuckDB connection."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Prepare a catalog around an open DuckDB connection.

        Args:
            connection: The DuckDB connection used for catalog operations.
        """
        self.connection = connection
        self.connection.execute("SET TimeZone = 'UTC'")
        self._create_schema()

    def _create_schema(self) -> None:
        """Create the three metadata tables when they do not exist."""
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                source VARCHAR NOT NULL,
                product VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                normalized_symbol VARCHAR NOT NULL,
                base_asset VARCHAR,
                quote_asset VARCHAR,
                status VARCHAR,
                PRIMARY KEY (source, product, symbol)
            );

            CREATE TABLE IF NOT EXISTS resources (
                source VARCHAR NOT NULL,
                product VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                day DATE NOT NULL,
                url VARCHAR NOT NULL,
                checksum_url VARCHAR NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'discovered',
                archive_sha256 VARCHAR,
                parquet_path VARCHAR,
                parquet_sha256 VARCHAR,
                parquet_size BIGINT,
                row_count BIGINT,
                first_timestamp TIMESTAMP,
                last_timestamp TIMESTAMP,
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
            SELECT symbol, normalized_symbol, base_asset, quote_asset, status
            FROM markets
            WHERE source = ? AND product = ?
            ORDER BY symbol
            """,
            [source, product],
        ).fetchall()
        return [Market(*row) for row in rows]

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
            )
            for market in markets
        ]
        with self._transaction():
            self.connection.execute(
                "UPDATE markets SET status = NULL WHERE source = ? AND product = ?",
                [source, product],
            )
            self.connection.executemany(
                """
                INSERT INTO markets VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (source, product, symbol) DO UPDATE SET
                    normalized_symbol = excluded.normalized_symbol,
                    base_asset = excluded.base_asset,
                    quote_asset = excluded.quote_asset,
                    status = excluded.status
                """,
                rows,
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
        rows = [
            (*key_values, resource.day, resource.url, resource.checksum_url)
            for resource in resources
        ]
        with self._transaction():
            if rows:
                self.connection.executemany(
                    """
                    INSERT INTO resources (
                        source, product, dataset, symbol, interval, day,
                        url, checksum_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (
                        source, product, dataset, symbol, interval, day
                    ) DO UPDATE SET
                        url = excluded.url,
                        checksum_url = excluded.checksum_url
                    """,
                    rows,
                )
            self.connection.execute(
                """
                INSERT INTO discoveries (
                    source, product, dataset, symbol, interval,
                    start_day, end_day
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    source, product, dataset, symbol, interval
                ) DO UPDATE SET
                    start_day = LEAST(discoveries.start_day, excluded.start_day),
                    end_day = GREATEST(discoveries.end_day, excluded.end_day),
                    scanned_at = excluded.scanned_at
                """,
                [*key_values, start_day, end_day],
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
                   parquet_path, parquet_sha256, parquet_size, row_count,
                   first_timestamp, last_timestamp, error, last_attempt_at
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
                parquet_sha256=row[6],
                parquet_size=row[7],
                row_count=row[8],
                first_timestamp=_utc_timestamp(row[9]),
                last_timestamp=_utc_timestamp(row[10]),
                error=row[11],
                last_attempt_at=_utc_timestamp(row[12]),
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
        row = self.connection.execute(
            """
            UPDATE resources SET
                status = 'ready',
                archive_sha256 = ?,
                parquet_path = ?,
                parquet_sha256 = ?,
                parquet_size = ?,
                row_count = ?,
                first_timestamp = ?,
                last_timestamp = ?,
                error = NULL,
                last_attempt_at = current_timestamp
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ? AND day = ?
            RETURNING day
            """,
            [
                metadata.archive_sha256,
                str(parquet_path),
                metadata.parquet_sha256,
                metadata.parquet_size,
                metadata.row_count,
                _database_timestamp(metadata.first_timestamp),
                _database_timestamp(metadata.last_timestamp),
                *_key_values(key),
                day,
            ],
        ).fetchone()
        if row is None:
            raise KeyError(f"resource {day.isoformat()} was not discovered")

    def mark_failed(self, key: ResourceKey, day: date, error: str) -> None:
        """Record a failed daily resource attempt.

        Args:
            key: The dataset identity containing the resource.
            day: The resource day that failed.
            error: The failure reported by the downloader.
        """
        row = self.connection.execute(
            """
            UPDATE resources SET
                status = 'failed',
                archive_sha256 = NULL,
                parquet_path = NULL,
                parquet_sha256 = NULL,
                parquet_size = NULL,
                row_count = NULL,
                first_timestamp = NULL,
                last_timestamp = NULL,
                error = ?,
                last_attempt_at = current_timestamp
            WHERE source = ? AND product = ? AND dataset = ?
              AND symbol = ? AND interval = ? AND day = ?
            RETURNING day
            """,
            [error, *_key_values(key), day],
        ).fetchone()
        if row is None:
            raise KeyError(f"resource {day.isoformat()} was not discovered")
