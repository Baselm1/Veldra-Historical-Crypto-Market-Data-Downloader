"""Test the DuckDB metadata catalog."""

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

from crypto_downloader.catalog import Catalog, open_catalog
from crypto_downloader.models import IngestedResource, Market, Resource, ResourceKey

KEY = ResourceKey("binance", "spot", "klines", "BTCUSDT", "1m")


def resource(day: int = 1, name: str = "original") -> Resource:
    """Create one daily resource for a January 2025 day.

    Args:
        day: The January day stored in the resource.
        name: A value used to distinguish its URLs.

    Returns:
        A discovered daily resource.
    """
    return Resource(
        date(2025, 1, day),
        f"https://data.example/{name}.zip",
        f"https://data.example/{name}.zip.CHECKSUM",
    )


def ingested() -> IngestedResource:
    """Create cache metadata for one two-row Parquet file.

    Returns:
        Representative verified file metadata.
    """
    return IngestedResource(
        archive_sha256="a" * 64,
        parquet_sha256="b" * 64,
        parquet_size=1234,
        parquet_mtime_ns=987654321,
        row_count=2,
        first_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        last_timestamp=datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
    )


@pytest.fixture
def catalog() -> Iterator[Catalog]:
    """Create an isolated in-memory catalog.

    Returns:
        A catalog backed by a temporary in-memory connection.
    """
    connection = duckdb.connect()
    value = Catalog(connection)
    yield value
    connection.close()


def test_catalog_creates_the_metadata_tables_and_uses_utc(catalog: Catalog) -> None:
    """Confirm a new catalog creates its planned schema and uses UTC."""
    tables = {
        row[0]
        for row in catalog.connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }

    assert tables == {"markets", "resources", "discoveries", "discovery_segments"}
    timezone = catalog.connection.execute(
        "SELECT current_setting('TimeZone')"
    ).fetchone()
    assert timezone is not None
    assert timezone[0] == "UTC"


def test_open_catalog_creates_parent_directories_and_persists(tmp_path: Path) -> None:
    """Confirm a file catalog survives closing and reopening."""
    path = tmp_path / "nested" / "catalog.duckdb"
    market = Market("BTCUSDT", "BTCUSDT", "BTC", "USDT", "TRADING")

    with open_catalog(path) as first:
        first.save_markets("binance", "spot", [market])

    assert path.is_file()
    with open_catalog(path) as second:
        assert second.markets("binance", "spot") == [market]


def test_market_snapshot_is_sorted_updated_and_isolated(catalog: Catalog) -> None:
    """Confirm market snapshots update their scope without mixing products."""
    bitcoin = Market("BTCUSDT", "BTCUSDT", "BTC", "USDT", "TRADING")
    ether = Market("ETHUSDT", "ETHUSDT", "ETH", "USDT", "TRADING")
    futures = Market("BTCUSDT", "BTCUSDT", "BTC", "USDT", "TRADING")

    catalog.save_markets("binance", "spot", [ether, bitcoin])
    catalog.save_markets("binance", "um", [futures])
    catalog.save_markets(
        "binance",
        "spot",
        [Market("BTCUSDT", "BTCUSDT", "BTC", "USDT", "BREAK")],
    )

    assert catalog.markets("binance", "spot") == [
        Market("BTCUSDT", "BTCUSDT", "BTC", "USDT", "BREAK"),
        Market("ETHUSDT", "ETHUSDT", "ETH", "USDT", None),
    ]
    assert catalog.markets("binance", "um") == [futures]
    assert catalog.markets("other", "spot") == []


def test_market_snapshot_accepts_archive_only_symbols_without_asset_metadata(
    catalog: Catalog,
) -> None:
    """Confirm an archive-only symbol may lack exchange-info asset fields."""
    archived = Market("OLDPAIR", "OLDPAIR")

    catalog.save_markets("binance", "spot", [archived])

    assert catalog.markets("binance", "spot") == [archived]


def test_market_snapshot_records_a_refresh_timestamp(catalog: Catalog) -> None:
    """Confirm market metadata records when its complete snapshot was refreshed."""
    assert catalog.market_snapshot_at("binance", "spot") is None

    catalog.save_markets(
        "binance",
        "spot",
        [Market("BTCUSDT", "BTCUSDT", "BTC", "USDT", "TRADING")],
    )

    refreshed_at = catalog.market_snapshot_at("binance", "spot")
    assert refreshed_at is not None
    assert refreshed_at.tzinfo == UTC


@pytest.mark.parametrize(
    "markets",
    [
        [],
        [
            Market("BTCUSDT", "BTCUSDT", "BTC", "USDT", "TRADING"),
            Market("BTCUSDT", "BTCUSDT", "BTC", "USDT", "BREAK"),
        ],
    ],
)
def test_invalid_market_snapshots_do_not_change_existing_rows(
    catalog: Catalog, markets: list[Market]
) -> None:
    """Confirm empty and duplicate snapshots fail before altering the catalog.

    Args:
        catalog: The isolated catalog used by the test.
        markets: An invalid complete snapshot.
    """
    original = Market("ETHUSDT", "ETHUSDT", "ETH", "USDT", "TRADING")
    catalog.save_markets("binance", "spot", [original])

    with pytest.raises(ValueError, match="snapshot"):
        catalog.save_markets("binance", "spot", markets)

    assert catalog.markets("binance", "spot") == [original]


def test_discovery_stores_sorted_resources_and_inclusive_boundaries(
    catalog: Catalog,
) -> None:
    """Confirm discovery writes daily URLs and the searched day range."""
    catalog.save_discovery(
        KEY,
        date(2025, 1, 1),
        date(2025, 1, 3),
        [resource(3, "three"), resource(1, "one")],
    )

    assert catalog.discovery_range(KEY) == (date(2025, 1, 1), date(2025, 1, 3))
    assert catalog.resources(KEY, date(2025, 1, 1), date(2025, 1, 3)) == [
        resource(1, "one"),
        resource(3, "three"),
    ]
    assert catalog.resources(KEY, date(2025, 1, 2), date(2025, 1, 2)) == []
    assert catalog.resource_bounds(KEY) == (date(2025, 1, 1), date(2025, 1, 3))


def test_resource_bounds_are_absent_without_discovered_files(catalog: Catalog) -> None:
    """Confirm empty discovery coverage does not invent availability bounds."""
    catalog.save_discovery(KEY, date(2025, 1, 1), date(2025, 1, 2), [])

    assert catalog.resource_bounds(KEY) is None


def test_empty_discovery_still_records_the_completed_range(catalog: Catalog) -> None:
    """Confirm a successful search with no files still advances discovery."""
    catalog.save_discovery(KEY, date(2025, 1, 1), date(2025, 1, 2), [])

    assert catalog.discovery_range(KEY) == (date(2025, 1, 1), date(2025, 1, 2))
    assert catalog.resources(KEY, date(2025, 1, 1), date(2025, 1, 2)) == []


def test_disjoint_discoveries_remain_distinct_coverage_segments(
    catalog: Catalog,
) -> None:
    """Confirm skipped dates are not hidden inside aggregate discovery bounds."""
    catalog.save_discovery(KEY, date(2025, 1, 1), date(2025, 1, 2), [resource(1)])
    catalog.save_discovery(KEY, date(2025, 1, 8), date(2025, 1, 9), [resource(8)])

    assert catalog.discovery_range(KEY) == (date(2025, 1, 1), date(2025, 1, 9))
    assert catalog.discovery_ranges(KEY) == [
        (date(2025, 1, 1), date(2025, 1, 2)),
        (date(2025, 1, 8), date(2025, 1, 9)),
    ]


def test_disjoint_discovery_segments_survive_catalog_reopening(tmp_path: Path) -> None:
    """Confirm compatibility migration never replaces real segments with bounds."""
    path = tmp_path / "catalog.duckdb"
    with open_catalog(path) as catalog:
        catalog.save_discovery(KEY, date(2025, 1, 1), date(2025, 1, 2), [resource(1)])
        catalog.save_discovery(KEY, date(2025, 1, 8), date(2025, 1, 9), [resource(8)])

    with open_catalog(path) as catalog:
        assert catalog.discovery_ranges(KEY) == [
            (date(2025, 1, 1), date(2025, 1, 2)),
            (date(2025, 1, 8), date(2025, 1, 9)),
        ]


def test_repeated_discovery_expands_bounds_and_preserves_cache_state(
    catalog: Catalog, tmp_path: Path
) -> None:
    """Confirm rescans refresh URLs without discarding valid cache metadata."""
    catalog.save_discovery(KEY, date(2025, 1, 2), date(2025, 1, 2), [resource(2)])
    catalog.mark_ready(KEY, date(2025, 1, 2), tmp_path / "two.parquet", ingested())

    catalog.save_discovery(
        KEY,
        date(2025, 1, 1),
        date(2025, 1, 3),
        [resource(2, "refreshed")],
    )
    found = catalog.resources(KEY, date(2025, 1, 2), date(2025, 1, 2))[0]

    assert catalog.discovery_range(KEY) == (date(2025, 1, 1), date(2025, 1, 3))
    assert found.url.endswith("refreshed.zip")
    assert found.checksum_url.endswith("refreshed.zip.CHECKSUM")
    assert found.status == "ready"
    assert found.archive_sha256 == "a" * 64
    assert found.parquet_path == tmp_path / "two.parquet"
    assert found.error is None


@pytest.mark.parametrize(
    ("start_day", "end_day", "resources", "message"),
    [
        (date(2025, 1, 2), date(2025, 1, 1), [], "range"),
        (
            date(2025, 1, 1),
            date(2025, 1, 2),
            [resource(1), resource(1, "duplicate")],
            "duplicate",
        ),
        (date(2025, 1, 1), date(2025, 1, 2), [resource(3)], "outside"),
    ],
)
def test_invalid_discovery_is_rejected_atomically(
    catalog: Catalog,
    start_day: date,
    end_day: date,
    resources: list[Resource],
    message: str,
) -> None:
    """Confirm invalid discovery input leaves no checkpoint or resource rows.

    Args:
        catalog: The isolated catalog used by the test.
        start_day: The proposed first searched day.
        end_day: The proposed last searched day.
        resources: The proposed discovered daily resources.
        message: The expected validation error text.
    """
    with pytest.raises(ValueError, match=message):
        catalog.save_discovery(KEY, start_day, end_day, resources)

    assert catalog.discovery_range(KEY) is None
    assert catalog.resources(KEY, date(2025, 1, 1), date(2025, 1, 3)) == []


def test_database_failure_rolls_back_resources_and_checkpoint(catalog: Catalog) -> None:
    """Confirm a failed database write leaves no partial discovery state."""
    invalid = resource(2)
    object.__setattr__(invalid, "url", None)

    with pytest.raises(duckdb.ConstraintException):
        catalog.save_discovery(
            KEY,
            date(2025, 1, 1),
            date(2025, 1, 2),
            [resource(1), invalid],
        )

    assert catalog.discovery_range(KEY) is None
    assert catalog.resources(KEY, date(2025, 1, 1), date(2025, 1, 2)) == []


def test_resource_queries_are_isolated_by_the_complete_key(catalog: Catalog) -> None:
    """Confirm similar symbols and datasets cannot leak into a resource query."""
    other = ResourceKey("binance", "spot", "trades", "BTCUSDT", "1m")
    catalog.save_discovery(
        KEY, date(2025, 1, 1), date(2025, 1, 1), [resource(1, "kline")]
    )
    catalog.save_discovery(
        other, date(2025, 1, 1), date(2025, 1, 1), [resource(1, "trade")]
    )

    assert catalog.resources(KEY, date(2025, 1, 1), date(2025, 1, 1)) == [
        resource(1, "kline")
    ]


def test_ready_resource_retains_all_integrity_metadata(
    catalog: Catalog, tmp_path: Path
) -> None:
    """Confirm a successful cache write records every required metadata value."""
    path = tmp_path / "BTCUSDT-1m-2025-01-01.parquet"
    metadata = ingested()
    catalog.save_discovery(KEY, date(2025, 1, 1), date(2025, 1, 1), [resource()])

    catalog.mark_ready(KEY, date(2025, 1, 1), path, metadata)
    found = catalog.resources(KEY, date(2025, 1, 1), date(2025, 1, 1))[0]

    assert found.status == "ready"
    assert found.archive_sha256 == metadata.archive_sha256
    assert found.parquet_path == path
    assert found.parquet_sha256 == metadata.parquet_sha256
    assert found.parquet_size == metadata.parquet_size
    assert found.parquet_mtime_ns == metadata.parquet_mtime_ns
    assert found.row_count == metadata.row_count
    assert found.first_timestamp == metadata.first_timestamp
    assert found.last_timestamp == metadata.last_timestamp
    assert found.error is None
    assert found.last_attempt_at is not None


def test_ready_resource_rejects_naive_integrity_timestamps(
    catalog: Catalog, tmp_path: Path
) -> None:
    """Confirm integrity timestamp metadata must state its timezone."""
    catalog.save_discovery(KEY, date(2025, 1, 1), date(2025, 1, 1), [resource()])
    metadata = replace(ingested(), first_timestamp=datetime(2025, 1, 1))

    with pytest.raises(ValueError, match="timezone"):
        catalog.mark_ready(KEY, date(2025, 1, 1), tmp_path / "one.parquet", metadata)

    found = catalog.resources(KEY, date(2025, 1, 1), date(2025, 1, 1))[0]
    assert found.status == "discovered"


def test_failed_resource_clears_stale_cache_metadata(
    catalog: Catalog, tmp_path: Path
) -> None:
    """Confirm a later failure cannot leave a resource appearing cache-ready."""
    catalog.save_discovery(KEY, date(2025, 1, 1), date(2025, 1, 1), [resource()])
    catalog.mark_ready(KEY, date(2025, 1, 1), tmp_path / "one.parquet", ingested())

    catalog.mark_failed(KEY, date(2025, 1, 1), "checksum mismatch")
    found = catalog.resources(KEY, date(2025, 1, 1), date(2025, 1, 1))[0]

    assert found.status == "failed"
    assert found.error == "checksum mismatch"
    assert found.archive_sha256 is None
    assert found.parquet_path is None
    assert found.parquet_sha256 is None
    assert found.parquet_size is None
    assert found.parquet_mtime_ns is None
    assert found.row_count is None
    assert found.first_timestamp is None
    assert found.last_timestamp is None
    assert found.last_attempt_at is not None


@pytest.mark.parametrize("operation", ["ready", "failed"])
def test_resource_state_changes_require_a_discovered_row(
    catalog: Catalog, tmp_path: Path, operation: str
) -> None:
    """Confirm cache results cannot silently create unknown resources.

    Args:
        catalog: The isolated catalog used by the test.
        tmp_path: The temporary location used for a prospective Parquet path.
        operation: The resource state change attempted by the test.
    """
    with pytest.raises(KeyError, match="2025-01-01"):
        if operation == "ready":
            catalog.mark_ready(
                KEY, date(2025, 1, 1), tmp_path / "missing.parquet", ingested()
            )
        else:
            catalog.mark_failed(KEY, date(2025, 1, 1), "missing")


def test_reversed_resource_query_range_is_rejected(catalog: Catalog) -> None:
    """Confirm resource queries reject a last day before the first day."""
    with pytest.raises(ValueError, match="range"):
        catalog.resources(KEY, date(2025, 1, 2), date(2025, 1, 1))
