"""Test physical archives, materializations, and logical catalog partitions."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from veldra.core.catalog import Catalog
from veldra.core.models import (
    ArchiveKey,
    ArchiveObject,
    IntegritySpec,
    LogicalPartition,
    Materialization,
)
from veldra.core.subjects import DataSubject

START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 2, tzinfo=UTC)
KEY = ArchiveKey(
    "okx",
    "spot",
    "klines",
    "archive",
    "all",
    "ANY",
    "daily",
    date(2025, 1, 1),
    date(2025, 1, 1),
    "spot-klines-ANY-2025-01-01.zip",
)


def archive(url: str = "https://signed.example/one?token=first") -> ArchiveObject:
    """Create one response-header-verified physical archive.

    Args:
        url: The current temporary download URL.

    Returns:
        A discovered archive object.
    """
    return ArchiveObject(
        KEY,
        url,
        url_expires_at=END,
        remote_size=1_000,
        integrity=IntegritySpec("response_header", algorithm="md5"),
        discovered_at=START,
    )


def materialization(path: Path) -> Materialization:
    """Create local metadata for one shared Parquet file.

    Args:
        path: The materialized Parquet path.

    Returns:
        A valid materialization declaration.
    """
    return Materialization(
        KEY,
        path,
        2,
        2_880,
        START,
        END - timedelta(minutes=1),
        42_000,
        local_mtime_ns=123,
        archive_revision="revision-one",
        ready_at=END,
    )


def partition(path: Path, symbol: str, rows: int = 1_440) -> LogicalPartition:
    """Create one instrument view of a shared materialization.

    Args:
        path: The shared Parquet path.
        symbol: The instrument selected from the file.
        rows: The number of rows belonging to the instrument.

    Returns:
        A logical instrument partition.
    """
    return LogicalPartition(
        "okx",
        "spot",
        "klines",
        DataSubject("instrument", symbol),
        "1m",
        START,
        END,
        path,
        "instrument_name",
        symbol,
        rows,
        source_day=date(2025, 1, 1),
    )


@pytest.fixture
def catalog() -> Catalog:
    """Create one isolated initialized catalog.

    Returns:
        An in-memory logical catalog.
    """
    return Catalog(duckdb.connect())


def test_archive_identity_excludes_temporary_signed_urls(catalog: Catalog) -> None:
    """Confirm URL refresh updates one stable physical archive row."""
    first = archive()
    refreshed = archive("https://signed.example/one?token=second")

    catalog.save_archives([first])
    catalog.save_archives([refreshed])

    assert first.key.archive_id == refreshed.key.archive_id
    assert catalog.archive(KEY) == refreshed
    assert catalog.connection.execute(
        "select count(*) from archive_objects"
    ).fetchone() == (1,)


def test_archive_failures_require_discovery_and_clear_ready_state(
    catalog: Catalog,
) -> None:
    """Confirm physical failures are recorded only for known objects."""
    catalog.save_archives([archive()])

    catalog.mark_archive_failed(KEY, "expired URL")
    failed = catalog.archive(KEY)

    assert failed is not None
    assert failed.status == "failed"
    assert failed.error == "expired URL"
    assert failed.last_attempt_at is not None
    with pytest.raises(KeyError, match="archive"):
        catalog.mark_archive_failed(
            ArchiveKey(
                "okx",
                "spot",
                "trades",
                "archive",
                "all",
                "ANY",
                "daily",
                date(2025, 1, 1),
                date(2025, 1, 1),
                "missing.zip",
            ),
            "missing",
        )


def test_catalog_returns_every_known_archive_state_in_a_range(
    catalog: Catalog,
) -> None:
    """Confirm coverage inspection can see discovered and failed objects."""
    discovered = archive()
    later_key = ArchiveKey(
        "okx",
        "spot",
        "klines",
        "archive",
        "instrument",
        "BTC-USDT",
        "daily",
        date(2025, 1, 2),
        date(2025, 1, 2),
        "BTC-USDT-2025-01-02.zip",
    )
    failed = ArchiveObject(later_key, "https://signed.example/two")
    catalog.save_archives([discovered, failed])
    catalog.mark_archive_failed(later_key, "source failure")

    found = catalog.archives_between(
        "okx", "spot", "klines", date(2025, 1, 1), date(2025, 1, 2)
    )

    assert [item.status for item in found] == ["discovered", "failed"]
    assert (
        catalog.archives_between(
            "okx", "spot", "klines", date(2025, 1, 3), date(2025, 1, 4)
        )
        == []
    )


def test_one_materialization_publishes_many_queryable_partitions(
    catalog: Catalog, tmp_path: Path
) -> None:
    """Confirm a shared archive is stored once and exposed per instrument."""
    path = tmp_path / "all.parquet"
    item = materialization(path)
    bitcoin = partition(path, "BTC-USDT")
    ether = partition(path, "ETH-USDT")
    catalog.save_archives([archive()])

    catalog.publish_materialization(item, [bitcoin, ether])

    assert catalog.partitions_between(
        "okx",
        "spot",
        "klines",
        DataSubject("instrument", "BTC-USDT"),
        "1m",
        START,
        END,
    ) == [bitcoin]
    assert catalog.partitions_between(
        "okx",
        "spot",
        "klines",
        DataSubject("instrument", "ETH-USDT"),
        "1m",
        START,
        END,
    ) == [ether]
    assert catalog.unreferenced_materializations() == []
    stored = catalog.archive(KEY)
    assert stored is not None and stored.status == "ready"


def test_materialization_publication_is_atomic_and_idempotent(
    catalog: Catalog, tmp_path: Path
) -> None:
    """Confirm invalid partitions publish nothing and repeats add no rows."""
    path = tmp_path / "all.parquet"
    item = materialization(path)
    valid = partition(path, "BTC-USDT")
    invalid = partition(tmp_path / "other.parquet", "ETH-USDT")
    catalog.save_archives([archive()])

    with pytest.raises(ValueError, match="path"):
        catalog.publish_materialization(item, [valid, invalid])
    assert catalog.connection.execute(
        "select count(*) from materializations"
    ).fetchone() == (0,)

    catalog.publish_materialization(item, [valid])
    catalog.publish_materialization(item, [valid])
    assert catalog.connection.execute(
        "select count(*) from materializations"
    ).fetchone() == (1,)
    assert catalog.connection.execute(
        "select count(*) from logical_partitions"
    ).fetchone() == (1,)


def test_materialization_deletion_is_reference_safe(
    catalog: Catalog, tmp_path: Path
) -> None:
    """Confirm physical metadata cannot be deleted while subjects use it."""
    path = tmp_path / "all.parquet"
    item = materialization(path)
    catalog.save_archives([archive()])
    catalog.publish_materialization(item, [partition(path, "BTC-USDT")])

    with pytest.raises(RuntimeError, match="referenced"):
        catalog.delete_materialization(item.materialization_id)
    assert (
        catalog.delete_partitions(
            "okx",
            "spot",
            "klines",
            DataSubject("instrument", "BTC-USDT"),
            "1m",
            START,
            END,
        )
        == 1
    )
    assert catalog.unreferenced_materializations() == [item]
    assert catalog.delete_materialization(item.materialization_id) == path
    assert catalog.unreferenced_materializations() == []
