"""Test incremental discovery, offline reuse, and durable caching."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Lock
from time import sleep

import duckdb
import httpx
import pytest

from crypto_downloader.cache import cache_resources, valid_cached_path
from crypto_downloader.catalog import Catalog, catalog_lock
from crypto_downloader.datasets import DatasetSpec, SPOT_KLINES
from crypto_downloader.discovery import discover_resources
from crypto_downloader.models import IngestedResource, Market, Resource, ResourceKey
from crypto_downloader.processing import file_sha256

KEY = ResourceKey("binance", "spot", "klines", "BTCUSDT", "1m")
START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 11, tzinfo=UTC)


class DurableSource:
    """Provide observable discovery and ingestion for durability tests."""

    code = "binance"
    products: tuple[str, ...] = ("spot",)
    active_statuses = frozenset({"TRADING"})
    max_concurrency = 2

    def __init__(self) -> None:
        """Create an empty record of source activity."""
        self.resource_calls: list[tuple[date, date]] = []
        self.ingest_calls: list[date] = []
        self.fail_once: set[date] = set()
        self.active = 0
        self.peak = 0
        self._counter_lock = Lock()

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return no markets because cache tests do not perform market discovery.

        Args:
            client: The unused HTTP client.
            product: The unused product name.

        Returns:
            An empty market list.
        """
        return []

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return one deterministic archive for every requested day.

        Args:
            client: The unused HTTP client.
            key: The unused resource identity.
            start_day: The first archive day to return.
            end_day: The last archive day to return.

        Returns:
            One resource for every day in the inclusive range.
        """
        self.resource_calls.append((start_day, end_day))
        return [resource(day) for day in days(start_day, end_day)]

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> Resource | None:
        """Return the first deterministic resource in a valid range.

        Args:
            client: The unused HTTP client.
            key: The unused resource identity.
            start_day: The earliest acceptable archive day.
            end_day: The latest acceptable archive day.

        Returns:
            The first daily resource when the range is nonempty.
        """
        if start_day > end_day:
            return None
        return resource(start_day)

    def ingest(
        self,
        client: httpx.Client,
        item: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Write a tiny file while recording bounded parallel activity.

        Args:
            client: The unused HTTP client.
            item: The resource being cached.
            dataset: The unused dataset specification.
            destination: The local file to write.

        Returns:
            Integrity metadata for the test file.
        """
        self.ingest_calls.append(item.day)
        if item.day in self.fail_once:
            self.fail_once.remove(item.day)
            raise RuntimeError("temporary failure")
        with self._counter_lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        sleep(0.01)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.day.isoformat().encode())
        stat = destination.stat()
        with self._counter_lock:
            self.active -= 1
        timestamp = datetime.combine(item.day, datetime.min.time(), UTC)
        return IngestedResource(
            "a" * 64,
            file_sha256(destination),
            stat.st_size,
            stat.st_mtime_ns,
            1,
            timestamp,
            timestamp,
        )


def days(start: date, end: date) -> list[date]:
    """Return inclusive dates for a short test range.

    Args:
        start: The first date to include.
        end: The last date to include.

    Returns:
        Every date in the inclusive range.
    """
    return [
        date.fromordinal(value)
        for value in range(start.toordinal(), end.toordinal() + 1)
    ]


def resource(day: date) -> Resource:
    """Create one test resource for a day.

    Args:
        day: The archive date represented by the resource.

    Returns:
        A deterministic resource with test URLs.
    """
    name = day.isoformat()
    return Resource(day, f"https://example/{name}.zip", f"https://example/{name}.sum")


def catalog() -> Catalog:
    """Create an isolated in-memory catalog.

    Returns:
        A catalog backed by a new in-memory DuckDB connection.
    """
    return Catalog(duckdb.connect(":memory:"))


def test_first_discovery_scans_the_complete_range() -> None:
    """Confirm a missing checkpoint causes one complete source scan."""
    source = DurableSource()
    store = catalog()

    found = discover_resources(source, store, httpx.Client(), KEY, START, END)

    assert source.resource_calls == [(date(2025, 1, 1), date(2025, 1, 10))]
    assert len(found) == 10


def test_inactive_discovery_reuses_a_complete_checkpoint() -> None:
    """Confirm finished markets do not repeat completed listings."""
    source = DurableSource()
    store = catalog()
    discover_resources(source, store, httpx.Client(), KEY, START, END)
    source.resource_calls.clear()

    found = discover_resources(
        source, store, httpx.Client(), KEY, START, END, active=False
    )

    assert source.resource_calls == []
    assert len(found) == 10


def test_active_discovery_rescans_only_the_recent_tail() -> None:
    """Confirm active markets revisit recent days without rescanning history."""
    source = DurableSource()
    store = catalog()
    discover_resources(source, store, httpx.Client(), KEY, START, END)
    source.resource_calls.clear()

    discover_resources(
        source,
        store,
        httpx.Client(),
        KEY,
        START,
        END,
        active=True,
        tail_days=3,
    )

    assert source.resource_calls == [(date(2025, 1, 8), date(2025, 1, 10))]


def test_discovery_scans_only_uncovered_checkpoint_edges() -> None:
    """Confirm older and newer unsearched days are listed without cached middle days."""
    source = DurableSource()
    store = catalog()
    middle_start = datetime(2025, 1, 3, tzinfo=UTC)
    middle_end = datetime(2025, 1, 8, tzinfo=UTC)
    discover_resources(source, store, httpx.Client(), KEY, middle_start, middle_end)
    source.resource_calls.clear()

    discover_resources(source, store, httpx.Client(), KEY, START, END, active=False)

    assert source.resource_calls == [
        (date(2025, 1, 1), date(2025, 1, 2)),
        (date(2025, 1, 8), date(2025, 1, 10)),
    ]


def test_discovery_scans_a_gap_between_disjoint_completed_ranges() -> None:
    """Confirm a skipped middle range remains eligible for later discovery."""
    source = DurableSource()
    store = catalog()
    store.save_discovery(
        KEY,
        date(2025, 1, 1),
        date(2025, 1, 2),
        [resource(day) for day in days(date(2025, 1, 1), date(2025, 1, 2))],
    )
    store.save_discovery(
        KEY,
        date(2025, 1, 8),
        date(2025, 1, 10),
        [resource(day) for day in days(date(2025, 1, 8), date(2025, 1, 10))],
    )

    discover_resources(source, store, httpx.Client(), KEY, START, END, active=False)

    assert source.resource_calls == [(date(2025, 1, 3), date(2025, 1, 7))]


def test_active_tail_merges_with_a_newer_uncovered_range() -> None:
    """Confirm overlapping active scans become one source request."""
    source = DurableSource()
    store = catalog()
    partial_end = datetime(2025, 1, 6, tzinfo=UTC)
    discover_resources(source, store, httpx.Client(), KEY, START, partial_end)
    source.resource_calls.clear()

    discover_resources(
        source,
        store,
        httpx.Client(),
        KEY,
        START,
        END,
        active=True,
        tail_days=3,
    )

    assert source.resource_calls == [(date(2025, 1, 6), date(2025, 1, 10))]


def test_discovery_rejects_a_nonpositive_tail() -> None:
    """Confirm active rediscovery cannot be configured with an empty tail."""
    with pytest.raises(ValueError, match="tail_days"):
        discover_resources(
            DurableSource(), catalog(), httpx.Client(), KEY, START, END, tail_days=0
        )


def test_refresh_repeats_the_complete_listing() -> None:
    """Confirm explicit refresh ignores a completed checkpoint."""
    source = DurableSource()
    store = catalog()
    discover_resources(source, store, httpx.Client(), KEY, START, END)
    source.resource_calls.clear()

    discover_resources(source, store, httpx.Client(), KEY, START, END, refresh=True)

    assert source.resource_calls == [(date(2025, 1, 1), date(2025, 1, 10))]


def test_offline_discovery_reads_only_the_catalog() -> None:
    """Confirm offline discovery never contacts the source."""
    source = DurableSource()
    store = catalog()
    store.save_discovery(
        KEY, date(2025, 1, 2), date(2025, 1, 2), [resource(date(2025, 1, 2))]
    )

    found = discover_resources(
        source, store, httpx.Client(), KEY, START, END, offline=True
    )

    assert source.resource_calls == []
    assert [item.day for item in found] == [date(2025, 1, 2)]


def test_cache_rejects_content_corruption_even_when_stat_is_preserved(
    tmp_path: Path,
) -> None:
    """Confirm SHA-256 catches corruption hidden behind unchanged file metadata.

    Args:
        tmp_path: The isolated cache directory.
    """
    path = tmp_path / "cached.parquet"
    path.write_bytes(b"valid")
    stat = path.stat()
    item = replace(
        resource(date(2025, 1, 1)),
        status="ready",
        parquet_path=path,
        parquet_sha256=file_sha256(path),
        parquet_size=stat.st_size,
        parquet_mtime_ns=stat.st_mtime_ns,
    )
    path.write_bytes(b"wrong")
    path.touch()
    path_stat = path.stat()
    item = replace(item, parquet_mtime_ns=path_stat.st_mtime_ns)

    assert valid_cached_path(item) is None


def test_offline_cache_reports_missing_files_without_ingestion(tmp_path: Path) -> None:
    """Confirm offline mode reports uncached archives without downloading them.

    Args:
        tmp_path: The isolated cache directory.
    """
    source = DurableSource()
    store = catalog()
    item = resource(date(2025, 1, 1))
    store.save_discovery(KEY, item.day, item.day, [item])

    coverage = cache_resources(
        source,
        store,
        httpx.Client(),
        KEY,
        SPOT_KLINES,
        [item],
        tmp_path,
        offline=True,
    )

    assert source.ingest_calls == []
    assert [problem.code for problem in coverage.problems] == ["offline_missing"]


def test_missing_ready_file_is_downloaded_again(tmp_path: Path) -> None:
    """Confirm a deleted Parquet partition is rebuilt from the source archive.

    Args:
        tmp_path: The isolated cache directory.
    """
    source = DurableSource()
    store = catalog()
    item = resource(date(2025, 1, 1))
    store.save_discovery(KEY, item.day, item.day, [item])
    first = cache_resources(
        source, store, httpx.Client(), KEY, SPOT_KLINES, [item], tmp_path
    )
    first.paths[0].unlink()
    ready = store.resources(KEY, item.day, item.day)

    second = cache_resources(
        source, store, httpx.Client(), KEY, SPOT_KLINES, ready, tmp_path
    )

    assert len(second.paths) == 1
    assert source.ingest_calls == [item.day, item.day]


def test_failed_resource_is_retried_on_the_next_request(tmp_path: Path) -> None:
    """Confirm a prior isolated failure does not permanently poison the cache.

    Args:
        tmp_path: The isolated cache directory.
    """
    source = DurableSource()
    store = catalog()
    item = resource(date(2025, 1, 1))
    source.fail_once.add(item.day)
    store.save_discovery(KEY, item.day, item.day, [item])

    first = cache_resources(
        source, store, httpx.Client(), KEY, SPOT_KLINES, [item], tmp_path
    )
    failed = store.resources(KEY, item.day, item.day)
    second = cache_resources(
        source, store, httpx.Client(), KEY, SPOT_KLINES, failed, tmp_path
    )

    assert len(first.problems) == 1
    assert second.problems == []
    assert len(second.paths) == 1
    assert source.ingest_calls == [item.day, item.day]


def test_downloads_are_concurrent_bounded_and_returned_in_day_order(
    tmp_path: Path,
) -> None:
    """Confirm daily ingestion uses the lowest configured concurrency limit.

    Args:
        tmp_path: The isolated cache directory.
    """
    source = DurableSource()
    store = catalog()
    items = [resource(day) for day in days(date(2025, 1, 1), date(2025, 1, 6))]
    store.save_discovery(KEY, items[0].day, items[-1].day, items)

    coverage = cache_resources(
        source,
        store,
        httpx.Client(),
        KEY,
        SPOT_KLINES,
        items,
        tmp_path,
        max_workers=8,
    )

    assert source.peak == 2
    assert [path.stem for path in coverage.paths] == [
        item.day.isoformat() for item in items
    ]


@pytest.mark.parametrize("workers", [0, -1])
def test_cache_rejects_nonpositive_worker_counts(tmp_path: Path, workers: int) -> None:
    """Confirm an invalid concurrency limit fails before starting work.

    Args:
        tmp_path: The isolated cache directory.
        workers: The invalid worker count.
    """
    with pytest.raises(ValueError, match="max_workers"):
        cache_resources(
            DurableSource(),
            catalog(),
            httpx.Client(),
            KEY,
            SPOT_KLINES,
            [],
            tmp_path,
            max_workers=workers,
        )


def test_cache_rejects_noninteger_worker_counts(tmp_path: Path) -> None:
    """Confirm Boolean concurrency values are not treated as integers.

    Args:
        tmp_path: The isolated cache directory.
    """
    with pytest.raises(TypeError, match="max_workers"):
        cache_resources(
            DurableSource(),
            catalog(),
            httpx.Client(),
            KEY,
            SPOT_KLINES,
            [],
            tmp_path,
            max_workers=True,
        )


def test_cache_rejects_an_invalid_source_limit(tmp_path: Path) -> None:
    """Confirm a broken source concurrency setting fails clearly.

    Args:
        tmp_path: The isolated cache directory.
    """
    source = DurableSource()
    source.max_concurrency = 0
    with pytest.raises(ValueError, match="source max_concurrency"):
        cache_resources(
            source,
            catalog(),
            httpx.Client(),
            KEY,
            SPOT_KLINES,
            [],
            tmp_path,
        )


def test_catalog_lock_serializes_the_same_database_path(tmp_path: Path) -> None:
    """Confirm two callers cannot mutate one catalog pipeline simultaneously.

    Args:
        tmp_path: The isolated catalog directory.
    """
    path = tmp_path / "catalog.duckdb"
    events: list[str] = []

    def worker(name: str) -> None:
        """Record entry and exit while holding the shared path lock.

        Args:
            name: The event prefix for this worker.
        """
        with catalog_lock(path):
            events.append(f"{name}-start")
            sleep(0.01)
            events.append(f"{name}-end")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(worker, "one")
        sleep(0.002)
        second = executor.submit(worker, "two")
        first.result()
        second.result()

    assert events == ["one-start", "one-end", "two-start", "two-end"]
