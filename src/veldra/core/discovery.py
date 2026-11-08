"""Discover requested daily resources and record them in the catalog."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import logging

import httpx

from veldra.core.catalog import Catalog, DiscoveryCheckpoint
from veldra.core.datasets import DatasetSpec
from veldra.core.reporting import Reporter
from veldra.core.models import Resource, ResourceKey
from veldra.core.connector import Connector

LOGGER = logging.getLogger(__name__)
DISCOVERY_TTL = timedelta(hours=24)
DEFAULT_DISCOVERY_WINDOW_DAYS = 31
type ArchiveBatch = Callable[[list[Resource]], None]


def record_source_boundary(catalog: Catalog, key: ResourceKey, first: Resource) -> None:
    """Persist the first archive under its cadence and the source start under the dataset.

    Args:
        catalog: Catalog that will store the source metadata.
        key: Dataset identity used when requesting the source boundary.
        first: Earliest archive reported by the source.
    """
    resource_key = replace(key, cadence=first.cadence)
    catalog.save_discovery(resource_key, first.day, first.last_day, [first])
    catalog.save_source_bounds(key, first.day, None)


def boundary_predates_known_cadence(
    requested_start: datetime, known_start: date, dataset: DatasetSpec
) -> bool:
    """Detect an older request whose cached boundary covers only the daily generation.

    Args:
        requested_start: First requested UTC timestamp.
        known_start: First archive day previously recorded in the catalog.
        dataset: Dataset declaring any historical monthly generation.

    Returns:
        Whether to discover the source start again for the older request.
    """
    transition = dataset.daily_archive_start
    return transition is not None and requested_start.date() < transition <= known_start


def covered_days(resources: list[Resource]) -> set[date]:
    """Return all calendar days represented by physical archives.

    Args:
        resources: Daily, monthly, or otherwise bounded source archives.

    Returns:
        Every inclusive source day covered by the supplied archives.
    """
    return {
        resource.day + timedelta(days=offset)
        for resource in resources
        for offset in range((resource.last_day - resource.day).days + 1)
    }


def requested_days(
    start: datetime, end: datetime, offset: timedelta = timedelta(0)
) -> tuple[date, date]:
    """Return the inclusive source days touched by an exclusive range.

    Args:
        start: The inclusive first requested timestamp.
        end: The exclusive final requested timestamp.
        offset: The source-local offset used to label archive days.

    Returns:
        The first and last daily archive dates.
    """
    if start >= end:
        raise ValueError("resource range must end after it starts")
    return (start + offset).date(), (end - timedelta(microseconds=1) + offset).date()


def latest_published_day(today: date, dataset: DatasetSpec) -> date:
    """Return the latest source-day label expected to be published.

    Args:
        today: The current UTC calendar date.
        dataset: The dataset declaring its publication delay.

    Returns:
        The newest complete source day eligible for discovery.
    """
    return today - timedelta(days=dataset.publication_delay_days)


def latest_published_end(today: date, dataset: DatasetSpec) -> datetime:
    """Return the exclusive UTC end of the latest published source day.

    Args:
        today: The current UTC calendar date.
        dataset: The dataset declaring its source-day offset and delay.

    Returns:
        The exact UTC timestamp after the newest publishable archive day.
    """
    final_day = latest_published_day(today, dataset)
    return (
        datetime.combine(final_day + timedelta(days=1), datetime.min.time(), UTC)
        - dataset.archive_day_offset
    )


def _validate_resources(
    resources: list[Resource], start_day: date, end_day: date
) -> None:
    """Reject duplicate resources or days outside the requested range.

    Args:
        resources: The source resources returned by discovery.
        start_day: The first requested archive day.
        end_day: The last requested archive day.
    """
    days = [resource.day for resource in resources]
    if len(days) != len(set(days)):
        raise ValueError("source discovery returned a duplicate resource day")
    if any(day < start_day or day > end_day for day in days):
        raise ValueError("source discovery returned a resource outside the range")


def discover_resources(
    source: Connector,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    start: datetime,
    end: datetime,
    *,
    dataset: DatasetSpec | None = None,
    active: bool = True,
    refresh: bool = False,
    offline: bool = False,
    tail_days: int = 7,
    reporter: Reporter | None = None,
    on_batch: ArchiveBatch | None = None,
) -> list[Resource]:
    """Discover and persist resources overlapping one request.

    Args:
        source: The source strategy used for discovery.
        catalog: The metadata catalog receiving discovered resources.
        client: The HTTPX client used for source requests.
        key: The requested source dataset identity.
        start: The inclusive first requested timestamp.
        end: The exclusive final requested timestamp.
        dataset: The optional dataset-specific source calendar.
        active: Whether recent source listings may still change.
        refresh: Whether to rescan the complete range explicitly.
        offline: Whether all source access must be skipped.
        tail_days: The recent active-market days eligible for periodic rescans.
        reporter: The optional activity reporter for actual source scans.
        on_batch: Optional consumer of settled archive batches during discovery.

    Returns:
        All cataloged resources in the requested daily range.
    """
    offset = dataset.archive_day_offset if dataset is not None else timedelta(0)
    start_day, end_day = requested_days(start, end, offset)
    if tail_days < 1:
        raise ValueError("tail_days must be positive")
    resources = catalog.resources_between(key, start, end)
    checkpoints = catalog.discovery_checkpoints(key)
    scan_ranges = _scan_ranges(
        start_day,
        end_day,
        checkpoints,
        known=[
            (resource.day, resource.last_day)
            for resource in resources
            if resource.status != "missing"
        ],
        active=active,
        refresh=refresh,
        offline=offline,
        tail_days=tail_days,
        now=datetime.now(UTC),
    )
    LOGGER.debug(
        "Resource discovery planned: key=%s requested=[%s, %s] checkpoints=%s "
        "scans=%s active=%s refresh=%s offline=%s",
        key,
        start_day,
        end_day,
        checkpoints,
        scan_ranges,
        active,
        refresh,
        offline,
    )
    display = reporter if reporter is not None else Reporter(False)
    if on_batch is not None:
        on_batch(
            [
                resource
                for resource in resources
                if all(
                    resource.last_day < first or resource.day > last
                    for first, last in scan_ranges
                )
            ]
        )
    if scan_ranges:
        windows = (
            _scan_windows(
                scan_ranges,
                getattr(source, "discovery_window_days", DEFAULT_DISCOVERY_WINDOW_DAYS),
            )
            if on_batch is not None
            else iter(scan_ranges)
        )
        with display.status(f"Discovering {key.symbol} {key.cadence} files"):
            for scan_start, scan_end in windows:
                found = source.resources(client, key, scan_start, scan_end)
                _validate_resources(found, scan_start, scan_end)
                catalog.save_discovery(key, scan_start, scan_end, found)
                if on_batch is not None and found:
                    on_batch(catalog.resources(key, scan_start, scan_end))
                LOGGER.debug(
                    "Resource range discovered: key=%s range=[%s, %s] resources=%d",
                    key,
                    scan_start,
                    scan_end,
                    len(found),
                )
        resources = catalog.resources_between(key, start, end)
    elif not offline:
        display.info(f"{key.symbol}: reused cached {key.cadence}-file discovery")
    LOGGER.info(
        "Resource discovery complete: key=%s resources=%d scans=%d",
        key,
        len(resources),
        len(scan_ranges),
    )
    return resources


def _scan_windows(
    ranges: list[tuple[date, date]], days: int
) -> Iterator[tuple[date, date]]:
    """Yield consecutive bounded ranges for incremental discovery."""
    if days < 1:
        raise ValueError("discovery window must be positive")
    for start, end in ranges:
        while start <= end:
            last = min(end, start + timedelta(days=days - 1))
            yield start, last
            start = last + timedelta(days=1)


def _scan_ranges(
    start_day: date,
    end_day: date,
    checkpoints: list[DiscoveryCheckpoint],
    *,
    known: list[tuple[date, date]],
    active: bool,
    refresh: bool,
    offline: bool,
    tail_days: int,
    now: datetime,
) -> list[tuple[date, date]]:
    """Return inclusive source ranges that still require discovery.

    Args:
        start_day: The first day whose availability is needed.
        end_day: The last day whose availability is needed.
        checkpoints: The inclusive ranges already searched and their scan times.
        known: The inclusive date ranges of archives already confirmed to exist.
        active: Whether recent listings may still change.
        refresh: Whether the caller requested a complete rescan.
        offline: Whether source access is forbidden.
        tail_days: The recent active-market window eligible for a stale rescan.
        now: The current aware timestamp used to evaluate checkpoint freshness.

    Returns:
        Ordered, merged inclusive ranges requiring source access.
    """
    if offline:
        return []
    if refresh:
        return [(start_day, end_day)]

    fresh_ranges = _fresh_ranges(checkpoints, now)
    ranges = _uncovered_ranges(start_day, end_day, [*known, *fresh_ranges])
    if active:
        tail_start = max(start_day, end_day - timedelta(days=tail_days - 1))
        ranges.extend(_uncovered_ranges(tail_start, end_day, fresh_ranges))
    return _merge_ranges(ranges)


def _fresh_ranges(
    checkpoints: list[DiscoveryCheckpoint], now: datetime
) -> list[tuple[date, date]]:
    """Return ranges checked no more than one discovery TTL ago.

    Args:
        checkpoints: The searched ranges and their aware scan timestamps.
        now: The current aware timestamp used to calculate age.

    Returns:
        Inclusive ranges whose latest scans are still fresh.
    """
    if now.tzinfo is None:
        raise ValueError("discovery clock must include a timezone")
    cutoff = now.astimezone(UTC) - DISCOVERY_TTL
    fresh: list[tuple[date, date]] = []
    for start_day, end_day, scanned_at in checkpoints:
        if scanned_at.tzinfo is None:
            raise ValueError("discovery checkpoint must include a timezone")
        if scanned_at.astimezone(UTC) >= cutoff:
            fresh.append((start_day, end_day))
    return fresh


def _uncovered_ranges(
    start_day: date,
    end_day: date,
    checkpoints: list[tuple[date, date]],
) -> list[tuple[date, date]]:
    """Return gaps inside a requested range after completed scans.

    Args:
        start_day: The first requested day.
        end_day: The last requested day.
        checkpoints: The inclusive ranges already searched.

    Returns:
        Ordered inclusive ranges that have never been searched.
    """
    cursor = start_day
    gaps: list[tuple[date, date]] = []
    for covered_start, covered_end in _merge_ranges(checkpoints):
        if covered_end < cursor or covered_start > end_day:
            continue
        if covered_start > cursor:
            gaps.append((cursor, min(end_day, covered_start - timedelta(days=1))))
        cursor = max(cursor, covered_end + timedelta(days=1))
        if cursor > end_day:
            break
    if cursor <= end_day:
        gaps.append((cursor, end_day))
    return gaps


def _merge_ranges(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Merge overlapping or adjacent inclusive date ranges.

    Args:
        ranges: The candidate inclusive date ranges.

    Returns:
        Ordered non-overlapping ranges.
    """
    merged: list[tuple[date, date]] = []
    for start_day, end_day in sorted(ranges):
        if start_day > end_day:
            continue
        if merged and start_day <= merged[-1][1] + timedelta(days=1):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_day))
        else:
            merged.append((start_day, end_day))
    return merged
