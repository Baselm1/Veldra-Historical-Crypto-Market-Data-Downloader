"""Discover requested daily resources and record them in the catalog."""

from datetime import UTC, date, datetime, timedelta
import logging

import httpx

from crypto_downloader._core.catalog import Catalog, DiscoveryCheckpoint
from crypto_downloader._core.reporting import Reporter
from crypto_downloader._core.models import Resource, ResourceKey
from crypto_downloader._core.source import Source

LOGGER = logging.getLogger(__name__)
DISCOVERY_TTL = timedelta(hours=24)


def requested_days(start: datetime, end: datetime) -> tuple[date, date]:
    """Return the inclusive source days touched by an exclusive range.

    Args:
        start: The inclusive first requested timestamp.
        end: The exclusive final requested timestamp.

    Returns:
        The first and last daily archive dates.
    """
    if start >= end:
        raise ValueError("resource range must end after it starts")
    return start.date(), (end - timedelta(microseconds=1)).date()


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
    source: Source,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    start: datetime,
    end: datetime,
    *,
    active: bool = True,
    refresh: bool = False,
    offline: bool = False,
    tail_days: int = 7,
    reporter: Reporter | None = None,
) -> list[Resource]:
    """Discover and persist resources overlapping one request.

    Args:
        source: The source strategy used for discovery.
        catalog: The metadata catalog receiving discovered resources.
        client: The HTTPX client used for source requests.
        key: The requested source dataset identity.
        start: The inclusive first requested timestamp.
        end: The exclusive final requested timestamp.
        active: Whether recent source listings may still change.
        refresh: Whether to rescan the complete range explicitly.
        offline: Whether all source access must be skipped.
        tail_days: The recent active-market days eligible for periodic rescans.
        reporter: The optional activity reporter for actual source scans.

    Returns:
        All cataloged resources in the requested daily range.
    """
    start_day, end_day = requested_days(start, end)
    if tail_days < 1:
        raise ValueError("tail_days must be positive")
    checkpoints = catalog.discovery_checkpoints(key)
    scan_ranges = _scan_ranges(
        start_day,
        end_day,
        checkpoints,
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
    if scan_ranges:
        with display.status(f"Discovering {key.symbol} daily files"):
            for scan_start, scan_end in scan_ranges:
                resources = source.resources(client, key, scan_start, scan_end)
                _validate_resources(resources, scan_start, scan_end)
                catalog.save_discovery(key, scan_start, scan_end, resources)
                LOGGER.debug(
                    "Resource range discovered: key=%s range=[%s, %s] resources=%d",
                    key,
                    scan_start,
                    scan_end,
                    len(resources),
                )
    elif not offline:
        display.info(f"{key.symbol}: reused cached daily-file discovery")
    resources = catalog.resources(key, start_day, end_day)
    LOGGER.info(
        "Resource discovery complete: key=%s resources=%d scans=%d",
        key,
        len(resources),
        len(scan_ranges),
    )
    return resources


def _scan_ranges(
    start_day: date,
    end_day: date,
    checkpoints: list[DiscoveryCheckpoint],
    *,
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
    if refresh or not checkpoints:
        return [(start_day, end_day)]

    covered_ranges = [(start, end) for start, end, _ in checkpoints]
    ranges = _uncovered_ranges(start_day, end_day, covered_ranges)
    if active:
        tail_start = max(start_day, end_day - timedelta(days=tail_days - 1))
        fresh_ranges = _fresh_ranges(checkpoints, now)
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
