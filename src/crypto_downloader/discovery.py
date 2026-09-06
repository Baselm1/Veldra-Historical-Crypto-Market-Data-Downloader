"""Discover requested daily resources and record them in the catalog."""

from datetime import date, datetime, timedelta
import logging

import httpx

from .catalog import Catalog
from .models import Resource, ResourceKey
from .source import Source

LOGGER = logging.getLogger(__name__)


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
        tail_days: The number of recent active-market days to rescan.

    Returns:
        All cataloged resources in the requested daily range.
    """
    start_day, end_day = requested_days(start, end)
    if tail_days < 1:
        raise ValueError("tail_days must be positive")
    checkpoint = catalog.discovery_range(key)
    scan_ranges = _scan_ranges(
        start_day,
        end_day,
        checkpoint,
        active=active,
        refresh=refresh,
        offline=offline,
        tail_days=tail_days,
    )
    LOGGER.debug(
        "Resource discovery planned: key=%s requested=[%s, %s] checkpoint=%s "
        "scans=%s active=%s refresh=%s offline=%s",
        key,
        start_day,
        end_day,
        checkpoint,
        scan_ranges,
        active,
        refresh,
        offline,
    )
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
    checkpoint: tuple[date, date] | None,
    *,
    active: bool,
    refresh: bool,
    offline: bool,
    tail_days: int,
) -> list[tuple[date, date]]:
    """Return inclusive source ranges that still require discovery.

    Args:
        start_day: The first day whose availability is needed.
        end_day: The last day whose availability is needed.
        checkpoint: The inclusive range already searched, when available.
        active: Whether recent listings may still change.
        refresh: Whether the caller requested a complete rescan.
        offline: Whether source access is forbidden.
        tail_days: The recent active-market window to revisit.

    Returns:
        Ordered, merged inclusive ranges requiring source access.
    """
    if offline:
        return []
    if refresh or checkpoint is None:
        return [(start_day, end_day)]

    scanned_start, scanned_end = checkpoint
    ranges: list[tuple[date, date]] = []
    if start_day < scanned_start:
        ranges.append((start_day, min(end_day, scanned_start - timedelta(days=1))))
    if end_day > scanned_end:
        ranges.append((max(start_day, scanned_end + timedelta(days=1)), end_day))
    if active:
        tail_start = max(start_day, end_day - timedelta(days=tail_days - 1))
        ranges.append((tail_start, end_day))
    return _merge_ranges(ranges)


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
