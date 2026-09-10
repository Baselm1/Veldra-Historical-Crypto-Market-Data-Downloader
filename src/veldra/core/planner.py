"""Select non-overlapping monthly and daily archives for a requested range."""

from calendar import monthrange
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
import logging

import httpx

from .catalog import Catalog
from .cache import valid_cached_path
from .datasets import DatasetSpec
from .discovery import (
    discover_resources,
    requested_days,
    _uncovered_ranges,
    _merge_ranges,
)
from .models import Resource, ResourceKey
from .reporting import Reporter
from .connector import Connector

LOGGER = logging.getLogger(__name__)


def covered_days(resources: list[Resource]) -> set[date]:
    """Return calendar days covered by the supplied physical archives."""
    return {
        resource.day + timedelta(days=i)
        for resource in resources
        for i in range((resource.last_day - resource.day).days + 1)
    }


def catalog_archives(
    catalog: Catalog, key: ResourceKey, first: date, last: date
) -> list[Resource]:
    """Return daily and monthly archive metadata overlapping inclusive dates."""
    return catalog.resources(key, first, last) + catalog.resources(
        replace(key, cadence="monthly"), first, last
    )


def catalog_archives_between(
    catalog: Catalog, key: ResourceKey, start: datetime, end: datetime
) -> list[Resource]:
    """Return daily and monthly archives overlapping an exact UTC range.

    Args:
        catalog: The metadata catalog containing physical archives.
        key: The daily resource identity.
        start: The inclusive UTC request timestamp.
        end: The exclusive UTC request timestamp.

    Returns:
        Every overlapping daily and monthly archive.
    """
    return catalog.resources_between(key, start, end) + catalog.resources_between(
        replace(key, cadence="monthly"), start, end
    )


def select_archives(resources: list[Resource], dataset: DatasetSpec) -> list[Resource]:
    """Select whole archives without overlaps, preferring ready files then months.

    Args:
        resources: All candidate archives for one dataset and pair.
        dataset: Schema used to check local cache metadata.

    Returns:
        Chronological non-overlapping physical archives.
    """
    ordered = sorted(
        resources,
        key=lambda r: (
            valid_cached_path(r, dataset) is None,
            r.cadence != "monthly",
            r.day,
        ),
    )
    selected: list[Resource] = []
    occupied: list[tuple[datetime, datetime]] = []
    for resource in ordered:
        start, end = resource.coverage
        if all(
            end <= other_start or start >= other_end
            for other_start, other_end in occupied
        ):
            selected.append(resource)
            occupied.append((start, end))
    return sorted(selected, key=lambda r: r.day)


def _months(first: date, last: date) -> list[tuple[date, date]]:
    """Return whole calendar months contained in inclusive requested dates."""
    months = []
    cursor = first.replace(day=1)
    while cursor <= last:
        end = cursor.replace(day=monthrange(cursor.year, cursor.month)[1])
        if cursor >= first and end <= last:
            months.append((cursor, end))
        cursor = end + timedelta(days=1)
    return months


def _monthly_candidates(
    resources: list[Resource],
    dataset: DatasetSpec,
    first: date,
    last: date,
    *,
    refresh: bool,
) -> list[Resource]:
    """Select ready files and usable whole-month downloads without overlaps.

    Args:
        resources: Known daily and monthly archives.
        dataset: Expected cache schema.
        first: First requested calendar day.
        last: Last requested calendar day.
        refresh: Whether failed monthly downloads should be retried.

    Returns:
        Ready files plus monthly downloads worth attempting.
    """
    candidates = []
    for resource in resources:
        ready = valid_cached_path(resource, dataset) is not None
        usable_month = (
            resource.cadence == "monthly"
            and (refresh or resource.status != "failed")
            and first <= resource.day
            and resource.last_day <= last
        )
        if ready or usable_month:
            candidates.append(resource)
    return select_archives(candidates, dataset)


def plan_archives(
    source: Connector,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    start: datetime,
    end: datetime,
    *,
    dataset: DatasetSpec,
    active: bool = True,
    refresh: bool = False,
    offline: bool = False,
    tail_days: int = 7,
    reporter: Reporter | None = None,
) -> list[Resource]:
    """Discover useful monthly archives and daily files for uncovered dates.

    Args:
        source: Exchange archive connector.
        catalog: Open metadata catalog.
        client: Shared HTTP client.
        key: Dataset and market identity.
        start: Inclusive request timestamp.
        end: Exclusive request timestamp.
        dataset: Requested source schema.
        active: Whether recent listings can change.
        refresh: Force remote metadata checks.
        offline: Use only cataloged resources.
        tail_days: Recent days subject to discovery freshness.
        reporter: Optional progress display.

    Returns:
        A non-overlapping set of physical archives for the request.
    """
    offset = getattr(source, "archive_day_offset", timedelta(0))
    first, last = requested_days(start, end, offset)

    def scan(
        scan_key: ResourceKey,
        scan_start: date,
        scan_end: date,
        *,
        monthly: bool = False,
    ) -> list[Resource]:
        """Discover an inclusive archive range using the request's cache settings."""
        return discover_resources(
            source,
            catalog,
            client,
            scan_key,
            datetime.combine(scan_start, time.min, UTC) - offset,
            datetime.combine(scan_end + timedelta(days=1), time.min, UTC) - offset,
            active=True if monthly else active,
            refresh=refresh,
            offline=offline,
            tail_days=(scan_end - scan_start).days + 1 if monthly else tail_days,
            reporter=reporter,
        )

    if key.dataset not in getattr(source, "monthly_datasets", ()):
        return scan(key, first, last)
    existing = catalog_archives_between(catalog, key, start, end)
    cached = [r for r in existing if valid_cached_path(r, dataset) is not None]
    month_key = replace(key, cadence="monthly")
    months = [
        (a, b)
        for a, b in _months(first, last)
        if refresh or not any(r.day <= b and r.last_day >= a for r in cached)
    ]
    for month_start, month_end in _merge_ranges(months):
        # Monthly publication can lag into the next month; retry absent months
        # after the same TTL as daily discovery, including inactive markets.
        try:
            scan(month_key, month_start, month_end, monthly=True)
        except Exception as error:
            LOGGER.warning(
                "Monthly listing unavailable; trying daily archives: key=%s error=%s",
                month_key,
                error,
            )
    candidates = catalog_archives_between(catalog, key, start, end)
    selected = _monthly_candidates(
        candidates,
        dataset,
        first,
        last,
        refresh=refresh,
    )
    for gap_start, gap_end in _uncovered_ranges(
        first, last, [(r.day, r.last_day) for r in selected if r.cadence == "monthly"]
    ):
        selected.extend(scan(key, gap_start, gap_end))
    return select_archives(selected, dataset)
