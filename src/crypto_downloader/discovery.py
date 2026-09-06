"""Discover requested daily resources and record them in the catalog."""

from datetime import date, datetime, timedelta

import httpx

from .catalog import Catalog
from .models import Resource, ResourceKey
from .source import Source


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
) -> list[Resource]:
    """Discover and persist resources overlapping one request.

    Args:
        source: The source strategy used for discovery.
        catalog: The metadata catalog receiving discovered resources.
        client: The HTTPX client used for source requests.
        key: The requested source dataset identity.
        start: The inclusive first requested timestamp.
        end: The exclusive final requested timestamp.

    Returns:
        All cataloged resources in the requested daily range.
    """
    start_day, end_day = requested_days(start, end)
    resources = source.resources(client, key, start_day, end_day)
    _validate_resources(resources, start_day, end_day)
    catalog.save_discovery(key, start_day, end_day, resources)
    return catalog.resources(key, start_day, end_day)
