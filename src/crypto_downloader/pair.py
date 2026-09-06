"""Run the download workflow for one requested pair."""

from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import httpx

from .catalog import Catalog
from .cache import cache_resources
from .datasets import DatasetSpec
from .discovery import discover_resources, requested_days
from .models import Market, Message, Resource, ResourceKey, Result
from .query import empty_frame, query_parquet
from .request import Request, normalize_pair
from .source import Source


def _result(pair: str, request: Request, dataset: DatasetSpec) -> Result:
    """Create an empty result for one pair workflow.

    Args:
        pair: The caller's original pair spelling.
        request: The validated shared request.
        dataset: The requested dataset schema.

    Returns:
        An empty result with its requested output schema.
    """
    columns = dataset.resolve_columns(request.columns)
    return Result(
        pair=pair,
        data=empty_frame(dataset, columns),
        requested_range=(request.start, request.end),
        product=request.product,
        dataset=request.dataset,
    )


def _market(pair: str, markets: list[Market]) -> Market | None:
    """Resolve one normalized pair only when it has one exact match.

    Args:
        pair: The caller's pair spelling.
        markets: The current source market snapshot.

    Returns:
        The unique normalized market, or ``None``.
    """
    normalized = normalize_pair(pair)
    matches = [market for market in markets if market.normalized_symbol == normalized]
    return matches[0] if len(matches) == 1 else None


def _missing_resources(
    resources: list[Resource], start: datetime, end: datetime
) -> list[Message]:
    """Describe requested days absent from a valid source listing.

    Args:
        resources: The resources found for the request.
        start: The inclusive first requested timestamp.
        end: The exclusive final requested timestamp.

    Returns:
        One problem for each unavailable daily archive.
    """
    first, last = requested_days(start, end)
    available = {resource.day for resource in resources}
    return [
        Message(
            "resource_unavailable",
            "No source file is available for this date.",
            first + timedelta(days=index),
        )
        for index in range((last - first).days + 1)
        if first + timedelta(days=index) not in available
    ]


def _availability(resources: list[Resource]) -> tuple[datetime, datetime] | None:
    """Return daily source bounds around discovered resources.

    Args:
        resources: The discovered resources to bound.

    Returns:
        The inclusive start and exclusive end, or ``None`` when empty.
    """
    if not resources:
        return None
    first = min(resource.day for resource in resources)
    last = max(resource.day for resource in resources) + timedelta(days=1)
    return datetime.combine(first, time.min, UTC), datetime.combine(last, time.min, UTC)


def process_pair(
    source: Source,
    catalog: Catalog,
    client: httpx.Client,
    data_dir: Path,
    markets: list[Market],
    pair: str,
    request: Request,
    dataset: DatasetSpec,
) -> Result:
    """Discover, cache, and query one requested market.

    Args:
        source: The source strategy serving the pair.
        catalog: The metadata catalog for discovery and cache state.
        client: The HTTPX client used for source requests.
        data_dir: The root downloader data directory.
        markets: The latest complete source market snapshot.
        pair: The caller's original pair spelling.
        request: The validated shared request.
        dataset: The requested dataset schema.

    Returns:
        The pair's data and structured outcome report.
    """
    result = _result(pair, request, dataset)
    market = _market(pair, markets)
    if market is None:
        result.errors.append(
            Message("unknown_pair", f"Pair '{pair}' was not found uniquely.")
        )
        return result

    result.pair = market.symbol
    key = ResourceKey(
        source.code,
        request.product,
        request.dataset,
        market.symbol,
        dataset.base_interval,
    )
    try:
        resources = discover_resources(
            source, catalog, client, key, request.start, request.end
        )
    except Exception as error:
        result.errors.append(Message("discovery_failed", str(error)))
        return result

    result.available_range = _availability(resources)
    result.problems.extend(_missing_resources(resources, request.start, request.end))
    coverage = cache_resources(
        source, catalog, client, key, dataset, resources, data_dir
    )
    result.problems.extend(coverage.problems)
    if not coverage.paths:
        return result

    columns = dataset.resolve_columns(request.columns)
    try:
        result.data = query_parquet(
            catalog.connection,
            coverage.paths,
            dataset,
            request.start,
            request.end,
            columns,
        )
    except Exception as error:
        result.errors.append(Message("query_failed", str(error)))
        return result
    result.used_range = (request.start, request.end)
    return result
