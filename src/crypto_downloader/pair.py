"""Run the download workflow for one requested pair."""

from datetime import UTC, date, datetime, time, timedelta
from difflib import get_close_matches
from pathlib import Path

import duckdb
import httpx

from .catalog import Catalog
from .cache import cache_resources
from .datasets import DatasetSpec
from .discovery import discover_resources, requested_days
from .models import Market, Message, MissingCandlesError, Resource, ResourceKey, Result
from .query import empty_frame, missing_ranges, query_parquet
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
        gap_policy=request.gap_policy,
    )


def _suggestions(pair: str, markets: list[Market]) -> tuple[str, ...]:
    """Return likely native symbols for one unknown pair spelling.

    Args:
        pair: The caller's pair spelling.
        markets: The current source market snapshot.

    Returns:
        Up to three ranked native symbol suggestions.
    """
    normalized = normalize_pair(pair)
    by_normalized: dict[str, list[str]] = {}
    for market in markets:
        by_normalized.setdefault(market.normalized_symbol, []).append(market.symbol)
    close = get_close_matches(normalized, by_normalized, n=3, cutoff=0.6)
    symbols = [symbol for candidate in close for symbol in by_normalized[candidate]]
    return tuple(symbols[:3])


def _resolve_market(
    pair: str, markets: list[Market]
) -> tuple[Market | None, Message | None]:
    """Resolve one normalized pair or describe why it is unresolved.

    Args:
        pair: The caller's pair spelling.
        markets: The current source market snapshot.

    Returns:
        The unique market and no error, or no market and a structured error.
    """
    native = [market for market in markets if market.symbol == pair]
    if len(native) == 1:
        return native[0], None
    normalized = normalize_pair(pair)
    matches = [market for market in markets if market.normalized_symbol == normalized]
    if len(matches) == 1:
        return matches[0], None
    if matches:
        suggestions = tuple(sorted(market.symbol for market in matches))
        return None, Message(
            "ambiguous_pair",
            f"Pair '{pair}' matches more than one source symbol.",
            suggestions=suggestions,
        )
    return None, Message(
        "unknown_pair",
        f"Pair '{pair}' was not found.",
        suggestions=_suggestions(pair, markets),
    )


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


def _availability(
    resources: list[Resource],
    market: Market,
    active_statuses: frozenset[str],
    today: date,
) -> tuple[datetime, datetime] | None:
    """Return daily source bounds around discovered resources.

    Args:
        resources: The discovered resources to bound.
        market: The resolved source market.
        active_statuses: Source-native statuses considered active.
        today: The current UTC date used as the active exclusive end.

    Returns:
        The inclusive start and exclusive end, or ``None`` when empty.
    """
    if not resources:
        return None
    first = min(resource.day for resource in resources)
    if market.status in active_statuses:
        last = today
    else:
        last = max(resource.day for resource in resources) + timedelta(days=1)
    return datetime.combine(first, time.min, UTC), datetime.combine(last, time.min, UTC)


def _clean_range(
    result: Result,
    request: Request,
    availability: tuple[datetime, datetime],
) -> tuple[datetime, datetime] | None:
    """Trim a request to known pair availability and report each edge.

    Args:
        result: The pair result receiving boundary warnings.
        request: The original validated request.
        availability: The pair's inclusive start and exclusive end.

    Returns:
        The usable range, or ``None`` when no timestamps overlap.
    """
    start = max(request.start, availability[0])
    end = min(request.end, availability[1])
    if start != request.start:
        result.warnings.append(
            Message(
                "start_trimmed",
                f"Earliest available date for {result.pair} is {availability[0].date()}.",
            )
        )
    if end != request.end:
        result.warnings.append(
            Message(
                "end_trimmed",
                f"Latest available end for {result.pair} is "
                f"{availability[1].date()} (exclusive).",
            )
        )
    if start >= end:
        result.warnings.append(
            Message("no_overlap", "The request does not overlap available data.")
        )
        return None
    return start, end


def _resources_in_range(
    resources: list[Resource], start: datetime, end: datetime
) -> list[Resource]:
    """Select daily resources touched by one cleaned timestamp range.

    Args:
        resources: Every discovered resource for the pair.
        start: The inclusive cleaned start timestamp.
        end: The exclusive cleaned end timestamp.

    Returns:
        Resources whose days overlap the cleaned request.
    """
    first, last = requested_days(start, end)
    return [resource for resource in resources if first <= resource.day <= last]


def _query_result(
    result: Result,
    connection: duckdb.DuckDBPyConnection,
    paths: list[Path],
    dataset: DatasetSpec,
    request: Request,
    used_range: tuple[datetime, datetime],
    source_code: str,
) -> None:
    """Detect gaps and populate one result from cached Parquet files.

    Args:
        result: The pair result to populate in place.
        connection: The DuckDB connection used for queries.
        paths: The valid local daily Parquet files.
        dataset: The schema describing cached rows.
        request: The validated caller request.
        used_range: The cleaned inclusive-start, exclusive-end range.
        source_code: The source identifier used in diagnostics.
    """
    result.gaps = missing_ranges(
        connection,
        paths,
        dataset,
        used_range[0],
        used_range[1],
    )
    if result.gaps:
        missing = sum(gap.count for gap in result.gaps)
        result.problems.append(
            Message(
                "missing_candles",
                f"{source_code} omitted {missing} candle(s) across "
                f"{len(result.gaps)} internal gap(s).",
            )
        )
        if request.gap_policy == "raise":
            raise MissingCandlesError(result.pair, result.gaps)
    columns = dataset.resolve_columns(request.columns)
    result.data = query_parquet(
        connection,
        paths,
        dataset,
        used_range[0],
        used_range[1],
        columns,
        gap_policy=request.gap_policy,
        interval=request.interval,
    )


def process_pair(
    source: Source,
    catalog: Catalog,
    client: httpx.Client,
    data_dir: Path,
    markets: list[Market],
    pair: str,
    request: Request,
    dataset: DatasetSpec,
    earliest_date: date,
    today: date,
    *,
    refresh: bool = False,
    offline: bool = False,
    discovery_tail_days: int = 7,
    max_workers: int = 16,
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
        earliest_date: The first daily archive date allowed by configuration.
        today: The current UTC date and exclusive active-market boundary.
        refresh: Whether to repeat complete resource discovery.
        offline: Whether source access and downloads must be skipped.
        discovery_tail_days: The recent active-market days to rediscover.
        max_workers: The maximum concurrent daily archive ingestions.

    Returns:
        The pair's data and structured outcome report.
    """
    result = _result(pair, request, dataset)
    result.source = source.code
    market, pair_error = _resolve_market(pair, markets)
    if market is None:
        if pair_error is None:
            raise RuntimeError("pair resolution returned no market or error")
        result.errors.append(pair_error)
        return result

    result.pair = market.symbol
    key = ResourceKey(
        source.code,
        request.product,
        request.dataset,
        market.symbol,
        dataset.base_interval,
    )
    discovery_start = datetime.combine(earliest_date, time.min, UTC)
    discovery_end = datetime.combine(today, time.min, UTC)
    active = market.status in source.active_statuses
    try:
        resources = discover_resources(
            source,
            catalog,
            client,
            key,
            discovery_start,
            discovery_end,
            active=active,
            refresh=refresh,
            offline=offline,
            tail_days=discovery_tail_days,
        )
    except Exception as error:
        result.errors.append(Message("discovery_failed", str(error)))
        return result

    availability = _availability(resources, market, source.active_statuses, today)
    result.available_range = availability
    if availability is None:
        result.errors.append(
            Message(
                "no_availability", f"No daily files are available for {result.pair}."
            )
        )
        return result
    used_range = _clean_range(result, request, availability)
    if used_range is None:
        return result
    result.used_range = used_range
    requested_resources = _resources_in_range(resources, *used_range)
    result.problems.extend(_missing_resources(requested_resources, *used_range))
    coverage = cache_resources(
        source,
        catalog,
        client,
        key,
        dataset,
        requested_resources,
        data_dir,
        offline=offline,
        max_workers=max_workers,
    )
    result.problems.extend(coverage.problems)
    if not coverage.paths:
        return result

    try:
        _query_result(
            result,
            catalog.connection,
            coverage.paths,
            dataset,
            request,
            used_range,
            source.code,
        )
    except MissingCandlesError:
        raise
    except Exception as error:
        result.errors.append(Message("query_failed", str(error)))
        return result
    return result
