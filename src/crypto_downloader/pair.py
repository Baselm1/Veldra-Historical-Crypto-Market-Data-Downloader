"""Run the download workflow for one requested pair."""

from datetime import UTC, date, datetime, time, timedelta
from difflib import get_close_matches
import logging
from pathlib import Path
from time import perf_counter

import duckdb
import httpx

from .catalog import Catalog
from .cache import cache_resources
from .datasets import DatasetSpec
from .display import Reporter, format_range, format_time
from .discovery import discover_resources, requested_days
from .models import Market, Message, MissingCandlesError, Resource, ResourceKey, Result
from .query import empty_frame, missing_ranges, query_parquet
from .request import Request, normalize_pair
from .source import Source

LOGGER = logging.getLogger(__name__)


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
        gap_policy=request.gap_policy or "keep",
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
    bounds: tuple[date, date] | None,
    active: bool,
    today: date,
) -> tuple[datetime, datetime] | None:
    """Return timestamp bounds around known daily resource days.

    Args:
        bounds: The inclusive first and last known resource days.
        active: Whether today's date is the dynamic exclusive end.
        today: The current UTC date used as the active exclusive end.

    Returns:
        The inclusive start and exclusive end, or ``None`` when empty.
    """
    if bounds is None:
        return None
    first, last_resource = bounds
    last = today if active else last_resource + timedelta(days=1)
    return datetime.combine(first, time.min, UTC), datetime.combine(last, time.min, UTC)


def _availability_range(
    source: Source,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    earliest_date: date,
    today: date,
    result: Result,
    reporter: Reporter,
    *,
    active: bool,
    refresh: bool,
    offline: bool,
    tail_days: int,
) -> tuple[datetime, datetime] | None:
    """Resolve pair bounds without listing active-market history.

    Args:
        source: The source strategy used for discovery.
        catalog: The metadata catalog containing known resources.
        client: The HTTPX client used for source requests.
        key: The requested source dataset identity.
        earliest_date: The first configured archive date.
        today: The current UTC day and exclusive active boundary.
        result: The result receiving discovery failures.
        reporter: The optional Rich activity reporter.
        active: Whether the market can receive new daily files.
        refresh: Whether source metadata must be refreshed.
        offline: Whether source access is forbidden.
        tail_days: The recent active-market rediscovery window.

    Returns:
        The known timestamp bounds, or ``None`` when no files exist.
    """
    broad_start = datetime.combine(earliest_date, time.min, UTC)
    broad_end = datetime.combine(today, time.min, UTC)
    if not active:
        resources = _discover(
            source,
            catalog,
            client,
            key,
            broad_start,
            broad_end,
            result,
            reporter,
            active=False,
            refresh=refresh,
            offline=offline,
            tail_days=tail_days,
        )
        if resources is None:
            return None
        return _availability(catalog.resource_bounds(key), False, today)

    bounds = catalog.resource_bounds(key)
    if bounds is None and not offline:
        try:
            with reporter.status(f"Finding the first {key.symbol} daily file"):
                first = source.first_resource(
                    client,
                    key,
                    earliest_date,
                    today - timedelta(days=1),
                )
            if first is not None:
                catalog.save_discovery(key, first.day, first.day, [first])
                bounds = (first.day, first.day)
        except Exception as error:
            LOGGER.exception("Earliest resource discovery failed: key=%s", key)
            result.errors.append(Message("discovery_failed", str(error)))
            return None
    return _availability(bounds, True, today)


def _recent_active_range(
    active: bool,
    used_range: tuple[datetime, datetime],
    today: date,
    tail_days: int,
) -> bool:
    """Return whether a request overlaps mutable active-market days.

    Args:
        active: Whether the market can receive new daily resources.
        used_range: The cleaned timestamp range.
        today: The current UTC day.
        tail_days: The number of recent days that may change.

    Returns:
        True only when recent active resources should be relisted.
    """
    if not active:
        return False
    _first, last = requested_days(*used_range)
    return last >= today - timedelta(days=tail_days)


def _clean_range(
    result: Result,
    request: Request,
    availability: tuple[datetime, datetime],
    reporter: Reporter,
) -> tuple[datetime, datetime] | None:
    """Trim a request to known pair availability and report each edge.

    Args:
        result: The pair result receiving boundary warnings.
        request: The original validated request.
        availability: The pair's inclusive start and exclusive end.
        reporter: The optional Rich activity reporter.

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
        reporter.warning(
            f"{result.pair}: start trimmed to {format_time(availability[0])}"
        )
    if end != request.end:
        result.warnings.append(
            Message(
                "end_trimmed",
                f"Latest available end for {result.pair} is "
                f"{availability[1].date()} (exclusive).",
            )
        )
        reporter.warning(
            f"{result.pair}: end trimmed to {format_time(availability[1])}"
        )
    if start >= end:
        result.warnings.append(
            Message("no_overlap", "The request does not overlap available data.")
        )
        reporter.warning(
            f"{result.pair}: requested dates do not overlap known availability"
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
    reporter: Reporter,
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
        reporter: The optional Rich activity reporter.
    """
    gap_policy = request.gap_policy
    if dataset.supports_gap_policy:
        if gap_policy is None:
            raise ValueError("candle dataset requires a resolved gap policy")
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
            reporter.warning(
                f"{result.pair}: {missing:,} missing candle(s) across "
                f"{len(result.gaps):,} internal gap(s); policy {gap_policy}"
            )
            LOGGER.warning(
                "Missing source candles: pair=%s count=%d gaps=%d policy=%s",
                result.pair,
                missing,
                len(result.gaps),
                gap_policy,
            )
            if gap_policy == "raise":
                raise MissingCandlesError(result.pair, result.gaps)
    columns = dataset.resolve_columns(request.columns)
    result.data = query_parquet(
        connection,
        paths,
        dataset,
        used_range[0],
        used_range[1],
        columns,
        gap_policy=gap_policy,
        interval=request.interval,
    )


def _finish(result: Result, reporter: Reporter, started: float) -> Result:
    """Report and return one completed or isolated pair outcome.

    Args:
        result: The pair result being completed.
        reporter: The optional Rich activity reporter.
        started: The monotonic start time for the pair workflow.

    Returns:
        The unchanged pair result.
    """
    if result.errors:
        reporter.error(f"{result.pair}: {result.errors[-1].message}")
    elif result.complete:
        reporter.success(f"{result.pair}: returned {len(result.data):,} rows")
    else:
        reporter.warning(
            f"{result.pair}: returned {len(result.data):,} rows with "
            f"{len(result.problems):,} problem(s) and "
            f"{len(result.warnings):,} warning(s)"
        )
    LOGGER.info(
        "Pair complete: pair=%s rows=%d complete=%s warnings=%d problems=%d "
        "errors=%d elapsed=%.3fs",
        result.pair,
        len(result.data),
        result.complete,
        len(result.warnings),
        len(result.problems),
        len(result.errors),
        perf_counter() - started,
    )
    return result


def _match_market(
    pair: str,
    markets: list[Market],
    result: Result,
    reporter: Reporter,
) -> Market | None:
    """Resolve and report one requested source market.

    Args:
        pair: The caller's pair spelling.
        markets: The current source market snapshot.
        result: The pair result receiving resolution errors.
        reporter: The optional Rich activity reporter.

    Returns:
        The unique market, or ``None`` after a structured resolution error.
    """
    market, error = _resolve_market(pair, markets)
    if market is None:
        if error is None:
            raise RuntimeError("pair resolution returned no market or error")
        result.errors.append(error)
        LOGGER.warning(
            "Pair resolution failed: pair=%s code=%s suggestions=%s",
            pair,
            error.code,
            error.suggestions,
        )
        return None
    assets = (
        f"{market.base_asset}/{market.quote_asset}"
        if market.base_asset and market.quote_asset
        else "assets unavailable"
    )
    reporter.info(
        f"{market.symbol}: matched {assets}; status {market.status or 'unknown'}"
    )
    LOGGER.debug(
        "Pair resolved: requested=%s symbol=%s base=%s quote=%s status=%s",
        pair,
        market.symbol,
        market.base_asset,
        market.quote_asset,
        market.status,
    )
    return market


def _discover(
    source: Source,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    start: datetime,
    end: datetime,
    result: Result,
    reporter: Reporter,
    *,
    active: bool,
    refresh: bool,
    offline: bool,
    tail_days: int,
) -> list[Resource] | None:
    """Discover one pair's resources while isolating source failures.

    Args:
        source: The source strategy used for discovery.
        catalog: The metadata catalog receiving discovered resources.
        client: The HTTPX client used for source requests.
        key: The requested source dataset identity.
        start: The inclusive discovery start.
        end: The exclusive discovery end.
        result: The result receiving a discovery error.
        reporter: The optional Rich activity reporter.
        active: Whether recent source files may still change.
        refresh: Whether to repeat complete discovery.
        offline: Whether source access must be skipped.
        tail_days: The recent active-market days to revisit.

    Returns:
        The known resources, or ``None`` after an isolated failure.
    """
    try:
        with reporter.status(f"Discovering {key.symbol} daily files"):
            return discover_resources(
                source,
                catalog,
                client,
                key,
                start,
                end,
                active=active,
                refresh=refresh,
                offline=offline,
                tail_days=tail_days,
            )
    except Exception as error:
        LOGGER.exception("Resource discovery failed: key=%s", key)
        result.errors.append(Message("discovery_failed", str(error)))
        return None


def _populate_query(
    result: Result,
    catalog: Catalog,
    paths: list[Path],
    dataset: DatasetSpec,
    request: Request,
    used_range: tuple[datetime, datetime],
    source_code: str,
    reporter: Reporter,
) -> bool:
    """Populate one result while isolating non-policy query failures.

    Args:
        result: The pair result to populate.
        catalog: The catalog providing the DuckDB connection.
        paths: The usable daily Parquet files.
        dataset: The requested dataset schema.
        request: The validated caller request.
        used_range: The cleaned query range.
        source_code: The source identifier used in diagnostics.
        reporter: The optional Rich activity reporter.

    Returns:
        True when querying completed, otherwise False.
    """
    try:
        _query_result(
            result,
            catalog.connection,
            paths,
            dataset,
            request,
            used_range,
            source_code,
            reporter,
        )
        return True
    except MissingCandlesError:
        raise
    except Exception as error:
        LOGGER.exception("Parquet query failed: pair=%s", result.pair)
        result.errors.append(Message("query_failed", str(error)))
        return False


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
    reporter: Reporter | None = None,
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
        reporter: The optional Rich activity reporter.

    Returns:
        The pair's data and structured outcome report.
    """
    started = perf_counter()
    display = reporter or Reporter(False)
    result = _result(pair, request, dataset)
    result.source = source.code
    LOGGER.debug(
        "Pair processing started: pair=%s product=%s dataset=%s",
        pair,
        request.product,
        request.dataset,
    )
    market = _match_market(pair, markets, result, display)
    if market is None:
        return _finish(result, display, started)

    result.pair = market.symbol
    key = ResourceKey(
        source.code,
        request.product,
        request.dataset,
        market.symbol,
        dataset.storage_interval,
    )
    active = market.status in source.active_statuses
    availability = _availability_range(
        source,
        catalog,
        client,
        key,
        earliest_date,
        today,
        result,
        display,
        active=active,
        refresh=refresh,
        offline=offline,
        tail_days=discovery_tail_days,
    )
    if result.errors:
        return _finish(result, display, started)
    result.available_range = availability
    if availability is None:
        result.errors.append(
            Message(
                "no_availability", f"No daily files are available for {result.pair}."
            )
        )
        return _finish(result, display, started)
    display.info(f"{market.symbol}: known availability {format_range(availability)}")
    used_range = _clean_range(result, request, availability, display)
    if used_range is None:
        return _finish(result, display, started)
    result.used_range = used_range
    resources = _discover(
        source,
        catalog,
        client,
        key,
        used_range[0],
        used_range[1],
        result,
        display,
        active=_recent_active_range(active, used_range, today, discovery_tail_days),
        refresh=refresh,
        offline=offline,
        tail_days=discovery_tail_days,
    )
    if resources is None:
        return _finish(result, display, started)
    noun = "file" if len(resources) == 1 else "files"
    display.info(f"{market.symbol}: found {len(resources):,} daily {noun}")
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
        reporter=display,
    )
    result.problems.extend(coverage.problems)
    if not coverage.paths:
        return _finish(result, display, started)

    if not _populate_query(
        result,
        catalog,
        coverage.paths,
        dataset,
        request,
        used_range,
        source.code,
        display,
    ):
        return _finish(result, display, started)
    return _finish(result, display, started)
