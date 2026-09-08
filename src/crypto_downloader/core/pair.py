"""Run the download workflow for one requested pair."""

from concurrent.futures import Executor
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
import logging
import math
from pathlib import Path
from time import perf_counter

import duckdb
import httpx

from crypto_downloader.core.catalog import Catalog
from crypto_downloader.core.cache import cache_resources, invalid_parquet_paths
from crypto_downloader.core.datasets import DatasetSpec
from crypto_downloader.core.reporting import Reporter, format_range, format_time
from crypto_downloader.core.discovery import discover_resources, requested_days
from crypto_downloader.core.models import (
    Market,
    Message,
    MissingCandlesError,
    Resource,
    ResourceKey,
    Result,
)
from crypto_downloader.core.matching import suggest_symbols
from crypto_downloader.core.query import (
    empty_frame,
    missing_ranges,
    query_parquet,
    suspect_gap_paths,
    empty_archive_days,
)
from crypto_downloader.core.request import Request, normalize_pair
from crypto_downloader.core.connector import Connector

from .planner import plan_archives, covered_days, catalog_archives

LOGGER = logging.getLogger(__name__)

type TimeRange = tuple[datetime, datetime]


@dataclass(frozen=True)
class Availability:
    """Separate full source availability from the configured usable range."""

    source_range: TimeRange
    usable_range: TimeRange
    configured_start: datetime | None


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


def _archive_symbol(market: Market, dataset: DatasetSpec, result: Result) -> str | None:
    """Resolve the source archive identifier declared by one dataset.

    Args:
        market: The public market selected by the caller.
        dataset: The dataset selecting a market identifier attribute.
        result: The result receiving a missing routing-context error.

    Returns:
        The alternate archive identifier, or ``None`` when the market symbol is used.
    """
    if dataset.archive_symbol_attribute == "symbol":
        return None
    value = market.pair
    if isinstance(value, str) and value:
        return value
    result.errors.append(
        Message(
            "archive_symbol_unavailable",
            f"{dataset.product}/{dataset.name} requires a source pair identifier.",
        )
    )
    return None


def _resource_key(
    source_code: str,
    pair: str,
    request: Request,
    dataset: DatasetSpec,
    markets: list[Market],
    result: Result,
    reporter: Reporter,
) -> tuple[Market, ResourceKey] | None:
    """Resolve one market and construct its dataset-specific resource key.

    Args:
        source_code: The identifier of the source serving the request.
        pair: The caller's original market spelling.
        request: The validated request selecting product and dataset.
        dataset: The resolved dataset declaration.
        markets: The current source market snapshot.
        result: The result receiving resolution errors and the native symbol.
        reporter: The optional activity reporter.

    Returns:
        The resolved market and source resource key, or ``None`` after an error.
    """
    market = _match_market(pair, markets, result, reporter)
    if market is None:
        return None
    result.pair = market.symbol
    archive_symbol = _archive_symbol(market, dataset, result)
    if result.errors:
        return None
    return market, ResourceKey(
        source_code,
        request.product,
        request.dataset,
        market.symbol,
        dataset.base_interval,
        archive_symbol=archive_symbol,
    )


def _suggestions(pair: str, markets: list[Market]) -> tuple[str, ...]:
    """Return likely native symbols for one unknown pair spelling.

    Args:
        pair: The caller's pair spelling.
        markets: The current source market snapshot.

    Returns:
        Up to three ranked native symbol suggestions.
    """
    return suggest_symbols(normalize_pair(pair), markets)


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
    available = covered_days(resources)
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
) -> TimeRange | None:
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


def _first_resource(
    source: Connector,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    today: date,
    result: Result,
    reporter: Reporter,
    *,
    refresh: bool,
    offline: bool,
) -> Resource | None:
    """Return and catalog the first source archive without a configured cutoff.

    Args:
        source: The source strategy used for discovery.
        catalog: The metadata catalog receiving the first resource.
        client: The HTTPX client used for source requests.
        key: The requested source dataset identity.
        today: The current UTC date and exclusive active boundary.
        result: The result receiving discovery failures.
        reporter: The optional Rich activity reporter.
        refresh: Whether to repeat source-boundary discovery.
        offline: Whether source access is forbidden.

    Returns:
        The first known resource, or ``None`` when no source file is available.
    """
    source_bounds = catalog.source_bounds(key)
    cached_day = source_bounds[0] if source_bounds is not None else None
    if cached_day is not None and not refresh:
        resources = catalog.resources(key, cached_day, cached_day)
        if resources:
            LOGGER.debug("Reused source boundary: key=%s day=%s", key, cached_day)
            return resources[0]
    if offline:
        fallback = catalog.resource_bounds(key)
        if fallback is None:
            return None
        day = fallback[0]
        resources = catalog.resources(key, day, day)
        return resources[0] if resources else None
    try:
        with reporter.status(f"Finding the first {key.symbol} daily file"):
            first = source.first_resource(
                client,
                key,
                None,
                today - timedelta(days=1),
            )
    except Exception as error:
        LOGGER.exception("Earliest resource discovery failed: key=%s", key)
        result.errors.append(Message("discovery_failed", str(error)))
        return None
    if first is not None:
        catalog.save_discovery(key, first.day, first.day, [first])
        catalog.save_source_bounds(key, first.day, None)
    return first


def _availability_range(
    source: Connector,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    earliest_date: date | None,
    today: date,
    result: Result,
    reporter: Reporter,
    *,
    active: bool,
    refresh: bool,
    offline: bool,
    tail_days: int,
) -> Availability | None:
    """Resolve pair bounds without listing active-market history.

    Args:
        source: The source strategy used for discovery.
        catalog: The metadata catalog containing known resources.
        client: The HTTPX client used for source requests.
        key: The requested source dataset identity.
        earliest_date: The optional first configured archive date.
        today: The current UTC day and exclusive active boundary.
        result: The result receiving discovery failures.
        reporter: The optional Rich activity reporter.
        active: Whether the market can receive new daily files.
        refresh: Whether source metadata must be refreshed.
        offline: Whether source access is forbidden.
        tail_days: The recent active-market rediscovery window.

    Returns:
        The source and configured timestamp bounds, or ``None`` when no files exist.
    """
    first = _first_resource(
        source,
        catalog,
        client,
        key,
        today,
        result,
        reporter,
        refresh=refresh,
        offline=offline,
    )
    if result.errors or first is None:
        return None
    if not active:
        broad_start = datetime.combine(first.day, time.min, UTC)
        broad_end = datetime.combine(today, time.min, UTC)
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
        bounds = catalog.resource_bounds(key)
        if bounds is not None:
            catalog.save_source_bounds(key, first.day, bounds[1])
    source_range = _availability(catalog.resource_bounds(key), active, today)
    if source_range is None:
        return None
    configured_start = (
        datetime.combine(earliest_date, time.min, UTC)
        if earliest_date is not None
        else None
    )
    usable_start = max(source_range[0], configured_start or source_range[0])
    return Availability(
        source_range=source_range,
        usable_range=(usable_start, source_range[1]),
        configured_start=configured_start,
    )


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
    availability: Availability,
    reporter: Reporter,
) -> TimeRange | None:
    """Trim a request to known pair availability and report each edge.

    Args:
        result: The pair result receiving boundary warnings.
        request: The original validated request.
        availability: The pair's full source and configured usable ranges.
        reporter: The optional Rich activity reporter.

    Returns:
        The usable range, or ``None`` when no timestamps overlap.
    """
    source_start, _source_end = availability.source_range
    usable_start, usable_end = availability.usable_range
    start = max(request.start, usable_start)
    end = min(request.end, usable_end)
    configured_start = availability.configured_start
    limited_by_configuration = (
        configured_start is not None
        and source_start < configured_start
        and request.start < configured_start
    )
    if start != request.start:
        if limited_by_configuration:
            assert configured_start is not None
            message = Message(
                "configured_start",
                f"Configured history for {result.pair} begins on "
                f"{configured_start.date()}; Binance archive data begins "
                f"on {source_start.date()}, so earlier files were not used.",
            )
            reporter.warning(
                f"{result.pair}: configured history begins "
                f"{format_time(configured_start)}; source archive "
                f"begins {format_time(source_start)}"
            )
        else:
            message = Message(
                "start_trimmed",
                f"Earliest source archive date for {result.pair} is "
                f"{source_start.date()}.",
            )
            reporter.warning(
                f"{result.pair}: start trimmed to {format_time(source_start)}"
            )
        result.warnings.append(message)
    if end != request.end:
        result.warnings.append(
            Message(
                "end_trimmed",
                f"Latest available end for {result.pair} is "
                f"{usable_end.date()} (exclusive).",
            )
        )
        reporter.warning(f"{result.pair}: end trimmed to {format_time(usable_end)}")
    if start >= end:
        overlap_message = (
            "The request does not overlap the configured history range."
            if limited_by_configuration
            else "The request does not overlap source archive availability."
        )
        result.warnings.append(Message("no_overlap", overlap_message))
        reporter.warning(f"{result.pair}: {overlap_message}")
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
    return [
        resource
        for resource in resources
        if resource.day <= last and resource.last_day >= first
    ]


def _with_contract_size(
    resources: list[Resource],
    market: Market,
    dataset: DatasetSpec,
    result: Result,
) -> list[Resource] | None:
    """Attach cataloged COIN-M contract size to resources that need it.

    Args:
        resources: The discovered daily resources required by the request.
        market: The resolved current or archived market metadata.
        dataset: The requested dataset capability declaration.
        result: The result receiving an unavailable-contract diagnostic.

    Returns:
        Original resources when no context is needed, enriched resources when a
        valid contract size exists, or ``None`` after recording an error.
    """
    if not dataset.requires_contract_size:
        return resources
    size = market.contract_size
    if (
        isinstance(size, bool)
        or not isinstance(size, (int, float))
        or not math.isfinite(size)
        or size <= 0
    ):
        result.errors.append(
            Message(
                "contract_size_unavailable",
                f"Contract size is unavailable for {market.symbol}; COIN-M "
                "quote notional cannot be derived.",
            )
        )
        LOGGER.error(
            "COIN-M contract size unavailable: symbol=%s dataset=%s value=%s",
            market.symbol,
            dataset.name,
            size,
        )
        return None
    return [replace(resource, contract_size=float(size)) for resource in resources]


def _usable_range(
    result: Result,
    request: Request,
    availability: Availability | None,
    reporter: Reporter,
) -> TimeRange | None:
    """Record one availability result and return its cleaned usable range.

    Args:
        result: The pair result receiving availability diagnostics.
        request: The original caller request to clean.
        availability: The source and configured ranges, if any source file exists.
        reporter: The optional Rich activity reporter.

    Returns:
        The cleaned request range, or ``None`` when no usable data exists.
    """
    if availability is None:
        result.errors.append(
            Message(
                "no_availability", f"No daily files are available for {result.pair}."
            )
        )
        return None
    result.available_range = availability.source_range
    reporter.info(
        f"{result.pair}: source archive availability "
        f"{format_range(availability.source_range)}"
    )
    if availability.configured_start is not None:
        reporter.info(
            f"{result.pair}: configured history begins "
            f"{format_time(availability.configured_start)}"
        )
    return _clean_range(result, request, availability, reporter)


def _query_result(
    result: Result,
    connection: duckdb.DuckDBPyConnection,
    paths: list[Path],
    gap_paths: list[Path],
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
        gap_paths: The partitions whose metadata cannot prove continuity.
        dataset: The schema describing cached rows.
        request: The validated caller request.
        used_range: The cleaned inclusive-start, exclusive-end range.
        source_code: The source identifier used in diagnostics.
        reporter: The optional Rich activity reporter.
    """
    gap_policy = request.gap_policy
    query_gap_policy = gap_policy
    if dataset.supports_gap_policy:
        if gap_policy is None:
            raise ValueError("candle dataset requires a resolved gap policy")
        result.gaps = missing_ranges(
            connection,
            gap_paths,
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
        else:
            query_gap_policy = "keep"
    columns = dataset.resolve_columns(request.columns)
    result.data = query_parquet(
        connection,
        paths,
        dataset,
        used_range[0],
        used_range[1],
        columns,
        gap_policy=query_gap_policy,
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
    source: Connector,
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
    dataset: DatasetSpec | None = None,
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
        if dataset is not None:
            return plan_archives(
                source,
                catalog,
                client,
                key,
                start,
                end,
                dataset=dataset,
                active=active,
                refresh=refresh,
                offline=offline,
                tail_days=tail_days,
                reporter=reporter,
            )
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
            reporter=reporter,
        )
    except Exception as error:
        LOGGER.exception("Resource discovery failed: key=%s", key)
        result.errors.append(Message("discovery_failed", str(error)))
        return None


def _populate_query(
    result: Result,
    catalog: Catalog,
    paths: list[Path],
    gap_paths: list[Path],
    dataset: DatasetSpec,
    request: Request,
    used_range: tuple[datetime, datetime],
    source_code: str,
    reporter: Reporter,
) -> Exception | None:
    """Populate one result while isolating non-policy query failures.

    Args:
        result: The pair result to populate.
        catalog: The catalog providing the DuckDB connection.
        paths: The usable daily Parquet files.
        gap_paths: The partitions that require row-level gap inspection.
        dataset: The requested dataset schema.
        request: The validated caller request.
        used_range: The cleaned query range.
        source_code: The source identifier used in diagnostics.
        reporter: The optional Rich activity reporter.

    Returns:
        ``None`` when querying completed, otherwise the isolated failure.
    """
    try:
        _query_result(
            result,
            catalog.connection,
            paths,
            gap_paths,
            dataset,
            request,
            used_range,
            source_code,
            reporter,
        )
        return None
    except MissingCandlesError:
        raise
    except Exception as error:
        LOGGER.warning("Parquet query failed: pair=%s error=%s", result.pair, error)
        return error


def _query_failure(result: Result, error: Exception) -> bool:
    """Record an unrecoverable Parquet query failure.

    Args:
        result: The pair result receiving the structured error.
        error: The query exception shown to the caller.

    Returns:
        False for direct use as a failed workflow outcome.
    """
    LOGGER.exception("Parquet query failed: pair=%s", result.pair, exc_info=error)
    result.errors.append(Message("query_failed", str(error)))
    return False


def _populate_cached_query(
    result: Result,
    catalog: Catalog,
    key: ResourceKey,
    paths: list[Path],
    dataset: DatasetSpec,
    request: Request,
    used_range: tuple[datetime, datetime],
    source_code: str,
    reporter: Reporter,
) -> Exception | None:
    """Query cached paths after selecting only possible gap partitions.

    Args:
        result: The pair result to populate.
        catalog: The catalog containing validated partition metadata.
        key: The requested dataset identity.
        paths: The local Parquet files to query.
        dataset: The requested dataset schema.
        request: The validated caller request.
        used_range: The cleaned query range.
        source_code: The source identifier used in diagnostics.
        reporter: The optional Rich activity reporter.

    Returns:
        ``None`` after a successful query, otherwise the isolated exception.
    """
    first_day, last_day = requested_days(*used_range)
    current_resources = catalog_archives(catalog, key, first_day, last_day)
    gap_paths = suspect_gap_paths(current_resources, paths, dataset)
    LOGGER.debug(
        "Gap scan planned: key=%s paths=%d suspect=%d",
        key,
        len(paths),
        len(gap_paths),
    )
    error = _populate_query(
        result,
        catalog,
        paths,
        gap_paths,
        dataset,
        request,
        used_range,
        source_code,
        reporter,
    )
    monthly = [
        resource
        for resource in current_resources
        if resource.cadence == "monthly" and resource.parquet_path in gap_paths
    ]
    if error is None and monthly:
        try:
            absent = empty_archive_days(
                catalog.connection, monthly, dataset, first_day, last_day
            )
        except duckdb.Error as failure:
            return failure
        result.problems.extend(
            Message(
                "resource_unavailable",
                "Source archive contains no candles for this date.",
                day,
            )
            for day in absent
        )
    return error


def _query_with_recovery(
    result: Result,
    source: Connector,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    resources: list[Resource],
    paths: list[Path],
    data_dir: Path,
    dataset: DatasetSpec,
    request: Request,
    used_range: tuple[datetime, datetime],
    reporter: Reporter,
    *,
    offline: bool,
    max_workers: int,
    ingestion_executor: Executor | None,
) -> bool:
    """Query cached partitions and rebuild unreadable files once.

    Args:
        result: The pair result to populate.
        source: The source strategy used for recovery downloads.
        catalog: The metadata catalog containing resource state.
        client: The HTTPX client used for recovery downloads.
        key: The requested source dataset identity.
        resources: The requested catalog resources.
        paths: The initially valid local partitions.
        data_dir: The root downloader data directory.
        dataset: The requested dataset schema.
        request: The validated caller request.
        used_range: The cleaned query range.
        reporter: The optional activity reporter.
        offline: Whether recovery downloads are forbidden.
        max_workers: The maximum concurrent recovery downloads.
        ingestion_executor: The request-wide cache executor, if one exists.

    Returns:
        True after a successful query, otherwise False with a result error.
    """
    error = _populate_cached_query(
        result,
        catalog,
        key,
        paths,
        dataset,
        request,
        used_range,
        source.code,
        reporter,
    )
    if error is None:
        return True
    invalid = invalid_parquet_paths(paths)
    if not invalid or offline:
        return _query_failure(result, error)
    for path in invalid:
        try:
            path.unlink()
        except OSError as remove_error:
            LOGGER.warning(
                "Unreadable Parquet could not be removed: path=%s error=%s",
                path,
                remove_error,
            )
            return _query_failure(result, error)
    reporter.warning(
        f"{result.pair}: rebuilding {len(invalid):,} unreadable cached file(s)"
    )
    # A failed monthly download may have been replaced by daily fallbacks.
    # Rebuild the resources that were actually queried, not the original plan.
    queried = [
        resource
        for resource in catalog_archives(catalog, key, *requested_days(*used_range))
        if resource.parquet_path in paths
    ]
    if queried:
        sizes = {r.archive_symbol: r.contract_size for r in resources}
        resources = [
            replace(r, contract_size=sizes.get(r.archive_symbol)) for r in queried
        ]
    recovered = cache_resources(
        source,
        catalog,
        client,
        key,
        dataset,
        resources,
        data_dir,
        max_workers=max_workers,
        reporter=reporter,
        executor=ingestion_executor,
        requested_range=requested_days(*used_range),
    )
    result.warnings.extend(recovered.warnings)
    result.problems.extend(recovered.problems)
    if not recovered.paths:
        return _query_failure(result, error)
    retry_error = _populate_cached_query(
        result,
        catalog,
        key,
        recovered.paths,
        dataset,
        request,
        used_range,
        source.code,
        reporter,
    )
    return True if retry_error is None else _query_failure(result, retry_error)


def process_pair(
    source: Connector,
    catalog: Catalog,
    client: httpx.Client,
    data_dir: Path,
    markets: list[Market],
    pair: str,
    request: Request,
    dataset: DatasetSpec,
    earliest_date: date | None,
    today: date,
    *,
    refresh: bool = False,
    offline: bool = False,
    discovery_tail_days: int = 7,
    max_workers: int = 16,
    reporter: Reporter | None = None,
    ingestion_executor: Executor | None = None,
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
        earliest_date: The optional first daily archive date allowed by configuration.
        today: The current UTC date and exclusive active-market boundary.
        refresh: Whether to repeat complete resource discovery.
        offline: Whether source access and downloads must be skipped.
        discovery_tail_days: The recent active-market days to rediscover.
        max_workers: The maximum concurrent daily archive ingestions.
        reporter: The optional Rich activity reporter.
        ingestion_executor: An optional request-wide cache executor.

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
    resolved = _resource_key(
        source.code,
        pair,
        request,
        dataset,
        markets,
        result,
        display,
    )
    if resolved is None:
        return _finish(result, display, started)
    market, key = resolved
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
    used_range = _usable_range(result, request, availability, display)
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
        dataset=dataset,
    )
    if resources is None:
        return _finish(result, display, started)
    noun = "file" if len(resources) == 1 else "files"
    display.info(f"{market.symbol}: found {len(resources):,} {noun}")
    requested_resources = _resources_in_range(resources, *used_range)
    contextual_resources = _with_contract_size(
        requested_resources,
        market,
        dataset,
        result,
    )
    if contextual_resources is None:
        return _finish(result, display, started)
    requested_resources = contextual_resources
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
        refresh=refresh,
        max_workers=max_workers,
        reporter=display,
        executor=ingestion_executor,
        requested_range=requested_days(*used_range),
    )
    result.warnings.extend(coverage.warnings)
    result.problems.extend(coverage.problems)
    if not coverage.paths:
        return _finish(result, display, started)

    if not _query_with_recovery(
        result,
        source,
        catalog,
        client,
        key,
        requested_resources,
        coverage.paths,
        data_dir,
        dataset,
        request,
        used_range,
        display,
        offline=offline,
        max_workers=max_workers,
        ingestion_executor=ingestion_executor,
    ):
        return _finish(result, display, started)
    return _finish(result, display, started)
