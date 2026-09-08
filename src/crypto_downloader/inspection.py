"""Inspect Binance markets and dataset coverage without exposing the catalog."""

from dataclasses import dataclass, replace
from datetime import date, timedelta
from difflib import get_close_matches
import logging
from pathlib import Path

import httpx

from .catalog import Catalog, catalog_lock, open_catalog
from .datasets import DatasetSpec, get_dataset
from .discovery import _merge_ranges, discover_resources, requested_days
from .display import Reporter
from .downloader import Downloader, _load_markets, _source_limit, utc_today
from .models import Availability, Market, Resource, ResourceKey
from .request import Request, normalize_pair, parse_identifier, parse_pairs

LOGGER = logging.getLogger(__name__)


def _boolean(value: object, name: str) -> bool:
    """Validate one strict Boolean option.

    Args:
        value: The proposed Boolean value.
        name: The option name used in failures.

    Returns:
        The validated Boolean value.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a Boolean")
    return value


def _product(downloader: Downloader, value: object) -> str:
    """Validate one product supported by the configured source.

    Args:
        downloader: The configured internal downloader.
        value: The proposed product identifier.

    Returns:
        The supported product identifier.
    """
    product = parse_identifier(value, name="product")
    if product not in downloader.source.products:
        raise ValueError(f"unsupported product for {downloader.source.code}: {product}")
    return product


def _filter(value: object, name: str) -> str | None:
    """Validate and normalize one optional exact market filter.

    Args:
        value: The optional source filter.
        name: The filter name used in failures.

    Returns:
        Uppercase filter text, or ``None`` when omitted.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _limit(value: object) -> int:
    """Validate a positive search result limit.

    Args:
        value: The proposed maximum result count.

    Returns:
        The validated positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("limit must be an integer")
    if value < 1:
        raise ValueError("limit must be positive")
    return value


def _query(value: object) -> str:
    """Validate and normalize one market search query.

    Args:
        value: The proposed native or human-formatted market text.

    Returns:
        Uppercase ASCII letters and digits used for matching.
    """
    if not isinstance(value, str):
        raise TypeError("query must be a string")
    normalized = normalize_pair(value)
    if not normalized:
        raise ValueError("query must contain ASCII letters or digits")
    return normalized


def _snapshot(
    downloader: Downloader,
    product: str,
    *,
    refresh: bool,
    offline: bool,
    progress: bool,
) -> list[Market]:
    """Load one fresh or cached market snapshot with public context.

    Args:
        downloader: The configured internal downloader.
        product: The validated Binance product.
        refresh: Whether to replace cached metadata now.
        offline: Whether source access is forbidden.
        progress: Whether to show Rich activity.

    Returns:
        Markets ordered by native symbol with source and product identity.
    """
    catalog_path = downloader.data_dir / "catalog.duckdb"
    if offline and not catalog_path.is_file():
        raise RuntimeError("offline mode requires cached market metadata")
    source_limit = _source_limit(downloader.source, downloader.max_workers)
    limits = httpx.Limits(
        max_connections=source_limit,
        max_keepalive_connections=source_limit,
    )
    with catalog_lock(catalog_path):
        with httpx.Client(
            transport=downloader.transport,
            follow_redirects=True,
            limits=limits,
        ) as client:
            with open_catalog(catalog_path) as catalog:
                markets = _load_markets(
                    downloader.source,
                    catalog,
                    client,
                    product,
                    Reporter(progress),
                    refresh=refresh,
                    offline=offline,
                    refresh_hours=downloader.market_refresh_hours,
                )
    return sorted(
        (
            replace(
                market,
                source=downloader.source.code,
                product=product,
            )
            for market in markets
        ),
        key=lambda market: market.symbol,
    )


def get_markets(
    downloader: Downloader,
    *,
    product: object = "spot",
    status: object = None,
    quote_asset: object = None,
    refresh: object = False,
    offline: object = False,
    progress: bool = True,
) -> list[Market]:
    """Return a filtered current or cached market snapshot.

    Args:
        downloader: The configured internal downloader.
        product: The Binance product to inspect.
        status: An optional native status filter.
        quote_asset: An optional quote asset filter.
        refresh: Whether to replace cached metadata now.
        offline: Whether source access is forbidden.
        progress: Whether to show Rich activity.

    Returns:
        Matching immutable markets ordered by native symbol.
    """
    selected_product = _product(downloader, product)
    selected_status = _filter(status, "status")
    selected_quote = _filter(quote_asset, "quote_asset")
    selected_refresh = _boolean(refresh, "refresh")
    selected_offline = _boolean(offline, "offline")
    if selected_refresh and selected_offline:
        raise ValueError("refresh and offline cannot both be enabled")
    markets = _snapshot(
        downloader,
        selected_product,
        refresh=selected_refresh,
        offline=selected_offline,
        progress=progress,
    )
    return [
        market
        for market in markets
        if (selected_status is None or market.status == selected_status)
        and (selected_quote is None or market.quote_asset == selected_quote)
    ]


def _fuzzy_matches(query: str, markets: list[Market]) -> list[Market]:
    """Return markets grouped by descending normalized similarity.

    Args:
        query: The normalized query text.
        markets: Candidate markets not already matched.

    Returns:
        Similar markets grouped in deterministic product and symbol order.
    """
    by_normalized: dict[str, list[Market]] = {}
    for market in markets:
        by_normalized.setdefault(market.normalized_symbol, []).append(market)
    if not by_normalized:
        return []
    close = get_close_matches(
        query,
        by_normalized,
        n=len(by_normalized),
        cutoff=0.6,
    )
    return [market for candidate in close for market in by_normalized[candidate]]


def _ranked_matches(query: str, markets: list[Market], limit: int) -> list[Market]:
    """Rank exact, prefix, and fuzzy matches without selecting one.

    Args:
        query: The normalized query text.
        markets: The filtered candidate markets.
        limit: The maximum number of matches.

    Returns:
        Ranked markets with no duplicates.
    """
    exact = [market for market in markets if market.normalized_symbol == query]
    prefix = [
        market
        for market in markets
        if market.normalized_symbol != query
        and market.normalized_symbol.startswith(query)
    ]
    remaining = [
        market
        for market in markets
        if market.normalized_symbol != query
        and not market.normalized_symbol.startswith(query)
    ]
    fuzzy = _fuzzy_matches(query, remaining)
    return [*exact, *prefix, *fuzzy][:limit]


def _search_products(downloader: Downloader, product: object) -> tuple[str, ...]:
    """Return one selected product or every source product.

    Args:
        downloader: The configured internal downloader.
        product: The optional product restriction.

    Returns:
        Product identifiers in stable source order.
    """
    if product is None:
        return downloader.source.products
    return (_product(downloader, product),)


def _matches_filters(
    market: Market,
    status: str | None,
    quote_asset: str | None,
) -> bool:
    """Return whether one market passes optional exact filters.

    Args:
        market: The market to inspect.
        status: The optional native status filter.
        quote_asset: The optional quote asset filter.

    Returns:
        True when every supplied filter matches.
    """
    status_matches = status is None or market.status == status
    quote_matches = quote_asset is None or market.quote_asset == quote_asset
    return status_matches and quote_matches


def _search_snapshots(
    downloader: Downloader,
    products: tuple[str, ...],
    status: str | None,
    quote_asset: str | None,
    *,
    refresh: bool,
    offline: bool,
    skip_missing: bool,
    progress: bool,
) -> list[Market]:
    """Load and filter the product snapshots participating in one search.

    Args:
        downloader: The configured internal downloader.
        products: The products to search in stable order.
        status: The optional native status filter.
        quote_asset: The optional quote asset filter.
        refresh: Whether to replace cached metadata now.
        offline: Whether source access is forbidden.
        skip_missing: Whether uncached offline products may be skipped.
        progress: Whether to show Rich activity.

    Returns:
        Filtered markets from every loaded product snapshot.
    """
    markets: list[Market] = []
    loaded_products = 0
    for product in products:
        try:
            snapshot = _snapshot(
                downloader,
                product,
                refresh=refresh,
                offline=offline,
                progress=progress,
            )
        except RuntimeError:
            if skip_missing:
                continue
            raise
        loaded_products += 1
        markets.extend(
            market
            for market in snapshot
            if _matches_filters(market, status, quote_asset)
        )
    if offline and not loaded_products:
        raise RuntimeError("offline mode requires cached market metadata")
    return markets


def find_markets(
    downloader: Downloader,
    query: object,
    *,
    product: object = None,
    status: object = None,
    quote_asset: object = None,
    limit: object = 10,
    refresh: object = False,
    offline: object = False,
    progress: bool = True,
) -> list[Market]:
    """Search one or every Binance product for likely markets.

    Args:
        downloader: The configured internal downloader.
        query: The native or normalized text to search for.
        product: An optional Binance product restriction.
        status: An optional native status filter.
        quote_asset: An optional quote asset filter.
        limit: The maximum number of matches.
        refresh: Whether to replace cached metadata now.
        offline: Whether source access is forbidden.
        progress: Whether to show Rich activity.

    Returns:
        Exact, prefix, then fuzzy matches with product context.
    """
    normalized_query = _query(query)
    selected_limit = _limit(limit)
    selected_status = _filter(status, "status")
    selected_quote = _filter(quote_asset, "quote_asset")
    selected_refresh = _boolean(refresh, "refresh")
    selected_offline = _boolean(offline, "offline")
    if selected_refresh and selected_offline:
        raise ValueError("refresh and offline cannot both be enabled")
    products = _search_products(downloader, product)
    markets = _search_snapshots(
        downloader,
        products,
        selected_status,
        selected_quote,
        refresh=selected_refresh,
        offline=selected_offline,
        skip_missing=selected_offline and product is None,
        progress=progress,
    )
    return _ranked_matches(normalized_query, markets, selected_limit)


def _unknown_market(requested: str, markets: list[Market]) -> ValueError:
    """Create an unknown-market error with optional fuzzy suggestions.

    Args:
        requested: The caller's original market spelling.
        markets: The market snapshot used for suggestions.

    Returns:
        The descriptive lookup error.
    """
    suggestions = _ranked_matches(normalize_pair(requested), markets, 3)
    if not suggestions:
        return ValueError(f"Pair '{requested}' was not found.")
    symbols = ", ".join(market.symbol for market in suggestions)
    return ValueError(f"Pair '{requested}' was not found. Suggestions: {symbols}.")


def _resolve_market(pair: object, markets: list[Market]) -> Market:
    """Resolve one native or normalized market without fuzzy substitution.

    Args:
        pair: The pair requested by the caller.
        markets: The product market snapshot to search.

    Returns:
        The unique matching market.
    """
    parsed, _single = parse_pairs(pair)
    requested = parsed[0]
    native = [market for market in markets if market.symbol == requested]
    if len(native) == 1:
        return native[0]
    normalized = [
        market
        for market in markets
        if market.normalized_symbol == normalize_pair(requested)
    ]
    if len(normalized) == 1:
        return normalized[0]
    matches = native or normalized
    if len(matches) > 1:
        symbols = ", ".join(market.symbol for market in matches)
        raise ValueError(f"Pair '{requested}' is ambiguous; matches: {symbols}")
    raise _unknown_market(requested, markets)


def _dataset(
    downloader: Downloader,
    product: object,
    dataset: object,
    interval: object,
) -> tuple[str, DatasetSpec, str | None]:
    """Resolve an inspection request to one stored dataset identity.

    Args:
        downloader: The configured internal downloader.
        product: The proposed Binance product.
        dataset: The proposed dataset name.
        interval: The optional output interval.

    Returns:
        The product, dataset declaration, and effective output interval.
    """
    selected_product = _product(downloader, product)
    selected_dataset = parse_identifier(dataset, name="dataset")
    specification = get_dataset(
        selected_product,
        selected_dataset,
        kline_base_interval=downloader.kline_base_interval,
    )
    return selected_product, specification, specification.resolve_interval(interval)


def _key(
    source: str,
    product: str,
    specification: DatasetSpec,
    market: Market,
) -> ResourceKey:
    """Build the stored resource identity for one resolved market.

    Args:
        source: The source identifier.
        product: The Binance product.
        specification: The dataset capability declaration.
        market: The resolved Binance market.

    Returns:
        The exact catalog resource key.
    """
    archive_symbol = None
    if specification.archive_symbol_attribute == "pair":
        archive_symbol = market.pair
        if not archive_symbol:
            raise ValueError(
                f"{product}/{specification.name} requires a source pair identifier"
            )
    return ResourceKey(
        source,
        product,
        specification.name,
        market.symbol,
        specification.base_interval,
        archive_symbol,
    )


def _local_size(resource: Resource, dataset: DatasetSpec) -> int | None:
    """Return a compatible ready file size without hashing its contents.

    Args:
        resource: The cataloged daily resource.
        dataset: The current schema expected by the caller.

    Returns:
        The current byte size, or ``None`` when local metadata is stale.
    """
    if (
        resource.status != "ready"
        or resource.parquet_path is None
        or resource.parquet_size is None
        or resource.parquet_mtime_ns is None
        or resource.schema_version != dataset.schema_version
        or resource.timestamp_column != dataset.time_column
    ):
        return None
    try:
        stat = resource.parquet_path.stat()
    except OSError:
        return None
    if (stat.st_size, stat.st_mtime_ns) != (
        resource.parquet_size,
        resource.parquet_mtime_ns,
    ):
        return None
    return stat.st_size


def _day_count(ranges: list[tuple[date, date]]) -> int:
    """Count distinct calendar days covered by inclusive ranges.

    Args:
        ranges: The possibly overlapping inclusive ranges.

    Returns:
        The number of distinct covered days.
    """
    return sum((end - start).days + 1 for start, end in _merge_ranges(ranges))


def _configured_start(
    remote_range: tuple[date, date] | None,
    earliest_date: date | None,
) -> date | None:
    """Return the configured or source-derived first usable day.

    Args:
        remote_range: The inclusive verified or active source bounds.
        earliest_date: The optional configured first usable day.

    Returns:
        The configured first day, a source fallback, or ``None``.
    """
    if earliest_date is not None:
        return earliest_date
    return remote_range[0] if remote_range is not None else None


def _configured_end(
    remote_range: tuple[date, date] | None,
    market: Market,
    today: date,
) -> date | None:
    """Return the active policy end or known inactive source end.

    Args:
        remote_range: The inclusive verified or active source bounds.
        market: The market whose activity controls the policy end.
        today: The current UTC day.

    Returns:
        The configured final day, or ``None`` when it cannot be known.
    """
    if market.active:
        return today - timedelta(days=1)
    return remote_range[1] if remote_range is not None else None


def _configured_range(
    remote_range: tuple[date, date] | None,
    earliest_date: date | None,
    market: Market,
    today: date,
) -> tuple[date, date] | None:
    """Return the configured policy window beside source availability.

    Args:
        remote_range: The inclusive verified or active source bounds.
        earliest_date: The optional configured first usable day.
        market: The market whose activity controls the policy end.
        today: The current UTC day.

    Returns:
        Inclusive configured bounds, or ``None`` without a usable policy window.
    """
    start = _configured_start(remote_range, earliest_date)
    end = _configured_end(remote_range, market, today)
    if start is None or end is None or start > end:
        return None
    return start, end


def _clip_ranges(
    ranges: list[tuple[date, date]],
    boundary: tuple[date, date] | None,
) -> list[tuple[date, date]]:
    """Clip inclusive ranges to a boundary and merge the remaining coverage.

    Args:
        ranges: The inclusive ranges to clip.
        boundary: The optional inclusive permitted boundary.

    Returns:
        Merged ranges that overlap the boundary.
    """
    if boundary is None:
        return []
    first, last = boundary
    return _merge_ranges(
        [
            (max(start, first), min(end, last))
            for start, end in ranges
            if start <= last and end >= first
        ]
    )


def _remote_range(
    catalog: Catalog,
    key: ResourceKey,
    market: Market,
    today: date,
) -> tuple[date, date] | None:
    """Return source bounds without confusing them with bounded discoveries.

    Args:
        catalog: The catalog containing verified source boundaries.
        key: The exact stored dataset identity.
        market: The market whose current activity controls the final day.
        today: The current UTC day.

    Returns:
        Inclusive source archive bounds, or ``None`` when no boundary is known.
    """
    bounds = catalog.source_bounds(key)
    if bounds is None:
        return None
    first, final = bounds
    if market.active:
        final = today - timedelta(days=1)
    elif final is None:
        known = catalog.resource_bounds(key)
        final = known[1] if known is not None else None
    if final is None or final < first:
        return None
    return first, final


@dataclass(frozen=True)
class _LocalCoverage:
    """Hold local resource state used to build public availability."""

    ready_days: frozenset[date]
    failed_days: frozenset[date]
    row_count: int
    local_bytes: int


def _local_coverage(resources: list[Resource], dataset: DatasetSpec) -> _LocalCoverage:
    """Classify local resource metadata without hashing cached files.

    Args:
        resources: The cataloged source resources.
        dataset: The current schema expected by the caller.

    Returns:
        Ready and failed days plus ready row and byte totals.
    """
    ready_days: set[date] = set()
    failed_days: set[date] = set()
    row_count = 0
    local_bytes = 0
    for resource in resources:
        size = _local_size(resource, dataset)
        if size is not None:
            ready_days.add(resource.day)
            row_count += resource.row_count or 0
            local_bytes += size
        elif resource.status == "failed":
            failed_days.add(resource.day)
    return _LocalCoverage(
        frozenset(ready_days),
        frozenset(failed_days),
        row_count,
        local_bytes,
    )


def _availability(
    downloader: Downloader,
    catalog: Catalog,
    key: ResourceKey,
    dataset: DatasetSpec,
    market: Market,
    output_interval: str | None,
) -> Availability:
    """Summarize remote discovery and local cache metadata.

    Args:
        downloader: The configured internal downloader.
        catalog: The open metadata catalog.
        key: The exact stored dataset identity.
        dataset: The current schema expected by the caller.
        market: The resolved market controlling active source bounds.
        output_interval: The effective caller-facing output interval.

    Returns:
        Immutable known coverage counts and bounds.
    """
    today = utc_today()
    remote_range = _remote_range(catalog, key, market, today)
    configured_range = _configured_range(
        remote_range,
        downloader.earliest_date,
        market,
        today,
    )
    usable_range = None
    if remote_range is not None and configured_range is not None:
        start = max(remote_range[0], configured_range[0])
        end = min(remote_range[1], configured_range[1])
        usable_range = (start, end) if start <= end else None
    coverage_range = usable_range or configured_range
    resources = (
        catalog.resources(key, *coverage_range) if coverage_range is not None else []
    )
    scanned = _clip_ranges(catalog.discovery_ranges(key), coverage_range)
    local = _local_coverage(resources, dataset)
    available_days = {resource.day for resource in resources}
    cached_range = (
        (min(local.ready_days), max(local.ready_days)) if local.ready_days else None
    )
    scanned_days = _day_count(scanned)
    return Availability(
        source=key.source,
        product=key.product,
        dataset=key.dataset,
        symbol=key.symbol,
        interval=output_interval,
        storage_interval=key.interval,
        remote_range=remote_range,
        configured_range=configured_range,
        cached_range=cached_range,
        scanned_ranges=tuple(scanned),
        scanned_days=scanned_days,
        available_days=len(available_days),
        cached_days=len(local.ready_days),
        missing_days=len(available_days - local.ready_days - local.failed_days),
        unavailable_days=max(0, scanned_days - len(available_days)),
        failed_days=len(local.failed_days),
        row_count=local.row_count,
        local_bytes=local.local_bytes,
    )


def get_availability(
    downloader: Downloader,
    pair: object,
    *,
    product: object,
    dataset: object,
    interval: object = None,
) -> Availability:
    """Read known coverage without creating a catalog or using the network.

    Args:
        downloader: The configured internal downloader.
        pair: The native or normalized Binance market.
        product: The Binance product.
        dataset: The dataset to inspect.
        interval: The optional Kline output interval.

    Returns:
        Already-cataloged remote and local coverage.
    """
    selected_product, specification, output_interval = _dataset(
        downloader, product, dataset, interval
    )
    parsed_pair, _single = parse_pairs(pair)
    catalog_path = downloader.data_dir / "catalog.duckdb"
    if not catalog_path.is_file():
        raise RuntimeError("local availability requires cached market metadata")
    with catalog_lock(catalog_path):
        with open_catalog(catalog_path) as catalog:
            markets = catalog.markets(downloader.source.code, selected_product)
            if not markets:
                raise RuntimeError("local availability requires cached market metadata")
            market = _resolve_market(parsed_pair[0], markets)
            key = _key(
                downloader.source.code,
                selected_product,
                specification,
                market,
            )
            return _availability(
                downloader,
                catalog,
                key,
                specification,
                market,
                output_interval,
            )


def _source_boundary(
    downloader: Downloader,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    *,
    refresh: bool,
    progress: bool,
) -> None:
    """Discover and cache the first source archive when it is not known.

    Args:
        downloader: The configured internal downloader.
        catalog: The catalog receiving source boundary metadata.
        client: The HTTPX client used for Binance requests.
        key: The exact stored dataset identity.
        refresh: Whether to repeat source-boundary discovery.
        progress: Whether to show Rich activity.
    """
    if catalog.source_bounds(key) is not None and not refresh:
        return
    with Reporter(progress).status(f"Finding the first {key.symbol} daily file"):
        first = downloader.source.first_resource(
            client,
            key,
            None,
            utc_today() - timedelta(days=1),
        )
    if first is None:
        return
    catalog.save_discovery(key, first.day, first.day, [first])
    catalog.save_source_bounds(key, first.day, None)


def discover_availability(
    downloader: Downloader,
    pair: object,
    start: object,
    end: object,
    *,
    product: object,
    dataset: object,
    interval: object = None,
    refresh: object = False,
    progress: bool = True,
) -> Availability:
    """Discover one bounded range without downloading source archives.

    Args:
        downloader: The configured internal downloader.
        pair: The native or normalized Binance market.
        start: The inclusive discovery start.
        end: The inclusive date or exclusive timestamp discovery end.
        product: The Binance product.
        dataset: The dataset to inspect.
        interval: The optional Kline output interval.
        refresh: Whether to repeat the complete bounded scan.
        progress: Whether to show Rich activity.

    Returns:
        Updated remote and local coverage.
    """
    selected_refresh = _boolean(refresh, "refresh")
    selected_product, specification, output_interval = _dataset(
        downloader, product, dataset, interval
    )
    request = Request.parse(
        pair,
        start,
        end,
        product=selected_product,
        dataset=specification.name,
        interval=interval,
        gap_policy=None,
    ).resolve_dataset(specification)
    catalog_path = downloader.data_dir / "catalog.duckdb"
    source_limit = _source_limit(downloader.source, downloader.max_workers)
    limits = httpx.Limits(
        max_connections=source_limit,
        max_keepalive_connections=source_limit,
    )
    with catalog_lock(catalog_path):
        with httpx.Client(
            transport=downloader.transport,
            follow_redirects=True,
            limits=limits,
        ) as client:
            with open_catalog(catalog_path) as catalog:
                markets = _load_markets(
                    downloader.source,
                    catalog,
                    client,
                    selected_product,
                    Reporter(progress),
                    refresh=selected_refresh,
                    offline=False,
                    refresh_hours=downloader.market_refresh_hours,
                )
                market = _resolve_market(request.pairs[0], markets)
                key = _key(
                    downloader.source.code,
                    selected_product,
                    specification,
                    market,
                )
                _source_boundary(
                    downloader,
                    catalog,
                    client,
                    key,
                    refresh=selected_refresh,
                    progress=progress,
                )
                _first, last = requested_days(request.start, request.end)
                recent = last >= utc_today() - timedelta(
                    days=downloader.discovery_tail_days
                )
                discover_resources(
                    downloader.source,
                    catalog,
                    client,
                    key,
                    request.start,
                    request.end,
                    active=market.active and recent,
                    refresh=selected_refresh,
                    tail_days=downloader.discovery_tail_days,
                )
                result = _availability(
                    downloader,
                    catalog,
                    key,
                    specification,
                    market,
                    output_interval,
                )
    LOGGER.info(
        "Availability discovered: product=%s dataset=%s symbol=%s "
        "range=[%s, %s) available_days=%d",
        selected_product,
        specification.name,
        result.symbol,
        request.start,
        request.end,
        result.available_days,
    )
    return result
