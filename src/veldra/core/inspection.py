"""Inspect source markets and dataset coverage without exposing the catalog."""

from dataclasses import dataclass, replace
from datetime import date, timedelta
import logging
from pathlib import Path
from typing import Protocol, cast

import httpx

from veldra.core.catalog import Catalog, catalog_lock, open_catalog
from veldra.core.datasets import DatasetSpec
from veldra.core.discovery import (
    _merge_ranges,
    discover_resources,
    requested_days,
)
from veldra.core.reporting import Reporter
from veldra.core.engine import (
    RetrievalEngine,
    _load_markets,
    _source_limit,
    utc_now,
    utc_today,
)
from veldra.core.models import Availability, Market, Resource, ResourceKey
from veldra.core.matching import rank_markets
from veldra.core.request import (
    Request,
    normalize_pair,
    parse_identifier,
    parse_pairs,
)

from .planner import plan_archives, catalog_archives, select_archives, covered_days

LOGGER = logging.getLogger(__name__)


class _QuoteVolumeSource(Protocol):
    """Describe optional quote-volume access for market inspection."""

    def quote_volumes(
        self,
        client: httpx.Client,
        product: str,
    ) -> dict[str, float]:
        """Return quote-asset volume indexed by native symbol."""
        pass


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


def _product(downloader: RetrievalEngine, value: object) -> str:
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


def _optional_limit(value: object) -> int | None:
    """Validate an optional positive market result limit.

    Args:
        value: The proposed limit or ``None`` for every result.

    Returns:
        The validated positive limit or ``None``.
    """
    return None if value is None else _limit(value)


def _sort_by(value: object) -> str:
    """Validate a supported market ordering.

    Args:
        value: The proposed market sort name.

    Returns:
        ``symbol`` or ``quote_volume``.
    """
    selected = parse_identifier(value, name="sort_by")
    if selected not in {"symbol", "quote_volume"}:
        raise ValueError("sort_by must be 'symbol' or 'quote_volume'")
    return selected


def _load_quote_volumes(
    downloader: RetrievalEngine,
    catalog: Catalog,
    client: httpx.Client,
    product: str,
    reporter: Reporter,
    *,
    refresh: bool,
    offline: bool,
) -> list[Market]:
    """Load cached quote volumes or refresh their rolling snapshot.

    Args:
        downloader: The configured internal downloader.
        catalog: The metadata catalog receiving fresh volume values.
        client: The HTTPX client used for source requests.
        product: The source product to inspect.
        reporter: The optional activity reporter.
        refresh: Whether to replace cached activity now.
        offline: Whether source access is forbidden.

    Returns:
        Markets enriched with rolling 24-hour quote volume.
    """
    snapshot = catalog.quote_volume_snapshot_at(downloader.source.code, product)
    cutoff = utc_now() - timedelta(hours=downloader.market_refresh_hours)
    fresh = snapshot is not None and snapshot >= cutoff
    if not offline and (refresh or not fresh):
        source = cast(_QuoteVolumeSource, downloader.source)
        with reporter.status(
            f"Refreshing {downloader.source.code.title()} {product} 24-hour volume"
        ):
            volumes = source.quote_volumes(client, product)
        catalog.save_quote_volumes(downloader.source.code, product, volumes)
    elif not fresh:
        raise RuntimeError("offline volume sorting requires cached market activity")
    return catalog.markets(downloader.source.code, product)


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
    downloader: RetrievalEngine,
    product: str,
    *,
    refresh: bool,
    offline: bool,
    progress: bool,
    with_volumes: bool = False,
) -> list[Market]:
    """Load one fresh or cached market snapshot with public context.

    Args:
        downloader: The configured internal downloader.
        product: The validated source product.
        refresh: Whether to replace cached metadata now.
        offline: Whether source access is forbidden.
        progress: Whether to show Rich activity.
        with_volumes: Whether rolling quote volume is required.

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
                reporter = Reporter(progress)
                markets = _load_markets(
                    downloader.source,
                    catalog,
                    client,
                    product,
                    reporter,
                    refresh=refresh,
                    offline=offline,
                    refresh_hours=downloader.market_refresh_hours,
                )
                if with_volumes:
                    markets = _load_quote_volumes(
                        downloader,
                        catalog,
                        client,
                        product,
                        reporter,
                        refresh=refresh,
                        offline=offline,
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


def _filtered_markets(
    markets: list[Market],
    status: str | None,
    quote_asset: str | None,
) -> list[Market]:
    """Apply optional exact market filters.

    Args:
        markets: The complete product market snapshot.
        status: The optional native status.
        quote_asset: The optional exact quote asset.

    Returns:
        Markets satisfying every requested filter.
    """
    return [
        market for market in markets if _matches_filters(market, status, quote_asset)
    ]


def _ordered_markets(
    markets: list[Market],
    sort_by: str,
    limit: int | None,
) -> list[Market]:
    """Order filtered markets and apply an optional result limit.

    Args:
        markets: The filtered market rows.
        sort_by: Native symbol or rolling quote-volume ordering.
        limit: The optional maximum result count.

    Returns:
        The requested ordered market slice.
    """
    if sort_by == "quote_volume":
        markets.sort(
            key=lambda market: (
                market.quote_volume_24h is None,
                -(market.quote_volume_24h or 0.0),
                market.symbol,
            )
        )
    return markets if limit is None else markets[:limit]


def get_markets(
    downloader: RetrievalEngine,
    *,
    product: object = "spot",
    status: object = None,
    quote_asset: object = None,
    sort_by: object = "symbol",
    limit: object = None,
    refresh: object = False,
    offline: object = False,
    progress: bool = True,
) -> list[Market]:
    """Return a filtered current or cached market snapshot.

    Args:
        downloader: The configured internal downloader.
        product: The source product to inspect.
        status: An optional native status filter.
        quote_asset: An optional quote asset filter.
        sort_by: Native symbol or rolling quote-volume ordering.
        limit: An optional positive maximum result count.
        refresh: Whether to replace cached metadata now.
        offline: Whether source access is forbidden.
        progress: Whether to show Rich activity.

    Returns:
        Matching immutable markets ordered by native symbol.
    """
    selected_product = _product(downloader, product)
    selected_status = _filter(status, "status")
    selected_quote = _filter(quote_asset, "quote_asset")
    selected_sort = _sort_by(sort_by)
    selected_limit = _optional_limit(limit)
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
        with_volumes=selected_sort == "quote_volume",
    )
    filtered = _filtered_markets(markets, selected_status, selected_quote)
    return _ordered_markets(filtered, selected_sort, selected_limit)


def _search_products(downloader: RetrievalEngine, product: object) -> tuple[str, ...]:
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
    status_matches = status is None or (
        market.status is not None and market.status.upper() == status
    )
    quote_matches = quote_asset is None or market.quote_asset == quote_asset
    return status_matches and quote_matches


def _search_snapshots(
    downloader: RetrievalEngine,
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
    downloader: RetrievalEngine,
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
    """Search one or every source product for likely markets.

    Args:
        downloader: The configured internal downloader.
        query: The native or normalized text to search for.
        product: An optional source product restriction.
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
    return rank_markets(normalized_query, markets, selected_limit)


def _unknown_market(requested: str, markets: list[Market]) -> ValueError:
    """Create an unknown-market error with optional fuzzy suggestions.

    Args:
        requested: The caller's original market spelling.
        markets: The market snapshot used for suggestions.

    Returns:
        The descriptive lookup error.
    """
    suggestions = rank_markets(normalize_pair(requested), markets, 3)
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
    downloader: RetrievalEngine,
    product: object,
    dataset: object,
    interval: object,
) -> tuple[str, DatasetSpec, str | None]:
    """Resolve an inspection request to one stored dataset identity.

    Args:
        downloader: The configured internal downloader.
        product: The proposed source product.
        dataset: The proposed dataset name.
        interval: The optional output interval.

    Returns:
        The product, dataset declaration, and effective output interval.
    """
    selected_product = _product(downloader, product)
    selected_dataset = parse_identifier(dataset, name="dataset")
    specification = downloader.dataset_resolver(
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
        product: The source product.
        specification: The dataset capability declaration.
        market: The resolved source market.

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
            ready_days.update(covered_days([resource]))
            row_count += resource.row_count or 0
            local_bytes += size
        elif resource.status == "failed":
            failed_days.update(covered_days([resource]))
    return _LocalCoverage(
        frozenset(ready_days),
        frozenset(failed_days),
        row_count,
        local_bytes,
    )


def _availability(
    downloader: RetrievalEngine,
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
        select_archives(catalog_archives(catalog, key, *coverage_range), dataset)
        if coverage_range is not None
        else []
    )
    scanned = _clip_ranges(
        _merge_ranges(
            catalog.discovery_ranges(key)
            + catalog.discovery_ranges(replace(key, cadence="monthly"))
        ),
        coverage_range,
    )
    local = _local_coverage(resources, dataset)
    available_days = covered_days(resources)
    ready_days, failed_days = set(local.ready_days), set(local.failed_days)
    if coverage_range is not None:
        first, last = coverage_range
        available_days = {day for day in available_days if first <= day <= last}
        ready_days &= available_days
        failed_days &= available_days
    cached_range = (min(ready_days), max(ready_days)) if ready_days else None
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
        cached_days=len(ready_days),
        missing_days=len(available_days - ready_days - failed_days),
        unavailable_days=max(0, scanned_days - len(available_days)),
        failed_days=len(failed_days),
        row_count=local.row_count,
        local_bytes=local.local_bytes,
    )


def get_availability(
    downloader: RetrievalEngine,
    pair: object,
    *,
    product: object,
    dataset: object,
    interval: object = None,
) -> Availability:
    """Read known coverage without creating a catalog or using the network.

    Args:
        downloader: The configured internal downloader.
        pair: The native or normalized source market.
        product: The source product.
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
    downloader: RetrievalEngine,
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
        client: The HTTPX client used for source requests.
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
    downloader: RetrievalEngine,
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
        pair: The native or normalized source market.
        start: The inclusive discovery start.
        end: The inclusive date or exclusive timestamp discovery end.
        product: The source product.
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
                plan_archives(
                    downloader.source,
                    catalog,
                    client,
                    key,
                    request.start,
                    request.end,
                    dataset=specification,
                    active=market.active and recent,
                    refresh=selected_refresh,
                    tail_days=downloader.discovery_tail_days,
                    reporter=Reporter(progress),
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
