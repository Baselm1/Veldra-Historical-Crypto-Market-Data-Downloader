"""Inspect OKX instruments and archive coverage without downloading data."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import math
from typing import TYPE_CHECKING

import httpx

from veldra.core.catalog import Catalog, catalog_lock, open_catalog
from veldra.core.discovery import latest_published_day
from veldra.core.matching import exact_markets, rank_markets
from veldra.core.models import (
    Availability,
    ArchiveObject,
    LogicalPartition,
    Market,
    ResourceKey,
)
from veldra.core.reporting import Reporter
from veldra.core.request import (
    Request,
    normalize_pair,
    parse_identifier,
    parse_timestamp,
)
from veldra.core.subjects import DataSubject
from veldra.okx.connector import PRODUCTS
from veldra.okx.datasets import get_dataset, manifest_spec
from veldra.okx.identities import (
    historical_future,
    historical_option,
    parse_currency,
    parse_option_id,
)
from veldra.okx.manifest import OKXManifestDiscovery
from veldra.okx.planner import OKXArchivePlanner

if TYPE_CHECKING:
    from veldra.core.datasets import DatasetSpec
    from veldra.okx.service import OKXService


def _boolean(value: object, name: str) -> bool:
    """Validate one strict Boolean option.

    Args:
        value: Proposed Boolean value.
        name: Public option name.

    Returns:
        The validated Boolean.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a Boolean")
    return value


def _product(value: object) -> str:
    """Validate one public OKX product.

    Args:
        value: Proposed product name.

    Returns:
        A supported product name.
    """
    product = parse_identifier(value, name="product")
    if product not in PRODUCTS:
        raise ValueError(f"unsupported OKX product {product!r}")
    return product


def _optional_text(value: object, name: str) -> str | None:
    """Normalize one optional exact text filter.

    Args:
        value: Optional text value.
        name: Public option name.

    Returns:
        Uppercase text or ``None``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _optional_active(value: object) -> bool | None:
    """Validate an optional active-market filter.

    Args:
        value: Boolean or ``None``.

    Returns:
        The validated filter.
    """
    if value is not None and not isinstance(value, bool):
        raise TypeError("active must be a Boolean or None")
    return value


def _limit(value: object, *, optional: bool) -> int | None:
    """Validate one optional or required positive result limit.

    Args:
        value: Proposed integer or ``None``.
        optional: Whether ``None`` is accepted.

    Returns:
        A positive integer or ``None``.
    """
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("limit must be an integer")
    if value < 1:
        raise ValueError("limit must be positive")
    return value


def _public_markets(markets: list[Market], product: str) -> list[Market]:
    """Attach source and product identity to catalog market rows.

    Args:
        markets: Stored product markets.
        product: Product represented by the rows.

    Returns:
        Markets with complete public context.
    """
    return [replace(item, source="okx", product=product) for item in markets]


def _snapshot(
    service: OKXService,
    product: str,
    *,
    refresh: bool,
    offline: bool,
    with_volumes: bool,
) -> list[Market]:
    """Load one current or cached OKX product snapshot.

    Args:
        service: Configured OKX service.
        product: Product to inspect.
        refresh: Whether current metadata must refresh.
        offline: Whether source access is forbidden.
        with_volumes: Whether native rolling volume is required.

    Returns:
        Product-scoped public markets.
    """
    catalog_path = service.data_dir / "catalog.duckdb"
    if offline and not catalog_path.is_file():
        raise RuntimeError("offline mode requires cached OKX market metadata")
    reporter = Reporter(service.progress)
    with catalog_lock(catalog_path):
        with service._client(offline=offline) as client:
            with open_catalog(catalog_path) as catalog:
                markets = service._markets(
                    catalog,
                    client,
                    product,
                    reporter,
                    refresh=refresh,
                    offline=offline,
                )
                if with_volumes:
                    snapshot = catalog.quote_volume_snapshot_at("okx", product)
                    cutoff = datetime.now(UTC) - timedelta(
                        hours=service.market_refresh_hours
                    )
                    fresh = snapshot is not None and snapshot >= cutoff
                    if not offline and (refresh or not fresh):
                        with reporter.status(
                            f"Refreshing OKX {product} 24-hour volume"
                        ):
                            volumes = service.connector.quote_volumes(client, product)
                        catalog.save_quote_volumes("okx", product, volumes)
                        markets = catalog.markets("okx", product)
                    elif not fresh:
                        raise RuntimeError(
                            "offline volume sorting requires cached OKX market activity"
                        )
    return _public_markets(markets, product)


def get_markets(
    service: OKXService,
    *,
    product: object = "spot",
    status: object = None,
    active: object = None,
    quote_asset: object = None,
    sort_by: object = "symbol",
    limit: object = None,
    refresh: object = False,
    offline: object = False,
) -> list[Market]:
    """Return filtered current or cached OKX instruments.

    Args:
        service: Configured OKX service.
        product: OKX product.
        status: Optional exact native state.
        active: Optional active-state filter.
        quote_asset: Optional exact quote or settlement currency.
        sort_by: ``symbol`` or native rolling ``quote_volume``.
        limit: Optional positive maximum result count.
        refresh: Whether current metadata must refresh.
        offline: Whether source access is forbidden.

    Returns:
        Matching markets in deterministic order.
    """
    selected_product = _product(product)
    selected_status = _optional_text(status, "status")
    selected_active = _optional_active(active)
    selected_quote = _optional_text(quote_asset, "quote_asset")
    selected_sort = parse_identifier(sort_by, name="sort_by")
    if selected_sort not in {"symbol", "quote_volume"}:
        raise ValueError("sort_by must be 'symbol' or 'quote_volume'")
    selected_limit = _limit(limit, optional=True)
    selected_refresh = _boolean(refresh, "refresh")
    selected_offline = _boolean(offline, "offline")
    if selected_refresh and selected_offline:
        raise ValueError("refresh and offline cannot both be enabled")
    values = _snapshot(
        service,
        selected_product,
        refresh=selected_refresh,
        offline=selected_offline,
        with_volumes=selected_sort == "quote_volume",
    )
    values = [
        item
        for item in values
        if (selected_status is None or (item.status or "").upper() == selected_status)
        and (selected_active is None or item.active is selected_active)
        and (selected_quote is None or item.quote_asset == selected_quote)
    ]
    if selected_sort == "quote_volume":
        values.sort(
            key=lambda item: (
                item.quote_volume_24h is None,
                -(item.quote_volume_24h or 0),
                item.symbol,
            )
        )
    else:
        values.sort(key=lambda item: item.symbol)
    return values if selected_limit is None else values[:selected_limit]


def find_markets(
    service: OKXService,
    query: object,
    *,
    product: object = None,
    status: object = None,
    active: object = None,
    quote_asset: object = None,
    limit: object = 10,
    refresh: object = False,
    offline: object = False,
) -> list[Market]:
    """Return exact, prefix, and high-confidence fuzzy OKX matches.

    Args:
        service: Configured OKX service.
        query: Native or human-formatted instrument text.
        product: Optional product restriction.
        status: Optional exact native state.
        active: Optional active-state filter.
        quote_asset: Optional exact quote or settlement currency.
        limit: Positive maximum match count.
        refresh: Whether current metadata must refresh.
        offline: Whether source access is forbidden.

    Returns:
        Ranked matches without automatic substitution.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    normalized = normalize_pair(query)
    if not normalized:
        raise ValueError("query must contain ASCII letters or digits")
    selected_limit = _limit(limit, optional=False)
    assert selected_limit is not None
    products = PRODUCTS if product is None else (_product(product),)
    values: list[Market] = []
    for selected_product in products:
        try:
            values.extend(
                get_markets(
                    service,
                    product=selected_product,
                    status=status,
                    active=active,
                    quote_asset=quote_asset,
                    refresh=refresh,
                    offline=offline,
                )
            )
        except RuntimeError:
            if product is not None or not offline:
                raise
    if offline and not values:
        raise RuntimeError("offline mode requires cached OKX market metadata")
    return rank_markets(normalized, values, selected_limit)


def get_contracts(
    service: OKXService,
    *,
    product: object,
    family: object,
    active: object = None,
    contract_style: object = None,
    refresh: object = False,
    offline: object = False,
) -> list[Market]:
    """Return current dated Futures contracts for one native family.

    Args:
        service: Configured OKX service.
        product: Linear- or inverse-margined Futures product.
        family: Exact native instrument family.
        active: Optional active-state filter.
        contract_style: Optional normal or X-Perp style.
        refresh: Whether current metadata must refresh.
        offline: Whether source access is forbidden.

    Returns:
        Matching current Futures contracts.
    """
    selected_product = _product(product)
    if selected_product not in {"linear_futures", "inverse_futures"}:
        raise ValueError("get_contracts requires a dated Futures product")
    selected_family = _optional_text(family, "family")
    assert selected_family is not None
    selected_style = _optional_text(contract_style, "contract_style")
    allowed = {None, "NORMAL", "XPERP", "PRE_MARKET_XPERP"}
    if selected_style not in allowed:
        raise ValueError("unsupported OKX contract style")
    values = get_markets(
        service,
        product=selected_product,
        active=active,
        refresh=refresh,
        offline=offline,
    )
    return [
        item
        for item in values
        if item.pair == selected_family
        and (selected_style is None or item.contract_type == selected_style)
    ]


def get_option_contracts(
    service: OKXService,
    *,
    family: object,
    expiry: object = None,
    option_type: object = None,
    strike_min: object = None,
    strike_max: object = None,
    active: object = None,
    refresh: object = False,
    offline: object = False,
) -> list[Market]:
    """Return current Option contracts matching native contract terms.

    Args:
        service: Configured OKX service.
        family: Exact native Option family.
        expiry: Optional exact UTC expiry date.
        option_type: Optional ``call`` or ``put``.
        strike_min: Optional inclusive minimum strike.
        strike_max: Optional inclusive maximum strike.
        active: Optional active-state filter.
        refresh: Whether current metadata must refresh.
        offline: Whether source access is forbidden.

    Returns:
        Matching current Option contracts.
    """
    selected_family = _optional_text(family, "family")
    assert selected_family is not None
    selected_expiry = parse_timestamp(expiry).date() if expiry is not None else None
    if option_type is not None and not isinstance(option_type, str):
        raise TypeError("option_type must be a string")
    if option_type not in {None, "call", "put"}:
        raise ValueError("option_type must be call or put")
    selected_type = {None: None, "call": "C", "put": "P"}[option_type]
    bounds: list[float | None] = []
    for name, value in (("strike_min", strike_min), ("strike_max", strike_max)):
        if value is None:
            bounds.append(None)
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        elif not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
        else:
            bounds.append(float(value))
    minimum, maximum = bounds
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("strike_min cannot exceed strike_max")
    values = get_markets(
        service,
        product="options",
        active=active,
        refresh=refresh,
        offline=offline,
    )
    selected: list[Market] = []
    for item in values:
        if item.pair != selected_family:
            continue
        contract_expiry, strike, kind = parse_option_id(item.symbol)
        if selected_expiry is not None and contract_expiry != selected_expiry:
            continue
        if selected_type is not None and kind != selected_type:
            continue
        if minimum is not None and strike < minimum:
            continue
        if maximum is not None and strike > maximum:
            continue
        selected.append(item)
    return selected


def _resolved_market(pair: str, product: str, markets: list[Market]) -> Market:
    """Resolve one exact current or conservatively parsed historical market.

    Args:
        pair: Caller-provided instrument ID.
        product: Requested OKX product.
        markets: Current product-scoped markets.

    Returns:
        Exact current or archive-derived market.
    """
    matches = exact_markets(pair, markets)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(sorted(item.symbol for item in matches))
        raise ValueError(f"Pair '{pair}' is ambiguous; matches: {names}")
    historical: Market | None = None
    try:
        if product in {"linear_futures", "inverse_futures"}:
            historical = historical_future(pair, product).market
        elif product == "options":
            historical = historical_option(pair).market
    except TypeError, ValueError:
        historical = None
    if historical is not None:
        return historical
    suggestions = rank_markets(pair, markets, 3)
    suffix = (
        " Suggestions: " + ", ".join(item.symbol for item in suggestions) + "."
        if suggestions
        else ""
    )
    raise ValueError(f"Pair '{pair}' was not found.{suffix}")


def _subject(specification: DatasetSpec, market: Market) -> DataSubject:
    """Return the physical manifest scope for one logical market.

    Args:
        specification: Requested dataset declaration.
        market: Resolved logical market.

    Returns:
        Instrument or family manifest subject.
    """
    kind = manifest_spec(specification.product, specification.name).subject_kind
    value = market.symbol if kind == "instrument" else market.pair
    if not value:
        raise ValueError("OKX market does not declare its archive family")
    return DataSubject(kind, value)


def _coverage_days(archives: list[ArchiveObject]) -> set[date]:
    """Expand physical archive periods into source calendar days.

    Args:
        archives: Relevant physical archives.

    Returns:
        Distinct inclusive source dates.
    """
    values: set[date] = set()
    for archive in archives:
        current = archive.key.period_start
        while current <= archive.key.period_end:
            values.add(current)
            current += timedelta(days=1)
    return values


def _merge_ranges(values: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Merge overlapping or adjacent inclusive date ranges.

    Args:
        values: Inclusive ranges in any order.

    Returns:
        Disjoint ordered ranges.
    """
    merged: list[tuple[date, date]] = []
    for first, last in sorted(values):
        if merged and first <= merged[-1][1] + timedelta(days=1):
            merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
        else:
            merged.append((first, last))
    return merged


def _partition_days(
    service: OKXService,
    partitions: list[LogicalPartition],
    specification: DatasetSpec,
) -> set[date]:
    """Return source dates represented by local logical partitions.

    Args:
        service: Configured OKX service.
        partitions: Exact logical subject partitions.
        specification: Dataset source calendar.

    Returns:
        Distinct cached source dates.
    """
    values: set[date] = set()
    for partition in partitions:
        first, last = service._source_days(
            partition.coverage_start, partition.coverage_end, specification
        )
        current = first
        while current <= last:
            values.add(current)
            current += timedelta(days=1)
    return values


def _availability(
    service: OKXService,
    catalog: Catalog,
    market: Market,
    subject: DataSubject,
    specification: DatasetSpec,
    output_interval: str | None,
) -> Availability:
    """Summarize known physical and logical OKX coverage.

    Args:
        service: Configured OKX service.
        catalog: Open local catalog.
        market: Logical instrument represented by the result.
        subject: Physical manifest subject.
        specification: Dataset declaration.
        output_interval: Caller-facing interval.

    Returns:
        Immutable known remote, scanned, and cached coverage.
    """
    archives = catalog.archives_between(
        "okx", specification.product, specification.name, date.min, date.max
    )
    relevant = [
        item
        for item in archives
        if item.key.remote_scope_kind == "all" or item.key.subject == subject
    ]
    available = _coverage_days([item for item in relevant if item.status != "missing"])
    remote_range = (min(available), max(available)) if available else None
    configured_start = service.earliest_date or (
        remote_range[0] if remote_range is not None else None
    )
    configured_end = (
        latest_published_day(datetime.now(UTC).date(), specification)
        if market.active
        else (remote_range[1] if remote_range is not None else None)
    )
    configured_range = (
        (configured_start, configured_end)
        if configured_start is not None
        and configured_end is not None
        and configured_start <= configured_end
        else None
    )
    if configured_range is not None:
        first, last = configured_range
        available = {day for day in available if first <= day <= last}
    query_start = datetime(1970, 1, 1, tzinfo=UTC)
    query_end = datetime(2100, 1, 1, tzinfo=UTC)
    partitions = catalog.partitions_between(
        "okx",
        specification.product,
        specification.name,
        DataSubject(
            "currency" if specification.name == "borrow_rates" else "instrument",
            market.symbol,
        ),
        specification.base_interval,
        query_start,
        query_end,
    )
    cached = _partition_days(service, partitions, specification)
    failed = _coverage_days([item for item in relevant if item.status == "failed"])
    if configured_range is not None:
        first, last = configured_range
        cached = {day for day in cached if first <= day <= last}
        failed = {day for day in failed if first <= day <= last}
    key = ResourceKey(
        "okx",
        specification.product,
        specification.name,
        subject.value,
        specification.base_interval,
        cadence="daily",
        subject=subject,
    )
    scanned = _merge_ranges(catalog.discovery_ranges(key))
    scanned_values = {
        day
        for first, last in scanned
        for day in (
            first + timedelta(days=offset) for offset in range((last - first).days + 1)
        )
    }
    paths = {item.materialization_path for item in partitions}
    local_bytes = sum(path.stat().st_size for path in paths if path.is_file())
    return Availability(
        "okx",
        specification.product,
        specification.name,
        market.symbol,
        output_interval,
        specification.base_interval,
        remote_range,
        configured_range,
        (min(cached), max(cached)) if cached else None,
        tuple(scanned),
        len(scanned_values),
        len(available),
        len(cached),
        len(available - cached - failed),
        len(scanned_values - available),
        len(failed),
        sum(item.row_count for item in partitions),
        local_bytes,
    )


def _inspection_identity(
    service: OKXService,
    catalog: Catalog,
    product: str,
    dataset: str,
    pair: str,
    *,
    client: httpx.Client | None = None,
    refresh: bool = False,
    offline: bool = True,
) -> tuple[Market, DataSubject, DatasetSpec]:
    """Resolve one public coverage request to its physical subject.

    Args:
        service: Configured OKX service.
        catalog: Open local catalog.
        product: Requested product.
        dataset: Requested dataset.
        pair: Instrument or currency value.
        client: Shared HTTP client for online market refresh.
        refresh: Whether current markets must refresh.
        offline: Whether source access is forbidden.

    Returns:
        Logical market, physical subject, and dataset declaration.
    """
    specification = get_dataset(product, dataset)
    if specification.name == "borrow_rates":
        currency = parse_currency(pair)
        market = Market(
            currency,
            currency,
            status="historical",
            source="okx",
            product="margin",
            active=True,
        )
        return market, DataSubject("currency", currency), specification
    if client is None:
        markets = catalog.markets("okx", product)
        if not markets:
            raise RuntimeError("local availability requires cached OKX market metadata")
    else:
        markets = service._markets(
            catalog,
            client,
            product,
            Reporter(service.progress),
            refresh=refresh,
            offline=offline,
        )
    market = _resolved_market(pair, product, _public_markets(markets, product))
    return market, _subject(specification, market), specification


def get_availability(
    service: OKXService,
    pair: object,
    *,
    product: object,
    dataset: object,
    interval: object = None,
) -> Availability:
    """Return cataloged OKX coverage without making a network request.

    Args:
        service: Configured OKX service.
        pair: Native instrument or currency.
        product: OKX product.
        dataset: Archive-backed dataset.
        interval: Optional Kline output interval.

    Returns:
        Known remote, scanned, and cached coverage.
    """
    if not isinstance(pair, str):
        raise TypeError("pair must be a string")
    selected_product = _product(product)
    selected_dataset = parse_identifier(dataset, name="dataset")
    catalog_path = service.data_dir / "catalog.duckdb"
    if not catalog_path.is_file():
        raise RuntimeError("local availability requires cached OKX market metadata")
    with catalog_lock(catalog_path):
        with open_catalog(catalog_path) as catalog:
            market, subject, specification = _inspection_identity(
                service,
                catalog,
                selected_product,
                selected_dataset,
                pair,
            )
            output_interval = specification.resolve_interval(interval)
            return _availability(
                service,
                catalog,
                market,
                subject,
                specification,
                output_interval,
            )


def discover_availability(
    service: OKXService,
    pair: object,
    start: object,
    end: object,
    *,
    product: object,
    dataset: object,
    interval: object = None,
    refresh: object = False,
) -> Availability:
    """Discover bounded OKX archive coverage without downloading files.

    Args:
        service: Configured OKX service.
        pair: Native instrument or currency.
        start: Inclusive UTC request start.
        end: Inclusive date or exclusive timestamp end.
        product: OKX product.
        dataset: Archive-backed dataset.
        interval: Optional Kline output interval.
        refresh: Whether current market metadata must refresh.

    Returns:
        Updated known remote, scanned, and cached coverage.
    """
    if not isinstance(pair, str):
        raise TypeError("pair must be a string")
    selected_refresh = _boolean(refresh, "refresh")
    selected_product = _product(product)
    selected_dataset = parse_identifier(dataset, name="dataset")
    specification = get_dataset(selected_product, selected_dataset)
    request = Request.parse(
        pair,
        start,
        end,
        product=selected_product,
        dataset=selected_dataset,
        interval=interval,
        gap_policy="keep" if specification.supports_gap_policy else None,
    ).resolve_dataset(specification)
    effective_start = service._effective_start(request)
    if effective_start >= request.end:
        raise ValueError("request ends before configured OKX history")
    first_day, last_day = service._source_days(
        effective_start, request.end, specification
    )
    catalog_path = service.data_dir / "catalog.duckdb"
    with catalog_lock(catalog_path):
        with service._client(offline=False) as client:
            with open_catalog(catalog_path) as catalog:
                market, subject, specification = _inspection_identity(
                    service,
                    catalog,
                    selected_product,
                    selected_dataset,
                    pair,
                    client=client,
                    refresh=selected_refresh,
                    offline=False,
                )
                cached = catalog.ready_archives_between(
                    "okx", selected_product, selected_dataset, first_day, last_day
                )
                discovery = OKXManifestDiscovery(service.connector._api(client))
                plan = OKXArchivePlanner(discovery, cached=cached).plan(
                    selected_product,
                    selected_dataset,
                    [subject],
                    first_day,
                    last_day,
                    transport="specific",
                )
                catalog.save_archives(plan.selected)
                key = ResourceKey(
                    "okx",
                    selected_product,
                    selected_dataset,
                    subject.value,
                    specification.base_interval,
                    cadence="daily",
                    subject=subject,
                )
                catalog.save_discovery(key, first_day, last_day, [])
                return _availability(
                    service,
                    catalog,
                    market,
                    subject,
                    specification,
                    specification.resolve_interval(interval),
                )
