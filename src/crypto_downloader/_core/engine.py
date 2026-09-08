"""Coordinate public cryptocurrency data requests."""

from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, date, datetime, time, timedelta
from functools import cache
import logging
import math
from pathlib import Path
from queue import LifoQueue
from ssl import SSLContext
from time import perf_counter

import httpx
import pandas as pd

from crypto_downloader._core.catalog import Catalog, catalog_lock, open_catalog
from crypto_downloader._core.config import Settings, load_settings
from crypto_downloader._core.datasets import DatasetSpec, get_dataset
from crypto_downloader._core.reporting import Reporter
from crypto_downloader._core.models import Market, Result
from crypto_downloader._core.pair import process_pair
from crypto_downloader._core.request import Request, normalize_pair, parse_timestamp
from crypto_downloader._core.source import Source
from crypto_downloader.binance.connector import BinanceSource

LOGGER = logging.getLogger(__name__)


def utc_today() -> date:
    """Return today's UTC calendar date.

    Returns:
        The current date at UTC.
    """
    return datetime.now(UTC).date()


def utc_now() -> datetime:
    """Return the current UTC timestamp.

    Returns:
        The current timezone-aware UTC timestamp.
    """
    return datetime.now(UTC)


@cache
def _ssl_context() -> SSLContext:
    """Create and retain the process-wide HTTP certificate context.

    Returns:
        The immutable TLS context shared by downloader request clients.
    """
    return httpx.create_ssl_context()


def _reject_offline_request(request: httpx.Request) -> httpx.Response:
    """Reject accidental HTTP access from an offline request.

    Args:
        request: The network request that offline mode attempted.

    Raises:
        RuntimeError: Always, because offline mode forbids HTTP access.
    """
    raise RuntimeError(f"offline mode attempted an HTTP request to {request.url}")


def _http_client(
    transport: httpx.BaseTransport | None,
    limits: httpx.Limits,
    *,
    offline: bool,
) -> httpx.Client:
    """Create one request client without repeating expensive TLS setup.

    Args:
        transport: An optional caller-supplied HTTP transport.
        limits: The source-wide connection-pool limits.
        offline: Whether every attempted HTTP request must fail locally.

    Returns:
        A configured synchronous HTTPX client.
    """
    if offline:
        return httpx.Client(
            transport=httpx.MockTransport(_reject_offline_request),
            follow_redirects=True,
            limits=limits,
        )
    if transport is not None:
        return httpx.Client(
            transport=transport,
            follow_redirects=True,
            limits=limits,
        )
    return httpx.Client(
        verify=_ssl_context(),
        follow_redirects=True,
        limits=limits,
    )


def _history_date(value: object) -> date | None:
    """Parse a configurable UTC history boundary.

    Args:
        value: The proposed date-like earliest boundary.

    Returns:
        A valid UTC calendar date, or ``None`` for all source history.
    """
    if value is None or value == "all":
        return None
    try:
        parsed = parse_timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("earliest_date must be an ISO date or 'all'") from error
    if parsed.time() != time.min:
        raise ValueError("earliest_date must be a UTC day or 'all'")
    return parsed.date()


def _source_limit(source: Source, max_workers: int) -> int:
    """Return the effective source-wide concurrency limit.

    Args:
        source: The source whose optional concurrency limit applies.
        max_workers: The caller's validated concurrency limit.

    Returns:
        The lower caller and source concurrency limit.
    """
    source_limit = getattr(source, "max_concurrency", max_workers)
    if (
        isinstance(source_limit, bool)
        or not isinstance(source_limit, int)
        or source_limit < 1
    ):
        raise ValueError("source max_concurrency must be a positive integer")
    return min(max_workers, source_limit)


def _load_markets(
    source: Source,
    catalog: Catalog,
    client: httpx.Client,
    product: str,
    reporter: Reporter,
    *,
    refresh: bool,
    offline: bool,
    refresh_hours: float,
) -> list[Market]:
    """Load cached markets or refresh a stale source snapshot.

    Args:
        source: The source serving market metadata.
        catalog: The catalog containing the cached snapshot.
        client: The HTTPX client used for source requests.
        product: The requested source product.
        reporter: The optional Rich activity reporter.
        refresh: Whether the caller requires fresh metadata now.
        offline: Whether all source access is forbidden.
        refresh_hours: Hours a market snapshot remains fresh.

    Returns:
        The complete market snapshot used by the request.
    """
    markets = catalog.markets(source.code, product)
    snapshot = catalog.market_snapshot_at(source.code, product)
    cutoff = utc_now() - timedelta(hours=refresh_hours)
    fresh = bool(markets) and snapshot is not None and snapshot >= cutoff
    should_refresh = not offline and (refresh or not fresh)
    if should_refresh:
        with reporter.status(f"Refreshing {source.code.title()} {product} markets"):
            markets = source.markets(client, product)
        catalog.save_markets(source.code, product, markets)
        LOGGER.info(
            "Market snapshot refreshed: source=%s product=%s markets=%d",
            source.code,
            product,
            len(markets),
        )
        reporter.market_summary(markets, refreshed=True)
        return markets
    if not markets:
        raise RuntimeError("offline mode requires cached market metadata")
    LOGGER.info(
        "Market snapshot loaded from cache: source=%s product=%s markets=%d age=%s",
        source.code,
        product,
        len(markets),
        utc_now() - snapshot if snapshot is not None else None,
    )
    reporter.market_summary(markets, refreshed=False)
    return markets


def _run_pair(
    source: Source,
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
    refresh: bool,
    offline: bool,
    discovery_tail_days: int,
    max_workers: int,
    reporter: Reporter,
    ingestion_executor: Executor,
) -> Result:
    """Run one pair against its dedicated catalog connection.

    Args:
        source: The source serving the pair.
        catalog: The pair's catalog connection.
        client: The shared source HTTP client.
        data_dir: The root downloader data directory.
        markets: The current market snapshot.
        pair: The caller's original pair spelling.
        request: The validated shared request.
        dataset: The requested dataset schema.
        earliest_date: The optional configured global history boundary.
        today: The current UTC day.
        refresh: Whether source metadata should be refreshed.
        offline: Whether source access is forbidden.
        discovery_tail_days: Recent active-market days to revisit.
        max_workers: Daily ingestion workers assigned to this pair.
        reporter: The optional Rich activity reporter.
        ingestion_executor: The request-wide cache executor.

    Returns:
        The pair's data and structured diagnostics.
    """
    return process_pair(
        source,
        catalog,
        client,
        data_dir,
        markets,
        pair,
        request,
        dataset,
        earliest_date,
        today,
        refresh=refresh,
        offline=offline,
        discovery_tail_days=discovery_tail_days,
        max_workers=max_workers,
        reporter=reporter,
        ingestion_executor=ingestion_executor,
    )


def _pair_workflow_keys(
    pairs: tuple[str, ...], markets: list[Market]
) -> list[tuple[str, str]]:
    """Group spellings that resolve uniquely to the same native market.

    Args:
        pairs: The caller's requested pair spellings.
        markets: The current source market snapshot.

    Returns:
        One stable workflow identity for each requested pair.
    """
    native_symbols = {market.symbol for market in markets}
    normalized_symbols: dict[str, list[str]] = {}
    for market in markets:
        normalized_symbols.setdefault(market.normalized_symbol, []).append(
            market.symbol
        )
    keys: list[tuple[str, str]] = []
    for pair in pairs:
        if pair in native_symbols:
            keys.append(("market", pair))
            continue
        matches = normalized_symbols.get(normalize_pair(pair), [])
        keys.append(("market", matches[0]) if len(matches) == 1 else ("request", pair))
    return keys


def _process_pairs(
    source: Source,
    catalog_path: Path,
    client: httpx.Client,
    data_dir: Path,
    markets: list[Market],
    request: Request,
    dataset: DatasetSpec,
    earliest_date: date | None,
    today: date,
    *,
    refresh: bool,
    offline: bool,
    discovery_tail_days: int,
    max_workers: int,
    reporter: Reporter,
) -> list[Result]:
    """Run distinct pair workflows concurrently within one source budget.

    Args:
        source: The source serving every requested pair.
        catalog_path: The shared catalog database path.
        client: The shared source HTTP client.
        data_dir: The root downloader data directory.
        markets: The current market snapshot.
        request: The validated shared request.
        dataset: The requested dataset schema.
        earliest_date: The optional configured global history boundary.
        today: The current UTC day.
        refresh: Whether source metadata should be refreshed.
        offline: Whether source access is forbidden.
        discovery_tail_days: Recent active-market days to revisit.
        max_workers: The total source-wide concurrency budget.
        reporter: The optional Rich activity reporter.

    Returns:
        Results in the caller's original pair order.
    """
    workflow_keys = _pair_workflow_keys(request.pairs, markets)
    representatives: dict[tuple[str, str], str] = {}
    for key, pair in zip(workflow_keys, request.pairs):
        representatives.setdefault(key, pair)
    unique_keys = list(representatives)
    unique_pairs = list(representatives.values())
    pair_workers = min(len(unique_pairs), max_workers)
    ingestion_workers = min(max_workers, dataset.max_concurrency)
    LOGGER.info(
        "Pair workflows planned: pairs=%d workers=%d ingestion_workers=%d",
        len(unique_pairs),
        pair_workers,
        ingestion_workers,
    )
    with ThreadPoolExecutor(max_workers=ingestion_workers) as ingestion_executor:
        with ExitStack() as stack:
            catalog_pool: LifoQueue[Catalog] = LifoQueue()
            for _ in range(pair_workers):
                catalog_pool.put(
                    stack.enter_context(open_catalog(catalog_path, initialize=False))
                )

            def run(pair: str) -> Result:
                """Run one pair with a catalog borrowed from the bounded pool.

                Args:
                    pair: The requested pair spelling.

                Returns:
                    The completed pair result.
                """
                catalog = catalog_pool.get()
                try:
                    return _run_pair(
                        source,
                        catalog,
                        client,
                        data_dir,
                        markets,
                        pair,
                        request,
                        dataset,
                        earliest_date,
                        today,
                        refresh=refresh,
                        offline=offline,
                        discovery_tail_days=discovery_tail_days,
                        max_workers=max_workers,
                        reporter=reporter,
                        ingestion_executor=ingestion_executor,
                    )
                finally:
                    catalog_pool.put(catalog)

            if pair_workers == 1:
                unique_results = [run(unique_pairs[0])]
            else:
                with ThreadPoolExecutor(max_workers=pair_workers) as executor:
                    unique_results = list(executor.map(run, unique_pairs))
    by_key = dict(zip(unique_keys, unique_results))
    return [by_key[key] for key in workflow_keys]


class Downloader:
    """Provide reusable downloader paths, source, and HTTP settings."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        *,
        source: Source | None = None,
        transport: httpx.BaseTransport | None = None,
        config_path: str | Path | None = None,
        earliest_date: object = None,
        max_workers: int = 32,
        discovery_tail_days: int = 7,
        market_refresh_hours: float = 24.0,
    ) -> None:
        """Create a reusable imported downloader service.

        Args:
            data_dir: The directory containing the catalog and Parquet cache.
            source: The optional source strategy, defaulting to Binance.
            transport: An optional HTTPX transport used for requests.
            config_path: An optional TOML file overriding installed defaults.
            earliest_date: An optional override for the configured history boundary.
            max_workers: The maximum concurrent daily archive downloads.
            discovery_tail_days: Recent active-market days rediscovered per request.
            market_refresh_hours: Hours before market metadata is refreshed again.
        """
        if not isinstance(data_dir, (str, Path)):
            raise TypeError("data_dir must be a path")
        if not str(data_dir).strip():
            raise ValueError("data_dir must not be empty")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.source: Source = source if source is not None else BinanceSource()
        self.transport = transport
        self.settings: Settings = load_settings(config_path)
        configured_date = self.settings.earliest_date
        self.earliest_date = _history_date(
            configured_date if earliest_date is None else earliest_date
        )
        if self.earliest_date is not None and self.earliest_date >= utc_today():
            raise ValueError("earliest_date must be before today in UTC")
        self.kline_base_interval = self.settings.kline_base_interval
        self.max_workers = _positive_integer(max_workers, "max_workers")
        self.discovery_tail_days = _positive_integer(
            discovery_tail_days, "discovery_tail_days"
        )
        self.market_refresh_hours = _positive_number(
            market_refresh_hours, "market_refresh_hours"
        )

    def get_results(
        self,
        pairs: object,
        starting_date: object,
        end_date: object,
        *,
        product: object = "spot",
        dataset: object = "klines",
        interval: object = None,
        desired_columns: object = None,
        refresh: bool = False,
        offline: bool = False,
        gap_policy: object = None,
        progress: bool = True,
    ) -> Result | list[Result]:
        """Return data and structured reports for requested pairs.

        Args:
            pairs: One pair string or an ordered list of pair strings.
            starting_date: The requested first date or timestamp.
            end_date: The requested inclusive date or exclusive timestamp.
            product: The source product identifier.
            dataset: The historical dataset identifier.
            interval: The optional output interval.
            desired_columns: Optional selected and renamed columns.
            refresh: Whether to repeat complete resource discovery.
            offline: Whether to use only cataloged markets and cached files.
            gap_policy: The behavior used for internal missing candles.
            progress: Whether to show optional Rich activity.

        Returns:
            One result for string input or an ordered result list.
        """
        request = Request.parse(
            pairs,
            starting_date,
            end_date,
            product=product,
            dataset=dataset,
            interval=interval,
            desired_columns=desired_columns,
            gap_policy=gap_policy,
        )
        specification = get_dataset(
            request.product,
            request.dataset,
            kline_base_interval=self.kline_base_interval,
        )
        request = request.resolve_dataset(specification)
        if request.product not in self.source.products:
            raise ValueError(
                f"unsupported product for {self.source.code}: {request.product}"
            )

        catalog_path = self.data_dir / "catalog.duckdb"
        refresh = _boolean(refresh, "refresh")
        offline = _boolean(offline, "offline")
        progress = _boolean(progress, "progress")
        if refresh and offline:
            raise ValueError("refresh and offline cannot both be enabled")
        reporter = Reporter(progress)
        reporter.request(
            self.source.code,
            request.product,
            request.dataset,
            len(request.pairs),
            request.start,
            request.end,
        )
        started = perf_counter()
        LOGGER.debug(
            "Request started: source=%s product=%s dataset=%s pairs=%s "
            "range=[%s, %s) interval=%s refresh=%s offline=%s",
            self.source.code,
            request.product,
            request.dataset,
            request.pairs,
            request.start,
            request.end,
            request.interval,
            refresh,
            offline,
        )
        source_limit = _source_limit(self.source, self.max_workers)
        limits = httpx.Limits(
            max_connections=source_limit,
            max_keepalive_connections=source_limit,
        )
        with catalog_lock(catalog_path):
            with _http_client(self.transport, limits, offline=offline) as client:
                with open_catalog(catalog_path) as catalog:
                    markets = _load_markets(
                        self.source,
                        catalog,
                        client,
                        request.product,
                        reporter,
                        refresh=refresh,
                        offline=offline,
                        refresh_hours=self.market_refresh_hours,
                    )
                results = _process_pairs(
                    self.source,
                    catalog_path,
                    client,
                    self.data_dir,
                    markets,
                    request,
                    specification,
                    self.earliest_date,
                    utc_today(),
                    refresh=refresh,
                    offline=offline,
                    discovery_tail_days=self.discovery_tail_days,
                    max_workers=source_limit,
                    reporter=reporter,
                )
        LOGGER.info(
            "Request complete: source=%s product=%s dataset=%s pairs=%d "
            "elapsed=%.3fs",
            self.source.code,
            request.product,
            request.dataset,
            len(results),
            perf_counter() - started,
        )
        return results[0] if request.single else results

    def get_data(
        self,
        pairs: object,
        starting_date: object,
        end_date: object,
        *,
        product: object = "spot",
        dataset: object = "klines",
        interval: object = None,
        desired_columns: object = None,
        refresh: bool = False,
        offline: bool = False,
        gap_policy: object = None,
        progress: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Return only DataFrames for requested pairs.

        Args:
            pairs: One pair string or an ordered list of pair strings.
            starting_date: The requested first date or timestamp.
            end_date: The requested inclusive date or exclusive timestamp.
            product: The source product identifier.
            dataset: The historical dataset identifier.
            interval: The optional output interval.
            desired_columns: Optional selected and renamed columns.
            refresh: Whether to repeat complete resource discovery.
            offline: Whether to use only cataloged markets and cached files.
            gap_policy: The behavior used for internal missing candles.
            progress: Whether to show optional Rich activity.

        Returns:
            One DataFrame for string input or an ordered DataFrame list.
        """
        results = self.get_results(
            pairs,
            starting_date,
            end_date,
            product=product,
            dataset=dataset,
            interval=interval,
            desired_columns=desired_columns,
            refresh=refresh,
            offline=offline,
            gap_policy=gap_policy,
            progress=progress,
        )
        if isinstance(results, Result):
            return results.frame()
        return [result.frame() for result in results]


def _positive_integer(value: object, name: str) -> int:
    """Validate one positive integer downloader setting.

    Args:
        value: The proposed setting value.
        name: The setting name used in errors.

    Returns:
        The validated positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_number(value: object, name: str) -> float:
    """Validate one positive finite numeric setting.

    Args:
        value: The proposed setting value.
        name: The setting name used in errors.

    Returns:
        The validated floating-point value.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if converted <= 0 or not math.isfinite(converted):
        raise ValueError(f"{name} must be positive and finite")
    return converted


def _boolean(value: object, name: str) -> bool:
    """Validate one strict Boolean request option.

    Args:
        value: The proposed option value.
        name: The option name used in errors.

    Returns:
        The validated Boolean.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a Boolean")
    return value
