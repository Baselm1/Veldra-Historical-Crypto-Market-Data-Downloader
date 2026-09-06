"""Coordinate public cryptocurrency data requests."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, date, datetime, time, timedelta
import logging
import math
from pathlib import Path
from time import perf_counter

import httpx
import pandas as pd

from .catalog import Catalog, catalog_lock, open_catalog
from .datasets import DatasetSpec, get_dataset
from .display import Reporter
from .models import Market, Result
from .pair import process_pair
from .request import Request
from .request import parse_timestamp
from .source import Source
from .sources.binance import Binance

MINIMUM_HISTORY_DATE = date(2018, 1, 1)
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


def _history_date(value: object) -> date:
    """Parse a configurable UTC history boundary.

    Args:
        value: The proposed date-like earliest boundary.

    Returns:
        A valid UTC calendar date on or after 2018.
    """
    try:
        parsed = parse_timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("earliest_date must be a valid date") from error
    if parsed.time() != time.min or parsed.date() < MINIMUM_HISTORY_DATE:
        raise ValueError("earliest_date must be a UTC day on or after 2018-01-01")
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
    earliest_date: date,
    today: date,
    *,
    refresh: bool,
    offline: bool,
    discovery_tail_days: int,
    max_workers: int,
    reporter: Reporter,
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
        earliest_date: The configured global history boundary.
        today: The current UTC day.
        refresh: Whether source metadata should be refreshed.
        offline: Whether source access is forbidden.
        discovery_tail_days: Recent active-market days to revisit.
        max_workers: Daily ingestion workers assigned to this pair.
        reporter: The optional Rich activity reporter.

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
    )


def _process_pairs(
    source: Source,
    catalog_path: Path,
    client: httpx.Client,
    data_dir: Path,
    markets: list[Market],
    request: Request,
    dataset: DatasetSpec,
    earliest_date: date,
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
        earliest_date: The configured global history boundary.
        today: The current UTC day.
        refresh: Whether source metadata should be refreshed.
        offline: Whether source access is forbidden.
        discovery_tail_days: Recent active-market days to revisit.
        max_workers: The total source-wide concurrency budget.
        reporter: The optional Rich activity reporter.

    Returns:
        Results in the caller's original pair order.
    """
    unique_pairs = list(dict.fromkeys(request.pairs))
    pair_workers = min(len(unique_pairs), max_workers)
    ingestion_workers = max(1, max_workers // pair_workers)
    LOGGER.info(
        "Pair workflows planned: pairs=%d workers=%d ingestion_workers_per_pair=%d",
        len(unique_pairs),
        pair_workers,
        ingestion_workers,
    )
    with ExitStack() as stack:
        catalogs = [
            stack.enter_context(open_catalog(catalog_path)) for _ in unique_pairs
        ]
        arguments = list(zip(catalogs, unique_pairs))

        def run(argument: tuple[Catalog, str]) -> Result:
            """Run one catalog and pair tuple.

            Args:
                argument: The dedicated catalog and requested pair.

            Returns:
                The completed pair result.
            """
            catalog, pair = argument
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
                max_workers=ingestion_workers,
                reporter=reporter,
            )

        if pair_workers == 1:
            unique_results = [run(arguments[0])]
        else:
            with ThreadPoolExecutor(max_workers=pair_workers) as executor:
                unique_results = list(executor.map(run, arguments))
    by_pair = dict(zip(unique_pairs, unique_results))
    return [by_pair[pair] for pair in request.pairs]


class Downloader:
    """Provide reusable downloader paths, source, and HTTP settings."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        *,
        source: Source | None = None,
        transport: httpx.BaseTransport | None = None,
        earliest_date: object = date(2020, 1, 1),
        max_workers: int = 32,
        discovery_tail_days: int = 7,
        market_refresh_hours: float = 24.0,
    ) -> None:
        """Create a reusable imported downloader service.

        Args:
            data_dir: The directory containing the catalog and Parquet cache.
            source: The optional source strategy, defaulting to Binance.
            transport: An optional HTTPX transport used for requests.
            earliest_date: The first daily archive date considered by discovery.
            max_workers: The maximum concurrent daily archive downloads.
            discovery_tail_days: Recent active-market days rediscovered per request.
            market_refresh_hours: Hours before market metadata is refreshed again.
        """
        if not isinstance(data_dir, (str, Path)):
            raise TypeError("data_dir must be a path")
        if not str(data_dir).strip():
            raise ValueError("data_dir must not be empty")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.source: Source = source if source is not None else Binance()
        self.transport = transport
        self.earliest_date = _history_date(earliest_date)
        if self.earliest_date >= utc_today():
            raise ValueError("earliest_date must be before today in UTC")
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
        specification = get_dataset(request.product, request.dataset)
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
            with httpx.Client(
                transport=self.transport,
                follow_redirects=True,
                limits=limits,
            ) as client:
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

    async def aget_data(
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
        """Run the DataFrame pipeline without blocking an async event loop.

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
        LOGGER.debug("Async downloader request delegated to a worker thread")
        return await asyncio.to_thread(
            self.get_data,
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


def get_results(
    pairs: object,
    starting_date: object,
    end_date: object,
    *,
    data_dir: str | Path = "data",
    product: object = "spot",
    dataset: object = "klines",
    interval: object = None,
    desired_columns: object = None,
    source: Source | None = None,
    transport: httpx.BaseTransport | None = None,
    earliest_date: object = date(2020, 1, 1),
    max_workers: int = 32,
    discovery_tail_days: int = 7,
    market_refresh_hours: float = 24.0,
    refresh: bool = False,
    offline: bool = False,
    gap_policy: object = None,
    progress: bool = True,
) -> Result | list[Result]:
    """Create a downloader and return requested data with reports.

    Args:
        pairs: One pair string or an ordered list of pair strings.
        starting_date: The requested first date or timestamp.
        end_date: The requested inclusive date or exclusive timestamp.
        data_dir: The directory containing the catalog and Parquet cache.
        product: The source product identifier.
        dataset: The historical dataset identifier.
        interval: The optional output interval.
        desired_columns: Optional selected and renamed columns.
        source: The optional source strategy, defaulting to Binance.
        transport: An optional HTTPX transport used for requests.
        earliest_date: The first daily archive date considered by discovery.
        max_workers: The maximum concurrent daily archive downloads.
        discovery_tail_days: Recent active-market days rediscovered per request.
        market_refresh_hours: Hours before market metadata is refreshed again.
        refresh: Whether to repeat complete resource discovery.
        offline: Whether to use only cataloged markets and cached files.
        gap_policy: The behavior used for internal missing candles.
        progress: Whether to show optional Rich activity.

    Returns:
        One result for string input or an ordered result list.
    """
    return Downloader(
        data_dir,
        source=source,
        transport=transport,
        earliest_date=earliest_date,
        max_workers=max_workers,
        discovery_tail_days=discovery_tail_days,
        market_refresh_hours=market_refresh_hours,
    ).get_results(
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


def get_data(
    pairs: object,
    starting_date: object,
    end_date: object,
    *,
    data_dir: str | Path = "data",
    product: object = "spot",
    dataset: object = "klines",
    interval: object = None,
    desired_columns: object = None,
    source: Source | None = None,
    transport: httpx.BaseTransport | None = None,
    earliest_date: object = date(2020, 1, 1),
    max_workers: int = 32,
    discovery_tail_days: int = 7,
    market_refresh_hours: float = 24.0,
    refresh: bool = False,
    offline: bool = False,
    gap_policy: object = None,
    progress: bool = True,
) -> pd.DataFrame | list[pd.DataFrame]:
    """Create a downloader and return only requested DataFrames.

    Args:
        pairs: One pair string or an ordered list of pair strings.
        starting_date: The requested first date or timestamp.
        end_date: The requested inclusive date or exclusive timestamp.
        data_dir: The directory containing the catalog and Parquet cache.
        product: The source product identifier.
        dataset: The historical dataset identifier.
        interval: The optional output interval.
        desired_columns: Optional selected and renamed columns.
        source: The optional source strategy, defaulting to Binance.
        transport: An optional HTTPX transport used for requests.
        earliest_date: The first daily archive date considered by discovery.
        max_workers: The maximum concurrent daily archive downloads.
        discovery_tail_days: Recent active-market days rediscovered per request.
        market_refresh_hours: Hours before market metadata is refreshed again.
        refresh: Whether to repeat complete resource discovery.
        offline: Whether to use only cataloged markets and cached files.
        gap_policy: The behavior used for internal missing candles.
        progress: Whether to show optional Rich activity.

    Returns:
        One DataFrame for string input or an ordered DataFrame list.
    """
    return Downloader(
        data_dir,
        source=source,
        transport=transport,
        earliest_date=earliest_date,
        max_workers=max_workers,
        discovery_tail_days=discovery_tail_days,
        market_refresh_hours=market_refresh_hours,
    ).get_data(
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


async def aget_data(
    pairs: object,
    starting_date: object,
    end_date: object,
    *,
    data_dir: str | Path = "data",
    product: object = "spot",
    dataset: object = "klines",
    interval: object = None,
    desired_columns: object = None,
    source: Source | None = None,
    transport: httpx.BaseTransport | None = None,
    earliest_date: object = date(2020, 1, 1),
    max_workers: int = 32,
    discovery_tail_days: int = 7,
    market_refresh_hours: float = 24.0,
    refresh: bool = False,
    offline: bool = False,
    gap_policy: object = None,
    progress: bool = True,
) -> pd.DataFrame | list[pd.DataFrame]:
    """Create a downloader and run it without blocking an async event loop.

    Args:
        pairs: One pair string or an ordered list of pair strings.
        starting_date: The requested first date or timestamp.
        end_date: The requested inclusive date or exclusive timestamp.
        data_dir: The directory containing the catalog and Parquet cache.
        product: The source product identifier.
        dataset: The historical dataset identifier.
        interval: The optional output interval.
        desired_columns: Optional selected and renamed columns.
        source: The optional source strategy, defaulting to Binance.
        transport: An optional HTTPX transport used for requests.
        earliest_date: The first daily archive date considered by discovery.
        max_workers: The maximum concurrent daily archive downloads.
        discovery_tail_days: Recent active-market days rediscovered per request.
        market_refresh_hours: Hours before market metadata is refreshed again.
        refresh: Whether to repeat complete resource discovery.
        offline: Whether to use only cataloged markets and cached files.
        gap_policy: The behavior used for internal missing candles.
        progress: Whether to show optional Rich activity.

    Returns:
        One DataFrame for string input or an ordered DataFrame list.
    """
    return await Downloader(
        data_dir,
        source=source,
        transport=transport,
        earliest_date=earliest_date,
        max_workers=max_workers,
        discovery_tail_days=discovery_tail_days,
        market_refresh_hours=market_refresh_hours,
    ).aget_data(
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
