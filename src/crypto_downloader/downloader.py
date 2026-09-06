"""Coordinate public cryptocurrency data requests."""

from datetime import UTC, date, datetime, time
import logging
from pathlib import Path
from time import perf_counter

import httpx
import pandas as pd

from .catalog import catalog_lock, open_catalog
from .datasets import get_dataset
from .display import Reporter
from .models import Result
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


class Downloader:
    """Provide reusable downloader paths, source, and HTTP settings."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        *,
        source: Source | None = None,
        transport: httpx.BaseTransport | None = None,
        earliest_date: object = date(2020, 1, 1),
        max_workers: int = 16,
        discovery_tail_days: int = 7,
    ) -> None:
        """Create a reusable imported downloader service.

        Args:
            data_dir: The directory containing the catalog and Parquet cache.
            source: The optional source strategy, defaulting to Binance.
            transport: An optional HTTPX transport used for requests.
            earliest_date: The first daily archive date considered by discovery.
            max_workers: The maximum concurrent daily archive downloads.
            discovery_tail_days: Recent active-market days rediscovered per request.
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
        gap_policy: object = "forward",
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
        specification.resolve_interval(request.interval)
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
        with catalog_lock(catalog_path):
            with (
                open_catalog(catalog_path) as catalog,
                httpx.Client(
                    transport=self.transport,
                    follow_redirects=True,
                ) as client,
            ):
                markets = catalog.markets(self.source.code, request.product)
                if not offline:
                    with reporter.status(
                        f"Refreshing {self.source.code.title()} "
                        f"{request.product} markets"
                    ):
                        markets = self.source.markets(client, request.product)
                    catalog.save_markets(self.source.code, request.product, markets)
                    LOGGER.info(
                        "Market snapshot refreshed: source=%s product=%s markets=%d",
                        self.source.code,
                        request.product,
                        len(markets),
                    )
                    reporter.market_summary(markets, refreshed=True)
                elif not markets:
                    raise RuntimeError("offline mode requires cached market metadata")
                else:
                    LOGGER.info(
                        "Market snapshot loaded from cache: source=%s product=%s "
                        "markets=%d",
                        self.source.code,
                        request.product,
                        len(markets),
                    )
                    reporter.market_summary(markets, refreshed=False)
                results = [
                    process_pair(
                        self.source,
                        catalog,
                        client,
                        self.data_dir,
                        markets,
                        pair,
                        request,
                        specification,
                        self.earliest_date,
                        utc_today(),
                        refresh=refresh,
                        offline=offline,
                        discovery_tail_days=self.discovery_tail_days,
                        max_workers=self.max_workers,
                        reporter=reporter,
                    )
                    for pair in request.pairs
                ]
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
        gap_policy: object = "forward",
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
    max_workers: int = 16,
    discovery_tail_days: int = 7,
    refresh: bool = False,
    offline: bool = False,
    gap_policy: object = "forward",
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
    max_workers: int = 16,
    discovery_tail_days: int = 7,
    refresh: bool = False,
    offline: bool = False,
    gap_policy: object = "forward",
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
