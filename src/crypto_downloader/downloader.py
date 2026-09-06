"""Coordinate public cryptocurrency data requests."""

from datetime import UTC, date, datetime, time
from pathlib import Path

import httpx
import pandas as pd

from .catalog import catalog_lock, open_catalog
from .datasets import get_dataset
from .models import Result
from .pair import process_pair
from .request import Request
from .request import parse_timestamp
from .source import Source
from .sources.binance import Binance

MINIMUM_HISTORY_DATE = date(2018, 1, 1)


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
        )
        specification = get_dataset(request.product, request.dataset)
        specification.resolve_interval(request.interval)
        if request.interval != specification.base_interval:
            raise ValueError("kline resampling is not available yet")
        if request.product not in self.source.products:
            raise ValueError(
                f"unsupported product for {self.source.code}: {request.product}"
            )

        catalog_path = self.data_dir / "catalog.duckdb"
        refresh = _boolean(refresh, "refresh")
        offline = _boolean(offline, "offline")
        if refresh and offline:
            raise ValueError("refresh and offline cannot both be enabled")
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
                    markets = self.source.markets(client, request.product)
                    catalog.save_markets(self.source.code, request.product, markets)
                elif not markets:
                    raise RuntimeError("offline mode requires cached market metadata")
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
                    )
                    for pair in request.pairs
                ]
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
