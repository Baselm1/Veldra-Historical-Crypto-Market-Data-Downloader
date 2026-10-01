"""Coordinate Bybit archive retrieval and cached public REST histories."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

from veldra.bybit.client import BybitClient
from veldra.bybit.connector import BybitConnector
from veldra.bybit.datasets import get_dataset
from veldra.bybit.history import BybitHistory
from veldra.bybit.rest import BybitRESTCache
from veldra.core.catalog import catalog_lock, open_catalog
from veldra.core.engine import RetrievalEngine
from veldra.core.subjects import DataSubject

type FrameOutput = pd.DataFrame | list[pd.DataFrame]


class BybitService:
    """Provide one source-owned service over Bybit's two transports."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        *,
        config_path: str | Path | None = None,
        earliest_date: object = None,
        max_workers: int = 32,
        discovery_tail_days: int = 7,
        market_refresh_hours: float = 24.0,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create archive, REST, cache, and rate-limit collaborators."""
        connector = BybitConnector(timeout=timeout, retries=retries, backoff=backoff)
        self.downloader = RetrievalEngine(
            data_dir,
            source=connector,
            dataset_resolver=get_dataset,
            transport=transport,
            config_path=config_path,
            earliest_date=earliest_date,
            max_workers=max_workers,
            discovery_tail_days=discovery_tail_days,
            market_refresh_hours=market_refresh_hours,
        )
        self._http = httpx.Client(transport=transport, follow_redirects=True)
        self.client = BybitClient(
            client=self._http,
            limiter=connector.limiter,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
        self.history = BybitHistory(self.client)

    @property
    def data_dir(self) -> Path:
        """Return the resolved catalog and Parquet root."""
        return self.downloader.data_dir

    def archives(
        self,
        pairs: object,
        start: object,
        end: object,
        *,
        product: object,
        dataset: object,
        columns: object = None,
        refresh: bool = False,
        offline: bool = False,
        progress: bool = True,
    ) -> FrameOutput:
        """Return one or several archive-backed Bybit datasets."""
        return self.downloader.get_data(
            pairs,
            start,
            end,
            product=product,
            dataset=dataset,
            desired_columns=columns,
            refresh=refresh,
            offline=offline,
            progress=progress,
        )

    def _rest(
        self,
        dataset: str,
        subject: DataSubject,
        start: datetime,
        end: datetime,
        *,
        product: str,
        fetch: Callable[[], pd.DataFrame],
        interval: str | None = None,
        variant: str | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return one REST dataset through its persistent range cache."""
        catalog_path = self.data_dir / "catalog.duckdb"
        with catalog_lock(catalog_path):
            with open_catalog(catalog_path) as catalog:
                return BybitRESTCache(catalog, self.data_dir).get(
                    dataset,
                    subject,
                    start,
                    end,
                    product=product,
                    fetch=fetch,
                    interval=interval,
                    variant=variant,
                    refresh=refresh,
                    offline=offline,
                )

    def klines(
        self,
        symbol: str,
        product: str,
        dataset: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return cached market, mark, index, or premium Klines."""
        return self._rest(
            dataset,
            DataSubject("instrument", symbol),
            start,
            end,
            product=product,
            interval=interval,
            refresh=refresh,
            offline=offline,
            fetch=lambda: self.history.klines(
                symbol, product, dataset, interval, start, end
            ),
        )

    def funding_rates(
        self,
        symbol: str,
        product: str,
        start: datetime,
        end: datetime,
        *,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return cached perpetual funding settlements."""
        return self._rest(
            "funding_rates",
            DataSubject("instrument", symbol),
            start,
            end,
            product=product,
            refresh=refresh,
            offline=offline,
            fetch=lambda: self.history.funding_rates(symbol, product, start, end),
        )

    def positions(
        self,
        symbol: str,
        product: str,
        dataset: str,
        period: str,
        start: datetime,
        end: datetime,
        *,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return cached open-interest or long/short histories."""
        return self._rest(
            dataset,
            DataSubject("instrument", symbol),
            start,
            end,
            product=product,
            variant=period,
            refresh=refresh,
            offline=offline,
            fetch=lambda: self.history.positions(
                symbol, product, dataset, period, start, end
            ),
        )

    def volatility(
        self,
        base_coin: str,
        period: int,
        start: datetime,
        end: datetime,
        *,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return cached Option historical volatility."""
        return self._rest(
            "historical_volatility",
            DataSubject("instrument_family", base_coin),
            start,
            end,
            product="options",
            variant=str(period),
            refresh=refresh,
            offline=offline,
            fetch=lambda: self.history.volatility(base_coin, period, start, end),
        )

    def delivery_prices(
        self,
        symbol: str,
        product: str,
        start: datetime,
        end: datetime,
        *,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return cached delivery records for one expired contract."""
        return self._rest(
            "delivery_prices",
            DataSubject("instrument", symbol),
            start,
            end,
            product=product,
            refresh=refresh,
            offline=offline,
            fetch=lambda: self.history.delivery_prices(symbol, product, start, end),
        )

    def close(self) -> None:
        """Close the facade-owned HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "BybitService":
        """Return this configured service from a context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the HTTP pool when leaving a context manager."""
        self.close()
