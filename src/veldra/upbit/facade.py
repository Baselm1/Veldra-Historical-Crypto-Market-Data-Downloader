"""Provide the public facade for Upbit historical data."""

from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from veldra.core.download import _validate_settings
from veldra.core.engine import RetrievalEngine
from veldra.core.inspection import (
    discover_availability as _discover_availability,
    find_markets as _find_markets,
    get_availability as _get_availability,
    get_markets as _get_markets,
)
from veldra.core.models import Availability, Market, Result
from veldra.upbit.connector import UpbitConnector
from veldra.upbit.datasets import UpbitDataset, get_dataset

type DateInput = str | date | datetime
type ColumnSelection = list[str] | dict[str, str] | None
type PairInput = str | list[str]
type FrameOutput = pd.DataFrame | list[pd.DataFrame]
type ResultOutput = Result | list[Result]
type GapPolicy = Literal["forward", "backward", "nan", "keep", "raise"]


class Upbit:
    """Provide declarative access to Upbit historical Spot data."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        *,
        config_path: str | Path | None = None,
        earliest_date: DateInput | Literal["all"] | None = None,
        max_workers: int = 32,
        discovery_tail_days: int = 7,
        market_refresh_hours: float = 24.0,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 0.5,
        progress: bool = True,
    ) -> None:
        """Create a configured Upbit service without performing I/O.

        Args:
            data_dir: The catalog and Parquet cache directory.
            config_path: An optional TOML file overriding installed defaults.
            earliest_date: An optional history boundary or ``"all"``.
            max_workers: The maximum concurrent archive workers.
            discovery_tail_days: Recent active-market days rescanned per request.
            market_refresh_hours: Hours before cached markets become stale.
            timeout: The timeout for each Upbit HTTP attempt in seconds.
            retries: The retries allowed after the first HTTP attempt.
            backoff: The initial exponential retry delay in seconds.
            progress: Whether calls show Rich progress output.
        """
        _validate_settings(timeout, retries, backoff)
        if not isinstance(progress, bool):
            raise TypeError("progress must be a Boolean")
        self._downloader = RetrievalEngine(
            data_dir,
            source=UpbitConnector(timeout=timeout, retries=retries, backoff=backoff),
            dataset_resolver=get_dataset,
            config_path=config_path,
            earliest_date=earliest_date,
            max_workers=max_workers,
            discovery_tail_days=discovery_tail_days,
            market_refresh_hours=market_refresh_hours,
        )
        self._progress = progress

    @property
    def data_dir(self) -> Path:
        """Return the resolved catalog and Parquet root."""
        return self._downloader.data_dir

    @property
    def earliest_date(self) -> date | None:
        """Return the configured usable history boundary."""
        return self._downloader.earliest_date

    @property
    def kline_base_interval(self) -> str:
        """Return the one-minute coarse Kline interval stored in the cache."""
        return self._downloader.kline_base_interval

    @property
    def max_workers(self) -> int:
        """Return the facade-wide worker ceiling."""
        return self._downloader.max_workers

    def get_results(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        dataset: UpbitDataset,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> ResultOutput:
        """Return Upbit data together with structured retrieval reports.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            dataset: Klines or individual trades.
            interval: The optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for internal missing candles.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One structured result or an ordered result list.
        """
        return self._downloader.get_results(
            pairs,
            start,
            end,
            product="spot",
            dataset=dataset,
            interval=interval,
            desired_columns=columns,
            gap_policy=gap_policy,
            refresh=refresh,
            offline=offline,
            progress=self._progress,
        )

    def _retrieve(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        dataset: UpbitDataset,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Delegate one typed request to the shared retrieval engine."""
        return self._downloader.get_data(
            pairs,
            start,
            end,
            product="spot",
            dataset=dataset,
            interval=interval,
            desired_columns=columns,
            gap_policy=gap_policy,
            refresh=refresh,
            offline=offline,
            progress=self._progress,
        )

    def get_klines(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "keep",
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Upbit trading candles for one or several Spot markets.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            interval: The output interval; ``1s`` uses second archives.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for naturally sparse candle intervals.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One Kline DataFrame or an ordered DataFrame list.
        """
        return self._retrieve(
            pairs,
            start,
            end,
            dataset="klines",
            interval=interval,
            columns=columns,
            gap_policy=gap_policy,
            refresh=refresh,
            offline=offline,
        )

    def get_trades(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Upbit individual trades for one or several Spot markets.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One trade DataFrame or an ordered DataFrame list.
        """
        return self._retrieve(
            pairs,
            start,
            end,
            dataset="trades",
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_markets(
        self,
        *,
        status: str | None = None,
        quote_asset: str | None = None,
        sort_by: Literal["symbol", "quote_volume"] = "symbol",
        limit: int | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> list[Market]:
        """Return Upbit Spot markets matching optional exact filters.

        Args:
            status: An optional native Upbit status.
            quote_asset: An optional exact quote asset.
            sort_by: Native symbol or rolling quote-volume ordering.
            limit: An optional positive maximum result count.
            refresh: Whether to replace cached market metadata now.
            offline: Whether to require cached market metadata.

        Returns:
            Matching immutable markets in the requested order.
        """
        return _get_markets(
            self._downloader,
            product="spot",
            status=status,
            quote_asset=quote_asset,
            sort_by=sort_by,
            limit=limit,
            refresh=refresh,
            offline=offline,
            progress=self._progress,
        )

    def find_markets(
        self,
        query: str,
        *,
        status: str | None = None,
        quote_asset: str | None = None,
        limit: int = 10,
        refresh: bool = False,
        offline: bool = False,
    ) -> list[Market]:
        """Return exact, prefix, and fuzzy Upbit Spot market matches.

        Args:
            query: The native or normalized market text to find.
            status: An optional native Upbit status.
            quote_asset: An optional exact quote asset.
            limit: The maximum number of matches to return.
            refresh: Whether to replace cached market metadata now.
            offline: Whether to require cached market metadata.

        Returns:
            Ranked immutable matches without automatic substitution.
        """
        return _find_markets(
            self._downloader,
            query,
            product="spot",
            status=status,
            quote_asset=quote_asset,
            limit=limit,
            refresh=refresh,
            offline=offline,
            progress=self._progress,
        )

    def get_availability(
        self,
        pair: str,
        *,
        dataset: UpbitDataset,
        interval: str | None = None,
    ) -> Availability:
        """Return already-cataloged remote and local Upbit coverage.

        Args:
            pair: The native or normalized Upbit market.
            dataset: Klines or individual trades.
            interval: An optional Kline output interval.

        Returns:
            Known coverage without making a network request.
        """
        return _get_availability(
            self._downloader,
            pair,
            product="spot",
            dataset=dataset,
            interval=interval,
        )

    def discover_availability(
        self,
        pair: str,
        start: DateInput,
        end: DateInput,
        *,
        dataset: UpbitDataset,
        interval: str | None = None,
        refresh: bool = False,
    ) -> Availability:
        """Discover bounded Upbit coverage without downloading archives.

        Args:
            pair: The native or normalized Upbit market.
            start: The inclusive discovery start.
            end: The inclusive date or exclusive timestamp discovery end.
            dataset: Klines or individual trades.
            interval: An optional Kline output interval.
            refresh: Whether to rescan the complete bounded range.

        Returns:
            Updated remote and local coverage for the dataset.
        """
        return _discover_availability(
            self._downloader,
            pair,
            start,
            end,
            product="spot",
            dataset=dataset,
            interval=interval,
            refresh=refresh,
            progress=self._progress,
        )
