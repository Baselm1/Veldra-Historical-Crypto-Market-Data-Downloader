"""Provide the public facade for Gate historical market data."""

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
from veldra.gate.connector import GateConnector
from veldra.gate.datasets import GateDataset, get_dataset

type DateInput = str | date | datetime
type ColumnSelection = list[str] | dict[str, str] | None
type PairInput = str | list[str]
type FrameOutput = pd.DataFrame | list[pd.DataFrame]
type ResultOutput = Result | list[Result]
type Product = Literal["spot", "um", "cm"]
type FuturesProduct = Literal["um", "cm"]
type GapPolicy = Literal["forward", "backward", "nan", "keep", "raise"]


class Gate:
    """Provide declarative access to Gate historical market data."""

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
        """Create a configured Gate service without performing I/O.

        Args:
            data_dir: The catalog and Parquet cache directory.
            config_path: An optional TOML file overriding installed defaults.
            earliest_date: An optional history boundary or ``"all"``.
            max_workers: The maximum concurrent archive workers.
            discovery_tail_days: Recent active-market days rescanned per request.
            market_refresh_hours: Hours before cached markets become stale.
            timeout: The timeout for each Gate HTTP attempt in seconds.
            retries: The retries allowed after the first HTTP attempt.
            backoff: The initial exponential retry delay in seconds.
            progress: Whether calls show Rich progress output.
        """
        _validate_settings(timeout, retries, backoff)
        if not isinstance(progress, bool):
            raise TypeError("progress must be a Boolean")
        self._downloader = RetrievalEngine(
            data_dir,
            source=GateConnector(timeout=timeout, retries=retries, backoff=backoff),
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
        """Return the one-minute ordinary Kline interval stored locally."""
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
        product: Product,
        dataset: GateDataset,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> ResultOutput:
        """Return Gate data together with structured retrieval reports.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: Spot, USDT-margined, or BTC-margined Futures.
            dataset: The supported Gate dataset to retrieve.
            interval: The optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for internal missing candles.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One structured result or an ordered result list.
        """
        effective_gap_policy = (
            "keep" if dataset == "klines" and gap_policy is None else gap_policy
        )
        return self._downloader.get_results(
            pairs,
            start,
            end,
            product=product,
            dataset=dataset,
            interval=interval,
            desired_columns=columns,
            gap_policy=effective_gap_policy,
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
        product: Product,
        dataset: GateDataset,
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
            product=product,
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
        product: Product = "spot",
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "keep",
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Gate trading candles for one or several markets.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: Spot, USDT-margined, or BTC-margined Futures.
            interval: The Kline interval, including native Futures ``10s``.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for sparse or missing candles.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One Kline DataFrame or an ordered DataFrame list.
        """
        return self._retrieve(
            pairs,
            start,
            end,
            product=product,
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
        product: Product = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Gate individual fills for one or several markets.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: Spot, USDT-margined, or BTC-margined Futures.
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
            product=product,
            dataset="trades",
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def _order_books(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: Product,
        dataset: Literal["order_book_updates", "order_book_snapshots"],
        columns: ColumnSelection,
        refresh: bool,
        offline: bool,
    ) -> FrameOutput:
        """Return one kind of Gate order-book history."""
        return self._retrieve(
            pairs,
            start,
            end,
            product=product,
            dataset=dataset,
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_order_book_updates(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Gate order-book price-level changes.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: Spot, USDT-margined, or BTC-margined Futures.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One update DataFrame or an ordered DataFrame list.
        """
        return self._order_books(
            pairs,
            start,
            end,
            product=product,
            dataset="order_book_updates",
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_order_book_snapshots(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return complete Gate order-book states with nested bid/ask levels.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: Spot, USDT-margined, or BTC-margined Futures.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One snapshot DataFrame or an ordered DataFrame list.
        """
        return self._order_books(
            pairs,
            start,
            end,
            product=product,
            dataset="order_book_snapshots",
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def _reference(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        dataset: Literal["mark_prices", "funding_rates", "funding_rate_updates"],
        columns: ColumnSelection,
        refresh: bool,
        offline: bool,
    ) -> FrameOutput:
        """Return one perpetual reference-price or funding history."""
        return self._retrieve(
            pairs,
            start,
            end,
            product=product,
            dataset=dataset,
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_mark_prices(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Gate perpetual mark, index, and last-price observations.

        Args:
            pairs: One native/normalized perpetual market or an ordered list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: USDT- or BTC-margined Futures.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One mark-price DataFrame or an ordered DataFrame list.
        """
        return self._reference(
            pairs,
            start,
            end,
            product=product,
            dataset="mark_prices",
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_funding_rates(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Gate perpetual funding rates applied at interval boundaries.

        Args:
            pairs: One native/normalized perpetual market or an ordered list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: USDT- or BTC-margined Futures.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One applied-funding DataFrame or an ordered DataFrame list.
        """
        return self._reference(
            pairs,
            start,
            end,
            product=product,
            dataset="funding_rates",
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_funding_rate_updates(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Gate expected next-interval funding-rate updates.

        Args:
            pairs: One native/normalized perpetual market or an ordered list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: USDT- or BTC-margined Futures.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One projected-funding DataFrame or an ordered DataFrame list.
        """
        return self._reference(
            pairs,
            start,
            end,
            product=product,
            dataset="funding_rate_updates",
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_markets(
        self,
        *,
        product: Product = "spot",
        status: str | None = None,
        quote_asset: str | None = None,
        sort_by: Literal["symbol", "quote_volume"] = "symbol",
        limit: int | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> list[Market]:
        """Return Gate markets matching optional exact filters.

        Args:
            product: Spot, USDT-margined, or BTC-margined Futures.
            status: An optional native Gate status.
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
            product=product,
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
        product: Product | None = None,
        status: str | None = None,
        quote_asset: str | None = None,
        limit: int = 10,
        refresh: bool = False,
        offline: bool = False,
    ) -> list[Market]:
        """Return exact, prefix, and fuzzy Gate market matches.

        Args:
            query: The native or normalized market text to find.
            product: An optional product restriction.
            status: An optional native Gate status.
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
            product=product,
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
        product: Product,
        dataset: GateDataset,
        interval: str | None = None,
    ) -> Availability:
        """Return already-cataloged remote and local Gate coverage.

        Args:
            pair: The native or normalized Gate market.
            product: Spot, USDT-margined, or BTC-margined Futures.
            dataset: The Gate dataset to inspect.
            interval: An optional Kline output interval.

        Returns:
            Known coverage without making a network request.
        """
        return _get_availability(
            self._downloader,
            pair,
            product=product,
            dataset=dataset,
            interval=interval,
        )

    def discover_availability(
        self,
        pair: str,
        start: DateInput,
        end: DateInput,
        *,
        product: Product,
        dataset: GateDataset,
        interval: str | None = None,
        refresh: bool = False,
    ) -> Availability:
        """Discover bounded Gate coverage without downloading archives.

        Args:
            pair: The native or normalized Gate market.
            start: The inclusive discovery start.
            end: The inclusive date or exclusive timestamp discovery end.
            product: Spot, USDT-margined, or BTC-margined Futures.
            dataset: The Gate dataset to inspect.
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
            product=product,
            dataset=dataset,
            interval=interval,
            refresh=refresh,
            progress=self._progress,
        )
