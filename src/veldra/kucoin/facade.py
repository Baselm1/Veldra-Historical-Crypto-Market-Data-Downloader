"""Provide the public facade for KuCoin historical data."""

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
from veldra.core.models import Availability, Market
from veldra.kucoin.connector import KuCoinConnector
from veldra.kucoin.datasets import KuCoinDataset, get_dataset

type DateInput = str | date | datetime
type ColumnSelection = list[str] | dict[str, str] | None
type PairInput = str | list[str]
type FrameOutput = pd.DataFrame | list[pd.DataFrame]
type Product = Literal["spot", "linear_futures", "inverse_futures"]
type FuturesProduct = Literal["linear_futures", "inverse_futures"]
type GapPolicy = Literal["forward", "backward", "nan", "keep", "raise"]


class KuCoin:
    """Provide declarative access to KuCoin historical data."""

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
        """Create a configured KuCoin service without performing I/O.

        Args:
            data_dir: The catalog and Parquet cache directory.
            config_path: An optional TOML file overriding installed defaults.
            earliest_date: An optional history boundary or ``"all"``.
            max_workers: The maximum concurrent archive workers.
            discovery_tail_days: Recent active-market days rescanned per request.
            market_refresh_hours: Hours before cached markets become stale.
            timeout: The timeout for each KuCoin HTTP attempt in seconds.
            retries: The retries allowed after the first HTTP attempt.
            backoff: The initial exponential retry delay in seconds.
            progress: Whether calls show Rich progress output.
        """
        _validate_settings(timeout, retries, backoff)
        if not isinstance(progress, bool):
            raise TypeError("progress must be a Boolean")
        self._downloader = RetrievalEngine(
            data_dir,
            source=KuCoinConnector(timeout=timeout, retries=retries, backoff=backoff),
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
        """Return the one-minute Kline interval stored in the cache."""
        return self._downloader.kline_base_interval

    @property
    def max_workers(self) -> int:
        """Return the facade-wide worker ceiling."""
        return self._downloader.max_workers

    def _retrieve(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: Product,
        dataset: KuCoinDataset,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Delegate one typed request to the shared retrieval engine.

        Args:
            pairs: One market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: The KuCoin product to query.
            dataset: The KuCoin dataset to query.
            interval: The optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for internal missing candles.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One DataFrame or an ordered list matching the input shape.
        """
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
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return KuCoin trading candles for one or several markets.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: Spot or one of the two perpetual products.
            interval: The optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for internal missing candles.
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
        """Return KuCoin individual trades for one or several markets.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: Spot or one of the two perpetual products.
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

    def _reference_klines(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        dataset: Literal["index_price_klines", "mark_price_klines"],
        interval: str | None,
        columns: ColumnSelection,
        gap_policy: GapPolicy,
        refresh: bool,
        offline: bool,
    ) -> FrameOutput:
        """Return one kind of perpetual reference-price candle.

        Args:
            pairs: One market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: The linear- or inverse-margined perpetual product.
            dataset: Index- or mark-price Klines.
            interval: The optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for internal missing candles.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One reference Kline DataFrame or an ordered DataFrame list.
        """
        return self._retrieve(
            pairs,
            start,
            end,
            product=product,
            dataset=dataset,
            interval=interval,
            columns=columns,
            gap_policy=gap_policy,
            refresh=refresh,
            offline=offline,
        )

    def get_index_price_klines(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return KuCoin perpetual index-price candles.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: The linear- or inverse-margined perpetual product.
            interval: The optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for internal missing candles.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One index-price DataFrame or an ordered DataFrame list.
        """
        return self._reference_klines(
            pairs,
            start,
            end,
            product=product,
            dataset="index_price_klines",
            interval=interval,
            columns=columns,
            gap_policy=gap_policy,
            refresh=refresh,
            offline=offline,
        )

    def get_mark_price_klines(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return KuCoin perpetual mark-price candles.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: The linear- or inverse-margined perpetual product.
            interval: The optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for internal missing candles.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One mark-price DataFrame or an ordered DataFrame list.
        """
        return self._reference_klines(
            pairs,
            start,
            end,
            product=product,
            dataset="mark_price_klines",
            interval=interval,
            columns=columns,
            gap_policy=gap_policy,
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
        """Return KuCoin perpetual funding-rate observations.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: The linear- or inverse-margined perpetual product.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One funding-rate DataFrame or an ordered DataFrame list.
        """
        return self._retrieve(
            pairs,
            start,
            end,
            product=product,
            dataset="funding_rates",
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
        """Return nested KuCoin level-50 order-book snapshots.

        Args:
            pairs: One native/normalized market or an ordered market list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: Spot or one of the two perpetual products.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One snapshot DataFrame or an ordered DataFrame list.
        """
        return self._retrieve(
            pairs,
            start,
            end,
            product=product,
            dataset="order_book_snapshots",
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
        """Return KuCoin markets matching optional exact filters.

        Args:
            product: Spot or one of the two perpetual products.
            status: An optional native KuCoin status.
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
        """Return exact, prefix, and fuzzy KuCoin market matches.

        Args:
            query: The native or normalized market text to find.
            product: An optional product restriction.
            status: An optional native KuCoin status.
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
        dataset: KuCoinDataset,
        interval: str | None = None,
    ) -> Availability:
        """Return already-cataloged remote and local KuCoin coverage.

        Args:
            pair: The native or normalized KuCoin market.
            product: Spot or one of the two perpetual products.
            dataset: The KuCoin dataset to inspect.
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
        dataset: KuCoinDataset,
        interval: str | None = None,
        refresh: bool = False,
    ) -> Availability:
        """Discover bounded KuCoin coverage without downloading archives.

        Args:
            pair: The native or normalized KuCoin market.
            start: The inclusive discovery start.
            end: The inclusive date or exclusive timestamp discovery end.
            product: Spot or one of the two perpetual products.
            dataset: The KuCoin dataset to inspect.
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
