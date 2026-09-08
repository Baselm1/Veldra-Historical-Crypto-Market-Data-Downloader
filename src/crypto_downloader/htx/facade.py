"""Provide the public facade for HTX historical data."""

from datetime import date, datetime
from pathlib import Path
from typing import Literal, overload

import pandas as pd

from crypto_downloader.core.download import _validate_settings
from crypto_downloader.core.engine import RetrievalEngine
from crypto_downloader.core.inspection import (
    discover_availability as _discover_availability,
    find_markets as _find_markets,
    get_availability as _get_availability,
    get_markets as _get_markets,
)
from crypto_downloader.core.models import Availability, Market
from crypto_downloader.htx.connector import HTXConnector
from crypto_downloader.htx.datasets import HTXDataset, get_dataset

type DateInput = str | date | datetime
type ColumnSelection = list[str] | dict[str, str] | None
type Product = Literal["spot", "linear_swap", "coin_swap"]
type FuturesProduct = Literal["linear_swap", "coin_swap"]
type GapPolicy = Literal["forward", "backward", "nan", "keep", "raise"]


class HTX:
    """Provide declarative access to HTX historical data."""

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
        """Create a configured HTX service without performing I/O.

        Args:
            data_dir: The directory containing the catalog and Parquet cache.
            config_path: An optional TOML file overriding installed defaults.
            earliest_date: An optional history boundary or ``"all"``.
            max_workers: The maximum HTX-wide concurrent archive workers.
            discovery_tail_days: Recent active-market days rescanned per request.
            market_refresh_hours: Hours before cached markets become stale.
            timeout: The timeout for each HTX HTTP attempt in seconds.
            retries: The retries allowed after the first HTTP attempt.
            backoff: The initial exponential retry delay in seconds.
            progress: Whether calls show Rich status and progress output.
        """
        _validate_settings(timeout, retries, backoff)
        if not isinstance(progress, bool):
            raise TypeError("progress must be a Boolean")
        self._downloader = RetrievalEngine(
            data_dir,
            source=HTXConnector(timeout=timeout, retries=retries, backoff=backoff),
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
        """Return the resolved catalog and Parquet root.

        Returns:
            The configured absolute data directory.
        """
        return self._downloader.data_dir

    @property
    def earliest_date(self) -> date | None:
        """Return the configured usable history boundary.

        Returns:
            The earliest usable UTC day, or ``None`` for all history.
        """
        return self._downloader.earliest_date

    @property
    def kline_base_interval(self) -> str:
        """Return the HTX Kline interval stored in the cache.

        Returns:
            The configured one-minute archive interval.
        """
        return self._downloader.kline_base_interval

    @property
    def max_workers(self) -> int:
        """Return the facade-wide worker ceiling.

        Returns:
            The maximum concurrent workers for one call.
        """
        return self._downloader.max_workers

    @overload
    def get_klines(
        self,
        pairs: str,
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Describe the return type for one Kline pair."""
        ...

    @overload
    def get_klines(
        self,
        pairs: list[str],
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> list[pd.DataFrame]:
        """Describe the return type for several Kline pairs."""
        ...

    def get_klines(
        self,
        pairs: str | list[str],
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Return HTX trading candles for one or several pairs.

        Args:
            pairs: One native/normalized pair or an ordered pair list.
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
        return self._downloader.get_data(
            pairs,
            start,
            end,
            product=product,
            dataset="klines",
            interval=interval,
            desired_columns=columns,
            refresh=refresh,
            offline=offline,
            gap_policy=gap_policy,
            progress=self._progress,
        )

    @overload
    def get_trades(
        self,
        pairs: str,
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Describe the return type for one trade pair."""
        ...

    @overload
    def get_trades(
        self,
        pairs: list[str],
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> list[pd.DataFrame]:
        """Describe the return type for several trade pairs."""
        ...

    def get_trades(
        self,
        pairs: str | list[str],
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Return HTX individual trades for one or several pairs.

        Args:
            pairs: One native/normalized pair or an ordered pair list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: Spot or one of the two perpetual products.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One trade DataFrame or an ordered DataFrame list.
        """
        return self._downloader.get_data(
            pairs,
            start,
            end,
            product=product,
            dataset="trades",
            desired_columns=columns,
            refresh=refresh,
            offline=offline,
            progress=self._progress,
        )

    def _get_reference_klines(
        self,
        dataset: Literal["index_price_klines", "mark_price_klines"],
        pairs: str | list[str],
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        interval: str | None,
        columns: ColumnSelection,
        gap_policy: GapPolicy,
        refresh: bool,
        offline: bool,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Return one kind of HTX perpetual reference-price candle.

        Args:
            dataset: The index- or mark-price Kline dataset.
            pairs: One native/normalized pair or an ordered pair list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: The linear- or coin-margined perpetual product.
            interval: The optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for internal missing candles.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One reference-price DataFrame or an ordered DataFrame list.
        """
        return self._downloader.get_data(
            pairs,
            start,
            end,
            product=product,
            dataset=dataset,
            interval=interval,
            desired_columns=columns,
            refresh=refresh,
            offline=offline,
            gap_policy=gap_policy,
            progress=self._progress,
        )

    @overload
    def get_index_price_klines(
        self,
        pairs: str,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Describe the return type for one index-price Kline pair."""
        ...

    @overload
    def get_index_price_klines(
        self,
        pairs: list[str],
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> list[pd.DataFrame]:
        """Describe the return type for several index-price Kline pairs."""
        ...

    def get_index_price_klines(
        self,
        pairs: str | list[str],
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Return HTX perpetual index-price candles.

        Args:
            pairs: One native/normalized pair or an ordered pair list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: The linear- or coin-margined perpetual product.
            interval: The optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for internal missing candles.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One index-price DataFrame or an ordered DataFrame list.
        """
        return self._get_reference_klines(
            "index_price_klines",
            pairs,
            start,
            end,
            product=product,
            interval=interval,
            columns=columns,
            gap_policy=gap_policy,
            refresh=refresh,
            offline=offline,
        )

    @overload
    def get_mark_price_klines(
        self,
        pairs: str,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Describe the return type for one mark-price Kline pair."""
        ...

    @overload
    def get_mark_price_klines(
        self,
        pairs: list[str],
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> list[pd.DataFrame]:
        """Describe the return type for several mark-price Kline pairs."""
        ...

    def get_mark_price_klines(
        self,
        pairs: str | list[str],
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "forward",
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Return HTX perpetual mark-price candles.

        Args:
            pairs: One native/normalized pair or an ordered pair list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: The linear- or coin-margined perpetual product.
            interval: The optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            gap_policy: The behavior for internal missing candles.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One mark-price DataFrame or an ordered DataFrame list.
        """
        return self._get_reference_klines(
            "mark_price_klines",
            pairs,
            start,
            end,
            product=product,
            interval=interval,
            columns=columns,
            gap_policy=gap_policy,
            refresh=refresh,
            offline=offline,
        )

    @overload
    def get_funding_rates(
        self,
        pairs: str,
        start: DateInput,
        end: DateInput,
        *,
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Describe the return type for one funding-rate pair."""
        ...

    @overload
    def get_funding_rates(
        self,
        pairs: list[str],
        start: DateInput,
        end: DateInput,
        *,
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> list[pd.DataFrame]:
        """Describe the return type for several funding-rate pairs."""
        ...

    def get_funding_rates(
        self,
        pairs: str | list[str],
        start: DateInput,
        end: DateInput,
        *,
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Return HTX linear-swap funding-rate observations.

        Args:
            pairs: One native/normalized pair or an ordered pair list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One funding-rate DataFrame or an ordered DataFrame list.
        """
        return self._downloader.get_data(
            pairs,
            start,
            end,
            product="linear_swap",
            dataset="funding_rates",
            desired_columns=columns,
            refresh=refresh,
            offline=offline,
            progress=self._progress,
        )

    @overload
    def get_order_book_updates(
        self,
        pairs: str,
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Describe the return type for one order-book pair."""
        ...

    @overload
    def get_order_book_updates(
        self,
        pairs: list[str],
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> list[pd.DataFrame]:
        """Describe the return type for several order-book pairs."""
        ...

    def get_order_book_updates(
        self,
        pairs: str | list[str],
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Return HTX order-book snapshots and incremental level updates.

        Args:
            pairs: One native/normalized pair or an ordered pair list.
            start: The inclusive request start.
            end: The inclusive date or exclusive timestamp request end.
            product: Spot or one of the two perpetual products.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether to repeat complete discovery for the range.
            offline: Whether to forbid all source requests.

        Returns:
            One order-book update DataFrame or an ordered DataFrame list.
        """
        return self._downloader.get_data(
            pairs,
            start,
            end,
            product=product,
            dataset="order_book_updates",
            desired_columns=columns,
            refresh=refresh,
            offline=offline,
            progress=self._progress,
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
        """Return HTX markets matching optional exact filters.

        Args:
            product: Spot or one of the two perpetual products.
            status: An optional native HTX status.
            quote_asset: An optional exact quote asset.
            sort_by: Native symbol or rolling Spot quote-volume ordering.
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
        """Return exact, prefix, and fuzzy HTX market matches.

        Args:
            query: The native or normalized market text to find.
            product: An optional product restriction.
            status: An optional native HTX status.
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
        dataset: HTXDataset,
        interval: str | None = None,
    ) -> Availability:
        """Return already-cataloged remote and local dataset coverage.

        Args:
            pair: The native or normalized HTX market.
            product: Spot or one of the two perpetual products.
            dataset: The HTX dataset to inspect.
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
        dataset: HTXDataset,
        interval: str | None = None,
        refresh: bool = False,
    ) -> Availability:
        """Discover bounded HTX coverage without downloading archives.

        Args:
            pair: The native or normalized HTX market.
            start: The inclusive discovery start.
            end: The inclusive date or exclusive timestamp discovery end.
            product: Spot or one of the two perpetual products.
            dataset: The HTX dataset to inspect.
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
