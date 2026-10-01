"""Provide the public facade for Bitget historical market data."""

from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

from veldra.bitget.client import BitgetClient, BitgetResponseError
from veldra.bitget.connector import BitgetConnector
from veldra.bitget.datasets import BitgetDataset, BitgetProduct, get_dataset
from veldra.bitget.reference import BitgetReferenceService, candle_frame, funding_frame
from veldra.bitget.rest import BitgetRESTCache
from veldra.core.catalog import catalog_lock, open_catalog
from veldra.core.download import _validate_settings
from veldra.core.engine import RetrievalEngine
from veldra.core.inspection import discover_availability as _discover_availability
from veldra.core.inspection import find_markets as _find_markets
from veldra.core.inspection import get_availability as _get_availability
from veldra.core.inspection import get_markets as _get_markets
from veldra.core.models import Availability, Market, Message, Result
from veldra.core.request import parse_timestamp
from veldra.core.request import parse_pairs
from veldra.core.subjects import DataSubject

type DateInput = str | date | datetime
type PairInput = str | list[str]
type FrameOutput = pd.DataFrame | list[pd.DataFrame]
type ColumnSelection = list[str] | dict[str, str] | None
type GapPolicy = Literal["forward", "backward", "nan", "keep", "raise"]
type FuturesProduct = Literal["usdt_futures", "usdc_futures", "coin_futures"]
type ReferenceDataset = Literal[
    "mark_price_klines", "index_price_klines", "premium_index_klines"
]


def _range(start: DateInput, end: DateInput) -> tuple[datetime, datetime]:
    """Return a valid half-open range, expanding inclusive date ends."""
    first = parse_timestamp(start)
    last = parse_timestamp(end)
    date_end = (isinstance(end, date) and not isinstance(end, datetime)) or (
        isinstance(end, str) and len(end.strip()) == 10
    )
    if date_end:
        last += timedelta(days=1)
    if first >= last:
        raise ValueError("start must be before end")
    return first, last


def _reference_pair(value: object) -> str:
    """Return one safe native Bitget reference-market symbol.

    Args:
        value: Caller-supplied Futures symbol.

    Returns:
        The validated uppercase native symbol.
    """
    pairs, single = parse_pairs(value)
    if not single:
        raise TypeError("pair must be a string")
    symbol = pairs[0].upper()
    if not symbol.isascii() or not symbol.isalnum():
        raise ValueError("pair must contain only ASCII letters and digits")
    return symbol


def _reported_reference(
    frame: pd.DataFrame,
    pair: str,
    requested: tuple[datetime, datetime],
    product: str,
    dataset: str,
    *,
    error: Message | None = None,
) -> pd.DataFrame:
    """Attach the normal Veldra report to one reference-history frame.

    Args:
        frame: Canonical source rows, possibly empty.
        pair: Validated native Futures symbol.
        requested: Exact half-open UTC request range.
        product: Futures settlement product.
        dataset: Reference-history dataset.
        error: Optional deterministic retrieval failure.

    Returns:
        The same tabular data with a serializable download report.
    """
    available = None if frame.empty else requested
    result = Result(
        pair,
        frame,
        requested,
        used_range=available,
        available_range=available,
        source="bitget",
        product=product,
        dataset=dataset,
        gap_policy=None,
    )
    if error is not None:
        result.errors.append(error)
    elif frame.empty:
        result.errors.append(
            Message(
                "range_unavailable",
                "Bitget returned no rows for the requested reference-data range.",
            )
        )
    return result.frame()


class Bitget:
    """Provide declarative access to Bitget historical market data."""

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
        """Create a configured Bitget service without performing I/O."""
        _validate_settings(timeout, retries, backoff)
        if not isinstance(progress, bool):
            raise TypeError("progress must be a Boolean")
        connector = BitgetConnector(timeout=timeout, retries=retries, backoff=backoff)
        self._downloader = RetrievalEngine(
            data_dir,
            source=connector,
            dataset_resolver=get_dataset,
            config_path=config_path,
            earliest_date=earliest_date,
            max_workers=max_workers,
            discovery_tail_days=discovery_tail_days,
            market_refresh_hours=market_refresh_hours,
        )
        self._reference_client = BitgetClient(
            limiter=connector.limiter,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
        self._reference = BitgetReferenceService(self._reference_client)
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
        """Return the physical Kline interval stored from daily archives."""
        return self._downloader.kline_base_interval

    @property
    def max_workers(self) -> int:
        """Return the facade-wide archive worker ceiling."""
        return self._downloader.max_workers

    def __enter__(self) -> "Bitget":
        """Return this open facade from a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Close the facade-owned REST client on context exit.

        Args:
            exc_type: Exception type raised inside the context, when present.
            exc_value: Exception raised inside the context, when present.
            traceback: Traceback associated with the exception, when present.
        """
        del exc_type, exc_value, traceback
        self.close()

    def _rest_data(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        *,
        product: str,
        dataset: str,
        interval: str | None,
        fetch: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """Return Bitget REST rows from local Parquet or fetch them once.

        Args:
            pair: Validated native Futures symbol.
            start: Inclusive UTC request boundary.
            end: Exclusive UTC request boundary.
            product: Bitget Futures settlement product.
            dataset: Canonical REST dataset name.
            interval: Stored Kline interval, or ``None`` for event data.
            fetch: Source request used only when local rows are absent.

        Returns:
            Exact canonical rows for the requested range.
        """
        catalog_path = self.data_dir / "catalog.duckdb"
        with catalog_lock(catalog_path):
            with open_catalog(catalog_path) as catalog:
                return BitgetRESTCache(catalog, self.data_dir).get(
                    dataset,
                    DataSubject("instrument", pair),
                    start,
                    end,
                    product=product,
                    interval=interval,
                    fetch=fetch,
                )

    def _retrieve(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: BitgetProduct,
        dataset: BitgetDataset,
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Delegate one archive request to the shared retrieval engine."""
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
        product: BitgetProduct = "spot",
        interval: str | None = None,
        columns: ColumnSelection = None,
        gap_policy: GapPolicy = "keep",
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Bitget market candles for one or several markets."""
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
        product: BitgetProduct = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Bitget individual public trades."""
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

    def get_best_book_snapshots(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: BitgetProduct = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Bitget top-of-book snapshots."""
        return self._retrieve(
            pairs,
            start,
            end,
            product=product,
            dataset="best_book_snapshots",
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
        product: BitgetProduct = "spot",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Bitget complete level-500 snapshots."""
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

    def get_reference_klines(
        self,
        pair: str,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct = "usdt_futures",
        dataset: ReferenceDataset = "mark_price_klines",
        interval: str = "1m",
    ) -> pd.DataFrame:
        """Return Bitget Futures mark, index, or premium candles."""
        declaration = get_dataset(product, dataset)
        declaration.resolve_interval(interval)
        first, last = _range(start, end)
        symbol = _reference_pair(pair)
        try:
            frame = self._rest_data(
                symbol,
                first,
                last,
                product=product,
                dataset=dataset,
                interval=interval,
                fetch=lambda: self._reference.candles(
                    symbol, product, dataset, interval, first, last
                ),
            )
            return _reported_reference(frame, symbol, (first, last), product, dataset)
        except BitgetResponseError as error:
            if error.code != "25100":
                raise
            return _reported_reference(
                candle_frame([]),
                symbol,
                (first, last),
                product,
                dataset,
                error=Message("unknown_pair", error.message),
            )

    def get_funding_rates(
        self,
        pair: str,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct = "usdt_futures",
    ) -> pd.DataFrame:
        """Return Bitget Futures funding settlements."""
        get_dataset(product, "funding_rates")
        first, last = _range(start, end)
        symbol = _reference_pair(pair)
        try:
            frame = self._rest_data(
                symbol,
                first,
                last,
                product=product,
                dataset="funding_rates",
                interval=None,
                fetch=lambda: self._reference.funding(symbol, product, first, last),
            )
            return _reported_reference(
                frame, symbol, (first, last), product, "funding_rates"
            )
        except BitgetResponseError as error:
            if error.code != "25100":
                raise
            return _reported_reference(
                funding_frame([]),
                symbol,
                (first, last),
                product,
                "funding_rates",
                error=Message("unknown_pair", error.message),
            )

    def get_mark_price_klines(
        self,
        pair: str,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct = "usdt_futures",
        interval: str = "1m",
    ) -> pd.DataFrame:
        """Return Bitget Futures mark-price candles.

        Args:
            pair: Native Futures market symbol.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Futures settlement product.
            interval: Native Bitget candle interval.

        Returns:
            Mark-price OHLC rows in UTC.
        """
        return self.get_reference_klines(
            pair,
            start,
            end,
            product=product,
            dataset="mark_price_klines",
            interval=interval,
        )

    def get_index_price_klines(
        self,
        pair: str,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct = "usdt_futures",
        interval: str = "1m",
    ) -> pd.DataFrame:
        """Return Bitget Futures index-price candles.

        Args:
            pair: Native Futures market symbol.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Futures settlement product.
            interval: Native Bitget candle interval.

        Returns:
            Index-price OHLC rows in UTC.
        """
        return self.get_reference_klines(
            pair,
            start,
            end,
            product=product,
            dataset="index_price_klines",
            interval=interval,
        )

    def get_premium_index_klines(
        self,
        pair: str,
        start: DateInput,
        end: DateInput,
        *,
        product: FuturesProduct = "usdt_futures",
        interval: str = "1m",
    ) -> pd.DataFrame:
        """Return Bitget Futures premium-index candles.

        Args:
            pair: Native Futures market symbol.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Futures settlement product.
            interval: Native Bitget candle interval.

        Returns:
            Premium-index OHLC rows in UTC.
        """
        return self.get_reference_klines(
            pair,
            start,
            end,
            product=product,
            dataset="premium_index_klines",
            interval=interval,
        )

    def get_markets(
        self,
        *,
        product: BitgetProduct = "spot",
        status: str | None = None,
        quote_asset: str | None = None,
        sort_by: Literal["symbol", "quote_volume"] = "symbol",
        limit: int | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> list[Market]:
        """Return Bitget markets matching optional exact filters."""
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
        product: BitgetProduct | None = None,
        status: str | None = None,
        quote_asset: str | None = None,
        limit: int = 10,
        refresh: bool = False,
        offline: bool = False,
    ) -> list[Market]:
        """Return exact, prefix, and fuzzy Bitget market matches."""
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
        product: BitgetProduct,
        dataset: BitgetDataset,
        interval: str | None = None,
    ) -> Availability:
        """Return already-cataloged remote and local Bitget coverage.

        Args:
            pair: Native or normalized Bitget market.
            product: Spot or one Futures settlement product.
            dataset: Archive dataset to inspect.
            interval: Optional Kline output interval.

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
        product: BitgetProduct,
        dataset: BitgetDataset,
        interval: str | None = None,
        refresh: bool = False,
    ) -> Availability:
        """Discover bounded Bitget coverage without downloading archives.

        Args:
            pair: Native or normalized Bitget market.
            start: Inclusive discovery start.
            end: Inclusive date or exclusive timestamp discovery end.
            product: Spot or one Futures settlement product.
            dataset: Archive dataset to inspect.
            interval: Optional Kline output interval.
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

    def close(self) -> None:
        """Close the facade-owned REST connection pool."""
        self._reference_client.close()
