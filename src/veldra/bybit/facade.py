"""Provide the public facade for Bybit historical market data."""

from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd

from veldra.bybit.client import BybitResponseError
from veldra.bybit.datasets import BybitDataset, get_dataset
from veldra.bybit.history import kline_frame, object_frame
from veldra.bybit.service import BybitService
from veldra.core.download import _validate_settings
from veldra.core.inspection import (
    discover_availability as _discover_availability,
    find_markets as _find_markets,
    get_availability as _get_availability,
    get_markets as _get_markets,
)
from veldra.core.models import Availability, Market, Message, Result
from veldra.core.request import parse_columns, parse_pairs, parse_range

type DateInput = str | date | datetime
type PairInput = str | list[str]
type FrameOutput = pd.DataFrame | list[pd.DataFrame]
type ColumnSelection = list[str] | dict[str, str] | None
type Product = Literal["spot", "linear", "inverse", "options"]
type TradingProduct = Literal["spot", "linear", "inverse"]
type DerivativeProduct = Literal["linear", "inverse"]
type ReferenceProduct = Literal["linear", "inverse", "options"]
type PositionDataset = Literal["open_interest", "long_short_ratios"]
type ReferenceDataset = Literal[
    "mark_price_klines", "index_price_klines", "premium_index_klines"
]

_UNKNOWN_CODES = frozenset({"10001", "110001"})


def _flag(value: object, name: str) -> bool:
    """Return one strict public Boolean option."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a Boolean")
    return value


def _project(frame: pd.DataFrame, columns: ColumnSelection) -> pd.DataFrame:
    """Apply the shared column selection contract to one REST result."""
    selected = parse_columns(columns)
    if selected is None:
        return frame
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise ValueError(f"unknown columns: {', '.join(missing)}")
    return frame[list(selected)].rename(columns=selected)


def _reported(
    frame: pd.DataFrame,
    pair: str,
    requested: tuple[datetime, datetime],
    product: str,
    dataset: str,
    *,
    error: Message | None = None,
) -> pd.DataFrame:
    """Attach Veldra's serializable retrieval report to one REST frame."""
    available = None if frame.empty else requested
    result = Result(
        pair,
        frame,
        requested,
        used_range=available,
        available_range=available,
        source="bybit",
        product=product,
        dataset=dataset,
        gap_policy=None,
    )
    if error is not None:
        result.errors.append(error)
    elif frame.empty:
        result.errors.append(
            Message("range_unavailable", "Bybit returned no rows for this range.")
        )
    return result.frame()


class Bybit:
    """Provide declarative access to Bybit historical market data."""

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
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create a configured Bybit facade without performing source I/O."""
        _validate_settings(timeout, retries, backoff)
        self._progress = _flag(progress, "progress")
        self._service = BybitService(
            data_dir,
            config_path=config_path,
            earliest_date=earliest_date,
            max_workers=max_workers,
            discovery_tail_days=discovery_tail_days,
            market_refresh_hours=market_refresh_hours,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            transport=transport,
        )

    @property
    def data_dir(self) -> Path:
        """Return the resolved catalog and Parquet root."""
        return self._service.data_dir

    @property
    def earliest_date(self) -> date | None:
        """Return the configured usable history boundary."""
        return self._service.downloader.earliest_date

    @property
    def max_workers(self) -> int:
        """Return the facade-wide archive worker ceiling."""
        return self._service.downloader.max_workers

    def _archives(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: Product,
        dataset: Literal["trades", "order_book_updates"],
        columns: ColumnSelection,
        refresh: bool,
        offline: bool,
    ) -> FrameOutput:
        """Delegate one archive-backed request to the shared engine."""
        return self._service.archives(
            pairs,
            start,
            end,
            product=product,
            dataset=dataset,
            columns=columns,
            refresh=refresh,
            offline=offline,
            progress=self._progress,
        )

    def _rest_frames(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: Product,
        dataset: str,
        columns: ColumnSelection,
        refresh: bool,
        offline: bool,
        retrieve: Callable[[str, datetime, datetime], pd.DataFrame],
        empty: Callable[[], pd.DataFrame],
    ) -> FrameOutput:
        """Retrieve independent REST histories without cross-pair failure."""
        _flag(refresh, "refresh")
        _flag(offline, "offline")
        if refresh and offline:
            raise ValueError("refresh and offline cannot both be enabled")
        requested = parse_range(start, end)
        values, single = parse_pairs(pairs)
        frames: list[pd.DataFrame] = []
        for value in values:
            symbol = value.upper()
            try:
                frame = _project(retrieve(symbol, *requested), columns)
                frames.append(_reported(frame, symbol, requested, product, dataset))
            except BybitResponseError as error:
                if error.code not in _UNKNOWN_CODES:
                    raise
                frames.append(
                    _reported(
                        _project(empty(), columns),
                        symbol,
                        requested,
                        product,
                        dataset,
                        error=Message("unknown_pair", error.message),
                    )
                )
        return frames[0] if single else frames

    def get_klines(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: TradingProduct = "spot",
        interval: str = "1m",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return native Bybit trading candles for one or several markets."""
        return self._kline_frames(
            pairs,
            start,
            end,
            product=product,
            dataset="klines",
            interval=interval,
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def _kline_frames(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: Product,
        dataset: ReferenceDataset | Literal["klines"],
        interval: str,
        columns: ColumnSelection,
        refresh: bool,
        offline: bool,
    ) -> FrameOutput:
        """Return one native Kline family through cached REST retrieval."""
        get_dataset(product, dataset, requested_interval=interval)
        return self._rest_frames(
            pairs,
            start,
            end,
            product=product,
            dataset=dataset,
            columns=columns,
            refresh=refresh,
            offline=offline,
            retrieve=lambda symbol, first, last: self._service.klines(
                symbol,
                product,
                dataset,
                interval,
                first,
                last,
                refresh=refresh,
                offline=offline,
            ),
            empty=lambda: kline_frame([], product, dataset),
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
        """Return individual public trades from Bybit's daily archives."""
        return self._archives(
            pairs,
            start,
            end,
            product=product,
            dataset="trades",
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
        """Return replayable Bybit order-book snapshots and deltas."""
        return self._archives(
            pairs,
            start,
            end,
            product=product,
            dataset="order_book_updates",
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_reference_klines(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: ReferenceProduct,
        dataset: ReferenceDataset,
        interval: str = "1m",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Bybit mark, index, or premium-index candles."""
        return self._kline_frames(
            pairs,
            start,
            end,
            product=product,
            dataset=dataset,
            interval=interval,
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_mark_price_klines(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: ReferenceProduct,
        interval: str = "1m",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return derivative or Option mark-price candles."""
        return self.get_reference_klines(
            pairs,
            start,
            end,
            product=product,
            dataset="mark_price_klines",
            interval=interval,
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_index_price_klines(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: DerivativeProduct,
        interval: str = "1m",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return linear or inverse index-price candles."""
        return self.get_reference_klines(
            pairs,
            start,
            end,
            product=product,
            dataset="index_price_klines",
            interval=interval,
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_premium_index_klines(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        interval: str = "1m",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return linear-contract premium-index candles."""
        return self.get_reference_klines(
            pairs,
            start,
            end,
            product="linear",
            dataset="premium_index_klines",
            interval=interval,
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
        product: DerivativeProduct,
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return linear or inverse perpetual funding settlements."""
        get_dataset(product, "funding_rates")
        return self._rest_frames(
            pairs,
            start,
            end,
            product=product,
            dataset="funding_rates",
            columns=columns,
            refresh=refresh,
            offline=offline,
            retrieve=lambda symbol, first, last: self._service.funding_rates(
                symbol,
                product,
                first,
                last,
                refresh=refresh,
                offline=offline,
            ),
            empty=lambda: object_frame([], "funding_rates"),
        )

    def _positions(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: DerivativeProduct,
        dataset: PositionDataset,
        period: str,
        columns: ColumnSelection,
        refresh: bool,
        offline: bool,
    ) -> FrameOutput:
        """Return one position-related analytical history."""
        get_dataset(product, dataset)
        return self._rest_frames(
            pairs,
            start,
            end,
            product=product,
            dataset=dataset,
            columns=columns,
            refresh=refresh,
            offline=offline,
            retrieve=lambda symbol, first, last: self._service.positions(
                symbol,
                product,
                dataset,
                period,
                first,
                last,
                refresh=refresh,
                offline=offline,
            ),
            empty=lambda: object_frame([], dataset),
        )

    def get_open_interest(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: DerivativeProduct,
        period: str = "1h",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Bybit open-interest history at one native period."""
        return self._positions(
            pairs,
            start,
            end,
            product=product,
            dataset="open_interest",
            period=period,
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_long_short_ratios(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: DerivativeProduct,
        period: str = "1h",
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return Bybit account long/short ratios at one native period."""
        return self._positions(
            pairs,
            start,
            end,
            product=product,
            dataset="long_short_ratios",
            period=period,
            columns=columns,
            refresh=refresh,
            offline=offline,
        )

    def get_historical_volatility(
        self,
        base_coin: str,
        start: DateInput,
        end: DateInput,
        *,
        period: int = 30,
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return one Option base coin's historical volatility."""
        _flag(refresh, "refresh")
        _flag(offline, "offline")
        if refresh and offline:
            raise ValueError("refresh and offline cannot both be enabled")
        values, single = parse_pairs(base_coin)
        if not single:
            raise TypeError("base_coin must be a string")
        symbol = values[0].upper()
        requested = parse_range(start, end)
        frame = self._service.volatility(
            symbol,
            period,
            *requested,
            refresh=refresh,
            offline=offline,
        )
        return _reported(
            _project(frame, columns),
            symbol,
            requested,
            "options",
            "historical_volatility",
        )

    def get_delivery_prices(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: ReferenceProduct,
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return delivery prices for dated Futures or Options."""
        get_dataset(product, "delivery_prices")
        return self._rest_frames(
            pairs,
            start,
            end,
            product=product,
            dataset="delivery_prices",
            columns=columns,
            refresh=refresh,
            offline=offline,
            retrieve=lambda symbol, first, last: self._service.delivery_prices(
                symbol,
                product,
                first,
                last,
                refresh=refresh,
                offline=offline,
            ),
            empty=lambda: object_frame([], "delivery_prices"),
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
        """Return Bybit markets matching optional exact filters."""
        return _get_markets(
            self._service.downloader,
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
        """Return exact, prefix, and fuzzy Bybit market matches."""
        return _find_markets(
            self._service.downloader,
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
        dataset: BybitDataset,
        interval: str | None = None,
    ) -> Availability:
        """Return already-cataloged remote and local Bybit coverage."""
        return _get_availability(
            self._service.downloader,
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
        dataset: Literal["trades", "order_book_updates"],
        refresh: bool = False,
    ) -> Availability:
        """Discover bounded archive coverage without downloading files."""
        return _discover_availability(
            self._service.downloader,
            pair,
            start,
            end,
            product=product,
            dataset=dataset,
            refresh=refresh,
            progress=self._progress,
        )

    def close(self) -> None:
        """Close the facade-owned persistent HTTP connection pool."""
        self._service.close()

    def __enter__(self) -> "Bybit":
        """Return this facade from a managed context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close owned resources when leaving a context manager."""
        self.close()
