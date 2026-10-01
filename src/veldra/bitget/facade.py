"""Provide the public facade for Bitget historical market data."""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

from veldra.bitget.client import BitgetClient
from veldra.bitget.connector import BitgetConnector
from veldra.bitget.datasets import BitgetDataset, BitgetProduct, get_dataset
from veldra.bitget.reference import BitgetReferenceService
from veldra.core.download import _validate_settings
from veldra.core.engine import RetrievalEngine
from veldra.core.inspection import find_markets as _find_markets
from veldra.core.inspection import get_markets as _get_markets
from veldra.core.models import Market
from veldra.core.request import parse_timestamp

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
        first, last = _range(start, end)
        return self._reference.candles(
            pair.strip().upper(), product, dataset, interval, first, last
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
        first, last = _range(start, end)
        return self._reference.funding(pair.strip().upper(), product, first, last)

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
        product: BitgetProduct = "spot",
        limit: int = 10,
        refresh: bool = False,
        offline: bool = False,
    ) -> list[Market]:
        """Return exact, prefix, and fuzzy Bitget market matches."""
        return _find_markets(
            self._downloader,
            query,
            product=product,
            status=None,
            quote_asset=None,
            limit=limit,
            refresh=refresh,
            offline=offline,
            progress=self._progress,
        )

    def close(self) -> None:
        """Close the facade-owned REST connection pool."""
        self._reference_client.close()
