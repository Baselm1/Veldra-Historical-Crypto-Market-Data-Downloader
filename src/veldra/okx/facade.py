"""Provide the public facade for OKX historical data."""

from datetime import date, datetime
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd

from veldra.core.models import Result
from veldra.okx.service import OKXService
from veldra.okx.reports import CacheReport

type DateInput = str | date | datetime
type PairInput = str | list[str]
type ColumnSelection = list[str] | dict[str, str] | None
type FrameOutput = pd.DataFrame | list[pd.DataFrame]
type Product = Literal[
    "spot",
    "margin",
    "linear_swap",
    "inverse_swap",
    "linear_futures",
    "inverse_futures",
    "options",
]
type GapPolicy = Literal["forward", "backward", "nan", "keep", "raise"]
type Transport = Literal["auto", "specific", "bulk"]


class OKX:
    """Provide declarative access to OKX public historical data."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        *,
        config_path: str | Path | None = None,
        earliest_date: DateInput | Literal["all"] | None = None,
        max_workers: int = 32,
        market_refresh_hours: float = 24,
        timeout: float = 30,
        retries: int = 3,
        backoff: float = 0.5,
        progress: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create a configured OKX facade without performing I/O.

        Args:
            data_dir: Catalog and Parquet cache directory.
            config_path: Optional TOML settings override.
            earliest_date: Optional history boundary or ``"all"``.
            max_workers: Maximum concurrent archive workers.
            market_refresh_hours: Hours current instruments remain fresh.
            timeout: Per-attempt source timeout.
            retries: Retries after the first source attempt.
            backoff: Initial retry delay.
            progress: Whether calls show Rich progress.
            transport: Optional custom HTTP transport.
        """
        self._service = OKXService(
            data_dir,
            config_path=config_path,
            earliest_date=earliest_date,
            max_workers=max_workers,
            market_refresh_hours=market_refresh_hours,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            progress=progress,
            transport=transport,
        )

    @property
    def data_dir(self) -> Path:
        """Return the resolved catalog and Parquet root."""
        return self._service.data_dir

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
        transport: Transport = "auto",
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return OKX candles for one or several instruments.

        Args:
            pairs: One native instrument or an ordered instrument list.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: OKX product.
            interval: Optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            gap_policy: Missing-candle behavior.
            transport: Automatic, specific, or bulk archive selection.
            refresh: Whether current markets must refresh.
            offline: Whether source access is forbidden.

        Returns:
            One DataFrame or a list matching the input shape.
        """
        result = self._service.get_results(
            pairs,
            start,
            end,
            product=product,
            dataset="klines",
            interval=interval,
            columns=columns,
            gap_policy=gap_policy,
            transport=transport,
            refresh=refresh,
            offline=offline,
        )
        if isinstance(result, Result):
            return result.frame()
        return [item.frame() for item in result]

    def get_trades(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: Product = "spot",
        columns: ColumnSelection = None,
        transport: Transport = "auto",
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return OKX individual trades for one or several instruments.

        Args:
            pairs: One native instrument or an ordered instrument list.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: OKX product.
            columns: Optional selected or renamed canonical columns.
            transport: Automatic, specific, or bulk archive selection.
            refresh: Whether current markets must refresh.
            offline: Whether source access is forbidden.

        Returns:
            One DataFrame or a list matching the input shape.
        """
        result = self._service.get_results(
            pairs,
            start,
            end,
            product=product,
            dataset="trades",
            interval=None,
            columns=columns,
            gap_policy=None,
            transport=transport,
            refresh=refresh,
            offline=offline,
        )
        if isinstance(result, Result):
            return result.frame()
        return [item.frame() for item in result]

    def get_funding_rates(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal["linear_swap", "inverse_swap"],
        columns: ColumnSelection = None,
        transport: Transport = "auto",
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return actual OKX perpetual funding-rate observations.

        Args:
            pairs: One native swap instrument or an ordered list.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Linear- or inverse-margined swap product.
            columns: Optional selected or renamed canonical columns.
            transport: Automatic, specific, or bulk archive selection.
            refresh: Whether current markets must refresh.
            offline: Whether source access is forbidden.

        Returns:
            One DataFrame or a list matching the input shape.
        """
        result = self._service.get_results(
            pairs,
            start,
            end,
            product=product,
            dataset="funding_rates",
            interval=None,
            columns=columns,
            gap_policy=None,
            transport=transport,
            refresh=refresh,
            offline=offline,
        )
        if isinstance(result, Result):
            return result.frame()
        return [item.frame() for item in result]

    def get_order_book_updates(
        self,
        pairs: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal["spot", "linear_swap", "inverse_swap"] = "spot",
        depth: Literal[400, 5000] = 400,
        columns: ColumnSelection = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> FrameOutput:
        """Return native historical OKX order-book snapshots and updates.

        Args:
            pairs: One native instrument or an ordered list.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Spot, linear-swap, or inverse-swap product.
            depth: Maximum native source depth, 400 or 5000.
            columns: Optional selected or renamed canonical columns.
            refresh: Whether current markets must refresh.
            offline: Whether source access is forbidden.

        Returns:
            One nested DataFrame or a list matching the input shape.
        """
        if isinstance(depth, bool) or depth not in {400, 5000}:
            raise ValueError("depth must be 400 or 5000")
        result = self._service.get_results(
            pairs,
            start,
            end,
            product=product,
            dataset=f"order_book_{depth}",
            interval=None,
            columns=columns,
            gap_policy=None,
            transport="specific",
            refresh=refresh,
            offline=offline,
        )
        if isinstance(result, Result):
            return result.frame()
        return [item.frame() for item in result]

    def cache_klines(
        self,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal["spot", "linear_swap", "inverse_swap"] = "spot",
        instruments: Literal["all"] = "all",
        dry_run: bool = False,
        refresh: bool = False,
        offline: bool = False,
    ) -> CacheReport:
        """Cache all available Kline instruments without returning their rows.

        Args:
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Spot or perpetual product.
            instruments: The required all-market selection.
            dry_run: Whether to return the physical plan without downloading.
            refresh: Whether current markets must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Physical plan and local cache totals.
        """
        return self._cache_all(
            start, end, product, "klines", instruments, dry_run, refresh, offline
        )

    def cache_trades(
        self,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal["spot", "linear_swap", "inverse_swap"] = "spot",
        instruments: Literal["all"] = "all",
        dry_run: bool = False,
        refresh: bool = False,
        offline: bool = False,
    ) -> CacheReport:
        """Cache all available individual trades without returning their rows.

        Args:
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Spot or perpetual product.
            instruments: The required all-market selection.
            dry_run: Whether to return the physical plan without downloading.
            refresh: Whether current markets must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Physical plan and local cache totals.
        """
        return self._cache_all(
            start, end, product, "trades", instruments, dry_run, refresh, offline
        )

    def cache_funding_rates(
        self,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal["linear_swap", "inverse_swap"],
        instruments: Literal["all"] = "all",
        dry_run: bool = False,
        refresh: bool = False,
        offline: bool = False,
    ) -> CacheReport:
        """Cache all available funding observations without returning their rows.

        Args:
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Linear- or inverse-margined swap product.
            instruments: The required all-market selection.
            dry_run: Whether to return the physical plan without downloading.
            refresh: Whether current markets must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Physical plan and local cache totals.
        """
        return self._cache_all(
            start,
            end,
            product,
            "funding_rates",
            instruments,
            dry_run,
            refresh,
            offline,
        )

    def _cache_all(
        self,
        start: DateInput,
        end: DateInput,
        product: str,
        dataset: str,
        instruments: object,
        dry_run: bool,
        refresh: bool,
        offline: bool,
    ) -> CacheReport:
        """Validate an all-market facade call and delegate cache planning.

        Args:
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: OKX product.
            dataset: Bulk-capable historical dataset.
            instruments: The required literal all-market selection.
            dry_run: Whether to plan without downloading.
            refresh: Whether current markets must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Physical plan and local cache totals.
        """
        if instruments != "all":
            raise ValueError("instruments must be 'all' for bulk caching")
        return self._service.cache_all(
            start,
            end,
            product=product,
            dataset=dataset,
            dry_run=dry_run,
            refresh=refresh,
            offline=offline,
        )
