"""Provide the public facade for OKX historical data."""

from datetime import date, datetime
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd

from veldra.core.models import Result
from veldra.core.request import parse_timestamp
from veldra.okx.chain import OptionChainFilter
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

    def get_borrow_rates(
        self,
        currencies: PairInput,
        start: DateInput,
        end: DateInput,
        *,
        transport: Transport = "auto",
        offline: bool = False,
    ) -> FrameOutput:
        """Return historical margin borrowing rates for currencies.

        Args:
            currencies: One currency or an ordered currency list.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            transport: Automatic, specific, or bulk archive selection.
            offline: Whether source access is forbidden.

        Returns:
            One DataFrame or a list matching the input shape.
        """
        result = self._service.get_currency_results(
            currencies,
            start,
            end,
            transport=transport,
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

    def get_futures_chain_klines(
        self,
        instrument_family: str,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal["linear_futures", "inverse_futures"],
        interval: str | None = None,
        columns: ColumnSelection = None,
        contract_style: Literal["normal", "xperp", "pre_market_xperp"] | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return Klines for every contract in one OKX Futures family.

        Args:
            instrument_family: Native family such as ``BTC-USD``.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Linear- or inverse-margined Futures product.
            interval: Optional output Kline interval.
            columns: Optional selected or renamed canonical columns.
            contract_style: Optional normal or X-Perp filter.
            refresh: Whether current instruments must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Contract-identified Kline rows for the family.
        """
        return self._service.get_chain(
            instrument_family,
            start,
            end,
            product=product,
            dataset="klines",
            interval=interval,
            columns=columns,
            contract_style=contract_style,
            option_filter=None,
            refresh=refresh,
            offline=offline,
        ).frame()

    def get_futures_chain_trades(
        self,
        instrument_family: str,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal["linear_futures", "inverse_futures"],
        columns: ColumnSelection = None,
        contract_style: Literal["normal", "xperp", "pre_market_xperp"] | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return trades for every contract in one OKX Futures family.

        Args:
            instrument_family: Native family such as ``BTC-USD``.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Linear- or inverse-margined Futures product.
            columns: Optional selected or renamed canonical columns.
            contract_style: Optional normal or X-Perp filter.
            refresh: Whether current instruments must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Contract-identified trade rows for the family.
        """
        return self._service.get_chain(
            instrument_family,
            start,
            end,
            product=product,
            dataset="trades",
            interval=None,
            columns=columns,
            contract_style=contract_style,
            option_filter=None,
            refresh=refresh,
            offline=offline,
        ).frame()

    def get_option_chain_klines(
        self,
        instrument_family: str,
        start: DateInput,
        end: DateInput,
        *,
        interval: str | None = None,
        columns: ColumnSelection = None,
        expiry: DateInput | None = None,
        strike_min: float | None = None,
        strike_max: float | None = None,
        option_type: Literal["call", "put"] | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return filtered Klines for every Option in one family.

        Args:
            instrument_family: Native Option family such as ``BTC-USD``.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            interval: Optional output Kline interval.
            columns: Optional selected or renamed canonical columns.
            expiry: Optional exact Option expiry.
            strike_min: Optional inclusive minimum strike.
            strike_max: Optional inclusive maximum strike.
            option_type: Optional call or put constraint.
            refresh: Whether current instruments must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Contract-identified Option Kline rows matching every filter.
        """
        return self._service.get_chain(
            instrument_family,
            start,
            end,
            product="options",
            dataset="klines",
            interval=interval,
            columns=columns,
            contract_style=None,
            option_filter=self._option_filter(
                expiry, strike_min, strike_max, option_type
            ),
            refresh=refresh,
            offline=offline,
        ).frame()

    def get_option_chain_trades(
        self,
        instrument_family: str,
        start: DateInput,
        end: DateInput,
        *,
        columns: ColumnSelection = None,
        expiry: DateInput | None = None,
        strike_min: float | None = None,
        strike_max: float | None = None,
        option_type: Literal["call", "put"] | None = None,
        refresh: bool = False,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return filtered trades for every Option in one family.

        Args:
            instrument_family: Native Option family such as ``BTC-USD``.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            columns: Optional selected or renamed canonical columns.
            expiry: Optional exact Option expiry.
            strike_min: Optional inclusive minimum strike.
            strike_max: Optional inclusive maximum strike.
            option_type: Optional call or put constraint.
            refresh: Whether current instruments must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Contract-identified Option trade rows matching every filter.
        """
        return self._service.get_chain(
            instrument_family,
            start,
            end,
            product="options",
            dataset="trades",
            interval=None,
            columns=columns,
            contract_style=None,
            option_filter=self._option_filter(
                expiry, strike_min, strike_max, option_type
            ),
            refresh=refresh,
            offline=offline,
        ).frame()

    @staticmethod
    def _option_filter(
        expiry: DateInput | None,
        strike_min: float | None,
        strike_max: float | None,
        option_type: Literal["call", "put"] | None,
    ) -> OptionChainFilter:
        """Normalize public Option constraints for the chain query.

        Args:
            expiry: Optional exact contract expiry.
            strike_min: Optional inclusive minimum strike.
            strike_max: Optional inclusive maximum strike.
            option_type: Optional call or put label.

        Returns:
            Validated internal Option chain filter.
        """
        native_type = {None: None, "call": "C", "put": "P"}.get(option_type)
        if option_type is not None and native_type is None:
            raise ValueError("option_type must be call or put")
        for name, value in (("strike_min", strike_min), ("strike_max", strike_max)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise TypeError(f"{name} must be numeric")
        parsed_expiry = parse_timestamp(expiry).date() if expiry is not None else None
        return OptionChainFilter(parsed_expiry, strike_min, strike_max, native_type)

    def get_index_price_klines(
        self,
        index: str,
        start: DateInput,
        end: DateInput,
        *,
        interval: str = "1m",
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return cached historical index-price Klines.

        Args:
            index: Native index such as ``BTC-USD``.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            interval: Native OKX Kline interval.
            offline: Whether source access is forbidden.

        Returns:
            Confirmed OHLC observations in UTC.
        """
        bar = self._rest_interval(interval)
        return self._rest(
            index,
            start,
            end,
            subject_kind="instrument",
            product="index",
            dataset="index_price_klines",
            params={"instId": index.upper(), "bar": bar},
            interval=interval,
            offline=offline,
        )

    def get_mark_price_klines(
        self,
        instrument: str,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal[
            "linear_swap",
            "inverse_swap",
            "linear_futures",
            "inverse_futures",
            "options",
        ],
        interval: str = "1m",
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return cached historical mark-price Klines.

        Args:
            instrument: Native derivative instrument ID.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Derivative product used for catalog separation.
            interval: Native OKX Kline interval.
            offline: Whether source access is forbidden.

        Returns:
            Confirmed OHLC observations in UTC.
        """
        bar = self._rest_interval(interval)
        return self._rest(
            instrument,
            start,
            end,
            subject_kind="instrument",
            product=product,
            dataset="mark_price_klines",
            params={"instId": instrument.upper(), "bar": bar},
            interval=interval,
            offline=offline,
        )

    def get_premium_history(
        self,
        instrument: str,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal["linear_swap", "inverse_swap"],
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return a perpetual instrument's historical premium index.

        Args:
            instrument: Native perpetual instrument ID.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Linear- or inverse-margined perpetual product.
            offline: Whether source access is forbidden.

        Returns:
            Timestamped signed premium observations.
        """
        return self._rest(
            instrument,
            start,
            end,
            subject_kind="instrument",
            product=product,
            dataset="premium_history",
            params={"instId": instrument.upper()},
            offline=offline,
        )

    def get_recent_funding_rates(
        self,
        instrument: str,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal["linear_swap", "inverse_swap"],
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return the mutable recent funding-rate REST tail.

        Args:
            instrument: Native perpetual instrument ID.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Linear- or inverse-margined perpetual product.
            offline: Whether source access is forbidden.

        Returns:
            Predicted and realized funding observations.
        """
        return self._rest(
            instrument,
            start,
            end,
            subject_kind="instrument",
            product=product,
            dataset="recent_funding_rates",
            params={"instId": instrument.upper()},
            offline=offline,
        )

    def get_settlement_history(
        self,
        instrument_family: str,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal["linear_futures", "inverse_futures"],
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return recent dated-Futures settlements for one family.

        Args:
            instrument_family: Native Futures family.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Linear- or inverse-margined Futures product.
            offline: Whether source access is forbidden.

        Returns:
            Flattened settlement events by exact contract.
        """
        return self._rest(
            instrument_family,
            start,
            end,
            subject_kind="instrument_family",
            product=product,
            dataset="settlements",
            params={"instFamily": instrument_family.upper()},
            offline=offline,
        )

    def get_delivery_exercise_history(
        self,
        instrument_family: str,
        start: DateInput,
        end: DateInput,
        *,
        product: Literal["linear_futures", "inverse_futures", "options"],
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return recent Futures delivery or Option exercise events.

        Args:
            instrument_family: Native Futures or Option family.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Dated Futures or Options product.
            offline: Whether source access is forbidden.

        Returns:
            Flattened lifecycle events by exact contract.
        """
        native = "OPTION" if product == "options" else "FUTURES"
        return self._rest(
            instrument_family,
            start,
            end,
            subject_kind="instrument_family",
            product=product,
            dataset="delivery_exercise",
            params={"instType": native, "instFamily": instrument_family.upper()},
            offline=offline,
        )

    def get_open_interest_history(
        self,
        currency: str,
        start: DateInput,
        end: DateInput,
        *,
        period: Literal["5m", "1h", "1d"] = "1h",
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return aggregate contract open interest and volume.

        Args:
            currency: Native underlying currency.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            period: Native analytical aggregation period.
            offline: Whether source access is forbidden.

        Returns:
            Timestamped open-interest and volume observations.
        """
        native_period = self._statistics_period(period)
        return self._rest(
            currency,
            start,
            end,
            subject_kind="currency",
            product="analytics",
            dataset="open_interest_history",
            params={"ccy": currency.upper(), "period": native_period},
            interval=period,
            offline=offline,
        )

    def get_taker_volume(
        self,
        currency: str,
        start: DateInput,
        end: DateInput,
        *,
        market: Literal["spot", "contracts"] = "contracts",
        period: Literal["5m", "1h", "1d"] = "1h",
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return aggregate taker buy and sell volume.

        Args:
            currency: Native underlying currency.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            market: Spot or combined derivatives statistics.
            period: Native analytical aggregation period.
            offline: Whether source access is forbidden.

        Returns:
            Timestamped taker-side volume observations.
        """
        if market not in {"spot", "contracts"}:
            raise ValueError("market must be spot or contracts")
        native_period = self._statistics_period(period)
        return self._rest(
            currency,
            start,
            end,
            subject_kind="currency",
            product="analytics",
            dataset="taker_volume",
            params={
                "ccy": currency.upper(),
                "instType": market.upper(),
                "period": native_period,
            },
            interval=period,
            offline=offline,
        )

    def get_long_short_ratio(
        self,
        currency: str,
        start: DateInput,
        end: DateInput,
        *,
        period: Literal["5m", "1h", "1d"] = "1h",
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return aggregate contract long/short account ratios.

        Args:
            currency: Native underlying currency.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            period: Native analytical aggregation period.
            offline: Whether source access is forbidden.

        Returns:
            Timestamped long/short ratios.
        """
        native_period = self._statistics_period(period)
        return self._rest(
            currency,
            start,
            end,
            subject_kind="currency",
            product="analytics",
            dataset="long_short_ratio",
            params={"ccy": currency.upper(), "period": native_period},
            interval=period,
            offline=offline,
        )

    def get_option_interest_volume(
        self,
        currency: str,
        start: DateInput,
        end: DateInput,
        *,
        period: Literal["8h", "1d"] = "1d",
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return aggregate Option open interest and volume.

        Args:
            currency: Native Option underlying currency.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            period: Native Option statistics period.
            offline: Whether source access is forbidden.

        Returns:
            Timestamped Option interest and volume observations.
        """
        if period not in {"8h", "1d"}:
            raise ValueError("Option statistics period must be 8h or 1d")
        native = {"8h": "8H", "1d": "1D"}[period]
        return self._rest(
            currency,
            start,
            end,
            subject_kind="currency",
            product="analytics",
            dataset="option_interest_volume",
            params={"ccy": currency.upper(), "period": native},
            interval=period,
            offline=offline,
        )

    def _rest(
        self,
        subject: str,
        start: DateInput,
        end: DateInput,
        *,
        subject_kind: str,
        product: str,
        dataset: str,
        params: dict[str, str],
        interval: str | None = None,
        offline: bool,
    ) -> pd.DataFrame:
        """Delegate one typed facade call to the cached REST service.

        Args:
            subject: Native logical subject.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            subject_kind: Instrument, family, or currency scope.
            product: Catalog product label.
            dataset: Registered REST dataset.
            params: Endpoint query parameters.
            interval: Optional native interval.
            offline: Whether source access is forbidden.

        Returns:
            Canonical cached historical frame.
        """
        return self._service.get_rest_history(
            subject,
            start,
            end,
            subject_kind=subject_kind,
            product=product,
            dataset=dataset,
            params=params,
            interval=interval,
            offline=offline,
        )

    @staticmethod
    def _rest_interval(value: object) -> str:
        """Return one native UTC-aligned historical Kline interval.

        Args:
            value: Public lowercase Veldra interval.

        Returns:
            Native OKX bar spelling.
        """
        mapping = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "6h": "6Hutc",
            "12h": "12Hutc",
            "1d": "1Dutc",
            "1w": "1Wutc",
            "1mo": "1Mutc",
        }
        if not isinstance(value, str) or value not in mapping:
            raise ValueError("unsupported OKX REST Kline interval")
        return mapping[value]

    @staticmethod
    def _statistics_period(value: object) -> str:
        """Return one supported native trading-statistics period.

        Args:
            value: Public lowercase period.

        Returns:
            Native OKX period spelling.
        """
        mapping = {"5m": "5m", "1h": "1H", "1d": "1D"}
        if not isinstance(value, str) or value not in mapping:
            raise ValueError("statistics period must be 5m, 1h, or 1d")
        return mapping[value]
