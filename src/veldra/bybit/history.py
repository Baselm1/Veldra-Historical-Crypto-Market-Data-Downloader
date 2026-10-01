"""Retrieve and normalize Bybit's bounded public REST histories."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
import math
from typing import Protocol, cast

import pandas as pd

from veldra.bybit.datasets import get_dataset
from veldra.bybit.identities import category
from veldra.bybit.client import BybitResponseError

_INTERVALS: Mapping[str, str] = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1mo": "M",
}
_KLINE_PATHS: Mapping[str, str] = {
    "klines": "/v5/market/kline",
    "mark_price_klines": "/v5/market/mark-price-kline",
    "index_price_klines": "/v5/market/index-price-kline",
    "premium_index_klines": "/v5/market/premium-index-price-kline",
}
_POSITION_PERIODS: Mapping[str, str] = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}
VOLATILITY_PERIODS = frozenset({7, 14, 21, 30, 60, 90, 180, 270})


class PublicClient(Protocol):
    """Describe the one client operation required by public histories."""

    def v5(self, path: str, params: Mapping[str, str] | None = None) -> object:
        """Return one validated public V5 result."""


def _milliseconds(value: object, field: str) -> pd.Timestamp:
    """Return one plausible UTC epoch-millisecond timestamp."""
    if isinstance(value, bool):
        raise BybitResponseError("invalid_data", f"invalid {field}")
    try:
        integer = int(str(value))
    except (TypeError, ValueError) as error:
        raise BybitResponseError("invalid_data", f"invalid {field}") from error
    if integer < 100_000_000_000 or integer >= 100_000_000_000_000:
        raise BybitResponseError("invalid_data", f"invalid {field} unit")
    return pd.Timestamp(integer, unit="ms", tz="UTC").as_unit("us")


def _number(value: object, field: str, *, nonnegative: bool = False) -> float:
    """Return one finite source number with optional nonnegative bounds."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise BybitResponseError("invalid_data", f"invalid {field}") from error
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise BybitResponseError("invalid_data", f"invalid {field}")
    return result


def _rows(value: object, dataset: str) -> list[list[object]]:
    """Return array-shaped rows from one validated V5 result object."""
    if not isinstance(value, dict) or not isinstance(value.get("list"), list):
        raise BybitResponseError("invalid_data", f"invalid {dataset} result")
    found = value["list"]
    if not all(isinstance(row, list) for row in found):
        raise BybitResponseError("invalid_data", f"invalid {dataset} rows")
    return cast(list[list[object]], found)


def _object_page(value: object, dataset: str) -> tuple[list[dict[str, object]], str]:
    """Return object rows and a validated optional cursor."""
    if not isinstance(value, dict) or not isinstance(value.get("list"), list):
        raise BybitResponseError("invalid_data", f"invalid {dataset} result")
    found = value["list"]
    if not all(isinstance(row, dict) for row in found):
        raise BybitResponseError("invalid_data", f"invalid {dataset} rows")
    cursor = value.get("nextPageCursor", "")
    if not isinstance(cursor, str):
        raise BybitResponseError("invalid_data", f"invalid {dataset} cursor")
    return cast(list[dict[str, object]], found), cursor


def _columns(product: str, dataset: str) -> tuple[str, ...]:
    """Return canonical columns for one Kline response shape."""
    declaration = get_dataset(product, dataset, requested_interval="1m")
    return declaration.stored_columns


def kline_frame(rows: Sequence[object], product: str, dataset: str) -> pd.DataFrame:
    """Normalize reverse-ordered Bybit Kline arrays into canonical rows."""
    columns = _columns(product, dataset)
    expected = len(columns)
    normalized: list[list[object]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != expected:
            raise BybitResponseError("invalid_data", f"invalid {dataset} row shape")
        prices = [
            _number(value, columns[index]) for index, value in enumerate(row[1:5], 1)
        ]
        values: list[object] = [_milliseconds(row[0], "open time"), *prices]
        values.extend(
            _number(value, columns[index], nonnegative=True)
            for index, value in enumerate(row[5:], 5)
        )
        if prices[2] > min(prices[0], prices[1], prices[3]):
            raise BybitResponseError("invalid_data", "Kline low exceeds OHLC")
        if prices[1] < max(prices[0], prices[2], prices[3]):
            raise BybitResponseError("invalid_data", "Kline high is below OHLC")
        normalized.append(values)
    if not normalized:
        typed = {column: pd.Series(dtype="float64") for column in columns[1:]}
        typed[columns[0]] = pd.Series(dtype="datetime64[us, UTC]")
        return pd.DataFrame(typed, columns=columns)
    return (
        pd.DataFrame(normalized, columns=columns)
        .sort_values("open_time", kind="stable")
        .drop_duplicates("open_time", keep="last")
        .reset_index(drop=True)
    )


def _epoch_milliseconds(value: datetime) -> int:
    """Return an exact integer millisecond boundary."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("history timestamps must be timezone-aware")
    return int(value.astimezone(UTC).timestamp() * 1_000)


def _integer(value: object, field: str) -> int:
    """Return one exact nonnegative source integer."""
    if isinstance(value, bool):
        raise BybitResponseError("invalid_data", f"invalid {field}")
    try:
        result = int(str(value))
    except (TypeError, ValueError) as error:
        raise BybitResponseError("invalid_data", f"invalid {field}") from error
    if result < 0:
        raise BybitResponseError("invalid_data", f"invalid {field}")
    return result


def _optional_number(value: object, field: str) -> float:
    """Return NaN for a source field introduced after older history."""
    return (
        float("nan") if value in {None, ""} else _number(value, field, nonnegative=True)
    )


def object_frame(rows: Sequence[object], dataset: str) -> pd.DataFrame:
    """Normalize one object-shaped Bybit history into canonical columns."""
    normalized: list[dict[str, object]] = []
    for value in rows:
        if not isinstance(value, dict):
            raise BybitResponseError("invalid_data", f"invalid {dataset} row")
        if dataset == "funding_rates":
            row = {
                "funding_time": _milliseconds(
                    value.get("fundingRateTimestamp"), "funding time"
                ),
                "funding_rate": _number(value.get("fundingRate"), "funding rate"),
            }
        elif dataset == "open_interest":
            row = {
                "event_time": _milliseconds(value.get("timestamp"), "event time"),
                "open_interest": _number(
                    value.get("openInterest"), "open interest", nonnegative=True
                ),
                "single_open_interest": _optional_number(
                    value.get("singleOpenInterest"), "single open interest"
                ),
            }
        elif dataset == "long_short_ratios":
            row = {
                "event_time": _milliseconds(value.get("timestamp"), "event time"),
                "buy_ratio": _number(
                    value.get("buyRatio"), "buy ratio", nonnegative=True
                ),
                "sell_ratio": _number(
                    value.get("sellRatio"), "sell ratio", nonnegative=True
                ),
            }
        elif dataset == "historical_volatility":
            row = {
                "event_time": _milliseconds(value.get("time"), "event time"),
                "period": _integer(value.get("period"), "period"),
                "volatility": _number(
                    value.get("value"), "volatility", nonnegative=True
                ),
            }
        elif dataset == "delivery_prices":
            row = {
                "delivery_time": _milliseconds(
                    value.get("deliveryTime"), "delivery time"
                ),
                "delivery_price": _number(
                    value.get("deliveryPrice"), "delivery price", nonnegative=True
                ),
            }
        else:
            raise ValueError(f"unsupported Bybit object history: {dataset}")
        normalized.append(row)
    time_column = {
        "funding_rates": "funding_time",
        "delivery_prices": "delivery_time",
    }.get(dataset, "event_time")
    columns = (
        tuple(normalized[0])
        if normalized
        else {
            "funding_rates": ("funding_time", "funding_rate"),
            "open_interest": (
                "event_time",
                "open_interest",
                "single_open_interest",
            ),
            "long_short_ratios": ("event_time", "buy_ratio", "sell_ratio"),
            "historical_volatility": ("event_time", "period", "volatility"),
            "delivery_prices": ("delivery_time", "delivery_price"),
        }[dataset]
    )
    if not normalized:
        typed: dict[str, pd.Series[object]] = {
            column: pd.Series(dtype="float64") for column in columns
        }
        typed[time_column] = pd.Series(dtype="datetime64[us, UTC]")
        if dataset == "historical_volatility":
            typed["period"] = pd.Series(dtype="int64")
        return pd.DataFrame(typed, columns=columns)
    return (
        pd.DataFrame(normalized, columns=columns)
        .sort_values(time_column, kind="stable")
        .drop_duplicates(time_column, keep="last")
        .reset_index(drop=True)
    )


class BybitHistory:
    """Page backward through Bybit's public historical REST endpoints."""

    def __init__(self, client: PublicClient) -> None:
        """Retain one shared, rate-limited public client."""
        self.client = client

    def klines(
        self,
        symbol: str,
        product: str,
        dataset: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        max_pages: int = 10_000,
    ) -> pd.DataFrame:
        """Return one exact half-open native Kline range."""
        get_dataset(product, dataset, requested_interval=interval)
        if start >= end:
            raise ValueError("history range must end after it starts")
        if (
            isinstance(max_pages, bool)
            or not isinstance(max_pages, int)
            or max_pages < 1
        ):
            raise ValueError("max_pages must be a positive integer")
        native_interval = _INTERVALS[interval]
        path = _KLINE_PATHS[dataset]
        start_ms = _epoch_milliseconds(start)
        cursor = _epoch_milliseconds(end) - 1
        found: list[object] = []
        for _page in range(max_pages):
            value = self.client.v5(
                path,
                {
                    "category": category(product),
                    "symbol": symbol,
                    "interval": native_interval,
                    "start": str(start_ms),
                    "end": str(cursor),
                    "limit": "1000",
                },
            )
            page = _rows(value, dataset)
            found.extend(page)
            if len(page) < 1000:
                break
            oldest = min(int(str(row[0])) for row in page)
            if oldest <= start_ms:
                break
            next_cursor = oldest - 1
            if next_cursor >= cursor:
                raise BybitResponseError("cursor_loop", f"{dataset} cursor repeated")
            cursor = next_cursor
        else:
            raise BybitResponseError("page_limit", f"{dataset} exceeded max_pages")
        frame = kline_frame(found, product, dataset)
        return frame[(frame.open_time >= start) & (frame.open_time < end)].copy()

    def _cursor_rows(
        self,
        path: str,
        params: dict[str, str],
        dataset: str,
        *,
        max_pages: int = 10_000,
    ) -> list[dict[str, object]]:
        """Collect one cursor-paginated object history without loops."""
        found: list[dict[str, object]] = []
        seen: set[str] = set()
        for _page in range(max_pages):
            rows, cursor = _object_page(self.client.v5(path, params), dataset)
            found.extend(rows)
            if not cursor:
                return found
            if cursor in seen:
                raise BybitResponseError("cursor_loop", f"{dataset} cursor repeated")
            seen.add(cursor)
            params["cursor"] = cursor
        raise BybitResponseError("page_limit", f"{dataset} exceeded max_pages")

    def funding_rates(
        self, symbol: str, product: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Return exact perpetual-funding settlements."""
        get_dataset(product, "funding_rates")
        start_ms = _epoch_milliseconds(start)
        cursor = _epoch_milliseconds(end) - 1
        found: list[dict[str, object]] = []
        for _page in range(10_000):
            value = self.client.v5(
                "/v5/market/funding/history",
                {
                    "category": category(product),
                    "symbol": symbol,
                    "startTime": str(start_ms),
                    "endTime": str(cursor),
                    "limit": "200",
                },
            )
            page, page_cursor = _object_page(value, "funding_rates")
            if page_cursor:
                raise BybitResponseError(
                    "invalid_data", "funding history returned an unexpected cursor"
                )
            found.extend(page)
            if len(page) < 200:
                break
            oldest = min(int(str(row.get("fundingRateTimestamp"))) for row in page)
            if oldest <= start_ms:
                break
            cursor = oldest - 1
        else:
            raise BybitResponseError("page_limit", "funding history exceeded max_pages")
        frame = object_frame(found, "funding_rates")
        return frame[(frame.funding_time >= start) & (frame.funding_time < end)].copy()

    def positions(
        self,
        symbol: str,
        product: str,
        dataset: str,
        period: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return open-interest or long/short history at one native period."""
        get_dataset(product, dataset)
        native_period = _POSITION_PERIODS.get(period)
        if native_period is None:
            choices = ", ".join(_POSITION_PERIODS)
            raise ValueError(
                f"unsupported Bybit position period; choose from {choices}"
            )
        path, name, limit = {
            "open_interest": ("/v5/market/open-interest", "intervalTime", "200"),
            "long_short_ratios": ("/v5/market/account-ratio", "period", "500"),
        }[dataset]
        params = {
            "category": category(product),
            "symbol": symbol,
            name: native_period,
            "startTime": str(_epoch_milliseconds(start)),
            "endTime": str(_epoch_milliseconds(end) - 1),
            "limit": limit,
        }
        frame = object_frame(self._cursor_rows(path, params, dataset), dataset)
        return frame[(frame.event_time >= start) & (frame.event_time < end)].copy()

    def volatility(
        self,
        base_coin: str,
        period: int,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return Option historical volatility across bounded 30-day windows."""
        get_dataset("options", "historical_volatility")
        if isinstance(period, bool) or period not in VOLATILITY_PERIODS:
            choices = ", ".join(str(item) for item in sorted(VOLATILITY_PERIODS))
            raise ValueError(f"unsupported volatility period; choose from {choices}")
        found: list[dict[str, object]] = []
        cursor = start
        while cursor < end:
            window_end = min(end, cursor + timedelta(days=30))
            value = self.client.v5(
                "/v5/market/historical-volatility",
                {
                    "category": "option",
                    "baseCoin": base_coin,
                    "period": str(period),
                    "startTime": str(_epoch_milliseconds(cursor)),
                    "endTime": str(_epoch_milliseconds(window_end) - 1),
                },
            )
            if not isinstance(value, list) or not all(
                isinstance(row, dict) for row in value
            ):
                raise BybitResponseError(
                    "invalid_data", "invalid historical_volatility result"
                )
            found.extend(cast(list[dict[str, object]], value))
            cursor = window_end
        frame = object_frame(found, "historical_volatility")
        return frame[(frame.event_time >= start) & (frame.event_time < end)].copy()

    def delivery_prices(
        self, symbol: str, product: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Return delivery prices for one dated contract or Option."""
        get_dataset(product, "delivery_prices")
        params = {
            "category": category(product),
            "symbol": symbol,
            "limit": "200",
        }
        found = self._cursor_rows(
            "/v5/market/delivery-price", params, "delivery_prices"
        )
        frame = object_frame(found, "delivery_prices")
        return frame[
            (frame.delivery_time >= start) & (frame.delivery_time < end)
        ].copy()
