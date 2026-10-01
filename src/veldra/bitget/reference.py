"""Retrieve bounded Bitget Futures reference candles and funding rates."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
import math
from typing import cast

import pandas as pd

from veldra.bitget.client import REST_URL, BitgetClient, BitgetResponseError
from veldra.bitget.markets import category

_CANDLE_TYPES: Mapping[str, str] = {
    "mark_price_klines": "mark",
    "index_price_klines": "index",
    "premium_index_klines": "premium",
}
_INTERVALS: Mapping[str, tuple[str, timedelta]] = {
    "1m": ("1m", timedelta(minutes=1)),
    "3m": ("3m", timedelta(minutes=3)),
    "5m": ("5m", timedelta(minutes=5)),
    "15m": ("15m", timedelta(minutes=15)),
    "30m": ("30m", timedelta(minutes=30)),
    "1h": ("1H", timedelta(hours=1)),
    "4h": ("4H", timedelta(hours=4)),
    "6h": ("6H", timedelta(hours=6)),
    "12h": ("12H", timedelta(hours=12)),
    "1d": ("1D", timedelta(days=1)),
}


def _time(value: object, field: str) -> pd.Timestamp:
    """Parse one required epoch-millisecond timestamp."""
    try:
        result = pd.to_datetime(int(str(value)), unit="ms", utc=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise BitgetResponseError("invalid_data", f"invalid Bitget {field}") from error
    return result.as_unit("us")


def _number(value: object, field: str) -> float:
    """Parse one required finite source number."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise BitgetResponseError("invalid_data", f"invalid Bitget {field}") from error
    if not math.isfinite(result):
        raise BitgetResponseError("invalid_data", f"invalid Bitget {field}")
    return result


def _rows(value: object, name: str) -> list[object]:
    """Return a validated endpoint row list."""
    if isinstance(value, dict) and "resultList" in value:
        value = value["resultList"]
    if not isinstance(value, list):
        raise BitgetResponseError("invalid_data", f"{name} data must be a list")
    return value


def candle_frame(rows: Sequence[object]) -> pd.DataFrame:
    """Convert Bitget array candles into canonical columns."""
    columns = (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
    )
    if not rows:
        frame = pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})
        frame["open_time"] = pd.Series(dtype="datetime64[us, UTC]")
        return frame
    values: list[list[object]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 7:
            raise BitgetResponseError("invalid_data", "candle row shape is invalid")
        values.append(
            [_time(row[0], "open_time"), *[_number(item, "candle") for item in row[1:]]]
        )
    return (
        pd.DataFrame(values, columns=columns)
        .drop_duplicates("open_time", keep="last")
        .sort_values("open_time", kind="stable")
        .reset_index(drop=True)
    )


def funding_frame(rows: Sequence[object]) -> pd.DataFrame:
    """Convert Bitget funding objects into canonical columns."""
    values: list[dict[str, object]] = []
    for value in rows:
        if not isinstance(value, dict):
            raise BitgetResponseError("invalid_data", "funding rows must be objects")
        row = cast(dict[str, object], value)
        values.append(
            {
                "funding_time": _time(row.get("fundingRateTimestamp"), "funding_time"),
                "funding_rate": _number(row.get("fundingRate"), "funding_rate"),
            }
        )
    if not values:
        return pd.DataFrame(
            {
                "funding_time": pd.Series(dtype="datetime64[us, UTC]"),
                "funding_rate": pd.Series(dtype="float64"),
            }
        )
    return (
        pd.DataFrame(values)
        .drop_duplicates("funding_time", keep="last")
        .sort_values("funding_time", kind="stable")
        .reset_index(drop=True)
    )


class BitgetReferenceService:
    """Page through the public Bitget historical reference endpoints."""

    def __init__(self, client: BitgetClient) -> None:
        """Retain one shared rate-limited HTTP client."""
        self.client = client

    def candles(
        self,
        symbol: str,
        product: str,
        dataset: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return an exact half-open range of reference-price candles."""
        candle_type = _CANDLE_TYPES.get(dataset)
        if candle_type is None:
            raise ValueError(f"unsupported Bitget reference dataset: {dataset}")
        native = _INTERVALS.get(interval)
        if native is None:
            raise ValueError(f"unsupported Bitget REST interval: {interval}")
        source_interval, duration = native
        found: list[object] = []
        cursor = start
        while cursor < end:
            page_end = min(end, cursor + duration * 1_000)
            data = self.client.request(
                "GET",
                f"{REST_URL}/api/v3/market/candles",
                policy_key="history_candles",
                params={
                    "category": category(product),
                    "symbol": symbol,
                    "interval": source_interval,
                    "type": candle_type,
                    "startTime": str(int(cursor.timestamp() * 1_000)),
                    "endTime": str(int(page_end.timestamp() * 1_000) - 1),
                    "limit": "1000",
                },
            )
            found.extend(_rows(data, "candle"))
            cursor = page_end
        frame = candle_frame(found)
        return frame[(frame.open_time >= start) & (frame.open_time < end)].copy()

    def funding(
        self, symbol: str, product: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Return funding settlements in one exact half-open UTC range."""
        found: list[object] = []
        for cursor in range(1, 101):
            data = self.client.request(
                "GET",
                f"{REST_URL}/api/v3/market/history-fund-rate",
                policy_key="history_funding",
                params={
                    "category": category(product),
                    "symbol": symbol,
                    "cursor": str(cursor),
                    "limit": "100",
                },
            )
            page = _rows(data, "funding")
            found.extend(page)
            if len(page) < 100:
                break
        frame = funding_frame(found)
        return frame[(frame.funding_time >= start) & (frame.funding_time < end)].copy()
