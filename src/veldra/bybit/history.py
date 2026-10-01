"""Retrieve and normalize Bybit's bounded public REST histories."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
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
