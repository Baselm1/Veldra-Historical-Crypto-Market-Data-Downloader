"""Exercise the Upbit facade through the complete archive pipeline."""

from datetime import date
import hashlib
import io
from pathlib import Path
from urllib.parse import parse_qs, unquote
import zipfile

import httpx
import pandas as pd
import pytest

from veldra import Upbit


def _archive(member: str, text: str) -> bytes:
    """Return one synthetic ZIP containing a CSV member."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return output.getvalue()


def _kline_archive(symbol: str, interval: str, day: date) -> bytes:
    """Return one sparse but valid synthetic Upbit Kline archive.

    Args:
        symbol: The native quote-first Upbit market.
        interval: The physical candle interval.
        day: The UTC archive day represented by the rows.

    Returns:
        A ZIP containing one official-shaped candle CSV.
    """
    stamp = day.strftime("%Y%m%d")
    name = f"{symbol}_candle-{interval}_{stamp}.csv"
    spacing = "00:00:01" if interval == "1s" else "00:01:00"
    text = (
        "date_time_utc,open,high,low,close,acc_trade_price,acc_trade_volume\n"
        f"{day.isoformat()}T00:00:00,100,102,99,101,202,2\n"
        f"{day.isoformat()}T{spacing},101,104,100,103,309,3\n"
    )
    return _archive(name, text)


def _trade_archive(symbol: str) -> bytes:
    """Return one out-of-order synthetic Upbit trade archive."""
    name = f"{symbol}_trade_20250101.csv"
    text = (
        "seq,timestamp,volume,price,ask_bid\n"
        "0,1735689601000,2,101,BID\n"
        "1,1735689600000,1,100,ASK\n"
    )
    return _archive(name, text)


class UpbitServer:
    """Serve a complete synthetic slice of Upbit's public services."""

    symbols = ("USDT-BTC", "BTC-USDT", "KRW-BTC")

    def __init__(self, days: tuple[date, ...] = (date(2025, 1, 1),)) -> None:
        """Build supported synthetic archives and an empty request log.

        Args:
            days: UTC source days made available by the archive portal.
        """
        self.requests: list[httpx.Request] = []
        self.archives: dict[str, bytes] = {}
        for symbol in self.symbols:
            for day in days:
                stamp = day.strftime("%Y%m%d")
                for interval in ("1s", "1m"):
                    key = (
                        f"candle/{symbol}/daily/{interval}/{day.year}/"
                        f"{symbol}_candle-{interval}_{stamp}.zip"
                    )
                    self.archives[key] = _kline_archive(symbol, interval, day)
            key = f"trade/{symbol}/daily/2025/{symbol}_trade_20250101.zip"
            self.archives[key] = _trade_archive(symbol)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return one official-shaped response for the requested service."""
        self.requests.append(request)
        if request.url.host == "api.upbit.com":
            return self._api(request)
        if request.url.host == "crix-data-api.upbit.com":
            return self._listing(request)
        return self._object(request)

    def _api(self, request: httpx.Request) -> httpx.Response:
        """Return synthetic current markets or rolling quote volumes."""
        if request.url.path.endswith("/market/all"):
            return httpx.Response(
                200,
                json=[
                    {"market": symbol, "market_event": {"warning": False}}
                    for symbol in self.symbols
                ],
            )
        return httpx.Response(
            200,
            json=[
                {"market": symbol, "acc_trade_price_24h": index + 1}
                for index, symbol in enumerate(self.symbols)
            ],
        )

    def _listing(self, request: httpx.Request) -> httpx.Response:
        """Return a synthetic archive directory or year listing."""
        prefix = parse_qs(request.url.query.decode())["prefix"][0]
        if prefix in {"candle", "trade"}:
            return self._directories(prefix, self.symbols)
        if prefix.endswith(("/daily/1s", "/daily/1m", "/daily")):
            return self._directories(prefix, ("2025",))
        keys = [key for key in self.archives if key.startswith(f"{prefix}/")]
        return httpx.Response(
            200,
            json=[
                {"key": key, "size": len(self.archives[key]), "type": "FILE"}
                for key in keys
            ],
        )

    @staticmethod
    def _directories(prefix: str, names: tuple[str, ...]) -> httpx.Response:
        """Return immediate listing directories beneath a prefix."""
        return httpx.Response(
            200,
            json=[
                {"key": f"{prefix}/{name}", "size": 0, "type": "DIRECTORY"}
                for name in names
            ],
        )

    def _object(self, request: httpx.Request) -> httpx.Response:
        """Return one archive object or its digest-only sidecar."""
        key = unquote(request.url.path.lstrip("/"))
        archive_key = key.removesuffix(".checksum")
        content = self.archives.get(archive_key)
        if content is None:
            return httpx.Response(404)
        if key.endswith(".checksum"):
            return httpx.Response(200, text=hashlib.sha256(content).hexdigest())
        return httpx.Response(200, content=content)

    def count(self, host: str, suffix: str = "") -> int:
        """Return logged requests matching one host and optional path suffix."""
        return sum(
            request.url.host == host and request.url.path.endswith(suffix)
            for request in self.requests
        )


def service(tmp_path: Path, server: UpbitServer) -> Upbit:
    """Return an Upbit facade backed by the synthetic transport."""
    facade = Upbit(tmp_path, earliest_date="all", retries=0, progress=False)
    facade._downloader.transport = httpx.MockTransport(server)
    return facade


def test_quote_first_aliases_exact_symbols_and_typos_are_isolated(
    tmp_path: Path,
) -> None:
    """Confirm mixed pair requests preserve Upbit's semantic identities."""
    server = UpbitServer()
    frames = service(tmp_path, server).get_klines(
        ["BTCUSDT", "BTC-USDT", "BTCSUDT", "NOTREALPAIR"],
        "2025-01-01T00:00:00Z",
        "2025-01-01T00:02:00Z",
        interval="1m",
    )

    assert isinstance(frames, list)
    assert len(frames[0]) == 2
    assert len(frames[1]) == 2
    assert frames[0].attrs["download"]["pair"] == "USDT-BTC"
    assert frames[1].attrs["download"]["pair"] == "BTC-USDT"
    assert frames[2].empty
    assert frames[2].attrs["download"]["errors"][0]["suggestions"][0] == "USDT-BTC"
    assert frames[3].attrs["download"]["errors"][0]["suggestions"] == []


def test_second_candles_use_the_one_second_physical_archive(tmp_path: Path) -> None:
    """Confirm explicit second requests never route through minute archives."""
    server = UpbitServer()
    frame = service(tmp_path, server).get_klines(
        "BTCUSDT",
        "2025-01-01T00:00:00Z",
        "2025-01-01T00:00:02Z",
        interval="1s",
    )

    assert isinstance(frame, pd.DataFrame)
    assert frame["open_time"].dt.second.tolist() == [0, 1]
    listing_queries = [request.url.query.decode() for request in server.requests]
    assert any(
        "daily%2F1s" in query or "daily/1s" in query for query in listing_queries
    )
    assert not any(
        "daily%2F1m" in query or "daily/1m" in query for query in listing_queries
    )


def test_verified_cache_supports_offline_reuse_and_online_corruption_recovery(
    tmp_path: Path,
) -> None:
    """Confirm cached trades avoid I/O and corrupt Parquet is rebuilt online."""
    server = UpbitServer()
    upbit = service(tmp_path, server)
    first = upbit.get_trades("KRW-BTC", "2025-01-01T00:00:00Z", "2025-01-01T00:00:02Z")
    assert isinstance(first, pd.DataFrame)
    assert first["event_number"].tolist() == [1, 0]
    requests_after_first = len(server.requests)

    offline = upbit.get_trades(
        "KRW-BTC",
        "2025-01-01T00:00:00Z",
        "2025-01-01T00:00:02Z",
        offline=True,
    )
    assert isinstance(offline, pd.DataFrame)
    assert offline.equals(first)
    assert len(server.requests) == requests_after_first

    parquet = next(tmp_path.rglob("*.parquet"))
    parquet.write_bytes(b"not parquet")
    recovered = upbit.get_trades(
        "KRW-BTC", "2025-01-01T00:00:00Z", "2025-01-01T00:00:02Z"
    )
    assert isinstance(recovered, pd.DataFrame)
    assert recovered.equals(first)
    assert parquet.stat().st_size > len(b"not parquet")
    assert len(server.requests) > requests_after_first


def test_refresh_relists_resources_but_reuses_valid_materialization(
    tmp_path: Path,
) -> None:
    """Confirm refresh updates discovery without downloading a ready ZIP again."""
    server = UpbitServer()
    upbit = service(tmp_path, server)
    upbit.get_klines("KRW-BTC", "2025-01-01", "2025-01-01", interval="1m")
    object_count = server.count("crix-data.upbit.com", ".zip")
    listing_count = server.count("crix-data-api.upbit.com")

    upbit.get_klines(
        "KRW-BTC",
        "2025-01-01",
        "2025-01-01",
        interval="1m",
        refresh=True,
    )

    assert server.count("crix-data-api.upbit.com") > listing_count
    assert server.count("crix-data.upbit.com", ".zip") == object_count


def test_mixed_existing_and_new_daily_archives_are_queryable_immediately(
    tmp_path: Path,
) -> None:
    """Confirm a larger request includes files downloaded during that same call.

    Args:
        tmp_path: The isolated cache and catalog directory.
    """
    server = UpbitServer((date(2025, 1, 1), date(2025, 1, 2)))
    upbit = service(tmp_path, server)
    first = upbit.get_klines("BTCUSDT", "2025-01-01", "2025-01-01", interval="1m")
    expanded = upbit.get_klines("BTCUSDT", "2025-01-01", "2025-01-02", interval="1d")

    assert isinstance(first, pd.DataFrame)
    assert isinstance(expanded, pd.DataFrame)
    assert len(first) == 2
    assert expanded["open_time"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
    ]
    assert expanded["base_volume"].tolist() == [5.0, 5.0]


def test_daily_archives_resample_to_canonical_coarse_intervals(
    tmp_path: Path,
) -> None:
    """Confirm one-minute Upbit archives produce daily through monthly candles.

    Args:
        tmp_path: The isolated cache and catalog directory.
    """
    days = tuple(date(2025, 1, day) for day in range(1, 32))
    upbit = service(tmp_path, UpbitServer(days))

    daily = upbit.get_klines("BTCUSDT", "2025-01-01", "2025-01-31", interval="1d")
    three_day = upbit.get_klines(
        "BTCUSDT",
        "2025-01-01",
        "2025-01-31",
        interval="3d",
        offline=True,
    )
    weekly = upbit.get_klines(
        "BTCUSDT",
        "2025-01-01",
        "2025-01-31",
        interval="1w",
        offline=True,
    )
    monthly = upbit.get_klines(
        "BTCUSDT",
        "2025-01-01",
        "2025-01-31",
        interval="1mo",
        offline=True,
    )

    assert all(
        isinstance(frame, pd.DataFrame) for frame in (daily, three_day, weekly, monthly)
    )
    assert len(daily) == 31
    assert len(three_day) == 11
    assert len(weekly) == 5
    assert len(monthly) == 1
    assert daily["base_volume"].tolist() == [5.0] * 31
    assert three_day["open_time"].iloc[0] == pd.Timestamp("2024-12-31T00:00:00Z")
    assert weekly["open_time"].iloc[0] == pd.Timestamp("2024-12-30T00:00:00Z")
    assert monthly["open_time"].tolist() == [pd.Timestamp("2025-01-01T00:00:00Z")]
    assert monthly["base_volume"].tolist() == [155.0]
    assert monthly["quote_volume"].tolist() == [15841.0]


def test_availability_changes_only_after_bounded_discovery(tmp_path: Path) -> None:
    """Confirm inspection distinguishes cached metadata from remote discovery."""
    server = UpbitServer()
    upbit = service(tmp_path, server)
    upbit.get_markets()
    before = upbit.get_availability("BTCUSDT", dataset="klines", interval="1h")

    discovered = upbit.discover_availability(
        "BTCUSDT",
        date(2025, 1, 1),
        date(2025, 1, 2),
        dataset="klines",
        interval="1h",
    )
    after = upbit.get_availability("BTCUSDT", dataset="klines", interval="1h")

    assert before.scanned_days == 0
    assert discovered.available_days == 1
    assert discovered.missing_days == 1
    assert after == discovered


@pytest.mark.parametrize(
    ("method", "kwargs", "message"),
    [
        ("get_klines", {"interval": "1hh"}, "interval"),
        ("get_results", {"dataset": "metrics"}, "unsupported dataset"),
        ("get_trades", {"offline": 1}, "offline"),
    ],
)
def test_invalid_requests_fail_before_network_access(
    tmp_path: Path,
    method: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Confirm malformed public requests cannot trigger source traffic."""
    server = UpbitServer()
    upbit = service(tmp_path, server)

    with pytest.raises((TypeError, ValueError), match=message):
        getattr(upbit, method)("BTCUSDT", "2025-01-01", "2025-01-01", **kwargs)

    assert server.requests == []
