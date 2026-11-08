"""Exercise the Gate facade through its complete archive pipeline."""

from datetime import UTC, date, datetime
import gzip
import hashlib
from pathlib import Path

import httpx
import pandas as pd
import pytest

from veldra import Gate
from veldra.core.catalog import open_catalog
from veldra.core.models import ResourceKey


class GateServer:
    """Serve one synthetic Gate Spot market and daily Kline archive."""

    def __init__(self) -> None:
        """Create one official-shaped Gzip payload and request log."""
        self.payload = gzip.compress(
            b"1735689660,2,101,102,99,100\n" b"1735689600,1,100,101,98,99\n"
        )
        self.digest = hashlib.md5(self.payload).hexdigest()
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return market metadata, archive metadata, or archive bytes."""
        self.requests.append(request)
        if request.url.path.endswith("/spot/currency_pairs"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "BTC_USDT",
                        "base": "BTC",
                        "quote": "USDT",
                        "trade_status": "tradable",
                        "sell_start": 1_735_689_600,
                    }
                ],
            )
        if request.url.path.endswith("/spot/tickers"):
            return httpx.Response(
                200,
                json=[{"currency_pair": "BTC_USDT", "quote_volume": "10"}],
            )
        if request.url.path.endswith("BTC_USDT-20250101.csv.gz"):
            return httpx.Response(
                200,
                content=b"" if request.method == "HEAD" else self.payload,
                headers={"ETag": f'"{self.digest}"'},
            )
        return httpx.Response(404)


def service(tmp_path: Path, server: GateServer) -> Gate:
    """Return a Gate facade backed by the synthetic transport.

    Args:
        tmp_path: The isolated catalog and Parquet directory.
        server: The synthetic Gate service.

    Returns:
        A facade using the mock transport.
    """
    facade = Gate(tmp_path, earliest_date="all", retries=0, progress=False)
    facade._downloader.transport = httpx.MockTransport(server)
    return facade


def test_spot_klines_are_discovered_verified_cached_and_queried(
    tmp_path: Path,
) -> None:
    """Confirm one facade call produces an exact DataFrame and reusable cache.

    Args:
        tmp_path: The isolated catalog and Parquet directory.
    """
    server = GateServer()
    gate = service(tmp_path, server)

    frame = gate.get_klines(
        "BTCUSDT",
        "2025-01-01T00:00:00Z",
        "2025-01-01T00:02:00Z",
        interval="1m",
    )

    assert isinstance(frame, pd.DataFrame)
    assert frame.open.tolist() == [99.0, 100.0]
    assert frame.attrs["download"]["pair"] == "BTC_USDT"
    requests = len(server.requests)

    cached = gate.get_klines(
        "BTC_USDT",
        "2025-01-01T00:00:00Z",
        "2025-01-01T00:02:00Z",
        interval="1m",
        offline=True,
    )

    assert isinstance(cached, pd.DataFrame)
    assert cached.equals(frame)
    assert len(server.requests) == requests


def test_unknown_pair_is_isolated_in_structured_results(tmp_path: Path) -> None:
    """Confirm invalid markets return suggestions instead of stopping the call.

    Args:
        tmp_path: The isolated catalog and Parquet directory.
    """
    result = service(tmp_path, GateServer()).get_results(
        "BTCSUDT",
        "2025-01-01",
        "2025-01-01",
        product="spot",
        dataset="klines",
        interval="1m",
    )

    assert not isinstance(result, list)
    assert result.data.empty
    assert result.errors[0].code == "unknown_pair"
    assert result.errors[0].suggestions == ("BTC_USDT",)


def test_spot_klines_cross_the_monthly_to_daily_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recover the older generation even if a daily-only boundary was cached."""
    transition = date(2024, 4, 1)
    monkeypatch.setattr("veldra.core.engine.utc_today", lambda: date(2024, 4, 2))
    monkeypatch.setattr("veldra.core.inspection.utc_today", lambda: date(2024, 4, 2))
    start = datetime(2024, 3, 31, 23, 59, tzinfo=UTC)
    middle = datetime(2024, 4, 1, tzinfo=UTC)
    end = datetime(2024, 4, 1, 0, 1, tzinfo=UTC)
    archives = {
        "BTC_USDT-202403.csv.gz": gzip.compress(
            f"{int(start.timestamp())},1,100,101,98,99\n".encode()
        ),
        "BTC_USDT-20240401.csv.gz": gzip.compress(
            f"{int(middle.timestamp())},2,101,102,99,100\n".encode()
        ),
    }
    market_server = GateServer()
    probed: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """Serve one older month and one recent day using Gate's URL layouts."""
        if request.url.path.endswith("/spot/currency_pairs"):
            markets = market_server(request).json()
            markets[0]["sell_start"] = int(datetime(2024, 3, 1, tzinfo=UTC).timestamp())
            return httpx.Response(200, json=markets)
        name = request.url.path.rsplit("/", 1)[-1]
        probed.append(name)
        payload = archives.get(name)
        if payload is None:
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers={"ETag": hashlib.md5(payload).hexdigest()},
            content=b"" if request.method == "HEAD" else payload,
        )

    key = ResourceKey("gate", "spot", "klines", "BTC_USDT", "1m")
    with open_catalog(tmp_path / "catalog.duckdb") as catalog:
        catalog.save_source_bounds(key, transition, None)

    gate = service(tmp_path, market_server)
    gate._downloader.transport = httpx.MockTransport(respond)
    discovered = gate.discover_availability(
        "BTCUSDT", start, end, product="spot", dataset="klines", interval="1m"
    )
    assert discovered.remote_range == (date(2024, 3, 1), transition)
    frame = gate.get_klines("BTCUSDT", start, end)
    assert isinstance(frame, pd.DataFrame)
    assert frame.open_time.tolist() == [start, middle]
    assert set(probed) == set(archives)
    cached = gate.get_klines("BTCUSDT", start, end, offline=True)
    assert isinstance(cached, pd.DataFrame)
    pd.testing.assert_frame_equal(frame, cached)


def test_sparse_order_book_day_discovers_and_refreshes_available_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load a day without hour zero and rebuild it when another hour appears."""
    monkeypatch.setattr("veldra.core.engine.utc_today", lambda: date(2025, 1, 2))
    market_server = GateServer()
    payloads = {"12": gzip.compress(b"1735732800,2,set,100,2,10,0\n")}

    def respond(request: httpx.Request) -> httpx.Response:
        """Serve the available depth hours and the ordinary market fixture."""
        if "/orderbooks/" not in request.url.path:
            return market_server(request)
        payload = payloads.get(request.url.path[-9:-7])
        if payload is None:
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers={"ETag": hashlib.md5(payload).hexdigest()},
            content=b"" if request.method == "HEAD" else payload,
        )

    gate = service(tmp_path, market_server)
    gate._downloader.transport = httpx.MockTransport(respond)
    day = "2025-01-01"
    first = gate.get_order_book_updates("BTCUSDT", day, day)
    assert isinstance(first, pd.DataFrame)
    assert first.update_id.tolist() == [10]

    payloads["13"] = gzip.compress(b"1735736400,1,make,101,3,11,1\n")
    refreshed = gate.get_order_book_updates("BTCUSDT", day, day, refresh=True)
    assert isinstance(refreshed, pd.DataFrame)
    assert refreshed.update_id.tolist() == [10, 11]
    cached = gate.get_order_book_updates("BTCUSDT", day, day, offline=True)
    assert isinstance(cached, pd.DataFrame)
    pd.testing.assert_frame_equal(cached, refreshed)


def test_partial_resampling_edges_are_omitted_and_reported(tmp_path: Path) -> None:
    """Confirm exact Gate ranges never expose incomplete coarse candles.

    Args:
        tmp_path: The isolated catalog and Parquet directory.
    """
    result = service(tmp_path, GateServer()).get_results(
        "BTCUSDT",
        "2025-01-01T00:01:00Z",
        "2025-01-01T00:02:00Z",
        product="spot",
        dataset="klines",
        interval="3m",
    )

    assert not isinstance(result, list)
    assert result.data.empty
    assert [warning.code for warning in result.warnings] == ["partial_buckets_trimmed"]
    assert result.complete is True
