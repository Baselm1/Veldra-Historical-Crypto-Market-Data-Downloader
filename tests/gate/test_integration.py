"""Exercise the Gate facade through its complete archive pipeline."""

import gzip
import hashlib
from pathlib import Path

import httpx
import pandas as pd

from veldra import Gate


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
