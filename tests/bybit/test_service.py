"""Test Bybit transport orchestration and cache identities."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from veldra.bybit.service import BybitService


def _kline_frame() -> pd.DataFrame:
    """Return one canonical Spot candle."""
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True).astype(
                "datetime64[us, UTC]"
            ),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "base_volume": [1.0],
            "quote_volume": [100.5],
        }
    )


def test_service_caches_history_without_a_second_source_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route Klines through the persistent REST range cache."""
    service = BybitService(tmp_path, retries=0)
    calls = 0

    def klines(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return _kline_frame()

    monkeypatch.setattr(service.history, "klines", klines)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 0, 1, tzinfo=UTC)
    first = service.klines("BTCUSDT", "spot", "klines", "1m", start, end)
    cached = service.klines("BTCUSDT", "spot", "klines", "1m", start, end, offline=True)
    service.close()
    assert first.equals(cached)
    assert calls == 1


def test_position_periods_have_distinct_cache_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never answer one native analytical period from another period's cache."""
    service = BybitService(tmp_path, retries=0)
    calls: list[str] = []

    def positions(
        symbol: str,
        product: str,
        dataset: str,
        period: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        del symbol, product, dataset, end
        calls.append(period)
        return pd.DataFrame(
            {
                "event_time": pd.Series([start], dtype="datetime64[us, UTC]"),
                "open_interest": [1.0],
                "single_open_interest": [float("nan")],
            }
        )

    monkeypatch.setattr(service.history, "positions", positions)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 1, tzinfo=UTC)
    service.positions("BTCUSDT", "linear", "open_interest", "5m", start, end)
    service.positions("BTCUSDT", "linear", "open_interest", "1h", start, end)
    service.close()
    assert calls == ["5m", "1h"]


def test_archive_requests_delegate_to_the_shared_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep archive orchestration in the source-neutral retrieval engine."""
    service = BybitService(tmp_path, retries=0)
    expected = pd.DataFrame({"event_time": []})
    monkeypatch.setattr(
        service.downloader, "get_data", lambda *args, **kwargs: expected
    )
    result = service.archives(
        "BTCUSDT",
        "2025-01-01",
        "2025-01-02",
        product="spot",
        dataset="trades",
        progress=False,
    )
    service.close()
    assert result is expected


def test_service_is_a_closing_context_manager(tmp_path: Path) -> None:
    """Release its owned persistent HTTP connection pool deterministically."""
    with BybitService(tmp_path, retries=0) as service:
        assert not service._http.is_closed
    assert service._http.is_closed
