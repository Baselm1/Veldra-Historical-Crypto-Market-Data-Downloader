"""Test the public Bybit facade contract."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from veldra import Bybit
from veldra.bybit.client import BybitResponseError
from veldra.bybit.history import object_frame
from veldra.core.models import Market


def _klines() -> pd.DataFrame:
    """Return one canonical Spot Kline."""
    return pd.DataFrame(
        {
            "open_time": pd.Series(
                [datetime(2025, 1, 1, tzinfo=UTC)], dtype="datetime64[us, UTC]"
            ),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "base_volume": [1.0],
            "quote_volume": [100.5],
        }
    )


def test_kline_facade_returns_frames_reports_and_pair_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep valid pairs usable when another REST pair is unknown."""
    api = Bybit(tmp_path, progress=False)

    def retrieve(symbol: str, *args: object, **kwargs: object) -> pd.DataFrame:
        if symbol == "BADUSDT":
            raise BybitResponseError("10001", "unknown symbol")
        return _klines()

    monkeypatch.setattr(api._service, "klines", retrieve)
    frames = api.get_klines(
        ["btcusdt", "BADUSDT"],
        "2025-01-01",
        "2025-01-01",
        columns={"open_time": "time", "close": "price"},
    )
    assert isinstance(frames, list)
    assert frames[0].columns.tolist() == ["time", "price"]
    assert frames[0].attrs["download"]["pair"] == "BTCUSDT"
    assert frames[1].empty
    assert frames[1].attrs["download"]["errors"][0]["code"] == "unknown_pair"
    api.close()


def test_empty_success_is_reported_as_an_unavailable_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinguish an empty valid response from a complete populated range."""
    api = Bybit(tmp_path, progress=False)
    monkeypatch.setattr(
        api._service,
        "funding_rates",
        lambda *args, **kwargs: object_frame([], "funding_rates"),
    )
    frame = api.get_funding_rates(
        "BTCUSDT", "2025-01-01", "2025-01-02", product="linear"
    )
    assert isinstance(frame, pd.DataFrame)
    assert frame.attrs["download"]["errors"][0]["code"] == "range_unavailable"
    api.close()


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("get_mark_price_klines", "mark_price_klines"),
        ("get_index_price_klines", "index_price_klines"),
    ],
)
def test_reference_convenience_methods_select_their_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expected: str,
) -> None:
    """Route named reference methods without duplicating retrieval logic."""
    api = Bybit(tmp_path, progress=False)
    captured: dict[str, object] = {}

    def reference(*args: object, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(api, "get_reference_klines", reference)
    getattr(api, method)("BTCUSDT", "2025-01-01", "2025-01-02", product="linear")
    assert captured["dataset"] == expected
    api.close()


def test_premium_convenience_method_is_linear_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind premium-index history to Bybit's supported linear product."""
    api = Bybit(tmp_path, progress=False)
    captured: dict[str, object] = {}

    def reference(*args: object, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(api, "get_reference_klines", reference)
    api.get_premium_index_klines("BTCUSDT", "2025-01-01", "2025-01-02")
    assert captured["product"] == "linear"
    assert captured["dataset"] == "premium_index_klines"
    api.close()


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("get_open_interest", "open_interest"),
        ("get_long_short_ratios", "long_short_ratios"),
    ],
)
def test_position_methods_select_distinct_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expected: str,
) -> None:
    """Keep open interest and account ratios as distinct schemas."""
    api = Bybit(tmp_path, progress=False)
    captured: dict[str, object] = {}

    def positions(*args: object, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(api, "_positions", positions)
    getattr(api, method)("BTCUSDT", "2025-01-01", "2025-01-02", product="inverse")
    assert captured["dataset"] == expected
    api.close()


def test_archive_methods_delegate_typed_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route trades and books through the archive service unchanged."""
    api = Bybit(tmp_path, progress=False)
    calls: list[str] = []

    def archives(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(str(kwargs["dataset"]))
        return pd.DataFrame()

    monkeypatch.setattr(api._service, "archives", archives)
    api.get_trades("BTCUSDT", "2025-01-01", "2025-01-02")
    api.get_order_book_updates("BTCUSDT", "2025-01-01", "2025-01-02")
    assert calls == ["trades", "order_book_updates"]
    api.close()


def test_volatility_and_delivery_return_reported_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose non-Kline derivative histories as ordinary DataFrames."""
    api = Bybit(tmp_path, progress=False)
    volatility = object_frame(
        [{"time": "1735689600000", "period": "30", "value": "0.5"}],
        "historical_volatility",
    )
    delivery = object_frame(
        [{"deliveryTime": "1735689600000", "deliveryPrice": "100"}],
        "delivery_prices",
    )
    monkeypatch.setattr(api._service, "volatility", lambda *args, **kwargs: volatility)
    monkeypatch.setattr(
        api._service, "delivery_prices", lambda *args, **kwargs: delivery
    )
    first = api.get_historical_volatility("btc", "2025-01-01", "2025-01-01")
    second = api.get_delivery_prices(
        "BTC-27DEC24", "2025-01-01", "2025-01-01", product="inverse"
    )
    assert first["volatility"].tolist() == [0.5]
    assert isinstance(second, pd.DataFrame)
    assert second.attrs["download"]["dataset"] == "delivery_prices"
    api.close()


@pytest.mark.parametrize(
    "call",
    [
        lambda api: api.get_klines(
            "BTCUSDT", "2025-01-01", "2025-01-02", interval="60m"
        ),
        lambda api: api.get_klines(
            "BTCUSDT", "2025-01-01", "2025-01-02", refresh="yes"
        ),
        lambda api: api.get_klines(
            "BTCUSDT",
            "2025-01-01",
            "2025-01-02",
            refresh=True,
            offline=True,
        ),
        lambda api: api.get_klines(
            "BTCUSDT",
            "2025-01-01",
            "2025-01-02",
            columns=["not_a_column"],
        ),
    ],
)
def test_rest_facade_rejects_invalid_public_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    call: Any,
) -> None:
    """Reject invalid intervals, flags, combinations, and projections."""
    api = Bybit(tmp_path, progress=False)
    monkeypatch.setattr(api._service, "klines", lambda *args, **kwargs: _klines())
    with pytest.raises((TypeError, ValueError)):
        call(api)
    api.close()


def test_inspection_methods_delegate_to_shared_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose the common market and coverage inspection contracts."""
    api = Bybit(tmp_path, progress=False)
    market = Market("BTCUSDT", "BTCUSDT", source="bybit", product="spot")
    monkeypatch.setattr(
        "veldra.bybit.facade._get_markets", lambda *args, **kwargs: [market]
    )
    monkeypatch.setattr(
        "veldra.bybit.facade._find_markets", lambda *args, **kwargs: [market]
    )
    marker = object()
    monkeypatch.setattr(
        "veldra.bybit.facade._get_availability", lambda *args, **kwargs: marker
    )
    monkeypatch.setattr(
        "veldra.bybit.facade._discover_availability", lambda *args, **kwargs: marker
    )
    assert api.get_markets() == [market]
    assert api.find_markets("btc") == [market]
    assert api.get_availability("BTCUSDT", product="spot", dataset="trades") is marker
    assert (
        api.discover_availability(
            "BTCUSDT",
            "2025-01-01",
            "2025-01-02",
            product="spot",
            dataset="trades",
        )
        is marker
    )
    api.close()


def test_facade_properties_and_context_manager(tmp_path: Path) -> None:
    """Expose configuration and close owned resources deterministically."""
    with Bybit(tmp_path, earliest_date="all", max_workers=7, progress=False) as api:
        assert api.data_dir == tmp_path.resolve()
        assert api.earliest_date is None
        assert api.max_workers == 7
        service = api._service
    assert service._http.is_closed
