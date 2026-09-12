"""Test the public KuCoin data facade."""

from datetime import date
import inspect
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import veldra.kucoin.facade as facade_module
from veldra import KuCoin
from veldra.core.engine import RetrievalEngine
from veldra.core.models import Availability, Market
from veldra.kucoin.connector import KuCoinConnector


def facade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[KuCoin, list[object]]:
    """Create a facade whose engine records calls without doing I/O.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace engine retrieval.

    Returns:
        The configured facade and mutable call record.
    """
    service = KuCoin(tmp_path, progress=False)
    calls: list[object] = []

    def fake_get_data(
        *args: object, **kwargs: object
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Record one engine call and preserve its pair-input shape."""
        calls.append((args, kwargs))
        pairs = args[0]
        if isinstance(pairs, str):
            return pd.DataFrame({"pair": [pairs]})
        assert isinstance(pairs, list)
        return [pd.DataFrame({"pair": [pair]}) for pair in pairs]

    monkeypatch.setattr(service._downloader, "get_data", fake_get_data)
    return service, calls


def availability() -> Availability:
    """Return an empty KuCoin coverage result."""
    return Availability(
        source="kucoin",
        product="spot",
        dataset="klines",
        symbol="BTC-USDT",
        interval="1h",
        storage_interval="1m",
        remote_range=None,
        configured_range=None,
        cached_range=None,
        scanned_ranges=(),
        scanned_days=0,
        available_days=0,
        cached_days=0,
        missing_days=0,
        unavailable_days=0,
        failed_days=0,
        row_count=0,
        local_bytes=0,
    )


def test_facade_construction_wires_kucoin_without_io(tmp_path: Path) -> None:
    """Confirm construction validates settings and creates no cache files.

    Args:
        tmp_path: The isolated parent of the proposed data directory.
    """
    data_dir = tmp_path / "cache"
    service = KuCoin(
        data_dir,
        earliest_date="2022-01-01",
        max_workers=12,
        discovery_tail_days=4,
        market_refresh_hours=6,
        timeout=8,
        retries=1,
        backoff=0.25,
        progress=False,
    )

    assert service.data_dir == data_dir.resolve()
    assert service.earliest_date == date(2022, 1, 1)
    assert service.kline_base_interval == "1m"
    assert service.max_workers == 12
    assert isinstance(service._downloader, RetrievalEngine)
    assert isinstance(service._downloader.source, KuCoinConnector)
    assert service._downloader.source.timeout == 8
    assert service._downloader.source.retries == 1
    assert service._downloader.source.backoff == 0.25
    assert not data_dir.exists()


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("timeout", 0, "timeout"),
        ("timeout", True, "timeout"),
        ("retries", -1, "retries"),
        ("retries", 1.5, "retries"),
        ("backoff", -0.1, "backoff"),
        ("backoff", float("nan"), "backoff"),
        ("progress", 1, "progress"),
    ],
)
def test_facade_rejects_invalid_network_and_display_settings(
    tmp_path: Path, option: str, value: object, message: str
) -> None:
    """Confirm invalid settings fail during construction.

    Args:
        tmp_path: The isolated configured data directory.
        option: The invalid constructor keyword.
        value: The invalid setting value.
        message: Text expected in the validation failure.
    """
    with pytest.raises((TypeError, ValueError), match=message):
        KuCoin(tmp_path, **{option: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method", "dataset", "product", "kline"),
    [
        ("get_klines", "klines", "spot", True),
        ("get_trades", "trades", "spot", False),
        ("get_index_price_klines", "index_price_klines", "linear_futures", True),
        ("get_mark_price_klines", "mark_price_klines", "inverse_futures", True),
        ("get_funding_rates", "funding_rates", "linear_futures", False),
        (
            "get_order_book_snapshots",
            "order_book_snapshots",
            "inverse_futures",
            False,
        ),
    ],
)
def test_dataset_methods_delegate_exact_engine_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    dataset: str,
    product: str,
    kline: bool,
) -> None:
    """Confirm public methods fix datasets and forward supported options.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace retrieval.
        method: The public facade method under test.
        dataset: The expected internal dataset identifier.
        product: The requested KuCoin product.
        kline: Whether Kline-only options are accepted.
    """
    service, calls = facade(tmp_path, monkeypatch)
    kwargs: dict[str, object] = {
        "product": product,
        "columns": ["open_time" if kline else "event_time"],
        "refresh": True,
        "offline": False,
    }
    if kline:
        kwargs.update(interval="1h", gap_policy="keep")

    result = getattr(service, method)("BTCUSDT", "2025-01-01", "2025-01-02", **kwargs)

    assert isinstance(result, pd.DataFrame)
    assert calls == [
        (
            ("BTCUSDT", "2025-01-01", "2025-01-02"),
            {
                "product": product,
                "dataset": dataset,
                "interval": "1h" if kline else None,
                "desired_columns": kwargs["columns"],
                "gap_policy": "keep" if kline else None,
                "refresh": True,
                "offline": False,
                "progress": False,
            },
        )
    ]


@pytest.mark.parametrize(
    "method", ["get_klines", "get_trades", "get_order_book_snapshots"]
)
def test_common_dataset_methods_default_to_spot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Confirm datasets shared with Spot use Spot by default.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace retrieval.
        method: The public facade method under test.
    """
    service, calls = facade(tmp_path, monkeypatch)

    getattr(service, method)("BTCUSDT", "2025-01-01", "2025-01-01")

    _, options = cast(tuple[tuple[object, ...], dict[str, object]], calls[0])
    assert options["product"] == "spot"


@pytest.mark.parametrize(
    "method",
    ["get_index_price_klines", "get_mark_price_klines", "get_funding_rates"],
)
def test_futures_only_methods_require_product_keyword(
    tmp_path: Path, method: str
) -> None:
    """Confirm Futures-only calls cannot silently assume a product.

    Args:
        tmp_path: The isolated configured data directory.
        method: The Futures-only facade method under test.
    """
    with pytest.raises(TypeError, match="product"):
        getattr(KuCoin(tmp_path, progress=False), method)(
            "BTCUSDT", "2025-01-01", "2025-01-01"
        )


@pytest.mark.parametrize(
    "method", ["get_trades", "get_funding_rates", "get_order_book_snapshots"]
)
def test_non_kline_signatures_exclude_kline_options(method: str) -> None:
    """Confirm event and snapshot methods exclude Kline-only options.

    Args:
        method: The public method whose signature is inspected.
    """
    parameters = inspect.signature(getattr(KuCoin, method)).parameters
    assert "interval" not in parameters
    assert "gap_policy" not in parameters


def test_pair_lists_preserve_engine_result_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm list requests retain order and duplicates.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace retrieval.
    """
    service, calls = facade(tmp_path, monkeypatch)
    frames = service.get_klines(
        ["BTCUSDT", "ETHUSDT", "BTCUSDT"], "2025-01-01", "2025-01-01"
    )

    assert isinstance(frames, list)
    assert [frame.loc[0, "pair"] for frame in frames] == [
        "BTCUSDT",
        "ETHUSDT",
        "BTCUSDT",
    ]
    arguments, _ = cast(tuple[tuple[object, ...], dict[str, object]], calls[0])
    assert arguments[0] == ["BTCUSDT", "ETHUSDT", "BTCUSDT"]


def test_market_inspection_methods_delegate_all_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm market listing and fuzzy search use the configured engine.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace inspection calls.
    """
    service = KuCoin(tmp_path, progress=False)
    expected = [Market("BTC-USDT", "BTCUSDT", active=True)]
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def get_markets(*args: object, **kwargs: object) -> list[Market]:
        """Record one market-list request."""
        calls.append(("get", args, kwargs))
        return expected

    def find_markets(*args: object, **kwargs: object) -> list[Market]:
        """Record one market-search request."""
        calls.append(("find", args, kwargs))
        return expected

    monkeypatch.setattr(facade_module, "_get_markets", get_markets)
    monkeypatch.setattr(facade_module, "_find_markets", find_markets)

    assert (
        service.get_markets(
            product="spot",
            status="online",
            quote_asset="USDT",
            sort_by="quote_volume",
            limit=5,
            refresh=True,
        )
        is expected
    )
    assert service.find_markets("BTCSUDT", product="spot", offline=True) is expected
    assert calls[0][2]["sort_by"] == "quote_volume"
    assert calls[0][2]["limit"] == 5
    assert calls[1][1][1:] == ("BTCSUDT",)
    assert calls[1][2]["offline"] is True


def test_availability_methods_delegate_bounded_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm known and remote coverage preserve dataset identity.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace inspection calls.
    """
    service = KuCoin(tmp_path, progress=False)
    expected = availability()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def get_availability(*args: object, **kwargs: object) -> Availability:
        """Record one local availability request."""
        calls.append(("get", args, kwargs))
        return expected

    def discover_availability(*args: object, **kwargs: object) -> Availability:
        """Record one remote availability request."""
        calls.append(("discover", args, kwargs))
        return expected

    monkeypatch.setattr(facade_module, "_get_availability", get_availability)
    monkeypatch.setattr(facade_module, "_discover_availability", discover_availability)

    assert (
        service.get_availability(
            "BTCUSDT", product="spot", dataset="klines", interval="1h"
        )
        is expected
    )
    assert (
        service.discover_availability(
            "BTCUSDT",
            "2025-01-01",
            "2025-01-07",
            product="spot",
            dataset="klines",
            interval="1h",
            refresh=True,
        )
        is expected
    )
    assert calls[0][2]["dataset"] == "klines"
    assert calls[1][1][1:] == ("BTCUSDT", "2025-01-01", "2025-01-07")
    assert calls[1][2]["refresh"] is True


def test_kucoin_module_publishes_only_the_facade() -> None:
    """Confirm users receive the facade without colliding free functions."""
    import veldra.kucoin as module

    assert module.__all__ == ("KuCoin",)
    assert not hasattr(module, "get_klines")
    assert not hasattr(module, "get_trades")
