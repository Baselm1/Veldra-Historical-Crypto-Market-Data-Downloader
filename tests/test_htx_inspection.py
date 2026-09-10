"""Test HTX facade delegation to source-neutral inspection services."""

from pathlib import Path

import pytest

import veldra.htx.facade as facade_module
from veldra.core.models import Availability, Market
from veldra.core.inspection import _matches_filters
from veldra.htx.facade import HTX


def availability() -> Availability:
    """Build a minimal immutable availability result.

    Returns:
        Empty known coverage for one HTX Spot dataset.
    """
    return Availability(
        source="htx",
        product="spot",
        dataset="klines",
        symbol="BTCUSDT",
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


def test_native_status_filters_are_case_insensitive() -> None:
    """Confirm lower-case HTX statuses match normalized public filters."""
    market = Market("BTCUSDT", "BTCUSDT", status="online", active=True)

    assert _matches_filters(market, "ONLINE", None)


def test_market_inspection_methods_delegate_all_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm market listing and fuzzy search use the configured engine.

    Args:
        tmp_path: The isolated data directory.
        monkeypatch: Pytest's attribute replacement helper.
    """
    service = HTX(tmp_path, progress=False)
    expected = [Market("BTCUSDT", "BTCUSDT", active=True)]
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
    assert (
        service.find_markets("BTCSUDT", product="spot", limit=3, offline=True)
        is expected
    )
    assert calls[0][2] == {
        "product": "spot",
        "status": "online",
        "quote_asset": "USDT",
        "sort_by": "quote_volume",
        "limit": 5,
        "refresh": True,
        "offline": False,
        "progress": False,
    }
    assert calls[1][1][1:] == ("BTCSUDT",)
    assert calls[1][2]["limit"] == 3
    assert calls[1][2]["offline"] is True


def test_availability_methods_delegate_bounded_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm local and remote coverage inspection preserve dataset identity.

    Args:
        tmp_path: The isolated data directory.
        monkeypatch: Pytest's attribute replacement helper.
    """
    service = HTX(tmp_path, progress=False)
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
            "2026-09-01",
            "2026-09-07",
            product="spot",
            dataset="klines",
            interval="1h",
            refresh=True,
        )
        is expected
    )
    assert calls[0][1][1:] == ("BTCUSDT",)
    assert calls[0][2] == {
        "product": "spot",
        "dataset": "klines",
        "interval": "1h",
    }
    assert calls[1][1][1:] == ("BTCUSDT", "2026-09-01", "2026-09-07")
    assert calls[1][2] == {
        "product": "spot",
        "dataset": "klines",
        "interval": "1h",
        "refresh": True,
        "progress": False,
    }
