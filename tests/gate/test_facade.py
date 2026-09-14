"""Test the public Gate historical-data facade."""

from datetime import date
import inspect
from pathlib import Path

import pandas as pd
import pytest

import veldra.gate.facade as facade_module
from veldra import Gate
from veldra.core.engine import RetrievalEngine
from veldra.core.models import Availability, Market, Result
from veldra.gate.connector import GateConnector

type RetrievalCall = tuple[str, tuple[object, ...], dict[str, object]]


def facade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Gate, list[RetrievalCall]]:
    """Return a Gate facade whose retrieval calls are recorded.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace engine retrieval.

    Returns:
        The facade and mutable call record.
    """
    service = Gate(tmp_path, progress=False)
    calls: list[RetrievalCall] = []

    def fake_get_data(
        *args: object, **kwargs: object
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Record one DataFrame retrieval call."""
        calls.append(("data", args, kwargs))
        pairs = args[0]
        if isinstance(pairs, str):
            return pd.DataFrame({"pair": [pairs]})
        assert isinstance(pairs, list)
        return [pd.DataFrame({"pair": [pair]}) for pair in pairs]

    def fake_get_results(*args: object, **kwargs: object) -> Result | list[Result]:
        """Record one structured retrieval call."""
        calls.append(("results", args, kwargs))
        start = pd.Timestamp("2025-01-01", tz="UTC")
        end = pd.Timestamp("2025-01-02", tz="UTC")
        return Result(str(args[0]), pd.DataFrame(), (start, end))

    monkeypatch.setattr(service._downloader, "get_data", fake_get_data)
    monkeypatch.setattr(service._downloader, "get_results", fake_get_results)
    return service, calls


def coverage_result() -> Availability:
    """Return an empty Gate coverage result."""
    return Availability(
        source="gate",
        product="spot",
        dataset="klines",
        symbol="BTC_USDT",
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


def test_facade_construction_wires_gate_without_io(tmp_path: Path) -> None:
    """Confirm construction creates the configured source without I/O.

    Args:
        tmp_path: The isolated parent of the proposed cache.
    """
    data_dir = tmp_path / "cache"
    service = Gate(
        data_dir,
        earliest_date="all",
        max_workers=12,
        discovery_tail_days=4,
        market_refresh_hours=6,
        timeout=8,
        retries=1,
        backoff=0.25,
        progress=False,
    )

    assert service.data_dir == data_dir.resolve()
    assert service.earliest_date is None
    assert service.kline_base_interval == "1m"
    assert service.max_workers == 12
    assert isinstance(service._downloader, RetrievalEngine)
    assert isinstance(service._downloader.source, GateConnector)
    assert service._downloader.source.timeout == 8
    assert service._downloader.source.retries == 1
    assert service._downloader.source.backoff == 0.25
    assert not data_dir.exists()


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("timeout", 0, "timeout"),
        ("retries", -1, "retries"),
        ("backoff", float("nan"), "backoff"),
        ("progress", 1, "progress"),
    ],
)
def test_facade_rejects_invalid_settings(
    tmp_path: Path, option: str, value: object, message: str
) -> None:
    """Confirm invalid constructor values fail immediately.

    Args:
        tmp_path: The isolated configured data directory.
        option: The invalid constructor keyword.
        value: The invalid setting value.
        message: Text expected in the validation failure.
    """
    with pytest.raises((TypeError, ValueError), match=message):
        Gate(tmp_path, **{option: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method", "dataset", "product", "kline"),
    [
        ("get_klines", "klines", "spot", True),
        ("get_trades", "trades", "um", False),
        ("get_order_book_updates", "order_book_updates", "cm", False),
        ("get_order_book_snapshots", "order_book_snapshots", "spot", False),
        ("get_mark_prices", "mark_prices", "um", False),
        ("get_funding_rates", "funding_rates", "cm", False),
        ("get_funding_rate_updates", "funding_rate_updates", "um", False),
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
        product: The requested Gate product.
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
    assert calls[0][2] == {
        "product": product,
        "dataset": dataset,
        "interval": "1h" if kline else None,
        "desired_columns": kwargs["columns"],
        "gap_policy": "keep" if kline else None,
        "refresh": True,
        "offline": False,
        "progress": False,
    }


@pytest.mark.parametrize(
    "method",
    ["get_klines", "get_trades", "get_order_book_updates", "get_order_book_snapshots"],
)
def test_common_datasets_default_to_spot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Confirm shared market datasets use Spot by default.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace retrieval.
        method: The public method under test.
    """
    service, calls = facade(tmp_path, monkeypatch)

    getattr(service, method)("BTCUSDT", "2025-01-01", "2025-01-01")

    assert calls[0][2]["product"] == "spot"


def test_explicit_none_uses_the_same_kline_gap_policy_as_omission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm every Gate Kline entry point treats ``None`` as ``keep``.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace retrieval.
    """
    service, calls = facade(tmp_path, monkeypatch)

    service.get_klines("BTCUSDT", "2025-01-01", "2025-01-01", gap_policy=None)
    service.get_results(
        "BTCUSDT",
        "2025-01-01",
        "2025-01-01",
        product="spot",
        dataset="klines",
        gap_policy=None,
    )

    assert [call[2]["gap_policy"] for call in calls] == ["keep", "keep"]


@pytest.mark.parametrize(
    "method", ["get_mark_prices", "get_funding_rates", "get_funding_rate_updates"]
)
def test_futures_reference_methods_require_product(tmp_path: Path, method: str) -> None:
    """Confirm Futures-only calls cannot silently assume a product.

    Args:
        tmp_path: The isolated configured data directory.
        method: The Futures-only public method under test.
    """
    with pytest.raises(TypeError, match="product"):
        getattr(Gate(tmp_path, progress=False), method)(
            "BTCUSDT", "2025-01-01", "2025-01-01"
        )


def test_structured_results_preserve_dataset_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm structured requests retain product, schema, and gap behavior.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace retrieval.
    """
    service, calls = facade(tmp_path, monkeypatch)

    result = service.get_results(
        "BTC_USDT",
        "2025-01-01",
        "2025-01-02",
        product="spot",
        dataset="klines",
    )

    assert isinstance(result, Result)
    assert calls[0][0] == "results"
    assert calls[0][2]["gap_policy"] == "keep"

    service.get_results(
        "BTC_USDT",
        "2025-01-01",
        "2025-01-02",
        product="um",
        dataset="funding_rates",
        columns={"event_time": "time"},
        refresh=True,
    )
    assert calls[1][2]["gap_policy"] is None
    assert calls[1][2]["desired_columns"] == {"event_time": "time"}


def test_pair_lists_preserve_input_shape_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm multi-market requests retain duplicates and order.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace retrieval.
    """
    service, _ = facade(tmp_path, monkeypatch)

    frames = service.get_trades(
        ["BTC_USDT", "ETH_USDT", "BTC_USDT"], "2025-01-01", "2025-01-01"
    )

    assert isinstance(frames, list)
    assert [frame.loc[0, "pair"] for frame in frames] == [
        "BTC_USDT",
        "ETH_USDT",
        "BTC_USDT",
    ]


@pytest.mark.parametrize(
    "method",
    [
        "get_trades",
        "get_order_book_updates",
        "get_order_book_snapshots",
        "get_mark_prices",
        "get_funding_rates",
        "get_funding_rate_updates",
    ],
)
def test_non_kline_signatures_exclude_kline_options(method: str) -> None:
    """Confirm non-Kline calls cannot receive Kline-only options.

    Args:
        method: The public method whose signature is inspected.
    """
    parameters = inspect.signature(getattr(Gate, method)).parameters
    assert "interval" not in parameters
    assert "gap_policy" not in parameters


def test_market_inspection_methods_delegate_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm market listing and fuzzy search use the shared inspector.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace inspection calls.
    """
    service = Gate(tmp_path, progress=False)
    expected = [Market("BTC_USDT", "BTCUSDT", active=True)]
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
            product="um",
            status="trading",
            quote_asset="USDT",
            sort_by="quote_volume",
            limit=5,
            refresh=True,
        )
        is expected
    )
    assert service.find_markets("BTCSUDT", product="spot", offline=True) is expected
    assert calls[0][2]["product"] == "um"
    assert calls[0][2]["sort_by"] == "quote_volume"
    assert calls[1][2]["offline"] is True


def test_availability_methods_delegate_bounded_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm local and remote coverage preserve Gate identities.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace inspection calls.
    """
    service = Gate(tmp_path, progress=False)
    expected = coverage_result()
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
            "BTC_USDT", product="spot", dataset="klines", interval="1h"
        )
        is expected
    )
    assert (
        service.discover_availability(
            "BTC_USDT",
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
    assert calls[1][2]["refresh"] is True


def test_gate_module_publishes_only_the_facade() -> None:
    """Confirm users receive the facade without colliding free functions."""
    import veldra.gate as module

    assert module.__all__ == ("Gate",)
    assert not hasattr(module, "get_klines")
    assert not hasattr(module, "get_trades")
