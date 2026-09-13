"""Test the public Upbit historical-data facade."""

from datetime import date
import inspect
from pathlib import Path

import pandas as pd
import pytest

import veldra.upbit.facade as facade_module
from veldra import Upbit
from veldra.core.engine import RetrievalEngine
from veldra.core.models import Availability, Market, Result
from veldra.upbit.connector import UpbitConnector

type RetrievalCall = tuple[str, tuple[object, ...], dict[str, object]]


def facade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Upbit, list[RetrievalCall]]:
    """Return an Upbit facade whose retrieval calls are recorded."""
    service = Upbit(tmp_path, progress=False)
    calls: list[RetrievalCall] = []

    def fake_get_data(
        *args: object, **kwargs: object
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Record one DataFrame retrieval call."""
        calls.append(("data", args, kwargs))
        return pd.DataFrame({"pair": [str(args[0])]})

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
    """Return an empty Upbit coverage result."""
    return Availability(
        source="upbit",
        product="spot",
        dataset="klines",
        symbol="USDT-BTC",
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


def test_facade_construction_wires_upbit_without_io(tmp_path: Path) -> None:
    """Confirm construction creates the configured source without I/O."""
    data_dir = tmp_path / "cache"
    service = Upbit(
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
    assert isinstance(service._downloader.source, UpbitConnector)
    assert service._downloader.source.timeout == 8
    assert service._downloader.source.retries == 1
    assert service._downloader.source.backoff == 0.25
    assert not data_dir.exists()


def test_history_boundary_uses_config_then_explicit_overrides(tmp_path: Path) -> None:
    """Confirm Upbit distinguishes configured, complete, and explicit history.

    Args:
        tmp_path: The isolated directory containing a custom configuration.
    """
    config = tmp_path / "upbit.toml"
    config.write_text(
        '[history]\nearliest_date = "2021-02-03"\n\n'
        '[klines]\nbase_interval = "1m"\n',
        encoding="utf-8",
    )

    configured = Upbit(tmp_path / "configured", config_path=config, progress=False)
    complete = Upbit(
        tmp_path / "complete",
        config_path=config,
        earliest_date="all",
        progress=False,
    )
    explicit = Upbit(
        tmp_path / "explicit",
        config_path=config,
        earliest_date=date(2022, 4, 5),
        progress=False,
    )

    assert configured.earliest_date == date(2021, 2, 3)
    assert complete.earliest_date is None
    assert explicit.earliest_date == date(2022, 4, 5)


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
    """Confirm invalid constructor values fail immediately."""
    with pytest.raises((TypeError, ValueError), match=message):
        Upbit(tmp_path, **{option: value})  # type: ignore[arg-type]


def test_dataset_methods_delegate_exact_engine_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm the two public dataset methods preserve their options."""
    service, calls = facade(tmp_path, monkeypatch)

    frame = service.get_klines(
        "BTCUSDT",
        date(2025, 1, 1),
        date(2025, 1, 2),
        interval="1h",
        columns=["open_time"],
        refresh=True,
    )
    service.get_trades(
        ["KRW-BTC", "USDT-BTC"],
        "2025-01-01",
        "2025-01-02",
        columns={"event_time": "time"},
        offline=True,
    )

    assert isinstance(frame, pd.DataFrame)
    assert calls[0][2] == {
        "product": "spot",
        "dataset": "klines",
        "interval": "1h",
        "desired_columns": ["open_time"],
        "gap_policy": "keep",
        "refresh": True,
        "offline": False,
        "progress": False,
    }
    assert calls[1][2]["dataset"] == "trades"
    assert calls[1][2]["gap_policy"] is None


def test_structured_results_delegate_without_changing_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm callers can request structured engine results explicitly."""
    service, calls = facade(tmp_path, monkeypatch)

    result = service.get_results(
        "KRW-BTC",
        "2025-01-01",
        "2025-01-02",
        dataset="trades",
        refresh=True,
    )

    assert isinstance(result, Result)
    assert calls[0][0] == "results"
    assert calls[0][2]["product"] == "spot"
    assert calls[0][2]["dataset"] == "trades"

    service.get_results(
        "KRW-BTC",
        "2025-01-01",
        "2025-01-02",
        dataset="klines",
    )
    assert calls[1][2]["gap_policy"] == "keep"


def test_non_kline_method_excludes_kline_options() -> None:
    """Confirm trade calls cannot accidentally receive Kline options."""
    parameters = inspect.signature(Upbit.get_trades).parameters
    assert "interval" not in parameters
    assert "gap_policy" not in parameters


def test_market_inspection_methods_fix_spot_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm market list and search calls are always scoped to Spot."""
    service = Upbit(tmp_path, progress=False)
    expected = [Market("USDT-BTC", "BTCUSDT", active=True)]
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def inspect_markets(*args: object, **kwargs: object) -> list[Market]:
        """Record one shared market-inspection call."""
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(facade_module, "_get_markets", inspect_markets)
    monkeypatch.setattr(facade_module, "_find_markets", inspect_markets)

    assert service.get_markets(sort_by="quote_volume", limit=5) is expected
    assert service.find_markets("BTCSUDT", offline=True) is expected
    assert calls[0][1]["product"] == "spot"
    assert calls[1][0][1] == "BTCSUDT"
    assert calls[1][1]["offline"] is True


def test_availability_methods_delegate_bounded_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm local and discovered coverage retain dataset identity."""
    service = Upbit(tmp_path, progress=False)
    expected = coverage_result()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def inspect_coverage(*args: object, **kwargs: object) -> Availability:
        """Record one shared coverage call."""
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(facade_module, "_get_availability", inspect_coverage)
    monkeypatch.setattr(facade_module, "_discover_availability", inspect_coverage)

    assert (
        service.get_availability("BTCUSDT", dataset="klines", interval="1h") is expected
    )
    assert (
        service.discover_availability(
            "BTCUSDT",
            "2025-01-01",
            "2025-01-07",
            dataset="trades",
            refresh=True,
        )
        is expected
    )
    assert calls[0][1]["product"] == "spot"
    assert calls[1][0][1:] == ("BTCUSDT", "2025-01-01", "2025-01-07")


def test_upbit_module_publishes_only_the_facade() -> None:
    """Confirm users receive a collision-free class-based API."""
    import veldra.upbit as module

    assert module.__all__ == ("Upbit",)
    assert not hasattr(module, "get_klines")
    assert not hasattr(module, "UpbitConnector")
