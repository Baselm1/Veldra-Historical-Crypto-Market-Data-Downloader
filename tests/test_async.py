"""Test the asynchronous adapter around the synchronous downloader."""

import asyncio
from pathlib import Path
from threading import get_ident

import pandas as pd
import pytest

import crypto_downloader as crypto
from crypto_downloader.downloader import Downloader
from crypto_downloader.source import Source


def test_downloader_async_adapter_runs_get_data_in_a_worker_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm async callers do not run blocking work on their event-loop thread.

    Args:
        tmp_path: The isolated downloader directory.
        monkeypatch: Pytest's replacement helper.
    """
    service = Downloader(tmp_path)
    expected = pd.DataFrame({"close": [100.0]})
    event_loop_thread = get_ident()
    calls: list[tuple[int, tuple[object, ...], dict[str, object]]] = []

    def fake_get_data(*args: object, **kwargs: object) -> pd.DataFrame:
        """Record the worker thread and return a deterministic frame."""
        calls.append((get_ident(), args, kwargs))
        return expected

    monkeypatch.setattr(service, "get_data", fake_get_data)

    actual = asyncio.run(
        service.aget_data(
            "BTCUSDT",
            "2025-01-01",
            "2025-01-02",
            interval="1h",
            desired_columns=["close"],
            offline=True,
            progress=False,
        )
    )

    assert actual is expected
    assert calls[0][0] != event_loop_thread
    assert calls[0][1] == ("BTCUSDT", "2025-01-01", "2025-01-02")
    assert calls[0][2]["interval"] == "1h"
    assert calls[0][2]["desired_columns"] == ["close"]
    assert calls[0][2]["offline"] is True
    assert calls[0][2]["progress"] is False


def test_downloader_async_adapter_propagates_pipeline_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm asynchronous use does not hide synchronous request failures.

    Args:
        tmp_path: The isolated downloader directory.
        monkeypatch: Pytest's replacement helper.
    """
    service = Downloader(tmp_path)

    def fail(*_args: object, **_kwargs: object) -> pd.DataFrame:
        """Raise one representative public validation failure."""
        raise ValueError("bad request")

    monkeypatch.setattr(service, "get_data", fail)

    with pytest.raises(ValueError, match="bad request"):
        asyncio.run(service.aget_data("BTCUSDT", "bad", "2025-01-02"))


def test_module_async_adapter_constructs_and_forwards_downloader_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm the public convenience wrapper retains every important option.

    Args:
        tmp_path: The isolated downloader directory.
        monkeypatch: Pytest's replacement helper.
    """
    expected = [pd.DataFrame({"close": [100.0]}), pd.DataFrame({"close": [200.0]})]
    calls: list[tuple[Downloader, tuple[object, ...], dict[str, object]]] = []

    async def fake_aget_data(
        service: Downloader, *args: object, **kwargs: object
    ) -> list[pd.DataFrame]:
        """Record the constructed service and forwarded request."""
        calls.append((service, args, kwargs))
        return expected

    monkeypatch.setattr(Downloader, "aget_data", fake_aget_data)

    actual = asyncio.run(
        crypto.aget_data(
            ["BTCUSDT", "ETHUSDT"],
            "2025-01-01",
            "2025-01-02",
            data_dir=tmp_path,
            max_workers=7,
            discovery_tail_days=3,
            market_refresh_hours=6,
            interval="4h",
            gap_policy="keep",
            refresh=True,
            progress=False,
        )
    )

    assert actual is expected
    assert calls[0][0].data_dir == tmp_path.resolve()
    assert calls[0][0].max_workers == 7
    assert calls[0][0].discovery_tail_days == 3
    assert calls[0][0].market_refresh_hours == 6
    assert calls[0][1][0] == ["BTCUSDT", "ETHUSDT"]
    assert calls[0][2] == {
        "product": "spot",
        "dataset": "klines",
        "interval": "4h",
        "desired_columns": None,
        "refresh": True,
        "offline": False,
        "gap_policy": "keep",
        "progress": False,
    }
