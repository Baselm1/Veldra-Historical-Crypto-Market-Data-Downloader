"""Test the public Binance data facade."""

from datetime import date
import inspect
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from crypto_downloader.binance.facade import Binance
from crypto_downloader._core.engine import Downloader
from crypto_downloader.binance.connector import BinanceConnector


def facade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Binance, list[object]]:
    """Create a facade whose engine records calls without doing I/O.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace engine retrieval.

    Returns:
        The configured facade and mutable call record.
    """
    service = Binance(tmp_path, progress=False)
    calls: list[object] = []

    def fake_get_data(
        *args: object, **kwargs: object
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Record one engine call and preserve the requested input shape."""
        calls.append((args, kwargs))
        pairs = args[0]
        if isinstance(pairs, str):
            frame = pd.DataFrame({"value": [1]})
            frame.attrs["download"] = {"pair": pairs, "complete": True}
            return frame
        return [pd.DataFrame({"value": [index]}) for index, _ in enumerate(pairs)]

    monkeypatch.setattr(service._downloader, "get_data", fake_get_data)
    return service, calls


def test_facade_construction_wires_validated_settings_without_io(
    tmp_path: Path,
) -> None:
    """Confirm construction configures one Binance engine without creating files.

    Args:
        tmp_path: The isolated parent of the proposed data directory.
    """
    data_dir = tmp_path / "cache"

    service = Binance(
        data_dir,
        earliest_date="2019-01-01",
        max_workers=12,
        discovery_tail_days=4,
        market_refresh_hours=6,
        timeout=8,
        retries=1,
        backoff=0.25,
        progress=False,
    )

    assert service.data_dir == data_dir.resolve()
    assert service.earliest_date == date(2019, 1, 1)
    assert service.kline_base_interval == "1m"
    assert service.max_workers == 12
    assert isinstance(service._downloader, Downloader)
    assert isinstance(service._downloader.source, BinanceConnector)
    assert service._downloader.source.timeout == 8
    assert service._downloader.source.retries == 1
    assert service._downloader.source.backoff == 0.25
    assert not data_dir.exists()


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("timeout", 0, "timeout"),
        ("timeout", "slow", "timeout"),
        ("timeout", None, "timeout"),
        ("timeout", True, "timeout"),
        ("timeout", float("nan"), "timeout"),
        ("timeout", float("inf"), "timeout"),
        ("retries", -1, "retries"),
        ("retries", True, "retries"),
        ("retries", 1.5, "retries"),
        ("retries", "three", "retries"),
        ("backoff", -0.1, "backoff"),
        ("backoff", "slow", "backoff"),
        ("backoff", None, "backoff"),
        ("backoff", True, "backoff"),
        ("backoff", float("nan"), "backoff"),
        ("backoff", float("inf"), "backoff"),
        ("progress", 1, "progress"),
        ("progress", None, "progress"),
    ],
)
def test_facade_rejects_invalid_network_and_display_settings(
    tmp_path: Path, option: str, value: object, message: str
) -> None:
    """Confirm invalid facade settings fail during construction.

    Args:
        tmp_path: The isolated configured data directory.
        option: The invalid constructor keyword.
        value: The invalid setting value.
        message: Text expected in the validation failure.
    """
    with pytest.raises((TypeError, ValueError), match=message):
        Binance(tmp_path, **{option: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method", "dataset", "product", "kline"),
    [
        ("get_klines", "klines", "spot", True),
        ("get_trades", "trades", "spot", False),
        ("get_agg_trades", "agg_trades", "spot", False),
        ("get_mark_price_klines", "mark_price_klines", "um", True),
        ("get_index_price_klines", "index_price_klines", "cm", True),
        ("get_premium_index_klines", "premium_index_klines", "um", True),
        ("get_metrics", "metrics", "cm", False),
        ("get_book_depth", "book_depth", "um", False),
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
    """Confirm each public method fixes its dataset and forwards common options.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace engine retrieval.
        method: The public facade method under test.
        dataset: The expected internal dataset identifier.
        product: The requested Binance product.
        kline: Whether the method accepts Kline-only options.
    """
    service, calls = facade(tmp_path, monkeypatch)
    kwargs: dict[str, object] = {
        "product": product,
        "columns": {"event_time" if not kline else "open_time": "time"},
        "refresh": True,
        "offline": False,
    }
    if kline:
        kwargs.update(interval="1h", gap_policy="keep")

    result = getattr(service, method)("BTCUSDT", "2025-01-01", "2025-01-02", **kwargs)

    assert isinstance(result, pd.DataFrame)
    assert result.attrs["download"]["pair"] == "BTCUSDT"
    assert calls == [
        (
            ("BTCUSDT", "2025-01-01", "2025-01-02"),
            {
                "product": product,
                "dataset": dataset,
                "interval": "1h" if kline else None,
                "desired_columns": kwargs["columns"],
                "refresh": True,
                "offline": False,
                "gap_policy": "keep" if kline else None,
                "progress": False,
            },
        )
    ]


@pytest.mark.parametrize("method", ["get_klines", "get_trades", "get_agg_trades"])
def test_common_dataset_methods_default_to_spot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Confirm datasets shared with Spot use Spot as their explicit default.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace engine retrieval.
        method: The public facade method under test.
    """
    service, calls = facade(tmp_path, monkeypatch)

    getattr(service, method)("BTCUSDT", "2025-01-01", "2025-01-01")

    _, options = cast(tuple[tuple[object, ...], dict[str, object]], calls[0])
    assert options["product"] == "spot"


@pytest.mark.parametrize(
    "method",
    [
        "get_mark_price_klines",
        "get_index_price_klines",
        "get_premium_index_klines",
        "get_metrics",
        "get_book_depth",
    ],
)
def test_futures_only_methods_require_product_keyword(
    tmp_path: Path, method: str
) -> None:
    """Confirm a Futures-only call cannot silently assume a product.

    Args:
        tmp_path: The isolated configured data directory.
        method: The Futures-only facade method under test.
    """
    service = Binance(tmp_path, progress=False)

    with pytest.raises(TypeError, match="product"):
        getattr(service, method)("BTCUSDT", "2025-01-01", "2025-01-01")


@pytest.mark.parametrize(
    "method",
    [
        "get_mark_price_klines",
        "get_index_price_klines",
        "get_premium_index_klines",
        "get_metrics",
        "get_book_depth",
    ],
)
def test_futures_only_methods_reject_spot_before_network(
    tmp_path: Path, method: str
) -> None:
    """Confirm a Futures-only dataset cannot be requested from Spot.

    Args:
        tmp_path: The isolated configured data directory.
        method: The Futures-only facade method under test.
    """
    service = Binance(tmp_path, progress=False)

    with pytest.raises(ValueError, match="unsupported dataset"):
        getattr(service, method)("BTCUSDT", "2025-01-01", "2025-01-01", product="spot")


@pytest.mark.parametrize(
    "method", ["get_trades", "get_agg_trades", "get_metrics", "get_book_depth"]
)
def test_event_and_snapshot_signatures_exclude_kline_options(method: str) -> None:
    """Confirm non-Kline APIs cannot receive interval or gap behavior.

    Args:
        method: The event or snapshot method whose signature is inspected.
    """
    parameters = inspect.signature(getattr(Binance, method)).parameters

    assert "interval" not in parameters
    assert "gap_policy" not in parameters


@pytest.mark.parametrize(
    "method",
    [
        "get_klines",
        "get_trades",
        "get_agg_trades",
        "get_mark_price_klines",
        "get_index_price_klines",
        "get_premium_index_klines",
        "get_metrics",
        "get_book_depth",
    ],
)
def test_dataset_signatures_hide_generic_engine_options(method: str) -> None:
    """Confirm facade methods do not expose downloader implementation choices.

    Args:
        method: The dataset method whose public signature is inspected.
    """
    parameters = inspect.signature(getattr(Binance, method)).parameters

    assert parameters.keys().isdisjoint(
        {"dataset", "data_dir", "source", "transport", "progress"}
    )


def test_pair_lists_preserve_the_engine_return_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm a list input returns the engine's ordered DataFrame list.

    Args:
        tmp_path: The isolated configured data directory.
        monkeypatch: The pytest helper used to replace engine retrieval.
    """
    service, calls = facade(tmp_path, monkeypatch)

    frames = service.get_klines(
        ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        "2025-01-01",
        "2025-01-01",
    )

    assert isinstance(frames, list)
    assert [frame.loc[0, "value"] for frame in frames] == [0, 1, 2]
    arguments, _ = cast(tuple[tuple[object, ...], dict[str, object]], calls[0])
    assert arguments[0] == ["BTCUSDT", "ETHUSDT", "BTCUSDT"]


def test_binance_module_does_not_publish_dataset_functions() -> None:
    """Confirm retrieval names exist only as configured facade methods."""
    import crypto_downloader.binance.facade as module

    assert not hasattr(module, "get_klines")
    assert not hasattr(module, "get_trades")
    assert not hasattr(module, "get_agg_trades")
