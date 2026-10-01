"""Test the public Bitget facade contract."""

from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from veldra import Bitget
from veldra.bitget.client import BitgetResponseError
from veldra.core.models import Availability
from veldra.bitget.facade import _range


def test_facade_constructs_without_network(tmp_path: Path) -> None:
    """Construction only configures services and local paths."""
    service = Bitget(tmp_path, progress=False)
    assert service.data_dir == tmp_path.resolve()
    service.close()


def test_facade_exposes_configuration_and_context_management(tmp_path: Path) -> None:
    """The facade reports its configuration and closes as a context manager."""
    service = Bitget(tmp_path, earliest_date="2024-01-01", max_workers=7)
    service._reference_client.close = Mock()  # type: ignore[method-assign]
    with service as entered:
        assert entered is service
        assert service.earliest_date == date(2024, 1, 1)
        assert service.kline_base_interval == "1m"
        assert service.max_workers == 7
    service._reference_client.close.assert_called_once_with()


def test_archive_helpers_delegate_declared_dataset(tmp_path: Path) -> None:
    """Typed helpers retain products, intervals, and dataset identities."""
    service = Bitget(tmp_path, progress=False)
    service._downloader.get_data = Mock(return_value=pd.DataFrame())  # type: ignore[method-assign]
    service.get_klines("BTCUSDT", "2025-01-01", "2025-01-02", interval="1h")
    assert service._downloader.get_data.call_args.kwargs["dataset"] == "klines"
    service.get_best_book_snapshots("BTCUSDT", "2025-01-01", "2025-01-02")
    assert (
        service._downloader.get_data.call_args.kwargs["dataset"]
        == "best_book_snapshots"
    )
    service.close()


def test_reference_helpers_use_exact_validated_range(tmp_path: Path) -> None:
    """Reference helpers expand date ends before delegation."""
    service = Bitget(tmp_path, progress=False)
    service._reference.candles = Mock(  # type: ignore[method-assign]
        return_value=pd.DataFrame(
            {
                "open_time": pd.Series(dtype="datetime64[us, UTC]"),
                "open": pd.Series(dtype="float64"),
                "high": pd.Series(dtype="float64"),
                "low": pd.Series(dtype="float64"),
                "close": pd.Series(dtype="float64"),
                "base_volume": pd.Series(dtype="float64"),
                "quote_volume": pd.Series(dtype="float64"),
            }
        )
    )
    service.get_reference_klines("btcusdt", "2025-01-01", "2025-01-01")
    args = service._reference.candles.call_args.args
    assert args[0] == "BTCUSDT"
    assert args[-1] - args[-2] == pd.Timedelta(days=1)
    service.close()


def test_reference_helpers_reuse_a_stored_covering_range(tmp_path: Path) -> None:
    """Repeated and narrower calls do not repeat Bitget REST requests."""
    service = Bitget(tmp_path, progress=False)
    row = pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2025-01-01T00:00:00Z"]).astype(
                "datetime64[us, UTC]"
            ),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "base_volume": [0.0],
            "quote_volume": [0.0],
        }
    )
    service._reference.candles = Mock(return_value=row)  # type: ignore[method-assign]

    service.get_mark_price_klines("BTCUSDT", "2025-01-01", "2025-01-01")
    service.get_mark_price_klines(
        "BTCUSDT",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 1, tzinfo=UTC),
    )

    service._reference.candles.assert_called_once()
    service.close()


@pytest.mark.parametrize(
    ("method", "pair", "product", "error"),
    [
        ("get_reference_klines", 1, "usdt_futures", TypeError),
        ("get_reference_klines", " ", "usdt_futures", ValueError),
        ("get_reference_klines", "BTCUSDT", "spot", ValueError),
        ("get_funding_rates", [], "usdt_futures", ValueError),
        ("get_funding_rates", "BTCUSDT", "options", ValueError),
    ],
)
def test_reference_helpers_reject_invalid_inputs_before_network(
    tmp_path: Path,
    method: str,
    pair: object,
    product: str,
    error: type[Exception],
) -> None:
    """Malformed symbols and products never become remote API errors."""
    service = Bitget(tmp_path, progress=False)
    operation = getattr(service, method)
    with pytest.raises(error):
        operation(pair, "2025-01-01", "2025-01-02", product=product)
    service.close()


def test_range_rejects_reversed_timestamps() -> None:
    """Reference calls cannot send ambiguous or reversed bounds."""
    with pytest.raises(ValueError, match="before"):
        _range(
            datetime(2025, 1, 2, tzinfo=UTC),
            datetime(2025, 1, 1, tzinfo=UTC),
        )


def test_market_search_forwards_common_filters(tmp_path: Path) -> None:
    """Bitget search accepts the same useful market filters as other facades."""
    service = Bitget(tmp_path, progress=False)
    with patch("veldra.bitget.facade._find_markets", return_value=[]) as find:
        service.find_markets(
            "btc", product=None, status="ONLINE", quote_asset="USDT", limit=3
        )
    assert find.call_args.kwargs["product"] is None
    assert find.call_args.kwargs["status"] == "ONLINE"
    assert find.call_args.kwargs["quote_asset"] == "USDT"
    service.close()


def test_availability_helpers_delegate_to_shared_inspection(tmp_path: Path) -> None:
    """Bitget exposes cataloged and bounded remote coverage inspection."""
    expected = Mock(spec=Availability)
    service = Bitget(tmp_path, progress=False)
    with patch(
        "veldra.bitget.facade._get_availability", return_value=expected
    ) as local:
        assert (
            service.get_availability(
                "BTCUSDT", product="spot", dataset="klines", interval="1h"
            )
            is expected
        )
    assert local.call_args.kwargs["dataset"] == "klines"
    with patch(
        "veldra.bitget.facade._discover_availability", return_value=expected
    ) as remote:
        assert (
            service.discover_availability(
                "BTCUSDT",
                "2025-01-01",
                "2025-01-02",
                product="spot",
                dataset="klines",
            )
            is expected
        )
    assert remote.call_args.kwargs["progress"] is False
    service.close()


def test_named_reference_helpers_select_exact_datasets(tmp_path: Path) -> None:
    """Dedicated reference methods retain explicit dataset identities."""
    service = Bitget(tmp_path, progress=False)
    service.get_reference_klines = Mock(return_value=pd.DataFrame())  # type: ignore[method-assign]
    service.get_mark_price_klines("BTCUSDT", "2025-01-01", "2025-01-02")
    assert service.get_reference_klines.call_args.kwargs["dataset"] == (
        "mark_price_klines"
    )
    service.get_index_price_klines("BTCUSDT", "2025-01-01", "2025-01-02")
    assert service.get_reference_klines.call_args.kwargs["dataset"] == (
        "index_price_klines"
    )
    service.get_premium_index_klines("BTCUSDT", "2025-01-01", "2025-01-02")
    assert service.get_reference_klines.call_args.kwargs["dataset"] == (
        "premium_index_klines"
    )
    service.close()


def test_reference_results_include_the_standard_download_report(tmp_path: Path) -> None:
    """Successful and empty REST frames carry ordinary Veldra result metadata."""
    service = Bitget(tmp_path, progress=False)
    row = pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2025-01-01T00:00:00Z"]),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "base_volume": [0.0],
            "quote_volume": [0.0],
        }
    )
    service._reference.candles = Mock(return_value=row)  # type: ignore[method-assign]
    result = service.get_mark_price_klines("BTCUSDT", "2025-01-01", "2025-01-01")
    assert result.attrs["download"]["complete"] is True
    assert result.attrs["download"]["dataset"] == "mark_price_klines"

    service._reference.candles = Mock(return_value=row.iloc[0:0])  # type: ignore[method-assign]
    empty = service.get_mark_price_klines("ETHUSDT", "2025-01-01", "2025-01-01")
    assert empty.attrs["download"]["complete"] is False
    assert empty.attrs["download"]["errors"][0]["code"] == "range_unavailable"
    service.close()


def test_unknown_reference_pair_returns_an_error_frame(tmp_path: Path) -> None:
    """A deterministic unknown symbol cannot terminate the caller's program."""
    service = Bitget(tmp_path, progress=False)
    service._reference.candles = Mock(  # type: ignore[method-assign]
        side_effect=BitgetResponseError("25100", "Trading pair does not exist")
    )
    frame = service.get_mark_price_klines("NOTREAL", "2025-01-01", "2025-01-01")
    assert frame.empty
    assert frame.attrs["download"]["errors"][0]["code"] == "unknown_pair"
    service.close()
