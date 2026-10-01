"""Test the public Bitget facade contract."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from veldra import Bitget
from veldra.bitget.facade import _range


def test_facade_constructs_without_network(tmp_path: Path) -> None:
    """Construction only configures services and local paths."""
    service = Bitget(tmp_path, progress=False)
    assert service.data_dir == tmp_path.resolve()
    service.close()


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
    service._reference.candles = Mock(return_value=pd.DataFrame())  # type: ignore[method-assign]
    service.get_reference_klines("btcusdt", "2025-01-01", "2025-01-01")
    args = service._reference.candles.call_args.args
    assert args[0] == "BTCUSDT"
    assert args[-1] - args[-2] == pd.Timedelta(days=1)
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
