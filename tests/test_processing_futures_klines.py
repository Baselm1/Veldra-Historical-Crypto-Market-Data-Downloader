"""Test USD-M and COIN-M perpetual Kline normalization and validation."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from crypto_downloader.core.datasets import DatasetSpec
from crypto_downloader.binance.datasets import CM_KLINES, UM_KLINES
from arrow_helpers import (
    DataValidationError,
    normalize_chunk,
    validate_chunk,
)

FIXTURES = Path(__file__).parent / "fixtures"


def source_frame(filename: str) -> pd.DataFrame:
    """Read one header-bearing Futures Kline fixture as raw source values.

    Args:
        filename: The fixture CSV filename.

    Returns:
        Unconverted source rows with their Binance headers.
    """
    return pd.read_csv(FIXTURES / filename, dtype=str)


@pytest.mark.parametrize(
    ("dataset", "filename", "volume_column", "expected_volumes"),
    [
        (UM_KLINES, "binance_um_klines_2024-01-01.csv", "base_volume", [1.5, 2.0]),
        (
            CM_KLINES,
            "binance_cm_klines_2024-01-01.csv",
            "contract_volume",
            [20.0, 30.0],
        ),
    ],
)
def test_futures_klines_normalize_explicit_quantity_units(
    dataset: DatasetSpec,
    filename: str,
    volume_column: str,
    expected_volumes: list[float],
) -> None:
    """Confirm each Futures product gives Binance volume an explicit unit.

    Args:
        dataset: The product-specific Futures Kline declaration.
        filename: The matching raw archive fixture.
        volume_column: The explicit canonical volume field expected in output.
        expected_volumes: The expected normalized source volumes.
    """
    normalized = normalize_chunk(source_frame(filename), dataset)

    assert tuple(normalized.columns) == dataset.stored_columns
    assert normalized[volume_column].tolist() == expected_volumes
    assert "volume" not in normalized.columns
    assert normalized["open_time"].dtype == "datetime64[us, UTC]"
    assert normalized["close_time"].dtype == "datetime64[us, UTC]"
    assert normalized["trade_count"].dtype == "int64"
    assert validate_chunk(normalized, dataset, date(2024, 1, 1)) == pd.Timestamp(
        "2024-01-01T00:01:00Z"
    )


@pytest.mark.parametrize(
    ("dataset", "filename"),
    [
        (UM_KLINES, "binance_um_klines_2024-01-01.csv"),
        (CM_KLINES, "binance_cm_klines_2024-01-01.csv"),
    ],
)
def test_futures_klines_reject_negative_declared_quantities(
    dataset: DatasetSpec, filename: str
) -> None:
    """Confirm product-specific Kline quantity fields cannot be negative.

    Args:
        dataset: The Futures Kline declaration under test.
        filename: The matching raw archive fixture.
    """
    normalized = normalize_chunk(source_frame(filename), dataset)
    quantity = dataset.resample_sum_columns[0]
    normalized.loc[0, quantity] = -1.0

    with pytest.raises(DataValidationError, match="nonnegative"):
        validate_chunk(normalized, dataset, date(2024, 1, 1))
