"""Test perpetual Futures index-price Kline normalization and validation."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from crypto_downloader._core.datasets import DatasetSpec
from crypto_downloader.binance.datasets import (
    CM_INDEX_PRICE_KLINES,
    UM_INDEX_PRICE_KLINES,
)
from crypto_downloader._core.processing import normalize_chunk, validate_chunk

FIXTURES = Path(__file__).parent / "fixtures"


def source_frame(filename: str) -> pd.DataFrame:
    """Read one raw Binance index-price CSV fixture.

    Args:
        filename: The fixture filename to read.

    Returns:
        Header-bearing source values as strings.
    """
    return pd.read_csv(FIXTURES / filename, dtype=str)


@pytest.mark.parametrize(
    ("dataset", "filename"),
    [
        (UM_INDEX_PRICE_KLINES, "binance_um_index_price_klines_2024-01-01.csv"),
        (CM_INDEX_PRICE_KLINES, "binance_cm_index_price_klines_2024-01-01.csv"),
    ],
)
def test_index_price_klines_normalize_to_price_only_candles(
    dataset: DatasetSpec, filename: str
) -> None:
    """Confirm both Futures products use the price-only canonical schema.

    Args:
        dataset: The product-specific index-price dataset declaration.
        filename: The representative raw archive fixture.
    """
    normalized = normalize_chunk(source_frame(filename), dataset)

    assert tuple(normalized.columns) == dataset.stored_columns
    assert normalized["sample_count"].tolist() == [60, 60]
    assert normalized["close"].tolist()[0] > 0
    assert validate_chunk(normalized, dataset, date(2024, 1, 1)) == pd.Timestamp(
        "2024-01-01T00:01:00Z"
    )
