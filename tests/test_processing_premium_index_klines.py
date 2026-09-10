"""Test perpetual Futures premium-index Kline normalization."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from veldra.core.datasets import DatasetSpec
from veldra.binance.datasets import (
    CM_PREMIUM_INDEX_KLINES,
    UM_PREMIUM_INDEX_KLINES,
)
from arrow_helpers import normalize_chunk, validate_chunk

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("dataset", "filename"),
    [
        (UM_PREMIUM_INDEX_KLINES, "binance_um_premium_index_klines_2024-01-01.csv"),
        (CM_PREMIUM_INDEX_KLINES, "binance_cm_premium_index_klines_2024-01-01.csv"),
    ],
)
def test_premium_index_klines_preserve_the_source_sample_count(
    dataset: DatasetSpec, filename: str
) -> None:
    """Confirm premium candles store their observed twelve source samples.

    Args:
        dataset: The product-specific premium-index declaration.
        filename: The representative raw Binance archive fixture.
    """
    normalized = normalize_chunk(pd.read_csv(FIXTURES / filename, dtype=str), dataset)

    assert tuple(normalized.columns) == dataset.stored_columns
    assert normalized["sample_count"].tolist() == [12, 12]
    assert validate_chunk(normalized, dataset, date(2024, 1, 1)) == pd.Timestamp(
        "2024-01-01T00:01:00Z"
    )


def test_premium_index_klines_accept_signed_premium_rates() -> None:
    """Confirm a negative premium is not treated as an invalid trade price."""
    frame = pd.read_csv(
        FIXTURES / "binance_um_premium_index_klines_2024-01-01.csv", dtype=str
    )
    frame.loc[0, ["open", "high", "low", "close"]] = [
        "-0.0008",
        "-0.0005",
        "-0.0010",
        "-0.0007",
    ]

    normalized = normalize_chunk(frame, UM_PREMIUM_INDEX_KLINES)

    assert normalized["close"].tolist()[0] == -0.0007
    assert validate_chunk(
        normalized, UM_PREMIUM_INDEX_KLINES, date(2024, 1, 1)
    ) == pd.Timestamp("2024-01-01T00:01:00Z")
