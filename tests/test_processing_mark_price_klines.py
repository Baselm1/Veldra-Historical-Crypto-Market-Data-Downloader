"""Test perpetual Futures mark-price Kline normalization and validation."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from crypto_downloader._core.datasets import DatasetSpec
from crypto_downloader.binance.datasets import (
    CM_MARK_PRICE_KLINES,
    UM_MARK_PRICE_KLINES,
)
from crypto_downloader._core.processing import (
    DataValidationError,
    normalize_chunk,
    validate_chunk,
)

FIXTURES = Path(__file__).parent / "fixtures"


def source_frame(filename: str) -> pd.DataFrame:
    """Read one raw Binance mark-price CSV fixture.

    Args:
        filename: The fixture filename to read.

    Returns:
        Header-bearing source values as strings.
    """
    return pd.read_csv(FIXTURES / filename, dtype=str)


@pytest.mark.parametrize(
    ("dataset", "filename"),
    [
        (UM_MARK_PRICE_KLINES, "binance_um_mark_price_klines_2024-01-01.csv"),
        (CM_MARK_PRICE_KLINES, "binance_cm_mark_price_klines_2024-01-01.csv"),
    ],
)
def test_mark_price_klines_keep_only_price_and_sample_fields(
    dataset: DatasetSpec, filename: str
) -> None:
    """Confirm structural zero-volume source fields are omitted from Parquet.

    Args:
        dataset: The product-specific mark-price dataset declaration.
        filename: The representative raw archive fixture.
    """
    normalized = normalize_chunk(source_frame(filename), dataset)

    assert tuple(normalized.columns) == dataset.stored_columns
    assert normalized["sample_count"].tolist() == [60, 60]
    assert normalized["sample_count"].dtype == "int64"
    assert normalized["open_time"].dtype == "datetime64[us, UTC]"
    assert normalized["close_time"].dtype == "datetime64[us, UTC]"
    assert not {"volume", "quote_volume", "taker_buy_volume"}.intersection(
        normalized.columns
    )
    assert validate_chunk(normalized, dataset, date(2024, 1, 1)) == pd.Timestamp(
        "2024-01-01T00:01:00Z"
    )


@pytest.mark.parametrize("dataset", [UM_MARK_PRICE_KLINES, CM_MARK_PRICE_KLINES])
def test_mark_price_klines_reject_nonzero_structural_volume(
    dataset: DatasetSpec,
) -> None:
    """Confirm a changed source volume field cannot be silently discarded.

    Args:
        dataset: The product-specific mark-price dataset declaration.
    """
    frame = source_frame("binance_um_mark_price_klines_2024-01-01.csv")
    frame.loc[0, "volume"] = "1"

    with pytest.raises(DataValidationError, match="structural field"):
        normalize_chunk(frame, dataset)
