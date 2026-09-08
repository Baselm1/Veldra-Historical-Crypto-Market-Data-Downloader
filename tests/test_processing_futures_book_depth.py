"""Test perpetual Futures book-depth normalization and validation."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from crypto_downloader._core.datasets import CM_BOOK_DEPTH, UM_BOOK_DEPTH, DatasetSpec
from crypto_downloader._core.processing import (
    DataValidationError,
    normalize_chunk,
    validate_chunk,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("dataset", "filename", "depth", "notional"),
    [
        (
            UM_BOOK_DEPTH,
            "binance_um_book_depth_2024-01-01.csv",
            "base_depth",
            "quote_notional",
        ),
        (
            CM_BOOK_DEPTH,
            "binance_cm_book_depth_2024-01-01.csv",
            "contract_depth",
            "base_notional",
        ),
    ],
)
def test_book_depth_normalizes_product_specific_depth_units(
    dataset: DatasetSpec, filename: str, depth: str, notional: str
) -> None:
    """Confirm each perpetual product retains its native depth units.

    Args:
        dataset: The product-specific book-depth declaration.
        filename: The representative raw source CSV fixture.
        depth: The expected native depth output field.
        notional: The expected product-specific notional output field.
    """
    source = pd.read_csv(FIXTURES / filename, dtype=str)
    normalized = normalize_chunk(source, dataset)

    assert tuple(normalized.columns) == dataset.stored_columns
    assert normalized[depth].notna().all()
    assert normalized[notional].notna().all()
    assert normalized["percentage_bucket"].dtype == "int64"
    assert normalized["percentage_bucket"].unique().tolist() == [
        -5,
        -4,
        -3,
        -2,
        -1,
        1,
        2,
        3,
        4,
        5,
    ]
    assert validate_chunk(normalized, dataset, date(2024, 1, 1)) == pd.Timestamp(
        normalized["event_time"].iloc[-1]
    )


@pytest.mark.parametrize(
    ("dataset", "filename"),
    [
        (UM_BOOK_DEPTH, "binance_um_book_depth_2024-01-01.csv"),
        (CM_BOOK_DEPTH, "binance_cm_book_depth_2024-01-01.csv"),
    ],
)
def test_book_depth_rejects_zero_percentage_buckets(
    dataset: DatasetSpec, filename: str
) -> None:
    """Confirm a depth row must describe either bid or ask side.

    Args:
        dataset: The product-specific book-depth declaration.
        filename: The representative raw source CSV fixture.
    """
    source = pd.read_csv(FIXTURES / filename, dtype=str)
    source.loc[0, "percentage"] = "0"
    normalized = normalize_chunk(source, dataset)

    with pytest.raises(DataValidationError, match="percentage bucket cannot be zero"):
        validate_chunk(normalized, dataset, date(2024, 1, 1))


def test_book_depth_rejects_negative_depth() -> None:
    """Confirm depth quantity cannot become negative after normalization."""
    source = pd.read_csv(FIXTURES / "binance_um_book_depth_2024-01-01.csv", dtype=str)
    normalized = normalize_chunk(source, UM_BOOK_DEPTH)
    normalized.loc[0, "base_depth"] = -1.0

    with pytest.raises(DataValidationError, match="depth values must be nonnegative"):
        validate_chunk(normalized, UM_BOOK_DEPTH, date(2024, 1, 1))
