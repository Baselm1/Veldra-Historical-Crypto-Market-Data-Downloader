"""Test perpetual Futures metrics normalization and snapshot validation."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from crypto_downloader._core.datasets import DatasetSpec
from crypto_downloader.binance.datasets import CM_METRICS, UM_METRICS
from crypto_downloader._core.processing import (
    DataValidationError,
    normalize_chunk,
    validate_chunk,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("dataset", "filename", "quantity", "value"),
    [
        (
            UM_METRICS,
            "binance_um_metrics_2024-01-01.csv",
            "open_interest_base_quantity",
            "open_interest_quote_value",
        ),
        (
            CM_METRICS,
            "binance_cm_metrics_2024-01-01.csv",
            "open_interest_contract_quantity",
            "open_interest_base_quantity",
        ),
    ],
)
def test_metrics_normalize_product_specific_open_interest_units(
    dataset: DatasetSpec, filename: str, quantity: str, value: str
) -> None:
    """Confirm metrics retain explicit product-specific open-interest units.

    Args:
        dataset: The product-specific metrics declaration.
        filename: The representative raw archive fixture.
        quantity: The expected native open-interest quantity field.
        value: The expected product-specific open-interest value field.
    """
    source = pd.read_csv(FIXTURES / filename, dtype=str)
    normalized = normalize_chunk(source, dataset)

    assert tuple(normalized.columns) == dataset.stored_columns
    assert normalized[quantity].notna().all()
    assert normalized[value].notna().all()
    assert normalized["event_time"].dtype == "datetime64[us, UTC]"
    assert validate_chunk(normalized, dataset, date(2024, 1, 1)) == pd.Timestamp(
        "2024-01-01T00:05:00Z"
    )


def test_cm_metrics_preserve_legitimate_missing_ratios() -> None:
    """Confirm absent COIN-M ratios remain null rather than becoming zero."""
    source = pd.read_csv(FIXTURES / "binance_cm_metrics_2024-01-01.csv", dtype=str)

    normalized = normalize_chunk(source, CM_METRICS)

    assert normalized["top_trader_account_long_short_ratio"].isna().all()
    assert normalized["account_long_short_ratio"].isna().all()
    assert normalized["taker_long_short_volume_ratio"].notna().all()


def test_metrics_reject_nonpositive_present_ratio() -> None:
    """Confirm a supplied long/short ratio must be positive when not null."""
    source = pd.read_csv(FIXTURES / "binance_um_metrics_2024-01-01.csv", dtype=str)
    source.loc[0, "count_long_short_ratio"] = "0"
    normalized = normalize_chunk(source, UM_METRICS)

    with pytest.raises(DataValidationError, match="ratio must be positive"):
        validate_chunk(normalized, UM_METRICS, date(2024, 1, 1))
