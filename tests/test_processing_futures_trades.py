"""Test USD-M and COIN-M perpetual trade normalization and validation."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from crypto_downloader._core.datasets import CM_TRADES, UM_TRADES, DatasetSpec
from crypto_downloader._core.processing import (
    DataValidationError,
    normalize_chunk,
    validate_chunk,
)

FIXTURES = Path(__file__).parent / "fixtures"


def source_frame(filename: str) -> pd.DataFrame:
    """Read one header-bearing Futures trade fixture as raw strings.

    Args:
        filename: The fixture CSV filename.

    Returns:
        Unconverted source event rows.
    """
    return pd.read_csv(FIXTURES / filename, dtype=str)


def test_um_trades_preserve_base_and_quote_quantities() -> None:
    """Confirm USD-M trades preserve Binance's native base and quote quantities."""
    normalized = normalize_chunk(
        source_frame("binance_um_trades_2024-01-01.csv"), UM_TRADES
    )

    assert tuple(normalized.columns) == UM_TRADES.stored_columns
    assert normalized["base_quantity"].tolist() == [0.5, 0.25]
    assert normalized["quote_quantity"].tolist() == [21000.0, 10502.5]
    assert validate_chunk(normalized, UM_TRADES, date(2024, 1, 1)) == pd.Timestamp(
        "2024-01-01T00:00:00.100000Z"
    )


def test_cm_trades_preserve_source_quantities_and_derive_quote_notional() -> None:
    """Confirm COIN-M trades retain quantities and derive USD notional from size."""
    normalized = normalize_chunk(
        source_frame("binance_cm_trades_2024-01-01.csv"),
        CM_TRADES,
        contract_size=100.0,
    )

    assert tuple(normalized.columns) == CM_TRADES.stored_columns
    assert normalized["contract_quantity"].tolist() == [3.0, 2.0]
    assert normalized["base_quantity"].tolist() == [0.00714285, 0.00476077]
    assert normalized["quote_notional"].tolist() == [300.0, 200.0]
    assert validate_chunk(normalized, CM_TRADES, date(2024, 1, 1)) == pd.Timestamp(
        "2024-01-01T00:00:00.100000Z"
    )


@pytest.mark.parametrize("contract_size", [None, 0.0, -1.0, float("inf"), True])
def test_cm_trades_require_a_positive_finite_contract_size(
    contract_size: float | None | bool,
) -> None:
    """Confirm COIN-M quote notional cannot be invented without contract metadata.

    Args:
        contract_size: The invalid cataloged contract size under test.
    """
    with pytest.raises(DataValidationError, match="contract size"):
        normalize_chunk(
            source_frame("binance_cm_trades_2024-01-01.csv"),
            CM_TRADES,
            contract_size=contract_size,
        )


@pytest.mark.parametrize(
    ("dataset", "filename", "column"),
    [
        (UM_TRADES, "binance_um_trades_2024-01-01.csv", "base_quantity"),
        (CM_TRADES, "binance_cm_trades_2024-01-01.csv", "contract_quantity"),
    ],
)
def test_futures_trades_reject_negative_quantities(
    dataset: DatasetSpec, filename: str, column: str
) -> None:
    """Confirm Futures source quantities cannot become negative after normalization.

    Args:
        dataset: The Futures trade schema under test.
        filename: The matching raw archive fixture.
        column: The canonical quantity column made invalid.
    """
    normalized = normalize_chunk(
        source_frame(filename),
        dataset,
        contract_size=100.0 if dataset is CM_TRADES else None,
    )
    normalized.loc[0, column] = -1.0

    with pytest.raises(DataValidationError, match="quantities"):
        validate_chunk(normalized, dataset, date(2024, 1, 1))
