"""Test USD-M and COIN-M perpetual aggregate-trade normalization."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from veldra.core.datasets import DatasetSpec
from veldra.binance.datasets import CM_AGG_TRADES, UM_AGG_TRADES
from arrow_helpers import (
    DataValidationError,
    normalize_chunk,
    validate_chunk,
)

FIXTURES = Path(__file__).parent / "fixtures"


def source_frame(filename: str) -> pd.DataFrame:
    """Read one header-bearing Futures aggregate-trade fixture as raw strings.

    Args:
        filename: The fixture CSV filename.

    Returns:
        Unconverted source aggregate-trade rows.
    """
    return pd.read_csv(FIXTURES / filename, dtype=str)


def test_um_aggregate_trades_preserve_component_ids_and_derive_quote_quantity() -> None:
    """Confirm USD-M aggregate trades retain IDs and use base quantity units."""
    normalized = normalize_chunk(
        source_frame("binance_um_agg_trades_2024-01-01.csv"), UM_AGG_TRADES
    )

    assert tuple(normalized.columns) == UM_AGG_TRADES.stored_columns
    assert normalized[
        ["agg_trade_id", "first_trade_id", "last_trade_id"]
    ].values.tolist() == [
        [300, 500, 501],
        [301, 502, 502],
    ]
    assert normalized["base_quantity"].tolist() == [0.5, 0.25]
    assert normalized["quote_quantity"].tolist() == [21000.0, 10502.5]
    assert validate_chunk(normalized, UM_AGG_TRADES, date(2024, 1, 1)) == pd.Timestamp(
        "2024-01-01T00:00:00.100000Z"
    )


def test_cm_aggregate_trades_preserve_contracts_and_derive_missing_quantities() -> None:
    """Confirm COIN-M aggregate trades retain contracts and derive units safely."""
    normalized = normalize_chunk(
        source_frame("binance_cm_agg_trades_2024-01-01.csv"),
        CM_AGG_TRADES,
        contract_size=100.0,
    )

    assert tuple(normalized.columns) == CM_AGG_TRADES.stored_columns
    assert normalized[
        ["agg_trade_id", "first_trade_id", "last_trade_id"]
    ].values.tolist() == [
        [400, 700, 701],
        [401, 702, 702],
    ]
    assert normalized["contract_quantity"].tolist() == [3.0, 2.0]
    assert normalized["quote_notional"].tolist() == [300.0, 200.0]
    assert normalized["base_quantity"].tolist() == [300.0 / 42000.0, 200.0 / 42010.0]
    assert validate_chunk(normalized, CM_AGG_TRADES, date(2024, 1, 1)) == pd.Timestamp(
        "2024-01-01T00:00:00.100000Z"
    )


@pytest.mark.parametrize(
    ("dataset", "filename", "contract_size"),
    [
        (UM_AGG_TRADES, "binance_um_agg_trades_2024-01-01.csv", None),
        (CM_AGG_TRADES, "binance_cm_agg_trades_2024-01-01.csv", 100.0),
    ],
)
def test_futures_aggregate_trades_reject_invalid_component_ids(
    dataset: DatasetSpec, filename: str, contract_size: float | None
) -> None:
    """Confirm aggregate trade component IDs remain nonnegative and ordered.

    Args:
        dataset: The Futures aggregate-trade schema under test.
        filename: The matching raw archive fixture.
        contract_size: The optional COIN-M contract size.
    """
    normalized = normalize_chunk(
        source_frame(filename),
        dataset,
        contract_size=contract_size,
    )
    normalized.loc[0, "last_trade_id"] = -1

    with pytest.raises(DataValidationError, match="aggregate trade IDs"):
        validate_chunk(normalized, dataset, date(2024, 1, 1))
