"""Test OKX perpetual funding archive behavior."""

import pandas as pd
import pytest

from veldra.core.models import DataValidationError
from veldra.okx.datasets import FUNDING_SOURCE_COLUMNS, get_dataset
from veldra.okx.processing import normalize_funding_rates


def funding_frame(**changes: object) -> pd.DataFrame:
    """Build valid funding rows with optional column replacements.

    Args:
        changes: Source columns replacing valid values.

    Returns:
        Mutable source observations.
    """
    values: dict[str, object] = {
        "instrument_name": ["BTC-USDT-SWAP", "BTC-USDT-SWAP"],
        "funding_rate": [0.0001, -0.0002],
        "funding_time": [1735689600000, 1735704000000],
    }
    values.update(changes)
    return pd.DataFrame(values, columns=FUNDING_SOURCE_COLUMNS)


@pytest.mark.parametrize(
    "product",
    ["linear_swap", "inverse_swap", "linear_futures", "inverse_futures"],
)
def test_funding_preserves_actual_irregular_timestamps(product: str) -> None:
    """Confirm no fixed funding interval is synthesized.

    Args:
        product: Linear or inverse swap or X-Perp Futures product.
    """
    frame = normalize_funding_rates(
        funding_frame(), get_dataset(product, "funding_rates")
    )
    assert frame["funding_rate"].tolist() == [0.0001, -0.0002]
    assert (
        frame.loc[1, "funding_time"] - frame.loc[0, "funding_time"]
    ).total_seconds() == 14_400


def test_funding_exact_duplicates_are_removed_and_conflicts_fail() -> None:
    """Confirm instrument and actual funding time identify one observation."""
    dataset = get_dataset("linear_swap", "funding_rates")
    raw = funding_frame()
    assert (
        len(
            normalize_funding_rates(
                pd.concat([raw, raw.iloc[[0]]], ignore_index=True), dataset
            )
        )
        == 2
    )
    conflict = raw.copy()
    conflict.loc[1, "funding_time"] = conflict.loc[0, "funding_time"]
    with pytest.raises(DataValidationError, match="conflicting"):
        normalize_funding_rates(conflict, dataset)


@pytest.mark.parametrize(
    "changes",
    [
        {"instrument_name": ["", "BTC-USDT-SWAP"]},
        {"funding_rate": ["bad", 0.1]},
        {"funding_time": ["bad", 1735704000000]},
    ],
)
def test_invalid_funding_rows_fail_visibly(changes: dict[str, object]) -> None:
    """Confirm malformed funding observations never enter Parquet.

    Args:
        changes: Invalid source values.
    """
    with pytest.raises(DataValidationError):
        normalize_funding_rates(
            funding_frame(**changes), get_dataset("linear_swap", "funding_rates")
        )
