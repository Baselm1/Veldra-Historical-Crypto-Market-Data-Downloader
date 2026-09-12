"""Test OKX linear and inverse perpetual archive normalization."""

import pandas as pd
import pytest

from veldra.core.models import DataValidationError
from veldra.okx.datasets import get_dataset
from veldra.okx.processing import normalize_klines, normalize_trades


def kline(instrument: str) -> pd.DataFrame:
    """Build one valid derivative Kline source row.

    Args:
        instrument: Native swap instrument ID.

    Returns:
        One source Kline row.
    """
    return pd.DataFrame(
        [
            {
                "instrument_name": instrument,
                "open": 100,
                "high": 102,
                "low": 99,
                "close": 101,
                "vol": 20,
                "vol_ccy": 0.2,
                "vol_quote": 20.2,
                "open_time": 1735689600000,
                "confirm": 1,
            }
        ]
    )


def trade(instrument: str, size: float = 2) -> pd.DataFrame:
    """Build one valid derivative trade source row.

    Args:
        instrument: Native swap instrument ID.
        size: Native contract quantity.

    Returns:
        One source trade row.
    """
    return pd.DataFrame(
        [
            {
                "instrument_name": instrument,
                "trade_id": 10,
                "side": "buy",
                "price": 100,
                "size": size,
                "created_time": 1735689600000,
            }
        ]
    )


@pytest.mark.parametrize("product", ["linear_swap", "inverse_swap"])
def test_perpetual_klines_keep_all_declared_volume_units(product: str) -> None:
    """Confirm source contract, base, and quote volumes are not conflated.

    Args:
        product: Linear or inverse perpetual product.
    """
    instrument = "BTC-USDT-SWAP" if product == "linear_swap" else "BTC-USD-SWAP"
    frame = normalize_klines(kline(instrument), get_dataset(product, "klines"))
    assert frame.loc[0, "contract_volume"] == 20
    assert frame.loc[0, "base_volume"] == 0.2
    assert frame.loc[0, "quote_volume"] == 20.2


def test_linear_trades_derive_base_and_quote_from_contract_metadata() -> None:
    """Confirm linear contract sizes produce explicit base and quote quantities."""
    frame = normalize_trades(
        trade("BTC-USDT-SWAP"),
        get_dataset("linear_swap", "trades"),
        {"BTC-USDT-SWAP": 0.01},
    )
    assert frame.loc[0, "contract_quantity"] == 2
    assert frame.loc[0, "base_quantity"] == 0.02
    assert frame.loc[0, "quote_quantity"] == 2


def test_inverse_trades_derive_quote_notional_and_base_quantity() -> None:
    """Confirm inverse USD face value is divided by execution price."""
    frame = normalize_trades(
        trade("BTC-USD-SWAP"),
        get_dataset("inverse_swap", "trades"),
        {"BTC-USD-SWAP": 100},
    )
    assert frame.loc[0, "contract_quantity"] == 2
    assert frame.loc[0, "quote_notional"] == 200
    assert frame.loc[0, "base_quantity"] == 2


def test_perpetual_trades_require_proven_contract_size() -> None:
    """Confirm missing unit metadata cannot silently create false quantities."""
    with pytest.raises(DataValidationError, match="contract size"):
        normalize_trades(
            trade("BTC-USDT-SWAP"), get_dataset("linear_swap", "trades"), {}
        )


def test_dated_futures_are_not_accepted_by_perpetual_specs() -> None:
    """Confirm expiry products do not enter the swap API accidentally."""
    with pytest.raises(ValueError, match="unsupported"):
        get_dataset("linear_futures", "klines")
