"""Test HTX linear- and coin-margined perpetual Klines and trades."""

from collections.abc import Sequence
from datetime import date

import pyarrow as pa
import pytest

from crypto_downloader.core.datasets import DatasetSpec
from crypto_downloader.core.models import DataValidationError
from crypto_downloader.htx.datasets import (
    COIN_KLINES,
    COIN_TRADES,
    LINEAR_KLINES,
    LINEAR_TRADES,
    NEW_KLINE_COLUMNS,
    NEW_TRADE_COLUMNS,
    OLD_COIN_TRADE_COLUMNS,
    OLD_KLINE_COLUMNS,
    OLD_LINEAR_TRADE_COLUMNS,
    get_dataset,
)
from crypto_downloader.htx.processing import normalize_chunk, validate_chunk


def raw_table(columns: tuple[str, ...], rows: Sequence[tuple[object, ...]]) -> pa.Table:
    """Build a string-valued perpetual source table.

    Args:
        columns: The exact source column names.
        rows: Source values in column order.

    Returns:
        An Arrow table shaped like CSV ingestion output.
    """
    return pa.table(
        {
            column: pa.array([str(row[index]) for row in rows], type=pa.string())
            for index, column in enumerate(columns)
        }
    )


@pytest.mark.parametrize(
    ("dataset", "columns", "row", "contracts", "base"),
    [
        (
            LINEAR_KLINES,
            OLD_KLINE_COLUMNS,
            (1735660800, 95406.7, 95324.2, 95449.8, 95324.1, 5202, 5.202),
            5202,
            5.202,
        ),
        (
            LINEAR_KLINES,
            NEW_KLINE_COLUMNS,
            (
                "BTC-USDT-PERP",
                78652.6,
                78700.5,
                78641.8,
                78700.5,
                6492,
                6.492,
                510719.6,
                1788795720,
            ),
            6492,
            6.492,
        ),
        (
            COIN_KLINES,
            OLD_KLINE_COLUMNS,
            (1735660800, 95442, 95362, 95442, 95362, 90, 0.09434733),
            90,
            0.09434733,
        ),
        (
            COIN_KLINES,
            NEW_KLINE_COLUMNS,
            (
                "BTC-USD-PERP",
                78632.8,
                78633,
                78632.8,
                78633,
                30,
                0.03815195,
                0,
                1788795960,
            ),
            30,
            0.03815195,
        ),
    ],
)
def test_perpetual_klines_preserve_contract_and_base_units(
    dataset: DatasetSpec,
    columns: tuple[str, ...],
    row: tuple[object, ...],
    contracts: float,
    base: float,
) -> None:
    """Confirm both source generations normalize without ambiguous volume.

    Args:
        dataset: The product-specific dataset declaration.
        columns: The old or new source fields.
        row: One representative Kline.
        contracts: The expected contract volume.
        base: The expected base-asset volume.
    """
    assert dataset in (LINEAR_KLINES, COIN_KLINES)
    normalized = normalize_chunk(raw_table(columns, [row]), dataset)
    assert normalized["contract_volume"][0].as_py() == pytest.approx(contracts)
    assert normalized["base_volume"][0].as_py() == pytest.approx(base)
    source_day = date(2025, 1, 1) if "id" in columns else date(2026, 9, 7)
    validate_chunk(normalized, dataset, source_day)


@pytest.mark.parametrize(
    (
        "dataset",
        "columns",
        "row",
        "contract_size",
        "contracts",
        "base",
        "quote_column",
        "quote",
    ),
    [
        (
            LINEAR_TRADES,
            OLD_LINEAR_TRADE_COLUMNS,
            (100046509040895, 1735660800604, 95406.7, 2, 0.002, 190.8134, "sell"),
            0.001,
            2,
            0.002,
            "quote_quantity",
            190.8134,
        ),
        (
            LINEAR_TRADES,
            NEW_TRADE_COLUMNS,
            ("BTC-USDT-PERP", 100127822479087, 79663.1, "sell", 62, 1788710400177),
            0.001,
            62,
            0.062,
            "quote_quantity",
            4939.1122,
        ),
        (
            COIN_TRADES,
            OLD_COIN_TRADE_COLUMNS,
            (100002510356464, 1735660828458, 95442, 17, 0.017811864797468619, "sell"),
            100,
            17,
            0.017811864797468619,
            "quote_notional",
            1700,
        ),
        (
            COIN_TRADES,
            NEW_TRADE_COLUMNS,
            ("BTC-USD-PERP", 100004811443505, 79663.7, "sell", 5, 1788710438243),
            100,
            5,
            500 / 79663.7,
            "quote_notional",
            500,
        ),
    ],
)
def test_perpetual_trades_preserve_and_derive_explicit_quantities(
    dataset: DatasetSpec,
    columns: tuple[str, ...],
    row: tuple[object, ...],
    contract_size: float,
    contracts: float,
    base: float,
    quote_column: str,
    quote: float,
) -> None:
    """Confirm old exact and new derived perpetual quantities share one schema.

    Args:
        dataset: The product-specific dataset declaration.
        columns: The old or new source fields.
        row: One representative trade.
        contract_size: The market contract size.
        contracts: The expected contract quantity.
        base: The expected base-asset quantity.
        quote_column: The expected quote quantity or notional field.
        quote: The expected quote-side value.
    """
    assert dataset in (LINEAR_TRADES, COIN_TRADES)
    normalized = normalize_chunk(raw_table(columns, [row]), dataset, contract_size)
    assert normalized["contract_quantity"][0].as_py() == pytest.approx(contracts)
    assert normalized["base_quantity"][0].as_py() == pytest.approx(base)
    assert normalized[quote_column][0].as_py() == pytest.approx(quote)
    source_day = date(2025, 1, 1) if "id" in columns else date(2026, 9, 7)
    validate_chunk(normalized, dataset, source_day)


def test_perpetual_trade_derivation_requires_contract_metadata() -> None:
    """Confirm new perpetual files never guess a missing contract size."""
    row = ("BTC-USDT-PERP", 1, 79663.1, "sell", 62, 1788710400177)
    with pytest.raises(DataValidationError, match="contract size"):
        normalize_chunk(raw_table(NEW_TRADE_COLUMNS, [row]), LINEAR_TRADES)


def test_perpetual_datasets_resolve_independently() -> None:
    """Confirm each perpetual product exposes Klines and trades."""
    assert get_dataset("linear_swap", "klines") is LINEAR_KLINES
    assert get_dataset("linear_swap", "trades") is LINEAR_TRADES
    assert get_dataset("coin_swap", "klines") is COIN_KLINES
    assert get_dataset("coin_swap", "trades") is COIN_TRADES
