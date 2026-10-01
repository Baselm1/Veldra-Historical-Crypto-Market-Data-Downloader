"""Test Bitget public-trade normalization and validation."""

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest

from veldra.bitget.datasets import get_dataset
from veldra.bitget.processing import normalize_chunk, validate_chunk
from veldra.core.models import DataValidationError


def raw(**changes: list[str]) -> pa.Table:
    """Return two representative unordered trade records."""
    values = {
        "trade_id": ["11", "10"],
        "timestamp": ["1735660800123", "1735660800123"],
        "price": ["100", "99"],
        "side": ["BUY", "sell"],
        "volume(quote)": ["200", "99"],
        "size(base)": ["2", "1"],
        "__row_number": [0, 1],
    }
    values.update(changes)
    return pa.table(values)


@pytest.mark.parametrize(
    "product", ["spot", "usdt_futures", "usdc_futures", "coin_futures"]
)
def test_normalizes_trade_products(product: str) -> None:
    """Every product exposes the same canonical trade contract."""
    dataset = get_dataset(product, "trades")
    table = normalize_chunk(raw(), dataset).sort_by(
        [(column, "ascending") for column in dataset.ordering_columns]
    )
    assert table["event_number"].to_pylist() == [10, 11]
    assert table["side"].to_pylist() == ["sell", "buy"]
    assert validate_chunk(table, dataset, date(2025, 1, 1)) == datetime(
        2024, 12, 31, 16, 0, 0, 123000, tzinfo=UTC
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"timestamp": ["1", "2"]}, "timestamp unit"),
        ({"trade_id": ["-1", "10"]}, "nonnegative"),
        ({"price": ["0", "99"]}, "positive"),
        ({"size(base)": ["0", "1"]}, "quantities"),
        ({"volume(quote)": ["0", "99"]}, "quantities"),
        ({"side": ["hold", "sell"]}, "buy or sell"),
        ({"timestamp": ["1735747200000", "1735747200001"]}, "outside"),
    ],
)
def test_rejects_invalid_trades(changes: dict[str, list[str]], message: str) -> None:
    """Bad identifiers, timestamps, quantities, and sides are rejected."""
    dataset = get_dataset("spot", "trades")
    with pytest.raises(DataValidationError, match=message):
        table = normalize_chunk(raw(**changes), dataset)
        table = table.sort_by(
            [(column, "ascending") for column in dataset.ordering_columns]
        )
        validate_chunk(table, dataset, date(2025, 1, 1))


def test_trade_timestamp_ties_are_valid_after_id_sort() -> None:
    """Multiple fills may share a millisecond and remain deterministic by ID."""
    dataset = get_dataset("spot", "trades")
    table = normalize_chunk(raw(), dataset).sort_by(
        [(column, "ascending") for column in dataset.ordering_columns]
    )
    validate_chunk(table, dataset, date(2025, 1, 1))
