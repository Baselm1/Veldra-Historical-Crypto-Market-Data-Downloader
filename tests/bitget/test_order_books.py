"""Test Bitget best-book and level-500 snapshot conversion."""

from datetime import date

import pyarrow as pa
import pytest

from veldra.bitget.datasets import get_dataset
from veldra.bitget.processing import normalize_chunk, validate_chunk
from veldra.core.models import DataValidationError


def best_book(**changes: list[str]) -> pa.Table:
    """Return one representative top-of-book worksheet."""
    values = {
        "timestamp": ["1735660800000"],
        "ask_price": ["101"],
        "bid_price": ["100"],
        "ask_volume": ["2"],
        "bid_volume": ["3"],
        "__row_number": ["7"],
    }
    values.update(changes)
    return pa.table(values)


def deep_book(**changes: list[str]) -> pa.Table:
    """Return one representative level-500 worksheet."""
    values = {
        "timestamp": ["1735660800000"],
        "asks": ['[["101", "2"], ["102", "1"]]'],
        "bids": ['[["100", "3"], ["99", "4"]]'],
        "__row_number": ["9"],
    }
    values.update(changes)
    return pa.table(values)


@pytest.mark.parametrize(
    "product", ["spot", "usdt_futures", "usdc_futures", "coin_futures"]
)
def test_normalizes_best_book_for_every_product(product: str) -> None:
    """Best-book data remains a flat complete snapshot."""
    dataset = get_dataset(product, "best_book_snapshots")
    table = normalize_chunk(best_book(), dataset)
    assert table.to_pydict()["event_number"] == [7]
    assert table.to_pydict()["ask_price"] == [101.0]
    validate_chunk(table, dataset, date(2025, 1, 1))


def test_normalizes_level_500_to_nested_structs() -> None:
    """Deep snapshots preserve typed price and quantity levels."""
    dataset = get_dataset("spot", "order_book_snapshots")
    table = normalize_chunk(deep_book(), dataset)
    assert pa.types.is_list(table["bids"].type)
    assert table["bids"].to_pylist()[0][0] == {"price": 100.0, "quantity": 3.0}
    validate_chunk(table, dataset, date(2025, 1, 1))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"ask_price": ["99"]}, "crossed"),
        ({"bid_volume": ["0"]}, "positive"),
    ],
)
def test_rejects_invalid_best_books(
    changes: dict[str, list[str]], message: str
) -> None:
    """Crossed books and empty quantities are invalid."""
    dataset = get_dataset("spot", "best_book_snapshots")
    table = normalize_chunk(best_book(**changes), dataset)
    with pytest.raises(DataValidationError, match=message):
        validate_chunk(table, dataset, date(2025, 1, 1))


@pytest.mark.parametrize("value", ["bad", "{}", "[]", '[["0", "2"]]', '[["100"]]'])
def test_rejects_invalid_deep_sides(value: str) -> None:
    """Malformed, empty, and nonpositive deep levels fail closed."""
    dataset = get_dataset("spot", "order_book_snapshots")
    with pytest.raises(DataValidationError):
        table = normalize_chunk(deep_book(bids=[value]), dataset)
        validate_chunk(table, dataset, date(2025, 1, 1))
