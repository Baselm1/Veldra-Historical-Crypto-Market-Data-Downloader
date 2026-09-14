"""Test Gate perpetual Futures Kline and trade processing."""

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest

from veldra.core.models import DataValidationError
from veldra.gate.datasets import get_dataset
from veldra.gate.processing import normalize_chunk, validate_chunk
from tests.gate.test_spot import raw


@pytest.mark.parametrize("product", ["um", "cm"])
def test_futures_klines_preserve_contract_volume(product: str) -> None:
    """Normalize both perpetual products without inventing asset quantities.

    Args:
        product: The USDT- or BTC-margined Gate product.
    """
    dataset = get_dataset(product, "klines")
    table = raw(
        dataset.source_columns,
        [["1735689600", "97614", "101", "102", "99", "100"]],
    )

    normalized = normalize_chunk(table, dataset, contract_size=0.0001)

    assert normalized["contract_volume"].to_pylist() == [97614.0]
    assert normalized["open"].to_pylist() == [100.0]
    assert "base_volume" not in normalized.column_names
    validate_chunk(
        normalized,
        dataset,
        date(2025, 1, 1),
        end_day=date(2025, 1, 31),
    )


@pytest.mark.parametrize("product", ["um", "cm"])
def test_futures_trades_separate_sign_from_contract_quantity(product: str) -> None:
    """Map positive fills to buys and negative fills to sells.

    Args:
        product: The USDT- or BTC-margined Gate product.
    """
    dataset = get_dataset(product, "trades")
    table = raw(
        dataset.source_columns,
        [
            ["1735689610.945195", "1", "93534.3", "-251"],
            ["1735689612.656877", "2", "93535.0", "1850"],
        ],
    )

    normalized = normalize_chunk(table, dataset)

    assert normalized["contract_quantity"].to_pylist() == [251.0, 1850.0]
    assert normalized["side"].to_pylist() == ["sell", "buy"]
    validate_chunk(
        normalized,
        dataset,
        date(2025, 1, 1),
        end_day=date(2025, 1, 31),
    )


def test_ten_second_klines_require_exact_physical_alignment() -> None:
    """Reject Futures 10s rows whose opening second is not divisible by ten."""
    dataset = get_dataset("um", "klines", requested_interval="10s")
    invalid = pa.table(
        {
            "open_time": pa.array(
                [datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC)],
                type=pa.timestamp("us", "UTC"),
            ),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "contract_volume": [1.0],
        }
    )

    with pytest.raises(DataValidationError, match="aligned to 10s"):
        validate_chunk(invalid, dataset, date(2025, 1, 1))


@pytest.mark.parametrize(
    ("dataset_name", "rows", "message"),
    [
        (
            "klines",
            [["1735689600", "-1", "101", "102", "99", "100"]],
            "nonnegative",
        ),
        (
            "trades",
            [["1735689610.945195", "1", "93534.3", "0"]],
            "positive",
        ),
    ],
)
def test_invalid_futures_quantities_are_rejected(
    dataset_name: str, rows: list[list[str]], message: str
) -> None:
    """Reject negative candle volume and zero-sized fills.

    Args:
        dataset_name: The Gate dataset under test.
        rows: Its malformed source rows.
        message: The expected validation message.
    """
    dataset = get_dataset("um", dataset_name)
    normalized = normalize_chunk(raw(dataset.source_columns, rows), dataset)
    with pytest.raises(DataValidationError, match=message):
        validate_chunk(
            normalized,
            dataset,
            date(2025, 1, 1),
            end_day=date(2025, 1, 31),
        )
