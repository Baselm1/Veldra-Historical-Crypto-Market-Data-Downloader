"""Test Gate perpetual mark-price and funding normalization."""

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest

from veldra.core.models import DataValidationError
from veldra.gate.datasets import get_dataset
from veldra.gate.processing import normalize_chunk, validate_chunk


def raw(columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> pa.Table:
    """Create one headerless-style Arrow source table.

    Args:
        columns: The declared source columns.
        rows: The string values in each source row.

    Returns:
        A raw Arrow table matching one Gate CSV.
    """
    return pa.table(
        {column: [row[index] for row in rows] for index, column in enumerate(columns)}
    )


@pytest.mark.parametrize("product", ["um", "cm"])
def test_mark_prices_are_normalized_and_sorted(product: str) -> None:
    """Normalize both perpetual products without losing subsecond times.

    Args:
        product: The USDT- or BTC-margined perpetual product.
    """
    dataset = get_dataset(product, "mark_prices")
    source = raw(
        dataset.source_columns,
        [
            ("1735689601.25", "100.1", "100.2", "100.3"),
            ("1735689600.125001", "99.1", "99.2", "99.3"),
        ],
    )

    frame = normalize_chunk(source, dataset)
    indices = pa.compute.sort_indices(frame, sort_keys=[("event_time", "ascending")])
    frame = frame.take(indices)

    assert frame.column_names == list(dataset.stored_columns)
    assert frame["event_time"][0].as_py() == datetime(
        2025, 1, 1, 0, 0, 0, 125001, tzinfo=UTC
    )
    assert frame["mark_price"].to_pylist() == [99.2, 100.2]
    assert validate_chunk(frame, dataset, date(2025, 1, 1)) == datetime(
        2025, 1, 1, 0, 0, 1, 250000, tzinfo=UTC
    )


@pytest.mark.parametrize(
    ("name", "row", "expected"),
    [
        ("funding_rates", ("1735689600", "-0.0001"), -0.0001),
        (
            "funding_rate_updates",
            (
                "1735689600.5",
                "0.0002",
                "0.0001",
                "-0.2",
                "0.3",
                "100.2",
                "100.1",
                "4",
            ),
            0.0002,
        ),
    ],
)
def test_funding_histories_preserve_signed_rates(
    name: str, row: tuple[str, ...], expected: float
) -> None:
    """Preserve signed applied and projected funding values.

    Args:
        name: The applied or update history dataset.
        row: One native source record.
        expected: The normalized funding rate.
    """
    dataset = get_dataset("um", name)
    frame = normalize_chunk(raw(dataset.source_columns, [row]), dataset)

    assert frame["funding_rate"][0].as_py() == expected
    validate_chunk(frame, dataset, date(2025, 1, 1))


@pytest.mark.parametrize(
    ("name", "column", "value", "message"),
    [
        ("mark_prices", "mark_price", "0", "mark_price must be positive"),
        ("mark_prices", "last_price", "-1", "last_price must be positive"),
        (
            "funding_rate_updates",
            "update_count",
            "-1",
            "update_count must be nonnegative",
        ),
        ("funding_rates", "funding_rate", "nan", "must be finite"),
    ],
)
def test_invalid_reference_values_are_rejected(
    name: str, column: str, value: str, message: str
) -> None:
    """Reject invalid prices, counts, and nonfinite rates.

    Args:
        name: The reference dataset under test.
        column: The field to corrupt.
        value: The invalid source value.
        message: The expected validation message.
    """
    dataset = get_dataset("um", name)
    defaults = {
        "timestamp": "1735689600",
        "funding_rate": "0.0001",
        "interest_rate": "0.0001",
        "bid_diff": "0.1",
        "ask_diff": "0.2",
        "mark_price": "100",
        "index_price": "100",
        "last_price": "100",
        "update_count": "1",
    }
    defaults[column] = value
    source = raw(
        dataset.source_columns,
        [tuple(defaults[item] for item in dataset.source_columns)],
    )

    with pytest.raises(DataValidationError, match=message):
        frame = normalize_chunk(source, dataset)
        validate_chunk(frame, dataset, date(2025, 1, 1))


def test_reference_schema_mismatch_is_rejected() -> None:
    """Reject a reference CSV with missing declared fields."""
    dataset = get_dataset("cm", "funding_rates")

    with pytest.raises(DataValidationError, match="reference-data schema"):
        normalize_chunk(pa.table({"timestamp": ["1735689600"]}), dataset)
