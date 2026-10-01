"""Test Bitget Kline normalization and validation."""

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest

from veldra.bitget.datasets import get_dataset
from veldra.bitget.processing import normalize_chunk, validate_chunk
from veldra.core.models import DataValidationError


def raw(**changes: list[str]) -> pa.Table:
    """Return two representative worksheet rows."""
    values = {
        "timestamp": ["1735660800", "1735660860"],
        "open": ["100", "102"],
        "high": ["103", "104"],
        "low": ["99", "101"],
        "close": ["102", "103"],
        "basevolume": ["2.5", "3"],
        "usdtvolume": ["251", "309"],
        "__row_number": [0, 1],
    }
    values.update(changes)
    return pa.table(values)


@pytest.mark.parametrize(
    ("product", "last_column"),
    [
        ("spot", "quote_volume"),
        ("usdt_futures", "quote_volume"),
        ("usdc_futures", "quote_volume"),
        ("coin_futures", "contract_volume"),
    ],
)
def test_normalizes_every_kline_product(product: str, last_column: str) -> None:
    """Every product preserves the declared quantity unit."""
    dataset = get_dataset(product, "klines")
    table = normalize_chunk(raw(), dataset)
    assert table.column_names[-1] == last_column
    assert table["open_time"].type == pa.timestamp("us", "UTC")
    assert table[last_column].to_pylist() == [251.0, 309.0]
    assert validate_chunk(table, dataset, date(2025, 1, 1)) == datetime(
        2024, 12, 31, 16, 1, tzinfo=UTC
    )


def test_accepts_legacy_camel_case_headers() -> None:
    """Older workbook quantity labels remain readable."""
    source = raw().rename_columns(
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "baseVolume",
            "usdtVolume",
            "__row_number",
        ]
    )
    table = normalize_chunk(source, get_dataset("spot", "klines"))
    assert table["base_volume"].to_pylist() == [2.5, 3.0]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"timestamp": ["bad", "1735660860"]}, "invalid integer"),
        ({"timestamp": ["1735660800000", "1735660860000"]}, "timestamp unit"),
        ({"open": ["nan", "102"]}, "finite"),
        ({"high": ["98", "104"]}, "high is below"),
        ({"low": ["104", "101"]}, "OHLC price"),
        ({"basevolume": ["-1", "3"]}, "nonnegative"),
        ({"timestamp": ["1735660801", "1735660860"]}, "aligned"),
        ({"timestamp": ["1735660860", "1735660800"]}, "increasing"),
        ({"timestamp": ["1735747200", "1735747260"]}, "outside"),
    ],
)
def test_rejects_bad_kline_values(changes: dict[str, list[str]], message: str) -> None:
    """Malformed source values never reach persisted Parquet."""
    dataset = get_dataset("spot", "klines")
    with pytest.raises(DataValidationError, match=message):
        table = normalize_chunk(raw(**changes), dataset)
        validate_chunk(table, dataset, date(2025, 1, 1))


def test_rejects_bad_source_shape_and_chunk_boundary() -> None:
    """Source schemas and preceding chunks are checked explicitly."""
    dataset = get_dataset("spot", "klines")
    with pytest.raises(DataValidationError, match="physical row"):
        normalize_chunk(raw().drop(["__row_number"]), dataset)
    table = normalize_chunk(raw(), dataset)
    with pytest.raises(DataValidationError, match="preceding"):
        validate_chunk(
            table,
            dataset,
            date(2025, 1, 1),
            datetime(2024, 12, 31, 16, tzinfo=UTC),
        )


def test_rejects_non_kline_dispatch() -> None:
    """Dataset dispatch cannot silently use a Kline schema."""
    with pytest.raises(ValueError, match="unsupported normalizer"):
        normalize_chunk(raw(), get_dataset("spot", "trades"))
