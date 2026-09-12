"""Test KuCoin perpetual Futures Kline normalization and validation."""

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest

from veldra.core.models import DataValidationError
from veldra.kucoin.datasets import (
    INVERSE_INDEX_PRICE_KLINES,
    INVERSE_KLINES,
    LINEAR_KLINES,
    LINEAR_MARK_PRICE_KLINES,
)
from veldra.kucoin.processing import normalize_chunk, validate_chunk


def raw_klines(*, reference: bool = False, volume: str = "2435") -> pa.Table:
    """Build a small raw Futures Kline or reference-price table."""
    values: dict[str, list[str]] = {
        "time": ["1735689600000", "1735689660000"],
        "open": ["93576", "93610"],
        "high": ["93620", "93630"],
        "low": ["93500", "93600"],
        "close": ["93610", "93620"],
    }
    if not reference:
        values["volume"] = [volume, "536"]
    return pa.table(values)


@pytest.mark.parametrize("dataset", [LINEAR_KLINES, INVERSE_KLINES])
def test_perpetual_klines_retain_explicit_contract_volume(dataset: object) -> None:
    """Confirm linear and inverse archives do not fabricate missing quantities."""
    table = normalize_chunk(raw_klines(), dataset)  # type: ignore[arg-type]

    assert table.column_names[-1] == "contract_volume"
    assert table["contract_volume"].to_pylist() == [2435.0, 536.0]
    assert validate_chunk(  # type: ignore[arg-type]
        table, dataset, date(2025, 1, 1)
    ) == datetime(2025, 1, 1, 0, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "dataset", [LINEAR_MARK_PRICE_KLINES, INVERSE_INDEX_PRICE_KLINES]
)
def test_reference_klines_add_resampleable_sample_counts(dataset: object) -> None:
    """Confirm each native reference observation contributes one sample."""
    table = normalize_chunk(raw_klines(reference=True), dataset)  # type: ignore[arg-type]

    assert table["sample_count"].to_pylist() == [1, 1]
    assert validate_chunk(table, dataset, date(2025, 1, 1))  # type: ignore[arg-type]


def test_futures_klines_reject_seconds_and_negative_contract_volume() -> None:
    """Confirm source units and contract counts are validated."""
    seconds = raw_klines().set_column(0, "time", pa.array(["1735689600", "1735689660"]))
    with pytest.raises(DataValidationError, match="timestamp unit"):
        normalize_chunk(seconds, LINEAR_KLINES)

    negative = normalize_chunk(raw_klines(volume="-1"), LINEAR_KLINES)
    with pytest.raises(DataValidationError, match="nonnegative"):
        validate_chunk(negative, LINEAR_KLINES, date(2025, 1, 1))


def test_reference_kline_schema_rejects_trade_price_volume() -> None:
    """Confirm a trade-price file cannot masquerade as an index archive."""
    with pytest.raises(DataValidationError, match="Futures Kline schema"):
        normalize_chunk(raw_klines(), INVERSE_INDEX_PRICE_KLINES)
