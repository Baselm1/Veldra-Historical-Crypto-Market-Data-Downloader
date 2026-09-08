"""Exercise exact IDs and streamed Arrow archive conversion."""

from datetime import date
import pyarrow as pa
import pytest
from crypto_downloader.binance.datasets import SPOT_TRADES
from crypto_downloader.binance.processing import (
    normalize_chunk,
    validate_chunk,
    DataValidationError,
)


def test_large_trade_ids_are_not_rounded() -> None:
    """Preserve signed 64-bit identifiers larger than float's exact range."""
    values = ["9007199254740993", "100", "2", "200", "1735689600010866", "true", "true"]
    source = pa.table(
        {name: [value] for name, value in zip(SPOT_TRADES.source_columns, values)}
    )
    normalized = normalize_chunk(source, SPOT_TRADES)
    assert normalized["trade_id"][0].as_py() == 9007199254740993
    assert (
        validate_chunk(normalized, SPOT_TRADES, date(2025, 1, 1)).microsecond == 10866
    )


@pytest.mark.parametrize(
    "identifier", ["1.5", "9223372036854775808", "-9223372036854775809"]
)
def test_invalid_integer_does_not_round_or_overflow(identifier: str) -> None:
    """Reject fractional or out-of-range identifiers before writing Parquet."""
    values = [identifier, "100", "2", "200", "1735689600010866", "true", "true"]
    source = pa.table(
        {name: [value] for name, value in zip(SPOT_TRADES.source_columns, values)}
    )
    with pytest.raises(DataValidationError, match="integer"):
        normalize_chunk(source, SPOT_TRADES)
