"""Test KuCoin perpetual Futures Kline normalization and validation."""

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest

from veldra.core.models import DataValidationError
from veldra.kucoin.datasets import (
    INVERSE_FUNDING_RATES,
    INVERSE_INDEX_PRICE_KLINES,
    INVERSE_KLINES,
    INVERSE_TRADES,
    LINEAR_FUNDING_RATES,
    LINEAR_KLINES,
    LINEAR_MARK_PRICE_KLINES,
    LINEAR_TRADES,
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


def raw_trades(**changes: list[str]) -> pa.Table:
    """Build a small raw KuCoin perpetual trade table."""
    values = {
        "trade_id": ["0d195e29", "5793017801"],
        "trade_time": ["1735689600051", "1735689600051"],
        "price": ["93548.8", "93548.8"],
        "size": ["36", "20"],
        "side": ["buy", "SELL"],
    }
    values.update(changes)
    return pa.table(values)


def raw_funding() -> pa.Table:
    """Build two funding observations without assuming a settlement cycle."""
    return pa.table(
        {
            "symbol": ["BTCUSDTM", "BTCUSDTM"],
            "time": ["1735689600000", "1735718400000"],
            "fundingRate": ["-0.0001", "0.0002"],
        }
    )


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


def test_linear_trades_derive_base_and_quote_quantities() -> None:
    """Confirm linear contract multipliers produce explicit asset quantities."""
    table = normalize_chunk(raw_trades(), LINEAR_TRADES, 0.001)

    assert table["trade_id"].to_pylist() == ["0d195e29", "5793017801"]
    assert table["contract_quantity"].to_pylist() == [36.0, 20.0]
    assert table["base_quantity"].to_pylist() == pytest.approx([0.036, 0.02])
    assert table["quote_quantity"].to_pylist() == pytest.approx([3367.7568, 1870.976])
    assert validate_chunk(table, LINEAR_TRADES, date(2025, 1, 1))


def test_inverse_trades_derive_quote_notional_and_base_quantity() -> None:
    """Confirm inverse contract face value is divided by the trade price."""
    table = normalize_chunk(raw_trades(), INVERSE_TRADES, 1.0)

    assert table["quote_notional"].to_pylist() == [36.0, 20.0]
    assert table["base_quantity"].to_pylist() == pytest.approx(
        [36 / 93548.8, 20 / 93548.8]
    )
    assert validate_chunk(table, INVERSE_TRADES, date(2025, 1, 1))


@pytest.mark.parametrize("value", [None, 0, float("inf"), True])
def test_futures_trades_require_a_valid_contract_multiplier(value: object) -> None:
    """Confirm quantity derivation never guesses missing contract metadata."""
    with pytest.raises(DataValidationError, match="contract multiplier"):
        normalize_chunk(raw_trades(), LINEAR_TRADES, value)  # type: ignore[arg-type]


@pytest.mark.parametrize("dataset", [LINEAR_FUNDING_RATES, INVERSE_FUNDING_RATES])
def test_funding_rates_preserve_actual_observation_times_and_sign(
    dataset: object,
) -> None:
    """Confirm funding rows remain signed point events without fixed schedules."""
    table = normalize_chunk(raw_funding(), dataset)  # type: ignore[arg-type]

    assert table["funding_rate"].to_pylist() == [-0.0001, 0.0002]
    assert table["funding_time"].to_pylist() == [
        datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 8, 0, tzinfo=UTC),
    ]
    assert validate_chunk(table, dataset, date(2025, 1, 1))  # type: ignore[arg-type]


def test_futures_trades_reject_invalid_sides_and_negative_contracts() -> None:
    """Confirm malformed perpetual events cannot enter the cache."""
    side = normalize_chunk(raw_trades(side=["hold", "buy"]), LINEAR_TRADES, 0.001)
    with pytest.raises(DataValidationError, match="side"):
        validate_chunk(side, LINEAR_TRADES, date(2025, 1, 1))
    quantity = normalize_chunk(raw_trades(size=["-1", "2"]), LINEAR_TRADES, 0.001)
    with pytest.raises(DataValidationError, match="nonnegative"):
        validate_chunk(quantity, LINEAR_TRADES, date(2025, 1, 1))
