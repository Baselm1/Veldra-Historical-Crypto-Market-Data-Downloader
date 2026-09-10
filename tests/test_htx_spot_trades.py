"""Test HTX Spot trade schemas, processing, and public facade."""

from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from veldra.core.models import DataValidationError
from veldra.core.query import empty_frame
from veldra.htx.datasets import (
    NEW_TRADE_COLUMNS,
    OLD_TRADE_COLUMNS,
    SPOT_TRADES,
    get_dataset,
)
from veldra.htx.facade import HTX
from veldra.htx.processing import normalize_chunk, validate_chunk


def raw_table(columns: tuple[str, ...], rows: Sequence[tuple[object, ...]]) -> pa.Table:
    """Build a string-valued trade source table.

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
    ("columns", "row"),
    [
        (
            OLD_TRADE_COLUMNS,
            (103130425211, 1735660802822, 95432.42, 0.00016, "buy"),
        ),
        (
            NEW_TRADE_COLUMNS,
            (
                "BTC-USDT",
                103628174219,
                79668.56,
                "SELL",
                0.00018,
                1788710400322,
            ),
        ),
    ],
)
def test_spot_trade_variants_normalize_with_derived_quote_quantity(
    columns: tuple[str, ...], row: tuple[object, ...]
) -> None:
    """Confirm old and new trades share one explicit quantity schema.

    Args:
        columns: The source schema under test.
        row: One representative source trade.
    """
    result = normalize_chunk(raw_table(columns, [row]), SPOT_TRADES)

    assert result.column_names == list(SPOT_TRADES.stored_columns)
    expected_id = row[0 if "id" in columns else 1]
    assert isinstance(expected_id, int)
    assert result["trade_id"][0].as_py() == expected_id
    assert result["quote_quantity"][0].as_py() == pytest.approx(
        result["price"][0].as_py() * result["base_quantity"][0].as_py()
    )
    assert result["side"][0].as_py() in {"buy", "sell"}


def test_spot_trades_sort_deterministically_with_repeated_timestamps() -> None:
    """Confirm timestamp ties use trade ID and retain a valid event stream."""
    rows = [
        (12, 1735660802822, 2, 3, "sell"),
        (10, 1735660801000, 2, 3, "buy"),
        (11, 1735660802822, 2, 3, "buy"),
    ]
    table = normalize_chunk(raw_table(OLD_TRADE_COLUMNS, rows), SPOT_TRADES)
    table = table.sort_by([("event_time", "ascending"), ("trade_id", "ascending")])

    last = validate_chunk(table, SPOT_TRADES, date(2025, 1, 1))

    assert table["trade_id"].to_pylist() == [10, 11, 12]
    assert last == datetime(2024, 12, 31, 16, 0, 2, 822000, tzinfo=UTC)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ((-1, 1735660802822, 2, 3, "buy"), "nonnegative"),
        ((1, 1735660802822, 0, 3, "buy"), "positive"),
        ((1, 1735660802822, 2, -3, "buy"), "nonnegative"),
        ((1, 1735660802822, 2, 3, "hold"), "buy or sell"),
        ((1, 1735574399000, 2, 3, "buy"), "outside"),
    ],
)
def test_spot_trade_validation_rejects_invalid_values(
    row: tuple[object, ...], message: str
) -> None:
    """Confirm invalid identifiers, values, sides, and bounds fail.

    Args:
        row: The malformed source trade.
        message: Text identifying the expected error.
    """
    table = normalize_chunk(raw_table(OLD_TRADE_COLUMNS, [row]), SPOT_TRADES)
    with pytest.raises(DataValidationError, match=message):
        validate_chunk(table, SPOT_TRADES, date(2025, 1, 1))


def test_duplicate_trade_id_at_one_timestamp_is_preserved() -> None:
    """Confirm HTX's repeated match identifiers do not discard trades."""
    rows = [
        (1, 1735660802822, 2, 3, "buy"),
        (1, 1735660802822, 2, 3, "sell"),
    ]
    table = normalize_chunk(raw_table(OLD_TRADE_COLUMNS, rows), SPOT_TRADES)

    last = validate_chunk(table, SPOT_TRADES, date(2025, 1, 1))

    assert len(table) == 2
    assert last == datetime(2024, 12, 31, 16, 0, 2, 822000, tzinfo=UTC)


def test_spot_trade_schema_exposes_text_side_in_empty_frames() -> None:
    """Confirm string columns retain a useful empty pandas dtype."""
    assert get_dataset("spot", "trades") is SPOT_TRADES
    frame = empty_frame(SPOT_TRADES, SPOT_TRADES.resolve_columns(None))
    assert str(frame.dtypes["side"]) == "string"


def test_htx_facade_delegates_spot_trades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm trade requests reach the shared engine without Kline options.

    Args:
        tmp_path: The isolated data directory.
        monkeypatch: Pytest's attribute replacement helper.
    """
    service = HTX(tmp_path, progress=False)
    expected = [pd.DataFrame({"trade_id": [1]}), pd.DataFrame({"trade_id": [2]})]
    received: dict[str, object] = {}

    def get_data(*args: object, **kwargs: object) -> list[pd.DataFrame]:
        """Record one delegated trade request."""
        received.update(kwargs)
        return expected

    monkeypatch.setattr(service._downloader, "get_data", get_data)
    actual = service.get_trades(
        ["BTCUSDT", "ETHUSDT"],
        "2025-01-01",
        "2025-01-02",
        columns=["event_time", "price"],
    )

    assert actual is expected
    assert received == {
        "product": "spot",
        "dataset": "trades",
        "desired_columns": ["event_time", "price"],
        "refresh": False,
        "offline": False,
        "progress": False,
    }
