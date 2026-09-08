"""Test normalization and validation of Binance Spot trade archives."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_downloader._core.datasets import DatasetSpec
from crypto_downloader.binance.datasets import SPOT_AGG_TRADES, SPOT_TRADES
from crypto_downloader._core.processing import (
    DataValidationError,
    normalize_chunk,
    validate_chunk,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("dataset", "fixture", "expected_time"),
    [
        (SPOT_TRADES, "binance_spot_trades_2024-01-01.csv", "2024-01-01T00:00:00Z"),
        (
            SPOT_TRADES,
            "binance_spot_trades_2025-01-01.csv",
            "2025-01-01T00:00:00.010866Z",
        ),
        (
            SPOT_AGG_TRADES,
            "binance_spot_agg_trades_2024-01-01.csv",
            "2024-01-01T00:00:00Z",
        ),
        (
            SPOT_AGG_TRADES,
            "binance_spot_agg_trades_2025-01-01.csv",
            "2025-01-01T00:00:00.010866Z",
        ),
    ],
)
def test_normalize_spot_trade_archives_across_timestamp_eras(
    dataset: DatasetSpec, fixture: str, expected_time: str
) -> None:
    """Confirm Spot trade families normalize their documented archive fields.

    Args:
        dataset: The declared Spot event dataset.
        fixture: The representative headerless CSV file.
        expected_time: The first canonical UTC timestamp.
    """
    source = pd.read_csv(
        FIXTURES / fixture,
        header=None,
        names=dataset.source_columns,
        dtype=str,
    )

    result = normalize_chunk(source, dataset)

    assert tuple(result.columns) == dataset.stored_columns
    assert result.loc[0, "event_time"] == pd.Timestamp(expected_time)
    assert result["event_time"].dtype == "datetime64[us, UTC]"
    assert result["buyer_is_maker"].dtype == np.dtype("bool")
    assert result.loc[0, "quote_quantity"] == pytest.approx(
        result.loc[0, "price"] * result.loc[0, "base_quantity"]
    )
    if dataset is SPOT_TRADES:
        assert result["trade_id"].dtype == np.dtype("int64")
    else:
        assert result["agg_trade_id"].dtype == np.dtype("int64")
        assert result["first_trade_id"].dtype == np.dtype("int64")
        assert result["last_trade_id"].dtype == np.dtype("int64")


@pytest.mark.parametrize("dataset", [SPOT_TRADES, SPOT_AGG_TRADES])
def test_validate_spot_events_allows_shared_timestamps_and_checks_daily_bounds(
    dataset: DatasetSpec,
) -> None:
    """Confirm real events may share a timestamp but cannot leave their day.

    Args:
        dataset: The Spot event dataset to validate.
    """
    fixture = (
        "binance_spot_trades_2024-01-01.csv"
        if dataset is SPOT_TRADES
        else "binance_spot_agg_trades_2024-01-01.csv"
    )
    source = pd.read_csv(
        FIXTURES / fixture,
        header=None,
        names=dataset.source_columns,
        dtype=str,
    )
    frame = normalize_chunk(source, dataset)

    assert (
        validate_chunk(frame, dataset, date(2024, 1, 1)) == frame.iloc[-1]["event_time"]
    )

    frame["event_time"] += pd.Timedelta(days=1)
    with pytest.raises(DataValidationError, match="resource day"):
        validate_chunk(frame, dataset, date(2024, 1, 1))


@pytest.mark.parametrize("dataset", [SPOT_TRADES, SPOT_AGG_TRADES])
def test_spot_trade_normalization_rejects_invalid_values(dataset: DatasetSpec) -> None:
    """Confirm malformed IDs, values, booleans, and epochs are rejected.

    Args:
        dataset: The Spot event dataset to normalize.
    """
    fixture = (
        "binance_spot_trades_2024-01-01.csv"
        if dataset is SPOT_TRADES
        else "binance_spot_agg_trades_2024-01-01.csv"
    )
    source = pd.read_csv(
        FIXTURES / fixture,
        header=None,
        names=dataset.source_columns,
        dtype=str,
    )
    id_column = "trade_id" if dataset is SPOT_TRADES else "agg_trade_id"

    for column, value in (
        (id_column, "1.5"),
        ("price", "nan"),
        ("base_quantity", "-1"),
        ("event_time", "bad"),
        ("is_buyer_maker", "perhaps"),
    ):
        invalid = source.copy()
        invalid.loc[0, column] = value
        with pytest.raises(DataValidationError):
            normalized = normalize_chunk(invalid, dataset)
            validate_chunk(normalized, dataset, date(2024, 1, 1))
