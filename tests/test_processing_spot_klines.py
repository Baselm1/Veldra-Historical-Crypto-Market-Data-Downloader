"""Test normalization and validation of Binance Spot klines."""

from datetime import date
from pathlib import Path
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from crypto_downloader.binance.datasets import SPOT_KLINES
from arrow_helpers import (
    DataValidationError,
    normalize_chunk,
    validate_chunk,
)

FIXTURES = Path(__file__).parent / "fixtures"


def source_frame(name: str) -> pd.DataFrame:
    """Read one headerless Spot kline fixture as source strings.

    Args:
        name: The fixture filename.

    Returns:
        Raw source rows with the declared Binance columns.
    """
    return pd.read_csv(
        FIXTURES / name,
        header=None,
        names=SPOT_KLINES.source_columns,
        dtype=str,
    )


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("binance_spot_klines_2024-01-01.csv", "2024-01-01 00:00:00+00:00"),
        ("binance_spot_klines_2025-01-01.csv", "2025-01-01 00:00:00+00:00"),
    ],
)
def test_normalize_chunk_handles_millisecond_and_microsecond_epochs(
    fixture: str, expected: str
) -> None:
    """Confirm both Binance timestamp eras become UTC timestamps.

    Args:
        fixture: The representative source file to normalize.
        expected: The first expected UTC timestamp.
    """
    source = source_frame(fixture)
    original = source.copy(deep=True)

    result = normalize_chunk(source, SPOT_KLINES)

    pd.testing.assert_frame_equal(source, original)
    assert tuple(result.columns) == SPOT_KLINES.stored_columns
    assert result.loc[0, "open_time"] == pd.Timestamp(expected)
    assert str(result["open_time"].dtype).endswith(", UTC]")
    assert str(result["close_time"].dtype).endswith(", UTC]")
    assert result["trade_count"].dtype == np.dtype("int64")
    assert all(
        result[column].dtype == np.dtype("float64")
        for column in SPOT_KLINES.stored_columns
        if column not in {"open_time", "close_time", "trade_count"}
    )
    assert "ignore" not in result


def test_normalize_chunk_requires_exact_source_schema() -> None:
    """Confirm missing, extra, or reordered source columns are rejected."""
    frame = source_frame("binance_spot_klines_2024-01-01.csv")

    for invalid in (
        frame.drop(columns="ignore"),
        frame.assign(extra="x"),
        frame[list(reversed(frame.columns))],
    ):
        with pytest.raises(DataValidationError, match="source columns"):
            normalize_chunk(invalid, SPOT_KLINES)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("open", "not-number", "open"),
        ("open", "nan", "finite"),
        ("high", "inf", "finite"),
        ("count", "1.5", "count"),
        ("open_time", "bad", "open_time"),
        ("open_time", "170406720000000", "timestamp unit"),
    ],
)
def test_normalize_chunk_rejects_invalid_values(
    column: str, value: str, message: str
) -> None:
    """Confirm invalid numeric and epoch values name their cause.

    Args:
        column: The source field to corrupt.
        value: The invalid source string.
        message: Text expected in the validation error.
    """
    frame = source_frame("binance_spot_klines_2024-01-01.csv")
    frame.loc[0, column] = value

    with pytest.raises(DataValidationError, match=message):
        normalize_chunk(frame, SPOT_KLINES)


def test_normalize_chunk_rejects_mixed_epoch_units() -> None:
    """Confirm one timestamp column cannot mix milliseconds and microseconds."""
    frame = source_frame("binance_spot_klines_2024-01-01.csv")
    frame.loc[1, "open_time"] = "1735689660000000"

    with pytest.raises(DataValidationError, match="timestamp unit"):
        normalize_chunk(frame, SPOT_KLINES)


def valid_frame() -> pd.DataFrame:
    """Return a normalized two-row daily Spot kline frame.

    Returns:
        Canonical rows suitable for focused validation mutations.
    """
    return normalize_chunk(
        source_frame("binance_spot_klines_2024-01-01.csv"), SPOT_KLINES
    )


def test_validate_chunk_accepts_valid_rows_and_an_internal_gap() -> None:
    """Confirm validation permits source outages without inventing rows."""
    frame = valid_frame()
    frame.loc[1, "open_time"] += pd.Timedelta(minutes=4)
    frame.loc[1, "close_time"] += pd.Timedelta(minutes=4)

    result = validate_chunk(frame, SPOT_KLINES, date(2024, 1, 1))

    assert result == frame.iloc[-1]["open_time"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.iloc[0:0], "empty"),
        (lambda frame: frame.drop(columns="volume"), "stored columns"),
        (lambda frame: frame.assign(extra=1), "stored columns"),
        (lambda frame: frame.iloc[::-1].reset_index(drop=True), "increasing"),
        (lambda frame: pd.concat([frame.iloc[[0]], frame.iloc[[0]]]), "increasing"),
        (
            lambda frame: frame.assign(
                open_time=frame["open_time"] + pd.Timedelta(seconds=1)
            ),
            "aligned",
        ),
        (
            lambda frame: frame.assign(
                close_time=frame["open_time"] - pd.Timedelta(microseconds=1)
            ),
            "close_time",
        ),
        (
            lambda frame: frame.assign(
                close_time=frame["open_time"] + pd.Timedelta(minutes=1)
            ),
            "close_time",
        ),
        (lambda frame: frame.assign(high=frame["open"] - 1), "high"),
        (lambda frame: frame.assign(low=frame["high"]), "low"),
        (lambda frame: frame.assign(open=0.0), "price"),
        (lambda frame: frame.assign(volume=-1.0), "nonnegative"),
        (lambda frame: frame.assign(trade_count=-1), "nonnegative"),
        (lambda frame: frame.assign(open=np.inf), "finite"),
    ],
)
def test_validate_chunk_rejects_invalid_rows(
    mutation: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    """Confirm structural and market-data errors are rejected.

    Args:
        mutation: A callable that corrupts an otherwise valid frame.
        message: Text expected in the validation error.
    """
    invalid = mutation(valid_frame())

    with pytest.raises(DataValidationError, match=message):
        validate_chunk(invalid, SPOT_KLINES, date(2024, 1, 1))


def test_validate_chunk_requires_rows_inside_the_resource_day() -> None:
    """Confirm timestamps cannot leak across a daily archive boundary."""
    frame = valid_frame()
    frame["open_time"] += pd.Timedelta(days=1)
    frame["close_time"] += pd.Timedelta(days=1)

    with pytest.raises(DataValidationError, match="resource day"):
        validate_chunk(frame, SPOT_KLINES, date(2024, 1, 1))


def test_validate_chunk_requires_order_across_chunks() -> None:
    """Confirm a later chunk must begin after the preceding chunk ends."""
    frame = valid_frame()

    with pytest.raises(DataValidationError, match="preceding chunk"):
        validate_chunk(
            frame,
            SPOT_KLINES,
            date(2024, 1, 1),
            previous_timestamp=frame.iloc[0]["open_time"],
        )


@pytest.mark.parametrize("column", ["open_time", "close_time"])
def test_validate_chunk_rejects_null_timestamps(column: str) -> None:
    """Confirm null timestamps are rejected before other comparisons.

    Args:
        column: The timestamp column made null.
    """
    frame = valid_frame()
    frame.loc[0, column] = pd.NaT

    with pytest.raises(DataValidationError, match="null"):
        validate_chunk(frame, SPOT_KLINES, date(2024, 1, 1))


@pytest.mark.parametrize("column", ["open_time", "close_time"])
def test_validate_chunk_requires_utc_timestamps(column: str) -> None:
    """Confirm naive timestamps cannot be mistaken for UTC values.

    Args:
        column: The timestamp column stripped of its timezone.
    """
    frame = valid_frame()
    frame[column] = frame[column].dt.tz_localize(None)

    with pytest.raises(DataValidationError, match=f"{column} must use UTC"):
        validate_chunk(frame, SPOT_KLINES, date(2024, 1, 1))


def test_processing_rejects_an_unsupported_dataset_spec() -> None:
    """Confirm Spot-specific processing cannot silently handle another dataset."""
    unsupported = replace(SPOT_KLINES, name="not_supported")
    source = source_frame("binance_spot_klines_2024-01-01.csv")

    with pytest.raises(ValueError, match="normalizer"):
        normalize_chunk(source, unsupported)
    with pytest.raises(ValueError, match="validator"):
        validate_chunk(valid_frame(), unsupported, date(2024, 1, 1))
