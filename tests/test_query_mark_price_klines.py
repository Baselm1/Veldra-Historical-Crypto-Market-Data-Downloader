"""Test DuckDB query behavior for perpetual Futures mark-price candles."""

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from crypto_downloader._core.datasets import (
    CM_MARK_PRICE_KLINES,
    UM_MARK_PRICE_KLINES,
    DatasetSpec,
)
from crypto_downloader._core.query import query_parquet


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Provide one isolated DuckDB connection.

    Returns:
        An in-memory database connection.
    """
    value = duckdb.connect()
    yield value
    value.close()


def price_frame(dataset: DatasetSpec) -> pd.DataFrame:
    """Build two consecutive price-only one-minute candles.

    Args:
        dataset: The UM or CM mark-price dataset declaration.

    Returns:
        Two canonical source candles.
    """
    opens = pd.date_range("2024-01-01", periods=2, freq="1min", tz=UTC).as_unit("us")
    return pd.DataFrame(
        {
            "open_time": opens,
            "open": [100.0, 101.0],
            "high": [102.0, 104.0],
            "low": [99.0, 100.0],
            "close": [101.0, 103.0],
            "close_time": opens + pd.Timedelta(seconds=59, microseconds=999999),
            "sample_count": pd.Series([60, 60], dtype="int64"),
        }
    ).loc[:, dataset.stored_columns]


@pytest.mark.parametrize("dataset", [UM_MARK_PRICE_KLINES, CM_MARK_PRICE_KLINES])
def test_mark_price_resampling_uses_only_prices_and_samples(
    connection: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    dataset: DatasetSpec,
) -> None:
    """Confirm price-only candles resample without trading-volume fields.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated Parquet directory.
        dataset: The product-specific mark-price declaration.
    """
    path = tmp_path / "mark-price.parquet"
    price_frame(dataset).to_parquet(path, index=False)

    result = query_parquet(
        connection,
        [path],
        dataset,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
        dataset.resolve_columns(None),
        interval="3m",
        gap_policy="keep",
    )

    assert len(result) == 1
    assert result["open"].tolist() == [100.0]
    assert result["high"].tolist() == [104.0]
    assert result["low"].tolist() == [99.0]
    assert result["close"].tolist() == [103.0]
    assert result["sample_count"].tolist() == [120]
    assert "volume" not in result.columns
