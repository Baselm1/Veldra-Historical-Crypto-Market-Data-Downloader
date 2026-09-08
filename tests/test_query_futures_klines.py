"""Test DuckDB resampling for perpetual Futures Kline quantity units."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from crypto_downloader._core.datasets import DatasetSpec
from crypto_downloader.binance.datasets import CM_KLINES, UM_KLINES
from crypto_downloader._core.query import query_parquet


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Provide an isolated DuckDB connection.

    Yields:
        A temporary in-memory DuckDB connection.
    """
    value = duckdb.connect()
    yield value
    value.close()


def futures_frame(dataset: DatasetSpec) -> pd.DataFrame:
    """Build two canonical Futures candles with distinct quantity units.

    Args:
        dataset: The UM or CM Kline schema used to name quantity fields.

    Returns:
        Two consecutive canonical one-minute candles.
    """
    opens = pd.date_range("2024-01-01", periods=2, freq="1min", tz=UTC).as_unit("us")
    values: dict[str, object] = {
        "open_time": opens,
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "close_time": opens + pd.Timedelta(seconds=59, microseconds=999999),
        "trade_count": pd.Series([3, 4], dtype="int64"),
    }
    for index, column in enumerate(dataset.resample_sum_columns):
        if column == "trade_count":
            continue
        values[column] = [float(index + 1), float((index + 1) * 10)]
    return pd.DataFrame(values).loc[:, dataset.stored_columns]


@pytest.mark.parametrize(
    ("dataset", "quantities"),
    [
        (UM_KLINES, ("base_volume", "quote_volume")),
        (CM_KLINES, ("contract_volume", "base_volume")),
    ],
)
def test_futures_resampling_sums_the_declared_product_quantities(
    connection: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    dataset: DatasetSpec,
    quantities: tuple[str, str],
) -> None:
    """Confirm resampling uses each schema instead of Spot volume field names.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated Parquet directory.
        dataset: The UM or CM Kline schema under test.
        quantities: The first two explicit quantity columns to inspect.
    """
    path = tmp_path / "futures.parquet"
    futures_frame(dataset).to_parquet(path, index=False)

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
    assert result[quantities[0]].tolist() == [11.0]
    assert result[quantities[1]].tolist() == [22.0]
    assert result["trade_count"].tolist() == [7]
    assert result["close"].tolist() == [102.0]
    assert "volume" not in result.columns
