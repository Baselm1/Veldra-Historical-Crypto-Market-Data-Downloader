"""Test DuckDB querying of raw perpetual Futures metrics snapshots."""

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from veldra.core.datasets import DatasetSpec
from veldra.binance.datasets import CM_METRICS, UM_METRICS
from veldra.core.query import query_parquet


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Provide an isolated in-memory DuckDB connection.

    Yields:
        An open DuckDB connection.
    """
    value = duckdb.connect()
    yield value
    value.close()


def metrics_frame(dataset: DatasetSpec) -> pd.DataFrame:
    """Build two canonical five-minute Futures metrics observations.

    Args:
        dataset: The UM or CM metrics declaration that names the quantities.

    Returns:
        A canonical raw metrics DataFrame.
    """
    values: dict[str, object] = {
        "event_time": pd.date_range(
            "2024-01-01", periods=2, freq="5min", tz=UTC
        ).as_unit("us"),
        "top_trader_account_long_short_ratio": [1.0, 1.1],
        "top_trader_position_long_short_ratio": [1.2, 1.3],
        "account_long_short_ratio": [1.4, 1.5],
        "taker_long_short_volume_ratio": [1.6, 1.7],
    }
    if dataset is UM_METRICS:
        values["open_interest_base_quantity"] = [100.0, 101.0]
        values["open_interest_quote_value"] = [4_000_000.0, 4_100_000.0]
    else:
        values["open_interest_contract_quantity"] = [10_000.0, 10_100.0]
        values["open_interest_base_quantity"] = [250.0, 251.0]
    return pd.DataFrame(values).loc[:, dataset.stored_columns]


@pytest.mark.parametrize("dataset", [UM_METRICS, CM_METRICS])
def test_metrics_query_filters_raw_snapshots_without_resampling(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path, dataset: DatasetSpec
) -> None:
    """Confirm exact DuckDB ranges return only real metrics observations.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated Parquet directory.
        dataset: The product-specific metrics declaration.
    """
    path = tmp_path / f"{dataset.product}-metrics.parquet"
    metrics_frame(dataset).to_parquet(path, index=False)

    result = query_parquet(
        connection,
        [path],
        dataset,
        datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
        dataset.resolve_columns(None),
        interval=None,
        gap_policy=None,
    )

    assert len(result) == 1
    assert result["event_time"].tolist() == [pd.Timestamp("2024-01-01T00:05:00Z")]
    assert "is_synthetic" not in result.columns
