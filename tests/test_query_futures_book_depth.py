"""Test DuckDB querying of raw perpetual Futures book-depth snapshots."""

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from crypto_downloader.datasets import CM_BOOK_DEPTH, UM_BOOK_DEPTH, DatasetSpec
from crypto_downloader.query import query_parquet


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Provide an isolated in-memory DuckDB connection.

    Yields:
        An open DuckDB connection.
    """
    value = duckdb.connect()
    yield value
    value.close()


def depth_frame(dataset: DatasetSpec) -> pd.DataFrame:
    """Build shuffled bid and ask buckets for one depth snapshot.

    Args:
        dataset: The product-specific declaration naming depth quantities.

    Returns:
        A canonical but deliberately unordered book-depth frame.
    """
    values: dict[str, object] = {
        "event_time": pd.Series(
            [pd.Timestamp("2024-01-01T00:00:00Z")] * 2,
            dtype="datetime64[us, UTC]",
        ),
        "percentage_bucket": pd.Series([1, -1], dtype="int64"),
    }
    if dataset is UM_BOOK_DEPTH:
        values["base_depth"] = [2.0, 1.0]
        values["quote_notional"] = [200.0, 100.0]
    else:
        values["contract_depth"] = [2.0, 1.0]
        values["base_notional"] = [0.02, 0.01]
    return pd.DataFrame(values).loc[:, dataset.stored_columns]


@pytest.mark.parametrize("dataset", [UM_BOOK_DEPTH, CM_BOOK_DEPTH])
def test_book_depth_query_filters_rows_and_applies_declared_bucket_order(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path, dataset: DatasetSpec
) -> None:
    """Confirm raw depth rows retain buckets and query in stable order.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated Parquet directory.
        dataset: The USD-M or COIN-M book-depth declaration.
    """
    path = tmp_path / f"{dataset.product}-book-depth.parquet"
    depth_frame(dataset).to_parquet(path, index=False)

    result = query_parquet(
        connection,
        [path],
        dataset,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
        dataset.resolve_columns(None),
        interval=None,
        gap_policy=None,
    )

    assert len(result) == 2
    assert result["percentage_bucket"].tolist() == [-1, 1]
    assert "is_synthetic" not in result.columns
