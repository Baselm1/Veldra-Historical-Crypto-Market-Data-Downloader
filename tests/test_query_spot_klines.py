"""Test DuckDB queries over cached Binance Spot klines."""

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from veldra.core.datasets import DatasetSpec
from veldra.binance.datasets import SPOT_AGG_TRADES, SPOT_KLINES, SPOT_TRADES
from arrow_helpers import normalize_chunk
from veldra.core.query import empty_frame, query_parquet


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Provide an isolated DuckDB connection.

    Yields:
        A temporary in-memory DuckDB connection.
    """
    value = duckdb.connect()
    yield value
    value.close()


def kline_frame(day: str, prices: tuple[float, float]) -> pd.DataFrame:
    """Create two canonical one-minute rows for one day.

    Args:
        day: The UTC calendar day used by the rows.
        prices: The opening price for each row.

    Returns:
        A canonical cached Spot kline DataFrame.
    """
    opens = pd.to_datetime([f"{day} 00:00:00Z", f"{day} 00:01:00Z"], utc=True).as_unit(
        "us"
    )
    return pd.DataFrame(
        {
            "open_time": opens,
            "open": prices,
            "high": [price + 2 for price in prices],
            "low": [price - 1 for price in prices],
            "close": [price + 1 for price in prices],
            "volume": [10.0, 20.0],
            "close_time": opens + pd.Timedelta(seconds=59, microseconds=999999),
            "quote_volume": [1000.0, 2000.0],
            "trade_count": pd.Series([10, 20], dtype="int64"),
            "taker_buy_base_volume": [4.0, 8.0],
            "taker_buy_quote_volume": [400.0, 800.0],
        }
    )


def parquet_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write two daily canonical Parquet fixtures.

    Args:
        tmp_path: The isolated fixture directory.

    Returns:
        Paths for January 1 and January 2 in chronological order.
    """
    first = tmp_path / "2025-01-01.parquet"
    second = tmp_path / "2025-01-02.parquet"
    kline_frame("2025-01-01", (100.0, 110.0)).to_parquet(first, index=False)
    kline_frame("2025-01-02", (200.0, 210.0)).to_parquet(second, index=False)
    return first, second


def selected_columns() -> dict[str, str]:
    """Return a small canonical projection used by query tests.

    Returns:
        Canonical column names mapped to caller-facing labels.
    """
    return {"open_time": "time", "close": "price", "trade_count": "trades"}


def raw_events() -> DatasetSpec:
    """Return a minimal raw event schema for generic query behavior.

    Returns:
        An interval-less dataset with deterministic event ordering.
    """
    return DatasetSpec(
        product="spot",
        name="events",
        remote_name="events",
        source_columns=("event_time", "trade_id", "is_buyer_maker", "price"),
        stored_columns=("event_time", "trade_id", "is_buyer_maker", "price"),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases={},
        timestamp_columns=("event_time",),
        integer_columns=("trade_id",),
        boolean_columns=("is_buyer_maker",),
        ordering_columns=("event_time", "trade_id"),
    )


def test_query_parquet_filters_end_exclusively_and_orders_multiple_files(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm exact timestamp filtering and chronological ordering."""
    first, second = parquet_files(tmp_path)

    result = query_parquet(
        connection,
        [second, first],
        SPOT_KLINES,
        datetime(2025, 1, 1, 0, 0, 30, tzinfo=UTC),
        datetime(2025, 1, 2, 0, 1, tzinfo=UTC),
        selected_columns(),
    )

    assert list(result.columns) == ["time", "price", "trades"]
    assert result["time"].tolist() == [
        pd.Timestamp("2025-01-01 00:01:00Z"),
        pd.Timestamp("2025-01-02 00:00:00Z"),
    ]
    assert result["price"].tolist() == [111.0, 201.0]
    assert result["trades"].tolist() == [20, 10]
    assert str(result["time"].dtype) == "datetime64[us, UTC]"
    assert result["price"].dtype == "float64"
    assert result["trades"].dtype == "int64"


def test_query_parquet_supports_all_columns_and_synthetic_marker(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm the default schema adds a false generated-row marker."""
    first, _ = parquet_files(tmp_path)
    columns = SPOT_KLINES.resolve_columns(None)

    result = query_parquet(
        connection,
        [first],
        SPOT_KLINES,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
        columns,
    )

    assert list(result.columns) == list(columns.values())
    assert result["is_synthetic"].tolist() == [False, False]
    assert result["is_synthetic"].dtype == "bool"


def test_query_parquet_quotes_unusual_output_labels(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm caller labels cannot alter the generated SQL."""
    first, _ = parquet_files(tmp_path)
    columns = {"open_time": 'time "UTC"', "close": "select"}

    result = query_parquet(
        connection,
        [first],
        SPOT_KLINES,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
        columns,
    )

    assert list(result.columns) == ['time "UTC"', "select"]
    assert len(result) == 1


def test_query_parquet_accepts_resolved_dataset_aliases(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm aliases are resolved before canonical Parquet projection."""
    first, _ = parquet_files(tmp_path)
    columns = SPOT_KLINES.resolve_columns({"base_volume": "size"})

    result = query_parquet(
        connection,
        [first],
        SPOT_KLINES,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
        columns,
    )

    assert list(result.columns) == ["size"]
    assert result["size"].tolist() == [10.0, 20.0]


@pytest.mark.parametrize(
    "columns",
    [
        {},
        {"unknown": "value"},
        {"open": "same", "close": "same"},
        {"open": " "},
        {"open\x00": "value"},
    ],
)
def test_query_parquet_rejects_invalid_projections(
    connection: duckdb.DuckDBPyConnection,
    columns: Mapping[str, str],
) -> None:
    """Confirm direct query calls reject invalid column mappings.

    Args:
        connection: The isolated DuckDB connection.
        columns: The invalid projection to reject.
    """
    with pytest.raises(ValueError, match="column"):
        query_parquet(
            connection,
            [],
            SPOT_KLINES,
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
            columns,
        )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (
            datetime(2025, 1, 1),
            datetime(2025, 1, 2, tzinfo=UTC),
            "timezone",
        ),
        (
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2),
            "timezone",
        ),
        (
            datetime(2025, 1, 2, tzinfo=UTC),
            datetime(2025, 1, 1, tzinfo=UTC),
            "range",
        ),
        (
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, tzinfo=UTC),
            "range",
        ),
        (
            datetime.fromisoformat("2025-01-01T00:00:00+02:00"),
            datetime.fromisoformat("2025-01-02T00:00:00+02:00"),
            "UTC",
        ),
    ],
)
def test_query_parquet_rejects_invalid_ranges(
    connection: duckdb.DuckDBPyConnection,
    start: datetime,
    end: datetime,
    message: str,
) -> None:
    """Confirm direct query calls require an increasing aware range.

    Args:
        connection: The isolated DuckDB connection.
        start: The proposed inclusive start.
        end: The proposed exclusive end.
        message: Text expected in the validation error.
    """
    with pytest.raises(ValueError, match=message):
        query_parquet(connection, [], SPOT_KLINES, start, end, selected_columns())


@pytest.mark.parametrize(
    "columns",
    [
        {"open_time": "time", "close": "price", "trade_count": "trades"},
        {"close_time": "finished", "volume": "size", "is_synthetic": "filled"},
    ],
)
def test_empty_frame_preserves_requested_order_labels_and_types(
    columns: Mapping[str, str],
) -> None:
    """Confirm no-file results retain a predictable public schema.

    Args:
        columns: The canonical projection and output labels.
    """
    result = empty_frame(SPOT_KLINES, columns)

    assert result.empty
    assert list(result.columns) == list(columns.values())
    for source, label in columns.items():
        if source.endswith("time"):
            assert str(result[label].dtype) == "datetime64[us, UTC]"
        elif source == "trade_count":
            assert result[label].dtype == "int64"
        elif source == "is_synthetic":
            assert result[label].dtype == "bool"
        else:
            assert result[label].dtype == "float64"


def test_query_without_paths_returns_the_typed_empty_frame(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Confirm DuckDB is not asked to read an empty path collection."""
    columns = selected_columns()

    result = query_parquet(
        connection,
        [],
        SPOT_KLINES,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
        columns,
    )

    pd.testing.assert_frame_equal(result, empty_frame(SPOT_KLINES, columns))


def test_existing_files_with_no_matching_rows_return_stable_types(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm an empty SQL result has the same schema as a no-file result."""
    first, _ = parquet_files(tmp_path)
    columns = SPOT_KLINES.resolve_columns(None)

    result = query_parquet(
        connection,
        [first],
        SPOT_KLINES,
        datetime(2025, 2, 1, tzinfo=UTC),
        datetime(2025, 2, 2, tzinfo=UTC),
        columns,
    )

    pd.testing.assert_frame_equal(result, empty_frame(SPOT_KLINES, columns))


def test_query_parquet_uses_raw_event_timestamp_and_secondary_ordering(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm event data filters by its timestamp and orders equal times by ID."""
    dataset = raw_events()
    path = tmp_path / "events.parquet"
    pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2025-01-01 00:01:00Z",
                    "2025-01-01 00:00:00Z",
                    "2025-01-01 00:00:00Z",
                ],
                utc=True,
            ).as_unit("us"),
            "trade_id": pd.Series([3, 2, 1], dtype="int64"),
            "is_buyer_maker": [True, False, True],
            "price": [103.0, 102.0, 101.0],
        }
    ).to_parquet(path, index=False)
    columns = {
        "event_time": "time",
        "trade_id": "id",
        "is_buyer_maker": "maker",
        "price": "price",
    }

    result = query_parquet(
        connection,
        [path],
        dataset,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
        columns,
        gap_policy=None,
        interval=None,
    )

    assert result["id"].tolist() == [1, 2]
    assert result["maker"].tolist() == [True, False]
    assert str(result["time"].dtype) == "datetime64[us, UTC]"
    assert result["id"].dtype == "int64"
    assert result["maker"].dtype == "bool"


def test_empty_frame_uses_declared_raw_event_types() -> None:
    """Confirm empty raw event results retain their declared column types."""
    columns = {
        "event_time": "time",
        "trade_id": "id",
        "is_buyer_maker": "maker",
        "price": "price",
    }

    result = empty_frame(raw_events(), columns)

    assert str(result["time"].dtype) == "datetime64[us, UTC]"
    assert result["id"].dtype == "int64"
    assert result["maker"].dtype == "bool"
    assert result["price"].dtype == "float64"


@pytest.mark.parametrize(
    ("dataset", "fixture", "id_column"),
    [
        (SPOT_TRADES, "binance_spot_trades_2025-01-01.csv", "trade_id"),
        (SPOT_AGG_TRADES, "binance_spot_agg_trades_2025-01-01.csv", "agg_trade_id"),
    ],
)
def test_query_parquet_filters_canonical_spot_event_data(
    connection: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    dataset: DatasetSpec,
    fixture: str,
    id_column: str,
) -> None:
    """Confirm declared Spot events use exact ranges and stable ID ordering.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The directory used for a temporary Parquet file.
        dataset: The declared Spot event dataset.
        fixture: The representative microsecond source CSV.
        id_column: The canonical event ID used for sorting.
    """
    source = pd.read_csv(
        Path(__file__).parent / "fixtures" / fixture,
        header=None,
        names=dataset.source_columns,
        dtype=str,
    )
    path = tmp_path / f"{dataset.name}.parquet"
    normalize_chunk(source, dataset).to_parquet(path, index=False)
    columns = {"event_time": "time", id_column: "id", "price": "price"}

    result = query_parquet(
        connection,
        [path],
        dataset,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 0, 0, 20_000, tzinfo=UTC),
        columns,
    )

    assert result["id"].tolist() == (
        [200, 201] if dataset is SPOT_TRADES else [400, 401]
    )
    assert result["time"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00.010866Z"),
        pd.Timestamp("2025-01-01T00:00:00.010866Z"),
    ]
