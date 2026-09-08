"""Test DuckDB resampling of cached one-minute Spot klines."""

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from crypto_downloader.binance.datasets import SPOT_KLINES
from crypto_downloader.core.query import query_parquet


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Provide an isolated DuckDB connection.

    Yields:
        A temporary in-memory DuckDB connection.
    """
    value = duckdb.connect()
    yield value
    value.close()


def frame(times: Sequence[str], prices: Sequence[float]) -> pd.DataFrame:
    """Create canonical kline rows at specified UTC timestamps.

    Args:
        times: ISO timestamps for each candle opening.
        prices: Opening prices corresponding to the timestamps.

    Returns:
        A canonical Spot kline frame.
    """
    opens = pd.to_datetime(times, utc=True).as_unit("us")
    price = pd.Series(prices, dtype="float64")
    count = len(opens)
    return pd.DataFrame(
        {
            "open_time": opens,
            "open": price,
            "high": price + 2,
            "low": price - 1,
            "close": price + 1,
            "volume": [10.0] * count,
            "close_time": opens + pd.Timedelta(seconds=59, microseconds=999999),
            "quote_volume": [1000.0] * count,
            "trade_count": pd.Series([10] * count, dtype="int64"),
            "taker_buy_base_volume": [4.0] * count,
            "taker_buy_quote_volume": [400.0] * count,
        }
    )


def minutes(count: int = 10) -> pd.DataFrame:
    """Create consecutive one-minute candles from midnight UTC.

    Args:
        count: The number of candles to create.

    Returns:
        Consecutive canonical kline rows.
    """
    times = pd.date_range("2024-01-01", periods=count, freq="1min", tz=UTC)
    return frame([value.isoformat() for value in times], range(100, 100 + count))


def parquet(tmp_path: Path, value: pd.DataFrame) -> Path:
    """Write one resampling fixture to Parquet.

    Args:
        tmp_path: The isolated fixture directory.
        value: The canonical rows to write.

    Returns:
        The resulting Parquet path.
    """
    path = tmp_path / "rows.parquet"
    value.to_parquet(path, index=False)
    return path


def query(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    start: datetime,
    end: datetime,
    interval: str,
    *,
    gap_policy: str = "keep",
    columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Query one fixture at a requested output interval.

    Args:
        connection: The isolated DuckDB connection.
        path: The cached Parquet fixture.
        start: The inclusive first timestamp.
        end: The exclusive final timestamp.
        interval: The requested output interval.
        gap_policy: The missing-candle policy applied before aggregation.
        columns: Optional canonical columns and output labels.

    Returns:
        The queried and possibly resampled frame.
    """
    return query_parquet(
        connection,
        [path],
        SPOT_KLINES,
        start,
        end,
        columns or SPOT_KLINES.resolve_columns(None),
        interval=interval,
        gap_policy=gap_policy,
    )


def test_five_minute_resampling_uses_correct_ohlcv_rules(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm fixed buckets use first, extrema, last, and sum rules.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated fixture directory.
    """
    path = parquet(tmp_path, minutes())

    result = query(
        connection,
        path,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
        "5m",
    )

    assert result["open_time"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00Z"),
        pd.Timestamp("2024-01-01 00:05:00Z"),
    ]
    assert result["open"].tolist() == [100.0, 105.0]
    assert result["high"].tolist() == [106.0, 111.0]
    assert result["low"].tolist() == [99.0, 104.0]
    assert result["close"].tolist() == [105.0, 110.0]
    assert result["volume"].tolist() == [50.0, 50.0]
    assert result["quote_volume"].tolist() == [5000.0, 5000.0]
    assert result["trade_count"].tolist() == [50, 50]
    assert result["taker_buy_base_volume"].tolist() == [20.0, 20.0]
    assert result["taker_buy_quote_volume"].tolist() == [2000.0, 2000.0]
    assert result["close_time"].tolist() == [
        pd.Timestamp("2024-01-01 00:04:59.999999Z"),
        pd.Timestamp("2024-01-01 00:09:59.999999Z"),
    ]
    assert result["trade_count"].dtype == "int64"
    assert not result["is_synthetic"].any()


def test_partial_edge_buckets_use_only_rows_inside_the_request(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm edge buckets aggregate the exact requested timestamp slice.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated fixture directory.
    """
    path = parquet(tmp_path, minutes())

    result = query(
        connection,
        path,
        datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 8, tzinfo=UTC),
        "5m",
    )

    assert result["open_time"].dt.minute.tolist() == [0, 5]
    assert result["open"].tolist() == [102.0, 105.0]
    assert result["close"].tolist() == [105.0, 108.0]
    assert result["volume"].tolist() == [30.0, 30.0]


def test_week_buckets_start_on_monday(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm Sunday and Monday belong to different Monday-based weeks.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated fixture directory.
    """
    value = frame(
        ["2024-01-07T12:00:00Z", "2024-01-08T12:00:00Z"],
        [100.0, 200.0],
    )
    path = parquet(tmp_path, value)

    result = query(
        connection,
        path,
        datetime(2024, 1, 7, tzinfo=UTC),
        datetime(2024, 1, 9, tzinfo=UTC),
        "1w",
    )

    assert result["open_time"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00Z"),
        pd.Timestamp("2024-01-08 00:00:00Z"),
    ]
    assert result["open"].tolist() == [100.0, 200.0]


def test_month_buckets_follow_calendar_boundaries(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm month aggregation does not use a fixed number of days.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated fixture directory.
    """
    value = frame(
        ["2024-01-31T23:59:00Z", "2024-02-01T00:00:00Z"],
        [100.0, 200.0],
    )
    path = parquet(tmp_path, value)

    result = query(
        connection,
        path,
        datetime(2024, 1, 31, tzinfo=UTC),
        datetime(2024, 2, 2, tzinfo=UTC),
        "1mo",
    )

    assert result["open_time"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00Z"),
        pd.Timestamp("2024-02-01 00:00:00Z"),
    ]


@pytest.mark.parametrize(
    "interval",
    ["3m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d"],
)
def test_every_fixed_output_interval_can_be_queried(
    connection: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    interval: str,
) -> None:
    """Confirm every declared fixed interval has executable DuckDB SQL.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated fixture directory.
        interval: The supported output interval under test.
    """
    path = parquet(tmp_path, minutes())

    result = query(
        connection,
        path,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
        interval,
    )

    assert not result.empty
    assert result["open_time"].is_monotonic_increasing


def test_synthetic_marker_and_values_propagate_through_resampling(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm a bucket records when any constituent candle was synthesized.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated fixture directory.
    """
    path = parquet(tmp_path, minutes(5).drop(index=[2]))

    result = query(
        connection,
        path,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        "5m",
        gap_policy="forward",
    )

    assert len(result) == 1
    assert result.iloc[0]["is_synthetic"]
    assert result.iloc[0]["volume"] == 40.0
    assert result.iloc[0]["close"] == 105.0


def test_nan_gap_makes_the_containing_output_bucket_unknown(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm nan policy does not hide missing source data through aggregation.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated fixture directory.
    """
    path = parquet(tmp_path, minutes(5).drop(index=[2]))

    result = query(
        connection,
        path,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        "5m",
        gap_policy="nan",
    )

    assert result.iloc[0]["is_synthetic"]
    assert result.iloc[0][["open", "high", "low", "close", "volume"]].isna().all()


def test_resampling_applies_column_selection_after_aggregation(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm requested labels refer to resampled rather than source values.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated fixture directory.
    """
    path = parquet(tmp_path, minutes(5))

    result = query(
        connection,
        path,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        "5m",
        columns={"open_time": "time", "volume": "total"},
    )

    assert list(result.columns) == ["time", "total"]
    assert result.iloc[0]["total"] == 50.0
