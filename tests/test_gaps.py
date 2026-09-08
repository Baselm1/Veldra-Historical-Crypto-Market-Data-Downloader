"""Test detection and handling of missing Spot kline candles."""

from crypto_downloader.binance.datasets import get_dataset


from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import httpx
import pandas as pd
import pytest

from crypto_downloader.core.datasets import DatasetSpec
from crypto_downloader.binance.datasets import SPOT_KLINES
from crypto_downloader.core.engine import RetrievalEngine
from crypto_downloader.core.models import (
    IngestedResource,
    Market,
    MissingCandlesError,
    Resource,
    ResourceKey,
    Result,
)
from crypto_downloader.core.query import (
    missing_ranges,
    query_parquet,
    suspect_gap_paths,
)
from crypto_downloader.core.request import Request, parse_gap_policy

DAY = date(2024, 1, 1)
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 1, 0, 5, tzinfo=UTC)


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Provide an isolated DuckDB connection.

    Yields:
        A temporary in-memory DuckDB connection.
    """
    value = duckdb.connect()
    yield value
    value.close()


def candles(periods: int = 5) -> pd.DataFrame:
    """Create canonical one-minute candles with distinguishable prices.

    Args:
        periods: The number of consecutive rows to create.

    Returns:
        A canonical Spot kline frame beginning at midnight UTC.
    """
    opens = pd.date_range(START, periods=periods, freq="1min").as_unit("us")
    prices = pd.Series([100.0 + value for value in range(periods)])
    return pd.DataFrame(
        {
            "open_time": opens,
            "open": prices,
            "high": prices + 2,
            "low": prices - 1,
            "close": prices + 1,
            "volume": [10.0] * periods,
            "close_time": opens + pd.Timedelta(seconds=59, microseconds=999999),
            "quote_volume": [1000.0] * periods,
            "trade_count": pd.Series([10] * periods, dtype="int64"),
            "taker_buy_base_volume": [4.0] * periods,
            "taker_buy_quote_volume": [400.0] * periods,
        }
    )


def write_frame(tmp_path: Path, frame: pd.DataFrame, name: str = "day.parquet") -> Path:
    """Write one canonical Parquet test partition.

    Args:
        tmp_path: The isolated cache directory.
        frame: The canonical rows to store.
        name: The desired file name.

    Returns:
        The written Parquet path.
    """
    path = tmp_path / name
    frame.to_parquet(path, index=False)
    return path


def test_gap_policy_parsing_accepts_only_supported_names() -> None:
    """Confirm request parsing has one explicit five-policy vocabulary."""
    for policy in ("forward", "backward", "nan", "keep", "raise"):
        assert parse_gap_policy(policy) == policy
    assert Request.parse("BTCUSDT", DAY, DAY).gap_policy == "forward"
    with pytest.raises(TypeError, match="gap_policy"):
        parse_gap_policy(None)
    with pytest.raises(ValueError, match="gap_policy"):
        parse_gap_policy("interpolate")
    with pytest.raises(ValueError, match="gap_policy"):
        parse_gap_policy("FORWARD")


def test_missing_ranges_reports_exact_consecutive_internal_candles(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm consecutive missing timestamps become one exclusive-end gap.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated cache directory.
    """
    path = write_frame(tmp_path, candles().drop(index=[2, 3]))

    gaps = missing_ranges(connection, [path], SPOT_KLINES, START, END)

    assert len(gaps) == 1
    assert gaps[0].start == datetime(2024, 1, 1, 0, 2, tzinfo=UTC)
    assert gaps[0].end == datetime(2024, 1, 1, 0, 4, tzinfo=UTC)
    assert gaps[0].count == 2


def test_missing_ranges_separates_nonconsecutive_gaps(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm separated missing candles remain distinct gap records.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated cache directory.
    """
    path = write_frame(tmp_path, candles().drop(index=[1, 3]))

    gaps = missing_ranges(connection, [path], SPOT_KLINES, START, END)

    assert [gap.count for gap in gaps] == [1, 1]
    assert [gap.start.minute for gap in gaps] == [1, 3]


def test_known_march_2023_outage_is_reported_as_eighty_candles(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm the known Binance outage is represented by its exact UTC range.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated cache directory.
    """
    frame = candles(83)
    opens = pd.date_range("2023-03-24 12:38:00Z", periods=83, freq="1min").as_unit("us")
    frame["open_time"] = opens
    frame["close_time"] = opens + pd.Timedelta(seconds=59, microseconds=999999)
    path = write_frame(tmp_path, frame.drop(index=range(2, 82)))

    gaps = missing_ranges(
        connection,
        [path],
        SPOT_KLINES,
        datetime(2023, 3, 24, 12, 38, tzinfo=UTC),
        datetime(2023, 3, 24, 14, 1, tzinfo=UTC),
    )

    assert len(gaps) == 1
    assert gaps[0].start == datetime(2023, 3, 24, 12, 40, tzinfo=UTC)
    assert gaps[0].end == datetime(2023, 3, 24, 14, 0, tzinfo=UTC)
    assert gaps[0].count == 80


def test_forward_fill_uses_the_previous_close_and_zero_activity(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm forward synthetic candles are constant and carry no fake volume.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated cache directory.
    """
    path = write_frame(tmp_path, candles().drop(index=[2, 3]))

    frame = query_parquet(
        connection,
        [path],
        SPOT_KLINES,
        START,
        END,
        SPOT_KLINES.resolve_columns(None),
        gap_policy="forward",
    )

    synthetic = frame[frame["is_synthetic"]]
    assert len(frame) == 5
    assert synthetic["open_time"].dt.minute.tolist() == [2, 3]
    assert (synthetic[["open", "high", "low", "close"]] == 102.0).all().all()
    assert (
        (
            synthetic[
                [
                    "volume",
                    "quote_volume",
                    "trade_count",
                    "taker_buy_base_volume",
                    "taker_buy_quote_volume",
                ]
            ]
            == 0
        )
        .all()
        .all()
    )
    assert synthetic["close_time"].tolist() == [
        pd.Timestamp("2024-01-01 00:02:59.999999Z"),
        pd.Timestamp("2024-01-01 00:03:59.999999Z"),
    ]
    assert synthetic["trade_count"].dtype == "int64"


def test_backward_fill_uses_the_next_open(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm backward synthetic candles use the following real opening price.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated cache directory.
    """
    path = write_frame(tmp_path, candles().drop(index=[2, 3]))

    frame = query_parquet(
        connection,
        [path],
        SPOT_KLINES,
        START,
        END,
        SPOT_KLINES.resolve_columns(None),
        gap_policy="backward",
    )

    synthetic = frame[frame["is_synthetic"]]
    assert (synthetic[["open", "high", "low", "close"]] == 104.0).all().all()


def test_nan_fill_preserves_missing_values_and_generated_timestamps(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm nan policy creates marked rows without invented market values.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated cache directory.
    """
    path = write_frame(tmp_path, candles().drop(index=[2, 3]))

    frame = query_parquet(
        connection,
        [path],
        SPOT_KLINES,
        START,
        END,
        SPOT_KLINES.resolve_columns(None),
        gap_policy="nan",
    )

    synthetic = frame[frame["is_synthetic"]]
    assert synthetic["open_time"].dt.minute.tolist() == [2, 3]
    assert synthetic.drop(columns=["open_time", "is_synthetic"]).isna().all().all()


def test_keep_policy_returns_only_real_candles(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm keep policy reports gaps without adding rows to the frame.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated cache directory.
    """
    path = write_frame(tmp_path, candles().drop(index=[2, 3]))

    frame = query_parquet(
        connection,
        [path],
        SPOT_KLINES,
        START,
        END,
        SPOT_KLINES.resolve_columns(None),
        gap_policy="keep",
    )

    assert frame["open_time"].dt.minute.tolist() == [0, 1, 4]
    assert not frame["is_synthetic"].any()


def test_edge_absences_and_completely_missing_days_are_never_filled(
    connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """Confirm synthesis stays between real rows within each local daily file.

    Args:
        connection: The isolated DuckDB connection.
        tmp_path: The isolated cache directory.
    """
    first = write_frame(tmp_path, candles().iloc[1:4], "2024-01-01.parquet")
    third_frame = candles(2).copy()
    third_frame["open_time"] += pd.Timedelta(days=2)
    third_frame["close_time"] += pd.Timedelta(days=2)
    third = write_frame(tmp_path, third_frame, "2024-01-03.parquet")
    end = datetime(2024, 1, 4, tzinfo=UTC)

    gaps = missing_ranges(connection, [first, third], SPOT_KLINES, START, end)
    frame = query_parquet(
        connection,
        [first, third],
        SPOT_KLINES,
        START,
        end,
        SPOT_KLINES.resolve_columns(None),
        gap_policy="forward",
    )

    assert gaps == []
    assert len(frame) == 5
    assert not frame["is_synthetic"].any()


def test_gap_helpers_handle_empty_paths_and_reject_other_base_intervals(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Confirm gap detection has explicit empty and unsupported-base behavior.

    Args:
        connection: The isolated DuckDB connection.
    """
    assert missing_ranges(connection, [], SPOT_KLINES, START, END) == []
    with pytest.raises(ValueError, match="1m base interval"):
        missing_ranges(
            connection,
            [],
            replace(SPOT_KLINES, base_interval="1s"),
            START,
            END,
        )


def test_catalog_metadata_selects_only_possible_gap_partitions(tmp_path: Path) -> None:
    """Confirm continuous partitions avoid row-level gap inspection.

    Args:
        tmp_path: The isolated cache directory.
    """
    continuous_path = tmp_path / "continuous.parquet"
    gapped_path = tmp_path / "gapped.parquet"
    unknown_path = tmp_path / "unknown.parquet"
    continuous = Resource(
        DAY,
        "archive",
        "checksum",
        status="ready",
        parquet_path=continuous_path,
        row_count=5,
        first_timestamp=START,
        last_timestamp=END - pd.Timedelta(minutes=1),
        timestamp_column="open_time",
        schema_version=SPOT_KLINES.schema_version,
    )
    gapped = replace(continuous, parquet_path=gapped_path, row_count=3)

    assert suspect_gap_paths(
        [continuous, gapped],
        [continuous_path, gapped_path, unknown_path],
        SPOT_KLINES,
    ) == [gapped_path, unknown_path]


def test_direct_query_rejects_an_unknown_gap_policy(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Confirm direct query callers receive the same policy validation.

    Args:
        connection: The isolated DuckDB connection.
    """
    with pytest.raises(ValueError, match="gap_policy"):
        query_parquet(
            connection,
            [],
            SPOT_KLINES,
            START,
            END,
            SPOT_KLINES.resolve_columns(None),
            gap_policy="linear",
        )


class GapSource:
    """Serve one available daily archive containing an internal candle gap."""

    code = "gap_source"
    products: tuple[str, ...] = ("spot",)
    active_statuses = frozenset({"TRADING"})
    max_concurrency = 1

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return the single test market.

        Args:
            client: The unused HTTP client.
            product: The requested source product.

        Returns:
            The available BTCUSDT test market.
        """
        return [
            Market(
                "BTCUSDT",
                "BTCUSDT",
                "BTC",
                "USDT",
                "TRADING",
                active=True,
            )
        ]

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return the gap archive when its day overlaps discovery.

        Args:
            client: The unused HTTP client.
            key: The requested resource identity.
            start_day: The first discovery day.
            end_day: The final discovery day.

        Returns:
            The daily test resource when its date overlaps the range.
        """
        if start_day <= DAY <= end_day:
            return [Resource(DAY, "archive", "checksum")]
        return []

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Return the gap archive when it follows the history boundary.

        Args:
            client: The unused HTTP client.
            key: The unused resource identity.
            start_day: The earliest acceptable archive day, or ``None`` for all days.
            end_day: The latest acceptable archive day.

        Returns:
            The daily test resource when it is inside the range.
        """
        if (start_day is None or start_day <= DAY) and DAY <= end_day:
            return Resource(DAY, "archive", "checksum")
        return None

    def ingest(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Write the test gap frame and return its integrity metadata.

        Args:
            client: The unused HTTP client.
            resource: The daily resource being ingested.
            dataset: The schema for the cached rows.
            destination: The final Parquet path.

        Returns:
            Integrity metadata describing the generated partition.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame = candles().drop(index=[2, 3])
        frame.to_parquet(destination, index=False)
        stat = destination.stat()
        return IngestedResource(
            "a" * 64,
            stat.st_size,
            stat.st_mtime_ns,
            len(frame),
            frame["open_time"].iloc[0].to_pydatetime(),
            frame["open_time"].iloc[-1].to_pydatetime(),
        )


def test_pipeline_reports_filled_gaps_and_remains_incomplete(tmp_path: Path) -> None:
    """Confirm filled data retains an explicit source-quality problem.

    Args:
        tmp_path: The isolated downloader directory.
    """
    result = RetrievalEngine(
        tmp_path, source=GapSource(), dataset_resolver=get_dataset
    ).get_results(
        "BTCUSDT",
        START,
        END,
        desired_columns=["open_time", "close", "is_synthetic"],
    )

    assert isinstance(result, Result)
    assert len(result.data) == 5
    assert result.data["is_synthetic"].sum() == 2
    assert result.data.loc[result.data["is_synthetic"], "close"].tolist() == [
        102.0,
        102.0,
    ]
    assert result.gaps[0].count == 2
    assert [problem.code for problem in result.problems] == ["missing_candles"]
    assert result.gap_policy == "forward"
    assert result.source == "gap_source"
    assert not result.complete


def test_raise_policy_raises_the_structured_missing_candles_error(
    tmp_path: Path,
) -> None:
    """Confirm strict callers can stop on source candle gaps.

    Args:
        tmp_path: The isolated downloader directory.
    """
    with pytest.raises(MissingCandlesError) as caught:
        RetrievalEngine(
            tmp_path, source=GapSource(), dataset_resolver=get_dataset
        ).get_results("BTCUSDT", START, END, gap_policy="raise")

    assert caught.value.pair == "BTCUSDT"
    assert caught.value.gaps[0].count == 2
