"""Test Upbit historical trade normalization and verified ingestion."""

from datetime import UTC, date, datetime
import hashlib
import io
from pathlib import Path
import zipfile

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from veldra.core.models import DataValidationError, Resource
from veldra.upbit.connector import UpbitConnector
from veldra.upbit.datasets import SPOT_TRADES
from veldra.upbit.processing import normalize_chunk, validate_chunk


def raw_trades(**changes: list[str]) -> pa.Table:
    """Build a small raw Upbit trade table with millisecond timestamps."""
    columns = {
        "seq": ["0", "1", "2"],
        "timestamp": ["1735689600257", "1735689600689", "1735689600824"],
        "volume": ["0.0018", "0.0004", "0.0001"],
        "price": ["93548.8", "93548.8", "93548.7"],
        "ask_bid": ["BID", "ask", " BID "],
    }
    columns.update(changes)
    return pa.table(columns)


def trade_archive(text: str, name: str) -> bytes:
    """Return one ZIP containing an Upbit trade CSV."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return output.getvalue()


def test_trades_normalize_time_sequence_quantities_and_side() -> None:
    """Confirm Upbit event fields become canonical typed Arrow columns."""
    table = normalize_chunk(raw_trades(), SPOT_TRADES)

    assert table.column_names == list(SPOT_TRADES.stored_columns)
    assert table["event_number"].type == pa.int64()
    assert table["event_number"].to_pylist() == [0, 1, 2]
    assert table["event_time"].to_pylist()[0] == datetime(
        2025, 1, 1, 0, 0, 0, 257000, tzinfo=UTC
    )
    assert table["side"].to_pylist() == ["buy", "sell", "buy"]
    assert table["quote_quantity"].to_pylist() == pytest.approx(
        [168.38784, 37.41952, 9.35487]
    )


def test_whole_second_trade_values_remain_millisecond_epochs() -> None:
    """Confirm old rows with zero millisecond tails keep the same time unit."""
    table = normalize_chunk(
        raw_trades(timestamp=["1651363200000", "1651363200000", "1651363201000"]),
        SPOT_TRADES,
    )

    assert table["event_time"].to_pylist() == [
        datetime(2022, 5, 1, 0, 0, tzinfo=UTC),
        datetime(2022, 5, 1, 0, 0, tzinfo=UTC),
        datetime(2022, 5, 1, 0, 0, 1, tzinfo=UTC),
    ]
    assert validate_chunk(table, SPOT_TRADES, date(2022, 5, 1)) == datetime(
        2022, 5, 1, 0, 0, 1, tzinfo=UTC
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"seq": ["0.5", "1", "2"]}, "event_number"),
        ({"timestamp": ["1651363200", "1651363200", "1651363201"]}, "unit"),
        ({"volume": ["-1", "1", "1"]}, "nonnegative"),
        ({"price": ["0", "1", "1"]}, "positive"),
        ({"ask_bid": ["hold", "BID", "ASK"]}, "BID or ASK"),
    ],
)
def test_trades_reject_malformed_source_values(
    changes: dict[str, list[str]], message: str
) -> None:
    """Confirm invalid event identifiers, values, and directions fail safely."""
    with pytest.raises(DataValidationError, match=message):
        table = normalize_chunk(raw_trades(**changes), SPOT_TRADES)
        validate_chunk(table, SPOT_TRADES, date(2025, 1, 1))


def test_trade_validation_allows_equal_times_but_rejects_reversals() -> None:
    """Confirm timestamp ties are valid and decreasing event order is not."""
    tied = normalize_chunk(
        raw_trades(timestamp=["1735689600257"] * 3),
        SPOT_TRADES,
    )
    assert validate_chunk(tied, SPOT_TRADES, date(2025, 1, 1)) == datetime(
        2025, 1, 1, 0, 0, 0, 257000, tzinfo=UTC
    )

    reversed_rows = normalize_chunk(
        raw_trades(timestamp=["1735689601000", "1735689600000", "1735689602000"]),
        SPOT_TRADES,
    )
    with pytest.raises(DataValidationError, match="increasing"):
        validate_chunk(reversed_rows, SPOT_TRADES, date(2025, 1, 1))


def test_verified_trade_archive_sorts_timestamp_inversions(
    tmp_path: Path,
) -> None:
    """Confirm source inversions are sorted by time and archive sequence.

    Args:
        tmp_path: The isolated cache directory.
    """
    filename = "KRW-BTC_trade_20250101"
    text = (
        "seq,timestamp,volume,price,ask_bid\n"
        "0,1735689601000,0.001,93548.8,BID\n"
        "1,1735689600000,0.002,93548.7,ASK\n"
        "2,1735689601000,0.003,93548.9,BID\n"
    )
    archive = trade_archive(text, f"{filename}.csv")
    digest = hashlib.sha256(archive).hexdigest()
    resource = Resource(
        date(2025, 1, 1),
        f"https://archive.example/{filename}.zip",
        f"https://archive.example/{filename}.zip.checksum",
        timestamp_column="event_time",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one digest-only sidecar and trade archive."""
        if request.url.path.endswith(".checksum"):
            return httpx.Response(200, text=digest)
        return httpx.Response(200, content=archive)

    destination = tmp_path / "trades.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        metadata = UpbitConnector(retries=0).ingest(
            client, resource, SPOT_TRADES, destination
        )

    table = pq.read_table(destination)
    assert metadata.row_count == 3
    assert table["event_number"].to_pylist() == [1, 0, 2]
    assert table["event_time"].to_pylist() == sorted(table["event_time"].to_pylist())


def test_trade_sequence_may_restart_in_each_daily_archive() -> None:
    """Confirm event numbers are validated locally rather than globally."""
    table = normalize_chunk(raw_trades(), SPOT_TRADES)
    previous = datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC)

    assert validate_chunk(
        table,
        SPOT_TRADES,
        date(2025, 1, 1),
        previous_timestamp=previous,
    ) == datetime(2025, 1, 1, 0, 0, 0, 824000, tzinfo=UTC)
