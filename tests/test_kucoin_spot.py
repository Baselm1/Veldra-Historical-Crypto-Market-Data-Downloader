"""Test KuCoin Spot Kline normalization and verified ingestion."""

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
from veldra.kucoin.connector import KuCoinConnector
from veldra.kucoin.datasets import SPOT_KLINES
from veldra.kucoin.processing import normalize_chunk, validate_chunk


def raw_klines(**changes: list[str]) -> pa.Table:
    """Build a small raw KuCoin Spot Kline table."""
    columns = {
        "time": ["1735689600", "1735689660"],
        "open": ["93576", "93610"],
        "close": ["93610", "93620"],
        "high": ["93620", "93630"],
        "low": ["93500", "93600"],
        "volume": ["8.2", "9.3"],
        "turnover": ["767000", "870000"],
    }
    columns.update(changes)
    return pa.table(columns)


def archive_bytes(text: str, name: str) -> bytes:
    """Create one ZIP containing the exact expected CSV member."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return output.getvalue()


def test_spot_klines_normalize_seconds_and_source_column_order() -> None:
    """Confirm KuCoin's unusual open-close-high-low schema becomes canonical."""
    table = normalize_chunk(raw_klines(), SPOT_KLINES)

    assert table.column_names == list(SPOT_KLINES.stored_columns)
    assert table["open_time"].type == pa.timestamp("us", "UTC")
    assert table["open_time"].to_pylist() == [
        datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
    ]
    assert table["high"].to_pylist() == [93620.0, 93630.0]
    assert table["quote_volume"].to_pylist() == [767000.0, 870000.0]


def test_spot_kline_validation_accepts_valid_sorted_rows() -> None:
    """Confirm canonical Spot Klines pass ordering and OHLC checks."""
    table = normalize_chunk(raw_klines(), SPOT_KLINES)

    assert validate_chunk(table, SPOT_KLINES, date(2025, 1, 1)) == datetime(
        2025, 1, 1, 0, 1, tzinfo=UTC
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"time": ["1735689600000", "1735689660000"]}, "timestamp unit"),
        ({"open": ["bad", "1"]}, "invalid open"),
        ({"high": ["93400", "93630"]}, "high is below"),
        ({"volume": ["-1", "1"]}, "nonnegative"),
    ],
)
def test_spot_klines_reject_invalid_source_values(
    changes: dict[str, list[str]], message: str
) -> None:
    """Confirm malformed timestamps, prices, and quantities fail safely."""
    with pytest.raises(DataValidationError, match=message):
        table = normalize_chunk(raw_klines(**changes), SPOT_KLINES)
        validate_chunk(table, SPOT_KLINES, date(2025, 1, 1))


def test_spot_kline_validation_rejects_wrong_day_and_duplicate_times() -> None:
    """Confirm rows cannot escape their resource or duplicate a candle opening."""
    table = normalize_chunk(raw_klines(), SPOT_KLINES)
    with pytest.raises(DataValidationError, match="outside"):
        validate_chunk(table, SPOT_KLINES, date(2025, 1, 2))
    duplicate = normalize_chunk(
        raw_klines(time=["1735689600", "1735689600"]), SPOT_KLINES
    )
    with pytest.raises(DataValidationError, match="increasing"):
        validate_chunk(duplicate, SPOT_KLINES, date(2025, 1, 1))


def test_verified_spot_kline_archive_is_written_as_atomic_parquet(
    tmp_path: Path,
) -> None:
    """Confirm MD5, CSV parsing, sorting, and Parquet metadata work together."""
    name = "BTCUSDT-1m-2025-01-01.csv"
    text = (
        "time,open,close,high,low,volume,turnover\n"
        "1735689660,93610,93620,93630,93600,9.3,870000\n"
        "1735689600,93576,93610,93620,93500,8.2,767000\n"
    )
    archive = archive_bytes(text, name)
    digest = hashlib.md5(archive).hexdigest()
    resource = Resource(
        date(2025, 1, 1),
        "https://archive.example/BTCUSDT-1m-2025-01-01.zip",
        "https://archive.example/BTCUSDT-1m-2025-01-01.zip.CHECKSUM",
        checksum_algorithm="md5",
        timestamp_column="open_time",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the matching MD5 sidecar or source archive."""
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=f"{digest}  BTCUSDT-1m-2025-01-01.zip")
        return httpx.Response(200, content=archive)

    destination = tmp_path / "day.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        metadata = KuCoinConnector(retries=0).ingest(
            client, resource, SPOT_KLINES, destination
        )

    table = pq.read_table(destination)
    assert metadata.archive_checksum == digest
    assert metadata.row_count == 2
    assert table["open_time"].to_pylist()[0] == datetime(2025, 1, 1, tzinfo=UTC)
    assert not destination.with_name("day.parquet.part").exists()
