"""Test Upbit Kline normalization, validation, and retrieval."""

from datetime import UTC, date, datetime
import hashlib
import io
from pathlib import Path
from urllib.parse import parse_qs
import zipfile

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from veldra.core.engine import RetrievalEngine
from veldra.core.models import DataValidationError, Resource
from veldra.upbit.connector import UpbitConnector
from veldra.upbit.datasets import MINUTE_KLINES, SECOND_KLINES, get_dataset
from veldra.upbit.processing import normalize_chunk, validate_chunk


def raw_klines(**changes: list[str]) -> pa.Table:
    """Build a small raw Upbit minute-candle table."""
    columns = {
        "date_time_utc": ["2025-01-01T00:00:00", "2025-01-01T00:01:00"],
        "open": ["93576", "93610"],
        "high": ["93620", "93630"],
        "low": ["93500", "93600"],
        "close": ["93610", "93620"],
        "acc_trade_price": ["767000", "870000"],
        "acc_trade_volume": ["8.2", "9.3"],
    }
    columns.update(changes)
    return pa.table(columns)


def archive_bytes(text: str, name: str) -> bytes:
    """Return one ZIP containing the expected CSV member."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return output.getvalue()


def test_minute_klines_normalize_text_timestamps_and_volume_units() -> None:
    """Confirm Upbit's candle fields become canonical Arrow columns."""
    table = normalize_chunk(raw_klines(), MINUTE_KLINES)

    assert table.column_names == list(MINUTE_KLINES.stored_columns)
    assert table["open_time"].type == pa.timestamp("us", "UTC")
    assert table["open_time"].to_pylist() == [
        datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
    ]
    assert table["base_volume"].to_pylist() == [8.2, 9.3]
    assert table["quote_volume"].to_pylist() == [767000.0, 870000.0]


def test_second_klines_validate_second_alignment() -> None:
    """Confirm physical one-second candles preserve microsecond UTC types."""
    raw = raw_klines(date_time_utc=["2025-01-01T00:00:00", "2025-01-01T00:00:01"])
    table = normalize_chunk(raw, SECOND_KLINES)

    assert validate_chunk(table, SECOND_KLINES, date(2025, 1, 1)) == datetime(
        2025, 1, 1, 0, 0, 1, tzinfo=UTC
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"date_time_utc": ["bad", "2025-01-01T00:01:00"]}, "open_time"),
        ({"open": ["bad", "1"]}, "invalid open"),
        ({"high": ["93400", "93630"]}, "high is below"),
        ({"acc_trade_volume": ["-1", "1"]}, "nonnegative"),
    ],
)
def test_klines_reject_malformed_source_values(
    changes: dict[str, list[str]], message: str
) -> None:
    """Confirm malformed timestamps, prices, and volumes fail safely."""
    with pytest.raises(DataValidationError, match=message):
        table = normalize_chunk(raw_klines(**changes), MINUTE_KLINES)
        validate_chunk(table, MINUTE_KLINES, date(2025, 1, 1))


def test_kline_validation_rejects_wrong_day_and_misalignment() -> None:
    """Confirm candles cannot escape their day or physical time grid."""
    table = normalize_chunk(raw_klines(), MINUTE_KLINES)
    with pytest.raises(DataValidationError, match="outside"):
        validate_chunk(table, MINUTE_KLINES, date(2025, 1, 2))

    misaligned = normalize_chunk(
        raw_klines(date_time_utc=["2025-01-01T00:00:30", "2025-01-01T00:01:00"]),
        MINUTE_KLINES,
    )
    with pytest.raises(DataValidationError, match="aligned"):
        validate_chunk(misaligned, MINUTE_KLINES, date(2025, 1, 1))


def test_verified_kline_archive_is_sorted_and_written_to_parquet(
    tmp_path: Path,
) -> None:
    """Confirm SHA-256, sorting, Arrow conversion, and Parquet work together.

    Args:
        tmp_path: The isolated cache directory.
    """
    filename = "USDT-BTC_candle-1m_20250101"
    text = (
        "date_time_utc,open,high,low,close,acc_trade_price,acc_trade_volume\n"
        "2025-01-01T00:01:00,93610,93630,93600,93620,870000,9.3\n"
        "2025-01-01T00:00:00,93576,93620,93500,93610,767000,8.2\n"
    )
    archive = archive_bytes(text, f"{filename}.csv")
    digest = hashlib.sha256(archive).hexdigest()
    resource = Resource(
        date(2025, 1, 1),
        f"https://archive.example/{filename}.zip",
        f"https://archive.example/{filename}.zip.checksum",
        timestamp_column="open_time",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a digest-only sidecar or its matching ZIP."""
        if request.url.path.endswith(".checksum"):
            return httpx.Response(200, text=digest)
        return httpx.Response(200, content=archive)

    destination = tmp_path / "day.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        metadata = UpbitConnector(retries=0).ingest(
            client, resource, MINUTE_KLINES, destination
        )

    table = pq.read_table(destination)
    assert metadata.archive_checksum == digest
    assert metadata.row_count == 2
    assert table["open_time"].to_pylist()[0] == datetime(2025, 1, 1, tzinfo=UTC)


def test_sparse_kline_retrieval_returns_exact_rows_without_false_gaps(
    tmp_path: Path,
) -> None:
    """Confirm a no-trade minute remains absent and the result stays complete.

    Args:
        tmp_path: The isolated cache directory.
    """
    filename = "USDT-BTC_candle-1m_20250101"
    text = (
        "date_time_utc,open,high,low,close,acc_trade_price,acc_trade_volume\n"
        "2025-01-01T00:00:00,93576,93620,93500,93610,767000,8.2\n"
        "2025-01-01T00:02:00,93610,93630,93600,93620,870000,9.3\n"
    )
    archive = archive_bytes(text, f"{filename}.csv")
    digest = hashlib.sha256(archive).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve current markets, portal listings, sidecar, and archive."""
        if request.url.host == "api.upbit.com":
            return httpx.Response(200, json=[{"market": "USDT-BTC"}])
        if request.url.host == "crix-data.upbit.com":
            if request.url.path.endswith(".checksum"):
                return httpx.Response(200, text=digest)
            return httpx.Response(200, content=archive)
        prefix = parse_qs(request.url.query.decode())["prefix"][0]
        if prefix in {"candle", "trade"}:
            return httpx.Response(
                200,
                json=[{"key": f"{prefix}/USDT-BTC", "size": 0, "type": "DIRECTORY"}],
            )
        if prefix == "candle/USDT-BTC/daily/1m":
            return httpx.Response(
                200,
                json=[{"key": f"{prefix}/2025", "size": 0, "type": "DIRECTORY"}],
            )
        key = f"{prefix}/{filename}.zip"
        return httpx.Response(
            200,
            json=[{"key": key, "size": len(archive), "type": "FILE"}],
        )

    engine = RetrievalEngine(
        tmp_path,
        source=UpbitConnector(retries=0),
        dataset_resolver=get_dataset,
        transport=httpx.MockTransport(handler),
    )
    result = engine.get_results(
        "BTCUSDT",
        "2025-01-01T00:00:00Z",
        "2025-01-01T00:03:00Z",
        interval="1m",
        gap_policy="keep",
        progress=False,
    )

    assert not isinstance(result, list)
    assert result.pair == "USDT-BTC"
    assert result.complete
    assert result.gaps == []
    assert result.data["open_time"].dt.minute.tolist() == [0, 2]
