"""Test verified Gzip CSV ingestion."""

from datetime import date
import gzip
import hashlib
from pathlib import Path

import httpx
import pandas as pd
import pytest

from veldra.binance.datasets import SPOT_KLINES
from veldra.binance.processing import normalize_chunk, validate_chunk
from veldra.core.ingest import ArchiveError, ingest_gzip_archive
from veldra.core.models import IntegritySpec, Resource

CSV = (
    b"1735689600000000,4.0,5.0,3.0,4.5,2.0,1735689659999999,9.0,3,1.0,4.5,0\n"
    b"1735689660000000,4.5,5.5,4.0,5.0,3.0,1735689719999999,15.0,4,2.0,10.0,0\n"
)


def _resource() -> Resource:
    """Return one response-header verified monthly resource."""
    return Resource(
        date(2025, 1, 1),
        "https://data.example/BTC_USDT-202501.csv.gz",
        None,
        end_day=date(2025, 1, 31),
        cadence="monthly",
        integrity=IntegritySpec("response_header", algorithm="md5"),
    )


def _client(payload: bytes) -> httpx.Client:
    """Return a client serving bytes with their plain MD5 ETag.

    Args:
        payload: The compressed response bytes.

    Returns:
        A mocked HTTPX client.
    """
    digest = hashlib.md5(payload).hexdigest()
    return httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=payload,
                headers={"ETag": f'"{digest}"'},
            )
        )
    )


def test_ingest_gzip_archive_streams_canonical_parquet(tmp_path: Path) -> None:
    """Confirm a verified Gzip CSV becomes one canonical Parquet file."""
    payload = gzip.compress(CSV)
    destination = tmp_path / "nested" / "result.parquet"

    with _client(payload) as client:
        metadata = ingest_gzip_archive(
            client,
            _resource(),
            SPOT_KLINES,
            destination,
            normalizer=normalize_chunk,
            validator=validate_chunk,
            chunk_rows=1,
        )

    frame = pd.read_parquet(destination)
    assert tuple(frame.columns) == SPOT_KLINES.stored_columns
    assert len(frame) == metadata.row_count == 2
    assert metadata.archive_checksum == hashlib.md5(payload).hexdigest()
    assert metadata.first_timestamp == frame.iloc[0].open_time.to_pydatetime()
    assert metadata.last_timestamp == frame.iloc[-1].open_time.to_pydatetime()
    assert not destination.with_name("result.parquet.part").exists()


def test_ingest_gzip_archive_rejects_corrupt_stream(tmp_path: Path) -> None:
    """Confirm invalid compressed bytes cannot publish a Parquet file."""
    destination = tmp_path / "result.parquet"

    with _client(b"not gzip") as client:
        with pytest.raises(ArchiveError, match="Gzip"):
            ingest_gzip_archive(
                client,
                _resource(),
                SPOT_KLINES,
                destination,
                normalizer=normalize_chunk,
                validator=validate_chunk,
                retries=0,
            )

    assert not destination.exists()
    assert not destination.with_name("result.parquet.part").exists()


def test_ingest_gzip_archive_limits_expanded_csv(tmp_path: Path) -> None:
    """Confirm the expanded-byte limit stops oversized Gzip payloads."""
    payload = gzip.compress(CSV)
    destination = tmp_path / "result.parquet"

    with _client(payload) as client:
        with pytest.raises(ArchiveError, match="uncompressed CSV size"):
            ingest_gzip_archive(
                client,
                _resource(),
                SPOT_KLINES,
                destination,
                normalizer=normalize_chunk,
                validator=validate_chunk,
                retries=0,
                max_csv_bytes=10,
            )

    assert not destination.exists()
