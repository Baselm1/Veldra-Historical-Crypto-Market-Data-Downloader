"""Test verified ZIP-to-Parquet ingestion."""

from datetime import date
import hashlib
from io import BytesIO
from pathlib import Path
import zipfile

import httpx
import pandas as pd
import pytest

from crypto_downloader.datasets import SPOT_KLINES
from crypto_downloader.http import ChecksumError
from crypto_downloader.ingest import ArchiveError, _member, ingest_archive
from crypto_downloader.models import Resource
from crypto_downloader.processing import DataValidationError, file_sha256
from crypto_downloader.sources.binance import Binance

FIXTURES = Path(__file__).parent / "fixtures"
DAY = date(2024, 1, 1)
ARCHIVE_NAME = "BTCUSDT-1m-2024-01-01.zip"
ARCHIVE_URL = f"https://data.example/{ARCHIVE_NAME}"
RESOURCE = Resource(DAY, ARCHIVE_URL, f"{ARCHIVE_URL}.CHECKSUM")


def archive_bytes(
    content: bytes | None = None,
    *,
    names: tuple[str, ...] = ("BTCUSDT-1m-2024-01-01.csv",),
) -> bytes:
    """Build an in-memory ZIP with chosen member names.

    Args:
        content: The bytes stored in every member.
        names: The archive member names to create.

    Returns:
        The complete ZIP bytes.
    """
    source = content
    if source is None:
        source = (FIXTURES / "binance_spot_klines_2024-01-01.csv").read_bytes()
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(name, source)
    return output.getvalue()


def client_for(payload: bytes) -> httpx.Client:
    """Create a client serving one checksum and archive.

    Args:
        payload: The archive response body.

    Returns:
        A mocked HTTPX client.
    """
    digest = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve the checksum sidecar or archive bytes."""
        if str(request.url).endswith(".CHECKSUM"):
            return httpx.Response(200, text=f"{digest}  {ARCHIVE_NAME}\n")
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "fixture",
    ["binance_spot_klines_2024-01-01.csv", "binance_spot_klines_2025-01-01.csv"],
)
def test_ingest_archive_writes_atomic_canonical_parquet(
    fixture: str, tmp_path: Path
) -> None:
    """Confirm representative timestamp eras survive a chunked round trip.

    Args:
        fixture: The source CSV fixture to ingest.
        tmp_path: The isolated output directory.
    """
    day = date.fromisoformat(fixture[-14:-4])
    name = f"BTCUSDT-1m-{day.isoformat()}.zip"
    resource = Resource(
        day, f"https://data.example/{name}", f"https://data.example/{name}.CHECKSUM"
    )
    payload = archive_bytes(
        (FIXTURES / fixture).read_bytes(),
        names=(name.removesuffix(".zip") + ".csv",),
    )
    destination = tmp_path / "nested" / "result.parquet"
    digest = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve the fixture with its exact checksum filename."""
        if str(request.url).endswith(".CHECKSUM"):
            return httpx.Response(200, text=f"{digest}  {name}\n")
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        metadata = ingest_archive(
            client, resource, SPOT_KLINES, destination, chunk_rows=1
        )

    frame = pd.read_parquet(destination)
    assert tuple(frame.columns) == SPOT_KLINES.stored_columns
    assert len(frame) == metadata.row_count == 2
    assert metadata.archive_sha256 == digest
    assert metadata.parquet_sha256 == file_sha256(destination)
    assert metadata.parquet_size == destination.stat().st_size
    assert metadata.parquet_mtime_ns == destination.stat().st_mtime_ns
    assert metadata.first_timestamp == frame.iloc[0]["open_time"].to_pydatetime()
    assert metadata.last_timestamp == frame.iloc[-1]["open_time"].to_pydatetime()
    assert not destination.with_name("result.parquet.part").exists()


@pytest.mark.parametrize(
    "names",
    [
        (),
        ("one.csv", "two.csv"),
        ("nested/one.csv",),
        ("nested\\one.csv",),
        ("/one.csv",),
        ("one.txt",),
        ("wrong.csv",),
        ("folder/",),
    ],
)
def test_ingest_archive_rejects_unsafe_or_unexpected_members(
    names: tuple[str, ...], tmp_path: Path
) -> None:
    """Confirm only the exact single expected CSV member is accepted.

    Args:
        names: The unsafe or malformed archive member set.
        tmp_path: The isolated output directory.
    """
    payload = archive_bytes(names=names)
    destination = tmp_path / "result.parquet"

    with client_for(payload) as client:
        with pytest.raises(ArchiveError):
            ingest_archive(client, RESOURCE, SPOT_KLINES, destination)

    assert not destination.exists()
    assert not destination.with_name("result.parquet.part").exists()


def test_ingest_archive_rejects_invalid_zip(tmp_path: Path) -> None:
    """Confirm checksum-valid non-ZIP bytes cannot produce Parquet."""
    with client_for(b"not a zip") as client:
        with pytest.raises(ArchiveError, match="ZIP"):
            ingest_archive(client, RESOURCE, SPOT_KLINES, tmp_path / "out.parquet")


def test_ingest_archive_rejects_oversized_csv(tmp_path: Path) -> None:
    """Confirm the uncompressed member limit is enforced."""
    payload = archive_bytes()
    with client_for(payload) as client:
        with pytest.raises(ArchiveError, match="size"):
            ingest_archive(
                client,
                RESOURCE,
                SPOT_KLINES,
                tmp_path / "out.parquet",
                max_csv_bytes=1,
            )


def test_ingest_archive_rejects_encrypted_member_metadata() -> None:
    """Confirm encrypted ZIP member metadata is rejected."""
    member = zipfile.ZipInfo("BTCUSDT-1m-2024-01-01.csv")
    member.flag_bits = 1

    class EncryptedArchive:
        """Supply encrypted member metadata to the archive validator."""

        def infolist(self) -> list[zipfile.ZipInfo]:
            """Return the one encrypted member.

            Returns:
                A list containing encrypted member metadata.
            """
            return [member]

    with pytest.raises(ArchiveError, match="encrypted"):
        _member(EncryptedArchive(), RESOURCE, 100)  # type: ignore[arg-type]


@pytest.mark.parametrize("content", [b"", b"1,2,3\n"])
def test_ingest_archive_rejects_empty_or_wrong_width_csv(
    content: bytes, tmp_path: Path
) -> None:
    """Confirm empty and malformed CSV data are not cached.

    Args:
        content: The invalid CSV bytes.
        tmp_path: The isolated output directory.
    """
    payload = archive_bytes(content)
    with client_for(payload) as client:
        with pytest.raises((ArchiveError, DataValidationError)):
            ingest_archive(client, RESOURCE, SPOT_KLINES, tmp_path / "out.parquet")


def test_ingest_failure_preserves_existing_destination(tmp_path: Path) -> None:
    """Confirm a later invalid chunk cannot replace a valid cached file."""
    good = (FIXTURES / "binance_spot_klines_2024-01-01.csv").read_text()
    bad_row = good.splitlines()[0].replace("42283.58", "not-number")
    payload = archive_bytes((good + bad_row + "\n").encode())
    destination = tmp_path / "result.parquet"
    destination.write_bytes(b"existing")

    with client_for(payload) as client:
        with pytest.raises(DataValidationError):
            ingest_archive(client, RESOURCE, SPOT_KLINES, destination, chunk_rows=2)

    assert destination.read_bytes() == b"existing"
    assert not destination.with_name("result.parquet.part").exists()


def test_checksum_failure_preserves_existing_destination(tmp_path: Path) -> None:
    """Confirm an unverified download cannot replace cached Parquet."""
    payload = archive_bytes()
    destination = tmp_path / "result.parquet"
    destination.write_bytes(b"existing")

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve an incorrect digest and otherwise valid ZIP."""
        if str(request.url).endswith(".CHECKSUM"):
            return httpx.Response(200, text=f"{'0' * 64}  {ARCHIVE_NAME}\n")
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ChecksumError):
            ingest_archive(
                client,
                RESOURCE,
                SPOT_KLINES,
                destination,
                retries=0,
            )

    assert destination.read_bytes() == b"existing"
    assert not destination.with_name("result.parquet.part").exists()


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("chunk_rows", 0),
        ("chunk_rows", True),
        ("max_archive_bytes", 0),
        ("max_csv_bytes", -1),
    ],
)
def test_ingest_archive_rejects_invalid_limits_without_http(
    setting: str, value: object, tmp_path: Path
) -> None:
    """Confirm invalid ingestion limits fail before network access.

    Args:
        setting: The keyword setting to replace.
        value: The invalid setting value.
        tmp_path: The isolated output directory.
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Count any unexpected request."""
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    arguments = {setting: value}
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            ingest_archive(
                client,
                RESOURCE,
                SPOT_KLINES,
                tmp_path / "out.parquet",
                **arguments,  # type: ignore[arg-type]
            )
    assert calls == 0


def test_binance_ingest_uses_source_http_settings(tmp_path: Path) -> None:
    """Confirm the source strategy exposes the generic ingestion workflow."""
    payload = archive_bytes()
    requests: list[httpx.Request] = []
    digest = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Record requests and serve one valid archive."""
        requests.append(request)
        if str(request.url).endswith(".CHECKSUM"):
            return httpx.Response(200, text=f"{digest}  {ARCHIVE_NAME}\n")
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = Binance(timeout=7.0, retries=0).ingest(
            client, RESOURCE, SPOT_KLINES, tmp_path / "out.parquet"
        )

    assert result.row_count == 2
    assert len(requests) == 2
    assert all(
        set(request.extensions["timeout"].values()) == {7.0} for request in requests
    )
