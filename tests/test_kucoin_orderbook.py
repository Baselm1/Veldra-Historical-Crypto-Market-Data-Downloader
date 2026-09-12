"""Test KuCoin level-50 order-book snapshot ingestion."""

from datetime import UTC, date, datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from veldra.core.datasets import DatasetSpec
from veldra.core.ingest import ArchiveError
from veldra.core.models import DataValidationError, Resource
from veldra.kucoin.datasets import (
    INVERSE_ORDER_BOOK_SNAPSHOTS,
    LINEAR_ORDER_BOOK_SNAPSHOTS,
    SPOT_KLINES,
    SPOT_ORDER_BOOK_SNAPSHOTS,
    get_dataset,
)
from veldra.kucoin.orderbook import ingest_order_book


def archive_bytes(name: str, bodies: list[bytes]) -> bytes:
    """Build a ZIP with the requested member bodies.

    Args:
        name: The first ZIP member name.
        bodies: The member bodies to store.

    Returns:
        Complete ZIP archive bytes.
    """
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, body in enumerate(bodies):
            member = name if index == 0 else f"extra-{index}.csv"
            archive.writestr(member, body)
    return output.getvalue()


def resource(
    name: str = "BTC-USDT-orderbooklv50-2025-01-01.zip",
) -> Resource:
    """Build one KuCoin order-book resource.

    Args:
        name: The exact remote archive filename.

    Returns:
        A resource with KuCoin's observed boundary tolerance.
    """
    return Resource(
        date(2025, 1, 1),
        f"https://example.test/{name}",
        f"https://example.test/{name}.CHECKSUM",
        checksum_algorithm="md5",
        coverage_start=datetime(2024, 12, 31, 23, 55, tzinfo=UTC),
        coverage_end=datetime(2025, 1, 2, 0, 5, tzinfo=UTC),
    )


def client_for(archive: bytes, name: str) -> httpx.Client:
    """Return a mock client serving one archive and MD5 sidecar.

    Args:
        archive: The exact source archive bytes.
        name: The expected archive filename.

    Returns:
        A mock HTTP client.
    """
    digest = hashlib.md5(archive).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the archive or matching checksum sidecar."""
        if request.url.path.endswith("CHECKSUM"):
            return httpx.Response(200, text=f"{digest}  {name}")
        return httpx.Response(200, content=archive)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "specification",
    [
        SPOT_ORDER_BOOK_SNAPSHOTS,
        LINEAR_ORDER_BOOK_SNAPSHOTS,
        INVERSE_ORDER_BOOK_SNAPSHOTS,
    ],
)
def test_order_book_declarations_preserve_nested_snapshot_semantics(
    specification: DatasetSpec,
) -> None:
    """Confirm each product registers bounded nested snapshot storage.

    Args:
        specification: One product-specific order-book declaration.
    """
    assert get_dataset(specification.product, specification.name) is specification
    assert specification.object_columns == ("bids", "asks")
    assert specification.discovery_lookahead_days == 1
    assert specification.max_concurrency == 4


def test_spot_order_book_streams_sorted_nested_snapshots(tmp_path: Path) -> None:
    """Confirm Spot levels retain structure and source quantities.

    Args:
        tmp_path: The isolated Parquet destination.
    """
    events = [
        {
            "asks": [["94001", "0.2"]],
            "bids": [["94000", "0.1"], ["93999", "0.3"]],
            "timestamp": 1735689601000,
        },
        {
            "asks": [["93991", "0.4"], ["93992", "0.5"]],
            "bids": [["93990", "0.6"]],
            "timestamp": 1735689600000,
        },
    ]
    body = b"data\n" + b"\n".join(json.dumps(event).encode() for event in events)
    name = "BTC-USDT-orderbooklv50-2025-01-01.zip"
    archive = archive_bytes(name.removesuffix(".zip") + ".csv", [body])
    destination = tmp_path / "spot.parquet"

    with client_for(archive, name) as client:
        metadata = ingest_order_book(
            client,
            resource(name),
            SPOT_ORDER_BOOK_SNAPSHOTS,
            destination,
            retries=0,
            chunk_rows=1,
        )

    table = pq.read_table(destination)
    assert metadata.row_count == 2
    assert table["event_time"].to_pylist() == [
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 0, 1, tzinfo=UTC),
    ]
    assert table["bids"].type == pa.list_(
        pa.struct(
            [
                pa.field("price", pa.float64()),
                pa.field("base_quantity", pa.float64()),
            ]
        )
    )
    assert table["bids"].to_pylist()[1][0] == {
        "price": 94000.0,
        "base_quantity": 0.1,
    }
    assert not destination.with_name("spot.parquet.part").exists()


@pytest.mark.parametrize(
    "specification",
    [LINEAR_ORDER_BOOK_SNAPSHOTS, INVERSE_ORDER_BOOK_SNAPSHOTS],
)
def test_futures_order_books_preserve_sequence_and_contract_quantity(
    tmp_path: Path, specification: DatasetSpec
) -> None:
    """Confirm both perpetual products retain sequence and contract units.

    Args:
        tmp_path: The isolated Parquet destination.
        specification: One perpetual order-book declaration.
    """
    event = {
        "asks": [["94001", "2"]],
        "bids": [["94000", "1"]],
        "sequence": 123,
        "timestamp": 1735689600000,
        "ts": 1735689600000,
    }
    name = "BTCUSDTM-orderbooklv50-2025-01-01.zip"
    body = b"data\n" + json.dumps(event).encode()
    archive = archive_bytes(name.removesuffix(".zip") + ".csv", [body])
    destination = tmp_path / f"{specification.product}.parquet"

    with client_for(archive, name) as client:
        metadata = ingest_order_book(
            client, resource(name), specification, destination, retries=0
        )

    table = pq.read_table(destination)
    assert metadata.row_count == 1
    assert table["sequence"].to_pylist() == [123]
    assert table["asks"].to_pylist()[0][0] == {
        "price": 94001.0,
        "contract_quantity": 2.0,
    }


@pytest.mark.parametrize(
    ("event", "message"),
    [
        ({"asks": [], "bids": [], "timestamp": True}, "timestamp"),
        ({"asks": [], "bids": [], "timestamp": 123}, "timestamp unit"),
        (
            {
                "asks": [],
                "bids": [],
                "timestamp": 1735689600000,
                "ts": 1735689601000,
            },
            "disagree",
        ),
        (
            {
                "asks": [["0", "1"]],
                "bids": [],
                "timestamp": 1735689600000,
            },
            "price",
        ),
        (
            {
                "asks": [["1", "-1"]],
                "bids": [],
                "timestamp": 1735689600000,
            },
            "quantity",
        ),
        (
            {
                "asks": [["2", "1"], ["1", "1"]],
                "bids": [],
                "timestamp": 1735689600000,
            },
            "ordered uniquely",
        ),
        (
            {
                "asks": [],
                "bids": [["1", "1"]] * 51,
                "timestamp": 1735689600000,
            },
            "at most 50",
        ),
        (
            {
                "asks": [],
                "bids": [],
                "timestamp": 1735689600000,
                "sequence": -1,
            },
            "sequence",
        ),
    ],
)
def test_malformed_snapshot_values_fail_atomically(
    tmp_path: Path, event: dict[str, object], message: str
) -> None:
    """Confirm malformed timestamps, levels, and sequences are rejected.

    Args:
        tmp_path: The isolated Parquet destination.
        event: The malformed Futures snapshot.
        message: The expected validation error fragment.
    """
    name = "BTCUSDTM-orderbooklv50-2025-01-01.zip"
    body = b"data\n" + json.dumps(event).encode()
    archive = archive_bytes(name.removesuffix(".zip") + ".csv", [body])
    destination = tmp_path / "invalid.parquet"

    with client_for(archive, name) as client:
        with pytest.raises(DataValidationError, match=message):
            ingest_order_book(
                client,
                resource(name),
                LINEAR_ORDER_BOOK_SNAPSHOTS,
                destination,
                retries=0,
            )
    assert not destination.exists()
    assert not destination.with_name("invalid.parquet.part").exists()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"wrong\n{}", "data header"),
        (b"data\n", "cannot be empty"),
        (b"data\nnot-json", "invalid JSON"),
        (
            b'data\n{"asks":[],"bids":[],"timestamp":1735689600000,'
            b'"timestamp":1735689600000}',
            "duplicate key",
        ),
        (b"data\n\n", "empty record"),
    ],
)
def test_malformed_snapshot_streams_are_rejected(
    tmp_path: Path, body: bytes, message: str
) -> None:
    """Confirm invalid headers, JSON, duplicates, and empty streams fail.

    Args:
        tmp_path: The isolated Parquet destination.
        body: The malformed member body.
        message: The expected error fragment.
    """
    name = "BTC-USDT-orderbooklv50-2025-01-01.zip"
    archive = archive_bytes(name.removesuffix(".zip") + ".csv", [body])

    with client_for(archive, name) as client:
        with pytest.raises((ArchiveError, DataValidationError), match=message):
            ingest_order_book(
                client,
                resource(name),
                SPOT_ORDER_BOOK_SNAPSHOTS,
                tmp_path / "invalid.parquet",
                retries=0,
            )


@pytest.mark.parametrize(
    ("member", "bodies", "limit", "message"),
    [
        ("wrong.csv", [b"data\n{}"], 1_000, "exact expected"),
        (
            "BTC-USDT-orderbooklv50-2025-01-01.csv",
            [b"data\n{}", b"data\n{}"],
            1_000,
            "only",
        ),
        (
            "BTC-USDT-orderbooklv50-2025-01-01.csv",
            [b"data\n" + b" " * 100],
            10,
            "uncompressed snapshot size",
        ),
    ],
)
def test_unsafe_snapshot_archives_are_rejected(
    tmp_path: Path,
    member: str,
    bodies: list[bytes],
    limit: int,
    message: str,
) -> None:
    """Confirm exact members, multiplicity, and expansion are bounded.

    Args:
        tmp_path: The isolated Parquet destination.
        member: The first archive member name.
        bodies: Member contents to store.
        limit: The accepted uncompressed size.
        message: The expected archive error fragment.
    """
    name = "BTC-USDT-orderbooklv50-2025-01-01.zip"
    archive = archive_bytes(member, bodies)

    with client_for(archive, name) as client:
        with pytest.raises(ArchiveError, match=message):
            ingest_order_book(
                client,
                resource(name),
                SPOT_ORDER_BOOK_SNAPSHOTS,
                tmp_path / "invalid.parquet",
                retries=0,
                max_snapshot_bytes=limit,
            )


@pytest.mark.parametrize(
    ("specification", "options", "message"),
    [
        (SPOT_KLINES, {}, "requires an order-book dataset"),
        (SPOT_ORDER_BOOK_SNAPSHOTS, {"chunk_rows": 0}, "positive integer"),
        (
            SPOT_ORDER_BOOK_SNAPSHOTS,
            {"max_json_line_bytes": True},
            "positive integer",
        ),
    ],
)
def test_order_book_ingestion_rejects_invalid_options_before_http(
    tmp_path: Path,
    specification: DatasetSpec,
    options: dict[str, object],
    message: str,
) -> None:
    """Confirm dataset and limit options fail before source requests.

    Args:
        tmp_path: The isolated Parquet destination.
        specification: The proposed dataset declaration.
        options: The invalid ingestion options.
        message: The expected error fragment.
    """
    with httpx.Client() as client:
        with pytest.raises(ValueError, match=message):
            ingest_order_book(
                client,
                resource(),
                specification,
                tmp_path / "invalid.parquet",
                **options,  # type: ignore[arg-type]
            )


def test_invalid_zip_is_rejected_atomically(tmp_path: Path) -> None:
    """Confirm checksum-valid non-ZIP bytes cannot produce Parquet.

    Args:
        tmp_path: The isolated Parquet destination.
    """
    name = "BTC-USDT-orderbooklv50-2025-01-01.zip"
    destination = tmp_path / "invalid.parquet"
    with client_for(b"not a zip", name) as client:
        with pytest.raises(ArchiveError, match="valid ZIP"):
            ingest_order_book(
                client,
                resource(name),
                SPOT_ORDER_BOOK_SNAPSHOTS,
                destination,
                retries=0,
            )
    assert not destination.exists()
