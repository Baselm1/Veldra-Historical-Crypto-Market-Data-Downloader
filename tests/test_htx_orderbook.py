"""Test safe HTX order-book update ingestion and facade access."""

from datetime import UTC, date, datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tarfile

import httpx
import pandas as pd
import pytest

from crypto_downloader.core.datasets import DatasetSpec
from crypto_downloader.core.ingest import ArchiveError
from crypto_downloader.core.models import Resource
from crypto_downloader.htx.connector import HTXConnector
from crypto_downloader.htx.datasets import (
    COIN_ORDER_BOOK_UPDATES,
    LINEAR_ORDER_BOOK_UPDATES,
    SPOT_ORDER_BOOK_UPDATES,
    SPOT_KLINES,
    get_dataset,
)
from crypto_downloader.htx.facade import HTX
from crypto_downloader.htx.orderbook import ingest_order_book


def tar_bytes(
    name: str, bodies: list[bytes], *, member_type: bytes | None = None
) -> bytes:
    """Build a TAR/GZIP archive with one or more named members.

    Args:
        name: The first member name.
        bodies: Member contents; later members receive numbered names.
        member_type: An optional unsafe TAR member type.

    Returns:
        Complete compressed archive bytes.
    """
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for index, body in enumerate(bodies):
            member = tarfile.TarInfo(name if index == 0 else f"extra-{index}.data")
            member.size = len(body)
            if member_type is not None:
                member.type = member_type
            archive.addfile(member, BytesIO(body))
    return output.getvalue()


def resource(name: str = "A8-USDT-l2orderbook-400lv-2026-09-05.tar.gz") -> Resource:
    """Build one new-tree order-book resource.

    Args:
        name: The remote archive filename.

    Returns:
        An HTX source resource with exact UTC+8 coverage.
    """
    return Resource(
        date(2026, 9, 5),
        f"https://example.test/{name}",
        f"https://example.test/{name}.CHECKSUM",
        archive_symbol="A8-USDT",
        coverage_start=datetime(2026, 9, 4, 16, tzinfo=UTC),
        coverage_end=datetime(2026, 9, 5, 16, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    "specification",
    [SPOT_ORDER_BOOK_UPDATES, LINEAR_ORDER_BOOK_UPDATES, COIN_ORDER_BOOK_UPDATES],
)
def test_order_book_declarations_preserve_event_semantics(
    specification: DatasetSpec,
) -> None:
    """Confirm every product exposes the same flattened event schema.

    Args:
        specification: One product-specific order-book declaration.
    """
    assert get_dataset(specification.product, specification.name) is specification
    assert specification.stored_columns == (
        "event_time",
        "event_number",
        "action",
        "side",
        "level_number",
        "price",
        "quantity",
    )
    assert specification.integer_columns == (
        "event_number",
        "level_number",
    )
    assert specification.string_columns == ("action", "side")


def test_verified_order_book_archive_streams_to_canonical_parquet(
    tmp_path: Path,
) -> None:
    """Confirm snapshots, updates, and deletion quantities survive ingestion.

    Args:
        tmp_path: The isolated Parquet destination.
    """
    events = [
        {
            "instId": "A8-USDT",
            "action": "snapshot",
            "ts": 1788566400000000,
            "asks": [["2", "3"], ["3", "4"]],
            "bids": [["1", "5"]],
        },
        {
            "instId": "A8-USDT",
            "action": "update",
            "ts": "1788566401000000",
            "asks": [["2", "0"]],
            "bids": [],
        },
    ]
    body = "\n".join(json.dumps(event) for event in events).encode()
    archive = tar_bytes("A8-USDT-l2orderbook-400lv-2026-09-05.data", [body])
    digest = sha256(archive).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the matching archive or checksum."""
        if request.url.path.endswith("CHECKSUM"):
            return httpx.Response(
                200,
                text=f"{digest}  A8-USDT-l2orderbook-400lv-2026-09-05.tar.gz\n",
            )
        return httpx.Response(200, content=archive)

    destination = tmp_path / "book.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        metadata = HTXConnector(retries=0).ingest(
            client, resource(), SPOT_ORDER_BOOK_UPDATES, destination
        )

    frame = pd.read_parquet(destination)
    assert metadata.row_count == 4
    assert frame["event_number"].tolist() == [0, 0, 0, 1]
    assert frame["action"].tolist() == ["snapshot"] * 3 + ["update"]
    assert frame["side"].tolist() == ["ask", "ask", "bid", "ask"]
    assert frame["level_number"].tolist() == [0, 1, 0, 0]
    assert frame["quantity"].tolist()[-1] == 0
    assert metadata.first_timestamp == datetime(2026, 9, 5, tzinfo=UTC)
    assert metadata.last_timestamp == datetime(2026, 9, 5, 0, 0, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (
            {"action": "snapshot", "ts": "1788566400000000", "asks": [], "bids": []},
            "instId",
        ),
        (
            {
                "instId": "OTHER",
                "action": "snapshot",
                "ts": "1788566400000000",
                "asks": [[1, 1]],
                "bids": [],
            },
            "symbol",
        ),
        (
            {
                "instId": "A8-USDT",
                "action": "replace",
                "ts": "1788566400000000",
                "asks": [[1, 1]],
                "bids": [],
            },
            "action",
        ),
        (
            {
                "instId": "A8-USDT",
                "action": "update",
                "ts": "123",
                "asks": [[1, 1]],
                "bids": [],
            },
            "timestamp",
        ),
        (
            {
                "instId": "A8-USDT",
                "action": "update",
                "ts": None,
                "asks": [[1, 1]],
                "bids": [],
            },
            "timestamp",
        ),
        (
            {
                "instId": "A8-USDT",
                "action": "update",
                "ts": "1788566400000000",
                "asks": None,
                "bids": [],
            },
            "levels",
        ),
        (
            {
                "instId": "A8-USDT",
                "action": "update",
                "ts": "1788566400000000",
                "asks": [["abc", 1]],
                "bids": [],
            },
            "numeric",
        ),
        (
            {
                "instId": "A8-USDT",
                "action": "update",
                "ts": "1788566400000000",
                "asks": [["nan", 1]],
                "bids": [],
            },
            "finite",
        ),
        (
            {
                "instId": "A8-USDT",
                "action": "update",
                "ts": "1788566400000000",
                "asks": [[1, True]],
                "bids": [],
            },
            "numeric",
        ),
        (
            {
                "instId": "A8-USDT",
                "action": "update",
                "ts": "1788566400000000",
                "asks": [],
                "bids": [],
            },
            "no levels",
        ),
        (
            {
                "instId": "A8-USDT",
                "action": "update",
                "ts": "1788566400000000",
                "asks": [[1]],
                "bids": [],
            },
            "level",
        ),
        (
            {
                "instId": "A8-USDT",
                "action": "update",
                "ts": "1788566400000000",
                "asks": [[0, 1]],
                "bids": [],
            },
            "price",
        ),
        (
            {
                "instId": "A8-USDT",
                "action": "update",
                "ts": "1788566400000000",
                "asks": [[1, -1]],
                "bids": [],
            },
            "quantity",
        ),
    ],
)
def test_malformed_order_book_events_are_rejected(
    tmp_path: Path, event: dict[str, object], message: str
) -> None:
    """Confirm invalid JSON event structure or values fail atomically.

    Args:
        tmp_path: The isolated Parquet destination.
        event: The malformed source event.
        message: The expected error fragment.
    """
    body = json.dumps(event).encode()
    archive = tar_bytes("A8-USDT-l2orderbook-400lv-2026-09-05.data", [body])
    digest = sha256(archive).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the malformed archive and its valid checksum."""
        if request.url.path.endswith("CHECKSUM"):
            return httpx.Response(
                200,
                text=f"{digest}  A8-USDT-l2orderbook-400lv-2026-09-05.tar.gz",
            )
        return httpx.Response(200, content=archive)

    destination = tmp_path / "book.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArchiveError, match=message):
            ingest_order_book(
                client,
                resource(),
                SPOT_ORDER_BOOK_UPDATES,
                destination,
                retries=0,
            )
    assert not destination.exists()
    assert not destination.with_name("book.parquet.part").exists()


@pytest.mark.parametrize(
    ("name", "bodies", "member_type", "limit", "message"),
    [
        ("../unsafe.data", [b"{}"], None, 100, "exact expected"),
        (
            "A8-USDT-l2orderbook-400lv-2026-09-05.data",
            [b"{}", b"{}"],
            None,
            100,
            "only",
        ),
        (
            "A8-USDT-l2orderbook-400lv-2026-09-05.data",
            [b"{}"],
            tarfile.SYMTYPE,
            100,
            "regular",
        ),
        ("A8-USDT-l2orderbook-400lv-2026-09-05.data", [b"too large"], None, 3, "size"),
    ],
)
def test_unsafe_order_book_archives_are_rejected(
    tmp_path: Path,
    name: str,
    bodies: list[bytes],
    member_type: bytes | None,
    limit: int,
    message: str,
) -> None:
    """Confirm TAR traversal, multiplicity, links, and expansion are bounded.

    Args:
        tmp_path: The isolated Parquet destination.
        name: The first TAR member name.
        bodies: The TAR member contents.
        member_type: Optional unsafe TAR type.
        limit: The accepted uncompressed byte ceiling.
        message: The expected error fragment.
    """
    archive = tar_bytes(name, bodies, member_type=member_type)
    digest = sha256(archive).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the unsafe archive and a valid checksum."""
        if request.url.path.endswith("CHECKSUM"):
            return httpx.Response(
                200,
                text=f"{digest}  A8-USDT-l2orderbook-400lv-2026-09-05.tar.gz",
            )
        return httpx.Response(200, content=archive)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArchiveError, match=message):
            ingest_order_book(
                client,
                resource(),
                SPOT_ORDER_BOOK_UPDATES,
                tmp_path / "book.parquet",
                retries=0,
                max_jsonl_bytes=limit,
            )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not-json", "JSON line"),
        (b"", "cannot be empty"),
        (
            b'{"instId":"A8-USDT","action":"update","ts":"1788566401000000","asks":[[1,1]],"bids":[]}\n'
            b'{"instId":"A8-USDT","action":"update","ts":"1788566400000000","asks":[[1,1]],"bids":[]}',
            "must not decrease",
        ),
    ],
)
def test_malformed_order_book_streams_are_rejected(
    tmp_path: Path, body: bytes, message: str
) -> None:
    """Confirm invalid JSON, empty streams, and reversed events fail.

    Args:
        tmp_path: The isolated Parquet destination.
        body: The malformed JSONL member contents.
        message: The expected error fragment.
    """
    archive = tar_bytes("A8-USDT-l2orderbook-400lv-2026-09-05.data", [body])
    digest = sha256(archive).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the malformed archive and matching checksum."""
        if request.url.path.endswith("CHECKSUM"):
            return httpx.Response(
                200,
                text=f"{digest}  A8-USDT-l2orderbook-400lv-2026-09-05.tar.gz",
            )
        return httpx.Response(200, content=archive)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArchiveError, match=message):
            ingest_order_book(
                client,
                resource(),
                SPOT_ORDER_BOOK_UPDATES,
                tmp_path / "book.parquet",
                retries=0,
            )


def test_order_book_json_line_size_is_bounded(tmp_path: Path) -> None:
    """Confirm a single oversized JSON event is rejected before parsing.

    Args:
        tmp_path: The isolated Parquet destination.
    """
    body = b"{" + b" " * 100 + b"}"
    archive = tar_bytes("A8-USDT-l2orderbook-400lv-2026-09-05.data", [body])
    digest = sha256(archive).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the oversized event and matching checksum."""
        if request.url.path.endswith("CHECKSUM"):
            return httpx.Response(
                200,
                text=f"{digest}  A8-USDT-l2orderbook-400lv-2026-09-05.tar.gz",
            )
        return httpx.Response(200, content=archive)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArchiveError, match="line exceeds"):
            ingest_order_book(
                client,
                resource(),
                SPOT_ORDER_BOOK_UPDATES,
                tmp_path / "book.parquet",
                retries=0,
                max_json_line_bytes=10,
            )


@pytest.mark.parametrize(
    ("dataset", "options", "message"),
    [
        (SPOT_KLINES, {}, "requires an order-book dataset"),
        (SPOT_ORDER_BOOK_UPDATES, {"chunk_rows": 0}, "positive integer"),
        (
            SPOT_ORDER_BOOK_UPDATES,
            {"max_archive_bytes": True},
            "positive integer",
        ),
    ],
)
def test_order_book_ingestion_rejects_invalid_options(
    tmp_path: Path,
    dataset: DatasetSpec,
    options: dict[str, object],
    message: str,
) -> None:
    """Confirm dataset and resource-limit options are validated eagerly.

    Args:
        tmp_path: The isolated Parquet destination.
        dataset: The proposed dataset declaration.
        options: Invalid ingestion options.
        message: The expected error fragment.
    """
    with httpx.Client() as client:
        with pytest.raises(ValueError, match=message):
            ingest_order_book(
                client,
                resource(),
                dataset,
                tmp_path / "book.parquet",
                **options,  # type: ignore[arg-type]
            )


def test_invalid_tar_gzip_archive_is_rejected(tmp_path: Path) -> None:
    """Confirm validly checksummed non-TAR bytes fail atomically.

    Args:
        tmp_path: The isolated Parquet destination.
    """
    body = b"not a tar archive"
    digest = sha256(body).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Return invalid TAR bytes and their matching checksum."""
        if request.url.path.endswith("CHECKSUM"):
            return httpx.Response(
                200,
                text=f"{digest}  A8-USDT-l2orderbook-400lv-2026-09-05.tar.gz",
            )
        return httpx.Response(200, content=body)

    destination = tmp_path / "book.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArchiveError, match="valid TAR/GZIP"):
            ingest_order_book(
                client,
                resource(),
                SPOT_ORDER_BOOK_UPDATES,
                destination,
                retries=0,
            )
    assert not destination.exists()


def test_order_book_facade_preserves_pair_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm order-book requests delegate without Kline-only options.

    Args:
        tmp_path: The isolated data directory.
        monkeypatch: Pytest's attribute replacement helper.
    """
    service = HTX(tmp_path, progress=False)
    expected = [pd.DataFrame({"action": ["snapshot"]})]
    received: dict[str, object] = {}

    def get_data(*args: object, **kwargs: object) -> list[pd.DataFrame]:
        """Record one delegated order-book request."""
        received.update(kwargs)
        return expected

    monkeypatch.setattr(service._downloader, "get_data", get_data)
    actual = service.get_order_book_updates(
        ["BTCUSDT"],
        "2026-09-05",
        "2026-09-05",
        product="linear_swap",
        columns=["event_time", "side", "price", "quantity"],
    )

    assert actual is expected
    assert received == {
        "product": "linear_swap",
        "dataset": "order_book_updates",
        "desired_columns": ["event_time", "side", "price", "quantity"],
        "refresh": False,
        "offline": False,
        "progress": False,
    }
