"""Test safe OKX nested order-book ingestion and facade access."""

from base64 import b64encode
from datetime import UTC, date, datetime
from hashlib import md5
from io import BytesIO
import json
from pathlib import Path
import tarfile

import httpx
import pandas as pd
import pytest

from veldra import OKX
from veldra.core.models import (
    ArchiveKey,
    ArchiveObject,
    DataValidationError,
    IntegritySpec,
)
from veldra.okx.datasets import get_dataset
from veldra.okx.orderbook import _event, _write, materialize_order_book


def physical(
    name: str = "BTC-USDT-L2orderbook-400lv-2025-01-01.tar.gz",
) -> ArchiveObject:
    """Build one physical Spot order-book object."""
    return ArchiveObject(
        ArchiveKey(
            "okx",
            "spot",
            "order_book_400",
            "module_4",
            "instrument",
            "BTC-USDT",
            "daily",
            date(2025, 1, 1),
            date(2025, 1, 1),
            name,
        ),
        f"https://files.test/{name}",
        integrity=IntegritySpec("response_header", algorithm="md5"),
    )


def event(
    *,
    action: str = "snapshot",
    timestamp: str = "1735689600000",
    instrument: str = "BTC-USDT",
    bids: object | None = None,
    asks: object | None = None,
) -> dict[str, object]:
    """Build one valid source order-book event."""
    return {
        "instId": instrument,
        "action": action,
        "ts": timestamp,
        "bids": [["99", "2", "3"]] if bids is None else bids,
        "asks": [["101", "1", "2"]] if asks is None else asks,
    }


def tar_bytes(name: str, rows: list[object], *, extra: bool = False) -> bytes:
    """Return a TAR/GZIP containing JSON Lines under the expected member."""
    output = BytesIO()
    body = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        member = tarfile.TarInfo(name.removesuffix(".tar.gz") + ".data")
        member.size = len(body)
        archive.addfile(member, BytesIO(body))
        if extra:
            other = tarfile.TarInfo("extra.data")
            other.size = 1
            archive.addfile(other, BytesIO(b"x"))
    return output.getvalue()


def test_order_book_declarations_keep_nested_native_units() -> None:
    """Confirm Spot and derivatives distinguish level quantity units."""
    spot = get_dataset("spot", "order_book_400")
    swap = get_dataset("linear_swap", "order_book_5000")
    assert spot.object_columns == ("bids", "asks")
    assert spot.max_concurrency == 4
    assert swap.max_concurrency == 2
    assert spot.archive_day_offset.total_seconds() == 0


def test_verified_order_book_streams_to_shared_nested_parquet(tmp_path: Path) -> None:
    """Confirm MD5 verification, streaming, native levels, and partitions."""
    item = physical()
    content = tar_bytes(
        item.key.remote_name,
        [event(), event(action="update", timestamp="1735689601000")],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one integrity-bearing source archive."""
        return httpx.Response(
            200,
            content=content,
            headers={"Content-MD5": b64encode(md5(content).digest()).decode()},
            request=request,
        )

    destination = tmp_path / "book.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        value = materialize_order_book(
            client,
            item,
            get_dataset("spot", "order_book_400"),
            destination,
            timeout=1,
            retries=0,
            backoff=0,
            max_archive_bytes=1_000_000,
            chunk_events=1,
        )
    frame = pd.read_parquet(destination)
    assert value.materialization.row_count == 2
    assert value.partitions[0].subject.value == "BTC-USDT"
    assert frame["action"].tolist() == ["snapshot", "update"]
    assert frame.loc[0, "bids"][0] == {
        "price": 99.0,
        "base_quantity": 2.0,
        "order_count": 3,
    }


def test_extra_tar_members_are_rejected_without_partial_output(tmp_path: Path) -> None:
    """Confirm an archive cannot smuggle an unvalidated second member."""
    item = physical()
    content = tar_bytes(item.key.remote_name, [event()], extra=True)

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one invalid multi-member archive with a valid digest."""
        return httpx.Response(
            200,
            content=content,
            headers={"ETag": md5(content).hexdigest()},
            request=request,
        )

    destination = tmp_path / "book.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataValidationError, match="more than one"):
            materialize_order_book(
                client,
                item,
                get_dataset("spot", "order_book_400"),
                destination,
                timeout=1,
                retries=0,
                backoff=0,
                max_archive_bytes=1_000_000,
            )
    assert not destination.exists()
    assert not destination.with_name("book.parquet.part").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"instrument": "ETH-USDT"}, "instrument"),
        ({"action": "partial"}, "action"),
        ({"timestamp": "bad"}, "timestamp"),
        ({"timestamp": "1735776000000"}, "outside"),
        ({"bids": [["99", "2"]]}, "three"),
        ({"bids": [["bad", "2", "3"]]}, "price"),
        ({"bids": [["99", "-1", "3"]]}, "quantity"),
        ({"bids": [["99", "1", "bad"]]}, "count"),
        ({"bids": "bad"}, "levels"),
    ],
)
def test_invalid_order_book_events_fail(
    change: dict[str, object], message: str
) -> None:
    """Confirm malformed source fields cannot become queryable rows."""
    with pytest.raises(DataValidationError, match=message):
        _event(
            json.dumps(event(**change)).encode(),
            physical(),
            get_dataset("spot", "order_book_400"),
            0,
        )


@pytest.mark.parametrize("raw", [b"not json", b"[]"])
def test_invalid_order_book_json_fails(raw: bytes) -> None:
    """Confirm invalid JSON and non-object JSON are rejected."""
    with pytest.raises(DataValidationError):
        _event(raw, physical(), get_dataset("spot", "order_book_400"), 0)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([event(action="update")], "begin with a snapshot"),
        (
            [
                event(timestamp="1735689601000"),
                event(action="update", timestamp="1735689600000"),
            ],
            "out of order",
        ),
    ],
)
def test_stream_requires_snapshot_and_ordered_timestamps(
    tmp_path: Path, rows: list[object], message: str
) -> None:
    """Confirm reconstruction prerequisites are checked per instrument."""
    source = BytesIO(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    with pytest.raises(DataValidationError, match=message):
        _write(
            source,
            physical(),
            get_dataset("spot", "order_book_400"),
            tmp_path / "bad.parquet",
            chunk_events=1,
            max_line_bytes=1_000_000,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_events": 0}, "chunk_events"),
        ({"max_line_bytes": 0}, "max_line_bytes"),
        ({"max_member_bytes": 0}, "max_member_bytes"),
    ],
)
def test_materializer_rejects_invalid_limits(
    tmp_path: Path, kwargs: dict[str, int], message: str
) -> None:
    """Confirm unsafe streaming configuration fails before network access."""
    with httpx.Client(transport=httpx.MockTransport(lambda request: None)) as client:
        with pytest.raises(ValueError, match=message):
            materialize_order_book(
                client,
                physical(),
                get_dataset("spot", "order_book_400"),
                tmp_path / "book.parquet",
                timeout=1,
                retries=0,
                backoff=0,
                max_archive_bytes=1,
                **kwargs,
            )


class BookFixture:
    """Serve current Spot metadata, a manifest, and one tiny book archive."""

    def __init__(self) -> None:
        """Create an archive request counter."""
        self.files = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return the response matching one public facade operation."""
        if request.url.path.endswith("/instruments"):
            row = {
                "instType": "SPOT",
                "instId": "BTC-USDT",
                "instFamily": "",
                "baseCcy": "BTC",
                "quoteCcy": "USDT",
                "settleCcy": "",
                "ctType": "",
                "ctVal": "",
                "ctMult": "",
                "ctValCcy": "",
                "state": "live",
                "ruleType": "normal",
                "listTime": "1609459200000",
                "expTime": "",
                "stk": "",
                "optType": "",
            }
            return httpx.Response(
                200, json={"code": "0", "msg": "", "data": [row]}, request=request
            )
        if request.url.path.endswith("/market-data-history"):
            name = "BTC-USDT-L2orderbook-400lv-2025-01-01.tar.gz"
            group = {
                "instId": "BTC-USDT",
                "instFamily": "",
                "groupDetails": [
                    {
                        "dateTs": "1735689600000",
                        "filename": name,
                        "sizeMB": "0.01",
                        "url": f"https://files.test/{name}",
                    }
                ],
            }
            return httpx.Response(
                200,
                json={"code": "0", "msg": "", "data": [{"details": [group]}]},
                request=request,
            )
        if request.url.host == "files.test":
            self.files += 1
            name = Path(request.url.path).name
            content = tar_bytes(name, [event()])
            return httpx.Response(
                200,
                content=content,
                headers={"Content-MD5": b64encode(md5(content).digest()).decode()},
                request=request,
            )
        raise AssertionError(f"unexpected request {request.url}")


def test_public_order_book_is_queryable_and_reused_offline(tmp_path: Path) -> None:
    """Confirm one declarative call returns nested rows and caches them."""
    fixture = BookFixture()
    api = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    frame = api.get_order_book_updates("BTC-USDT", "2025-01-01", "2025-01-01")
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert list(frame.columns) == [
        "event_time",
        "event_number",
        "action",
        "bids",
        "asks",
    ]
    cached = api.get_order_book_updates(
        "BTC-USDT", "2025-01-01", "2025-01-01", offline=True
    )
    assert isinstance(cached, pd.DataFrame)
    pd.testing.assert_frame_equal(frame, cached)
    assert fixture.files == 1


def test_public_order_book_rejects_unknown_depth(tmp_path: Path) -> None:
    """Confirm the facade accepts only implemented native depths."""
    api = OKX(tmp_path, progress=False)
    with pytest.raises(ValueError, match="400 or 5000"):
        api.get_order_book_updates("BTC-USDT", "2025-01-01", "2025-01-01", depth=50)  # type: ignore[arg-type]
