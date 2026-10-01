"""Test Bybit order-book discovery and streaming ingestion."""

from datetime import UTC, date, datetime
import io
import json
from pathlib import Path
import zipfile

import httpx
import pyarrow.parquet as pq
import pytest

from veldra.bybit.client import BybitClient
from veldra.bybit.datasets import get_dataset
from veldra.bybit.manifest import BybitOrderBookDiscovery
from veldra.bybit.orderbook import ingest_order_book, parse_event
from veldra.core.ingest import ArchiveError
from veldra.core.models import IntegritySpec, Resource
from veldra.core.subjects import DataSubject


class Limiter:
    """Provide no-op limits for mock source requests."""

    def acquire(self, key: str, *, cost: int = 1) -> None:
        """Accept one reservation."""
        del key, cost

    def penalize(self, key: str, retry_after: float) -> None:
        """Accept one penalty."""
        del key, retry_after


def _event(
    *,
    action: str = "snapshot",
    timestamp: int = 1735689600000,
    sequence: int = 100,
    bids: object = None,
    asks: object = None,
) -> dict[str, object]:
    """Return one representative Bybit depth event."""
    return {
        "topic": "orderbook.200.BTCUSDT",
        "type": action,
        "ts": timestamp,
        "cts": timestamp - 1,
        "data": {
            "s": "BTCUSDT",
            "b": [["100", "2"]] if bids is None else bids,
            "a": [["101", "3"]] if asks is None else asks,
            "u": sequence,
            "seq": sequence,
        },
    }


def _zip(name: str, events: list[dict[str, object]]) -> bytes:
    """Return one safe source ZIP containing JSON Lines."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, b"\n".join(json.dumps(item).encode() for item in events))
    return output.getvalue()


def _raw_zip(entries: dict[str, bytes]) -> bytes:
    """Return a ZIP with explicitly supplied members."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return output.getvalue()


def _resource(name: str = "2025-01-01_BTCUSDT_ob200.data.zip") -> Resource:
    """Return one representative order-book resource."""
    return Resource(
        date(2025, 1, 1),
        f"https://quote-saver.bycsi.com/orderbook/spot/BTCUSDT/{name}",
        None,
        integrity=IntegritySpec("archive_only"),
    )


def test_parse_event_preserves_snapshot_and_delta_semantics() -> None:
    """Keep zero-quantity deletions and source sequencing losslessly."""
    snapshot = parse_event(_event(), 0, "2025-01-01_BTCUSDT_ob200.data")
    delta = parse_event(
        _event(action="delta", sequence=101, bids=[["100", "0"]], asks=[]),
        1,
        "2025-01-01_BTCUSDT_ob200.data",
    )
    assert snapshot["source_depth"] == 200
    assert snapshot["action"] == "snapshot"
    assert delta["event_number"] == 1
    assert delta["bids"] == [{"price": 100.0, "quantity": 0.0}]
    assert not delta["asks"]


@pytest.mark.parametrize(
    "event",
    [
        {},
        _event(action="partial"),
        _event(bids=[["100"]]),
        _event(bids=[["100", "-1"]]),
        _event(bids=[["100", "0"]]),
        _event(bids=[], asks=[]),
        {**_event(), "topic": "orderbook.200.ETHUSDT"},
        {**_event(), "ts": True},
    ],
)
def test_invalid_book_events_fail_closed(event: object) -> None:
    """Reject malformed actions, levels, depths, and identities."""
    with pytest.raises(ArchiveError):
        parse_event(event, 0, "2025-01-01_BTCUSDT_ob200.data")


def test_event_and_filename_depth_must_agree() -> None:
    """Reject archives whose physical depth contradicts their topic."""
    with pytest.raises(ArchiveError, match="inconsistent"):
        parse_event(_event(), 0, "2025-01-01_BTCUSDT_ob500.data")


def test_streaming_ingestion_keeps_physical_event_order(tmp_path: Path) -> None:
    """Write snapshots and deltas as nested Parquet without reconstruction."""
    name = "2025-01-01_BTCUSDT_ob200.data"
    events = [
        _event(),
        _event(action="delta", timestamp=1735689600001, sequence=101),
    ]
    payload = _zip(name, events)
    destination = tmp_path / "book.parquet"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload)
        )
    ) as client:
        metadata = ingest_order_book(
            client,
            _resource(),
            get_dataset("spot", "order_book_updates"),
            destination,
            chunk_rows=1,
        )
    table = pq.read_table(destination)
    assert metadata.row_count == 2
    assert table["event_number"].to_pylist() == [0, 1]
    assert table["action"].to_pylist() == ["snapshot", "delta"]
    assert table["bids"].to_pylist()[0][0] == {"price": 100.0, "quantity": 2.0}


def test_option_book_ingestion_retains_only_the_requested_instrument(
    tmp_path: Path,
) -> None:
    """Filter one shared family archive into an exact logical Option book."""
    target = "BTC-20SEP26-81000-C-USDT"
    other = "BTC-20SEP26-82000-C-USDT"

    def option_event(instrument: str, sequence: int) -> dict[str, object]:
        event = _event(sequence=sequence)
        event["topic"] = f"orderbook.25.{instrument}"
        event["data"]["s"] = instrument  # type: ignore[index]
        return event

    name = "2025-01-01_BTC_USDT.ob25"
    payload = _zip(name, [option_event(other, 1), option_event(target, 2)])
    resource = Resource(
        date(2025, 1, 1),
        f"https://public.bybit.com/{name}.zip",
        None,
        archive_symbol=target,
        integrity=IntegritySpec("archive_only"),
    )
    destination = tmp_path / "option-book.parquet"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload)
        )
    ) as client:
        metadata = ingest_order_book(
            client,
            resource,
            get_dataset("options", "order_book_updates"),
            destination,
        )
    assert metadata.row_count == 1
    assert pq.read_table(destination)["instrument"].to_pylist() == [target]


def test_streaming_ingestion_deduplicates_the_rollover_snapshot(
    tmp_path: Path,
) -> None:
    """Keep post-midnight deltas but omit the next archive's opening snapshot."""
    name = "2025-01-01_BTCUSDT_ob200.data"
    events = [
        _event(),
        _event(action="delta", timestamp=1735776000001, sequence=101),
        _event(action="snapshot", timestamp=1735776000002, sequence=102),
    ]
    payload = _zip(name, events)
    destination = tmp_path / "book.parquet"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload)
        )
    ) as client:
        metadata = ingest_order_book(
            client,
            _resource(),
            get_dataset("spot", "order_book_updates"),
            destination,
        )
    table = pq.read_table(destination)
    assert metadata.row_count == 2
    assert table["event_number"].to_pylist() == [0, 1]
    assert table["event_time"].to_pylist()[-1] == datetime(
        2025, 1, 2, 0, 0, 0, 1000, UTC
    )


def test_rollover_snapshot_replaces_an_equivalent_final_delta(tmp_path: Path) -> None:
    """Drop the last delta when Bybit repeats its state as the next snapshot."""
    events = [
        _event(),
        _event(action="delta", timestamp=1735776000001, sequence=101),
        _event(action="snapshot", timestamp=1735776000001, sequence=101),
    ]
    payload = _zip("2025-01-01_BTCUSDT_ob200.data", events)
    destination = tmp_path / "book.parquet"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload)
        )
    ) as client:
        metadata = ingest_order_book(
            client,
            _resource(),
            get_dataset("spot", "order_book_updates"),
            destination,
        )
    assert metadata.row_count == 1
    assert pq.read_table(destination)["cross_sequence"].to_pylist() == [100]


def test_rollover_snapshot_must_be_the_final_event(tmp_path: Path) -> None:
    """Reject a malformed stream that continues after its duplicate snapshot."""
    events = [
        _event(),
        _event(action="snapshot", timestamp=1735776000001, sequence=101),
        _event(action="delta", timestamp=1735776000002, sequence=102),
    ]
    payload = _zip("2025-01-01_BTCUSDT_ob200.data", events)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload)
        )
    ) as client:
        with pytest.raises(ArchiveError, match="final event"):
            ingest_order_book(
                client,
                _resource(),
                get_dataset("spot", "order_book_updates"),
                tmp_path / "book.parquet",
            )


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([_event(action="delta")], "precedes"),
        ([_event(), _event(action="delta", sequence=100)], "must increase"),
        ([_event(timestamp=1735776360000)], "outside"),
    ],
)
def test_ingestion_rejects_unreplayable_event_streams(
    events: list[dict[str, object]], message: str, tmp_path: Path
) -> None:
    """Reject delta streams without a valid daily replay origin."""
    payload = _zip("2025-01-01_BTCUSDT_ob200.data", events)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload)
        )
    ) as client:
        with pytest.raises(ArchiveError, match=message):
            ingest_order_book(
                client,
                _resource(),
                get_dataset("spot", "order_book_updates"),
                tmp_path / "book.parquet",
            )


def test_large_archive_guard_rejects_before_processing(tmp_path: Path) -> None:
    """Require explicit configuration for unusually large depth archives."""
    payload = _zip("2025-01-01_BTCUSDT_ob200.data", [_event()])
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"Content-Length": str(len(payload))}, content=payload
            )
        )
    ) as client:
        with pytest.raises(Exception, match="size|larger|exceeds"):
            ingest_order_book(
                client,
                _resource(),
                get_dataset("spot", "order_book_updates"),
                tmp_path / "book.parquet",
                max_archive_bytes=1,
            )


@pytest.mark.parametrize(
    ("payload", "message", "max_line_bytes"),
    [
        (_raw_zip({"unexpected.data": b"{}"}), "unsafe|unexpected", 1024),
        (
            _raw_zip(
                {
                    "2025-01-01_BTCUSDT_ob200.data": b"{}",
                    "other.data": b"{}",
                }
            ),
            "exactly one",
            1024,
        ),
        (
            _raw_zip({"2025-01-01_BTCUSDT_ob200.data": b"not-json"}),
            "invalid JSON",
            1024,
        ),
        (
            _raw_zip({"2025-01-01_BTCUSDT_ob200.data": b"not-json"}),
            "line exceeds",
            1,
        ),
    ],
)
def test_malformed_book_archives_fail_closed(
    payload: bytes, message: str, max_line_bytes: int, tmp_path: Path
) -> None:
    """Reject unsafe members and malformed or oversized JSON lines."""
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload)
        )
    ) as client:
        with pytest.raises(ArchiveError, match=message):
            ingest_order_book(
                client,
                _resource(),
                get_dataset("spot", "order_book_updates"),
                tmp_path / "book.parquet",
                max_line_bytes=max_line_bytes,
            )


def test_ingestion_settings_must_be_positive(tmp_path: Path) -> None:
    """Reject Boolean and nonpositive resource limits before downloading."""
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    ) as client:
        with pytest.raises(ValueError, match="positive integer"):
            ingest_order_book(
                client,
                _resource(),
                get_dataset("spot", "order_book_updates"),
                tmp_path / "book.parquet",
                chunk_rows=True,
            )


@pytest.mark.parametrize(
    ("day", "depth"),
    [(date(2025, 8, 20), 500), (date(2025, 8, 21), 200)],
)
def test_derivative_discovery_tracks_the_depth_transition(
    day: date, depth: int
) -> None:
    """Use depth 500 through August 20 and depth 200 afterward."""

    def response(request: httpx.Request) -> httpx.Response:
        assert f"_ob{depth}.data.zip" in request.url.path
        return httpx.Response(200, headers={"Content-Length": "10"})

    with httpx.Client(transport=httpx.MockTransport(response)) as http:
        api = BybitClient(client=http, limiter=Limiter(), retries=0)
        found = BybitOrderBookDiscovery(api).discover(
            "linear", DataSubject("instrument", "BTCUSDT"), day, day
        )
    assert len(found) == 1
    assert found[0].key.dataset == "order_book_updates"


@pytest.mark.parametrize(
    ("product", "subject", "depth"),
    [
        ("spot", DataSubject("instrument", "BTCUSDT"), 200),
        ("options", DataSubject("instrument_family", "BTC"), 25),
    ],
)
def test_portal_book_discovery_preserves_physical_scope(
    product: str, subject: DataSubject, depth: int
) -> None:
    """Discover hidden Spot files and shared Option-family files."""
    native = "option" if product == "options" else "spot"
    filename = (
        f"2026-09-20_BTC_USDT.ob{depth}.zip"
        if product == "options"
        else f"2026-09-20_BTCUSDT_ob{depth}.data.zip"
    )
    folder = "option/BTC" if product == "options" else "spot/BTCUSDT"
    row = {
        "bizType": native,
        "productId": "orderbook",
        "interval": "daily",
        "symbol": subject.value,
        "date": "2026-09-20",
        "filename": filename,
        "size": "100",
        "url": f"https://quote-saver.bycsi.com/orderbook/{folder}/{filename}",
    }
    payload = {"ret_code": 0, "result": {"list": [row]}}
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as http:
        api = BybitClient(client=http, limiter=Limiter(), retries=0)
        found = BybitOrderBookDiscovery(api).discover(
            product, subject, date(2026, 9, 20), date(2026, 9, 20)
        )
    assert found[0].key.remote_scope_kind == subject.kind
    assert found[0].remote_size == 100


def test_dataset_orders_replay_by_cross_sequence() -> None:
    """Query order-book updates by exchange sequence, not wall-clock jitter."""
    dataset = get_dataset("linear", "order_book_updates")
    assert dataset.ordering_columns == ("cross_sequence", "event_number")
    assert dataset.archive_day_offset.total_seconds() == -300
