"""Stream Bybit order-book snapshots and deltas into nested Parquet rows."""

from datetime import UTC, date, datetime, time, timedelta
from dataclasses import dataclass, field as dataclass_field
import json
import logging
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import cast
from urllib.parse import unquote, urlsplit
import zipfile

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from veldra.core.datasets import DatasetSpec
from veldra.core.download import download
from veldra.core.ingest import ArchiveError
from veldra.core.models import IngestedResource, Resource

LOGGER = logging.getLogger(__name__)
_DEPTH = re.compile(r"(?:_ob|orderbook\.)([0-9]+)")
_ROLLOVER_GRACE = timedelta(minutes=5)
_LEVEL_TYPE = pa.list_(
    pa.struct([pa.field("price", pa.float64()), pa.field("quantity", pa.float64())])
)
_SCHEMA = pa.schema(
    [
        pa.field("event_time", pa.timestamp("us", "UTC")),
        pa.field("engine_time", pa.timestamp("us", "UTC")),
        pa.field("event_number", pa.int64()),
        pa.field("update_id", pa.int64()),
        pa.field("cross_sequence", pa.int64()),
        pa.field("instrument", pa.string()),
        pa.field("action", pa.string()),
        pa.field("source_depth", pa.int64()),
        pa.field("bids", _LEVEL_TYPE),
        pa.field("asks", _LEVEL_TYPE),
    ]
)


def _positive_integer(value: object, name: str) -> int:
    """Return one positive non-Boolean integer setting."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _integer(value: object, field: str) -> int:
    """Return one exact nonnegative source integer."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ArchiveError(f"invalid order-book {field}")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ArchiveError(f"invalid order-book {field}") from error
    if parsed < 0:
        raise ArchiveError(f"invalid order-book {field}")
    return parsed


def _timestamp(value: object, field: str) -> datetime:
    """Convert one epoch-millisecond source time to UTC."""
    milliseconds = _integer(value, field)
    if milliseconds < 100_000_000_000 or milliseconds >= 100_000_000_000_000:
        raise ArchiveError(f"invalid timestamp unit for {field}")
    return datetime.fromtimestamp(milliseconds / 1_000, UTC)


def _text(value: object, field: str) -> str:
    """Return one nonempty single-line source string."""
    if not isinstance(value, str) or not value.strip() or not value.isascii():
        raise ArchiveError(f"invalid order-book {field}")
    result = value.strip()
    if "\n" in result or "\r" in result:
        raise ArchiveError(f"invalid order-book {field}")
    return result


def _levels(
    value: object, action: str, side: str, depth: int
) -> list[dict[str, float]]:
    """Parse one bounded bid or ask update list."""
    # A delta can remove stale levels and add replacements in one message, so
    # its change count can legitimately exceed the resulting book depth.
    maximum = depth if action == "snapshot" else depth * 2
    if not isinstance(value, list) or len(value) > maximum:
        raise ArchiveError(f"invalid order-book {side} levels")
    found: list[dict[str, float]] = []
    seen: set[float] = set()
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ArchiveError(f"invalid order-book {side} level")
        try:
            price, quantity = float(item[0]), float(item[1])
        except (TypeError, ValueError) as error:
            raise ArchiveError(f"invalid order-book {side} level") from error
        if not (price > 0 and quantity >= 0):
            raise ArchiveError(f"invalid order-book {side} level")
        if action == "snapshot" and quantity == 0:
            raise ArchiveError("order-book snapshots cannot contain deletions")
        if price in seen:
            raise ArchiveError(f"duplicate order-book {side} price")
        seen.add(price)
        found.append({"price": price, "quantity": quantity})
    return found


def _depth(topic: str, remote_name: str) -> int:
    """Return matching source depth declared by topic or filename."""
    values = [int(match) for match in _DEPTH.findall(f"{topic} {remote_name}")]
    if not values or len(set(values)) != 1:
        raise ArchiveError("order-book depth is missing or inconsistent")
    return values[0]


def parse_event(
    value: object, event_number: int, remote_name: str
) -> dict[str, object]:
    """Parse one Bybit JSONL snapshot or delta event."""
    if not isinstance(value, dict) or not isinstance(value.get("data"), dict):
        raise ArchiveError("order-book event must contain a data object")
    topic = _text(value.get("topic"), "topic")
    action = _text(value.get("type"), "action").lower()
    if action not in {"snapshot", "delta"}:
        raise ArchiveError("order-book action must be snapshot or delta")
    data = value["data"]
    instrument = _text(data.get("s"), "instrument")
    if not topic.endswith(f".{instrument}"):
        raise ArchiveError("order-book topic does not match its instrument")
    source_depth = _depth(topic, remote_name)
    bids = _levels(data.get("b"), action, "bid", source_depth)
    asks = _levels(data.get("a"), action, "ask", source_depth)
    if not bids and not asks:
        raise ArchiveError("order-book event has no changed levels")
    return {
        "event_time": _timestamp(value.get("ts"), "event time"),
        "engine_time": _timestamp(value.get("cts"), "engine time"),
        "event_number": event_number,
        "update_id": _integer(data.get("u"), "update ID"),
        "cross_sequence": _integer(data.get("seq"), "cross sequence"),
        "instrument": instrument,
        "action": action,
        "source_depth": source_depth,
        "bids": bids,
        "asks": asks,
    }


def _member(
    archive: zipfile.ZipFile, resource: Resource, max_json_bytes: int
) -> zipfile.ZipInfo:
    """Return the single safe JSONL member in one source ZIP."""
    members = archive.infolist()
    if len(members) != 1 or members[0].is_dir():
        raise ArchiveError("order-book ZIP must contain exactly one file")
    member = members[0]
    actual = member.filename
    remote = unquote(Path(urlsplit(resource.url).path).name).removesuffix(".zip")
    names = {remote, f"{remote}.data", f"{remote}.json", f"{remote}.jsonl"}
    if actual not in names or "/" in actual or "\\" in actual or member.flag_bits & 1:
        raise ArchiveError("order-book ZIP contains an unsafe or unexpected member")
    if member.file_size > max_json_bytes:
        raise ArchiveError("uncompressed order-book data exceeds configured limit")
    return member


def _table(rows: list[dict[str, object]]) -> pa.Table:
    """Return one explicitly typed nested Arrow table."""
    return pa.Table.from_pylist(rows, schema=_SCHEMA)


def _validate_batch(
    rows: list[dict[str, object]],
    day: date,
    snapshots: set[str],
    previous_sequence: dict[str, int],
) -> None:
    """Validate source-day bounds and replay order for one parsed batch."""
    start = datetime.combine(day, time.min, UTC)
    end = start + timedelta(days=1) + _ROLLOVER_GRACE
    for row in rows:
        instrument = str(row["instrument"])
        action = str(row["action"])
        timestamp = row["event_time"]
        if not isinstance(timestamp, datetime) or timestamp < start or timestamp >= end:
            raise ArchiveError("order-book event falls outside its archive day")
        if action == "snapshot":
            snapshots.add(instrument)
        elif instrument not in snapshots:
            raise ArchiveError("order-book delta precedes its daily snapshot")
        sequence = cast(int, row["cross_sequence"])
        if sequence <= previous_sequence.get(instrument, -1):
            raise ArchiveError("order-book cross sequence must increase")
        previous_sequence[instrument] = sequence


@dataclass
class _WriterState:
    """Hold bounded state for one streaming Parquet conversion."""

    day: date
    partial: Path
    chunk_rows: int
    writer: pq.ParquetWriter | None = None
    pending: list[dict[str, object]] = dataclass_field(default_factory=list)
    snapshots: set[str] = dataclass_field(default_factory=set)
    sequences: dict[str, int] = dataclass_field(default_factory=dict)
    rows: int = 0
    first: datetime | None = None
    last: datetime | None = None
    held: dict[str, object] | None = None
    rollover_snapshot: bool = False


def _flush(state: _WriterState) -> None:
    """Validate and persist the state's buffered events."""
    if not state.pending:
        return
    _validate_batch(state.pending, state.day, state.snapshots, state.sequences)
    table = _table(state.pending)
    times = [cast(datetime, row["event_time"]) for row in state.pending]
    batch_first = min(times)
    batch_last = max(times)
    state.first = batch_first if state.first is None else min(state.first, batch_first)
    state.last = batch_last if state.last is None else max(state.last, batch_last)
    if state.writer is None:
        state.writer = pq.ParquetWriter(
            state.partial, _SCHEMA, compression="zstd", use_dictionary=False
        )
    state.writer.write_table(table)
    state.rows += len(state.pending)
    state.pending.clear()


def _queue_held(state: _WriterState) -> None:
    """Move the one-event look-behind into the bounded output buffer."""
    if state.held is None:
        return
    state.pending.append(state.held)
    state.held = None
    if len(state.pending) >= state.chunk_rows:
        _flush(state)


def _same_update(left: dict[str, object], right: dict[str, object]) -> bool:
    """Return whether a delta and rollover snapshot identify one update."""
    marker = ("event_time", "update_id", "cross_sequence")
    return all(left[field] == right[field] for field in marker)


def _push_event(state: _WriterState, event: dict[str, object]) -> None:
    """Validate source-day rollover semantics and retain one event."""
    timestamp = cast(datetime, event["event_time"])
    start = datetime.combine(state.day, time.min, UTC)
    boundary = start + timedelta(days=1)
    if timestamp < start or timestamp >= boundary + _ROLLOVER_GRACE:
        raise ArchiveError("order-book event falls outside its archive day")
    if state.rollover_snapshot:
        raise ArchiveError("order-book rollover snapshot must be the final event")
    if timestamp >= boundary and event["action"] == "snapshot":
        if state.held is not None and not _same_update(state.held, event):
            _queue_held(state)
        else:
            state.held = None
        state.rollover_snapshot = True
        return
    _queue_held(state)
    state.held = event


def _decode_event(
    encoded: bytes, event_number: int, member_name: str, max_line_bytes: int
) -> dict[str, object]:
    """Decode and validate one bounded JSONL record."""
    if len(encoded) > max_line_bytes:
        raise ArchiveError("order-book JSON line exceeds configured limit")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchiveError("order-book archive contains invalid JSON") from error
    return parse_event(value, event_number, member_name)


def _write_events(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    resource: Resource,
    partial: Path,
    *,
    chunk_rows: int,
    max_line_bytes: int,
    target_instrument: str | None = None,
) -> tuple[int, datetime, datetime]:
    """Stream one JSONL member through validation into Parquet."""
    state = _WriterState(resource.day, partial, chunk_rows)
    try:
        with archive.open(member, "r") as source:
            for event_number, encoded in enumerate(source):
                event = _decode_event(
                    encoded, event_number, member.filename, max_line_bytes
                )
                if (
                    target_instrument is None
                    or event["instrument"] == target_instrument
                ):
                    _push_event(state, event)
        _queue_held(state)
        _flush(state)
    finally:
        if state.writer is not None:
            state.writer.close()
    if not state.rows or state.first is None or state.last is None:
        raise ArchiveError("order-book archive cannot be empty")
    return state.rows, state.first, state.last


def ingest_order_book(
    client: httpx.Client,
    resource: Resource,
    dataset: DatasetSpec,
    destination: Path,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.5,
    chunk_rows: int = 50_000,
    max_archive_bytes: int = 512 * 1024 * 1024,
    max_json_bytes: int = 8 * 1024 * 1024 * 1024,
    max_line_bytes: int = 8 * 1024 * 1024,
) -> IngestedResource:
    """Download and stream one lossless Bybit order-book archive."""
    if dataset.name != "order_book_updates":
        raise ValueError("order-book ingestion requires order_book_updates")
    if dataset.product == "options" and not resource.archive_symbol:
        raise ValueError("Option archive requires a target instrument")
    for value, name in (
        (chunk_rows, "chunk_rows"),
        (max_archive_bytes, "max_archive_bytes"),
        (max_json_bytes, "max_json_bytes"),
        (max_line_bytes, "max_line_bytes"),
    ):
        _positive_integer(value, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    try:
        with TemporaryDirectory(prefix="veldra-bybit-book-") as directory:
            archive_path = Path(directory) / "source.zip"
            archive_checksum = download(
                client,
                resource,
                archive_path,
                timeout=timeout,
                retries=retries,
                backoff=backoff,
                max_bytes=max_archive_bytes,
            )
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    member = _member(archive, resource, max_json_bytes)
                    rows, first, last = _write_events(
                        archive,
                        member,
                        resource,
                        partial,
                        chunk_rows=chunk_rows,
                        max_line_bytes=max_line_bytes,
                        target_instrument=(
                            resource.archive_symbol
                            if dataset.product == "options"
                            else None
                        ),
                    )
            except zipfile.BadZipFile as error:
                raise ArchiveError("source file is not a valid ZIP archive") from error
        partial.replace(destination)
        stat = destination.stat()
        return IngestedResource(
            archive_checksum=archive_checksum,
            parquet_size=stat.st_size,
            parquet_mtime_ns=stat.st_mtime_ns,
            row_count=rows,
            first_timestamp=first,
            last_timestamp=last,
            timestamp_column=dataset.time_column,
            schema_version=dataset.schema_version,
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        LOGGER.exception("Bybit order-book ingestion failed: %s", resource.url)
        raise
