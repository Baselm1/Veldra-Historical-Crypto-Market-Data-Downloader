"""Stream OKX TAR/GZIP JSONL order-book updates into nested Parquet."""

from collections.abc import Mapping
from datetime import UTC, datetime, time, timedelta
import json
import math
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from typing import IO

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from veldra.core.datasets import DatasetSpec
from veldra.core.download import download
from veldra.core.models import (
    ArchiveObject,
    DataValidationError,
    LogicalPartition,
    Materialization,
    Resource,
)
from veldra.core.providers import MaterializedArchive
from veldra.core.subjects import DataSubject

type BookLevel = dict[str, float | int]
type BookRecord = dict[str, object]


def _positive_integer(value: object, name: str) -> int:
    """Return one positive integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _source_coverage(resource: ArchiveObject) -> tuple[datetime, datetime]:
    """Return the exact UTC source-day bounds for an order-book archive."""
    start = datetime.combine(resource.key.period_start, time.min, UTC)
    end = datetime.combine(resource.key.period_end + timedelta(days=1), time.min, UTC)
    return start, end


def _member(
    archive: tarfile.TarFile, resource: ArchiveObject, limit: int
) -> tarfile.TarInfo:
    """Return the one safe data member from a streaming TAR reader."""
    member = archive.next()
    expected = resource.key.remote_name.removesuffix(".tar.gz") + ".data"
    if (
        member is None
        or not member.isfile()
        or member.name != expected
        or "/" in member.name
        or "\\" in member.name
    ):
        raise DataValidationError("OKX TAR must contain its one expected data member")
    if member.size > limit:
        raise DataValidationError("OKX order-book member exceeds its size limit")
    return member


def _timestamp(value: object) -> datetime:
    """Convert one exact epoch-millisecond source timestamp to UTC."""
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise DataValidationError("OKX order-book timestamp must be milliseconds")
    milliseconds = int(value)
    if not 100_000_000_000 <= milliseconds < 100_000_000_000_000:
        raise DataValidationError("OKX order-book timestamp has an invalid unit")
    try:
        return datetime.fromtimestamp(milliseconds / 1_000, UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise DataValidationError("OKX order-book timestamp is invalid") from error


def _number(value: object, name: str, *, positive: bool) -> float:
    """Return one finite level number with the requested sign constraint."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise DataValidationError(f"OKX order-book {name} must be numeric")
    try:
        result = float(value)
    except ValueError as error:
        raise DataValidationError(f"OKX order-book {name} must be numeric") from error
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise DataValidationError(f"OKX order-book {name} is invalid")
    return result


def _count(value: object) -> int:
    """Return one nonnegative native order count."""
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise DataValidationError("OKX order-book order count must be an integer")
    result = int(value)
    if result < 0:
        raise DataValidationError("OKX order-book order count cannot be negative")
    return result


def _levels(
    value: object, name: str, quantity_name: str, maximum: int
) -> list[BookLevel]:
    """Validate native price, quantity, and order-count triplets."""
    if not isinstance(value, list) or len(value) > maximum:
        raise DataValidationError(
            f"OKX order-book {name} must contain at most {maximum} levels"
        )
    result: list[BookLevel] = []
    for level in value:
        if not isinstance(level, list) or len(level) != 3:
            raise DataValidationError("OKX order-book level must contain three values")
        result.append(
            {
                "price": _number(level[0], "price", positive=True),
                quantity_name: _number(level[1], "quantity", positive=False),
                "order_count": _count(level[2]),
            }
        )
    return result


def _matches_scope(instrument: str, resource: ArchiveObject) -> bool:
    """Return whether an instrument belongs to the manifest archive scope."""
    scope = resource.key.subject
    if scope.kind == "instrument":
        return instrument == scope.value
    if scope.kind == "instrument_family":
        return instrument == f"{scope.value}-SWAP" or instrument.startswith(
            f"{scope.value}-"
        )
    return False


def _event(
    raw: bytes,
    resource: ArchiveObject,
    dataset: DatasetSpec,
    event_number: int,
) -> BookRecord:
    """Decode and validate one source JSONL order-book event."""
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataValidationError("OKX order-book contains invalid JSON") from error
    if not isinstance(value, dict):
        raise DataValidationError("OKX order-book event must be an object")
    instrument = value.get("instId")
    if (
        not isinstance(instrument, str)
        or not instrument
        or not _matches_scope(instrument, resource)
    ):
        raise DataValidationError(
            "OKX order-book instrument does not match its archive"
        )
    action = value.get("action")
    if action not in {"snapshot", "update"}:
        raise DataValidationError("OKX order-book action must be snapshot or update")
    event_time = _timestamp(value.get("ts"))
    start, end = _source_coverage(resource)
    if not start <= event_time < end:
        raise DataValidationError("OKX order-book timestamp is outside its source day")
    maximum = 400 if dataset.name == "order_book_400" else 5000
    quantity = "base_quantity" if dataset.product == "spot" else "contract_quantity"
    return {
        "instrument_id": instrument,
        "event_time": event_time,
        "event_number": event_number,
        "action": action,
        "bids": _levels(value.get("bids"), "bids", quantity, maximum),
        "asks": _levels(value.get("asks"), "asks", quantity, maximum),
    }


def _schema(dataset: DatasetSpec) -> pa.Schema:
    """Return the product-specific nested Arrow order-book schema."""
    quantity = "base_quantity" if dataset.product == "spot" else "contract_quantity"
    level = pa.struct(
        (
            pa.field("price", pa.float64()),
            pa.field(quantity, pa.float64()),
            pa.field("order_count", pa.int64()),
        )
    )
    return pa.schema(
        (
            pa.field("instrument_id", pa.string()),
            pa.field("event_time", pa.timestamp("us", "UTC")),
            pa.field("event_number", pa.int64()),
            pa.field("action", pa.string()),
            pa.field("bids", pa.list_(level)),
            pa.field("asks", pa.list_(level)),
        )
    )


def _write(
    source: IO[bytes],
    resource: ArchiveObject,
    dataset: DatasetSpec,
    destination: Path,
    *,
    chunk_events: int,
    max_line_bytes: int,
) -> tuple[int, datetime, datetime, dict[str, int]]:
    """Stream validated JSON events into a bounded-memory Parquet writer."""
    schema = _schema(dataset)
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    pending: list[BookRecord] = []
    counts: dict[str, int] = {}
    first: datetime | None = None
    last: datetime | None = None
    previous: dict[str, datetime] = {}
    seen: set[str] = set()

    def flush() -> None:
        """Write and clear one bounded event batch."""
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=schema))
            pending.clear()

    try:
        for event_number, raw in enumerate(source):
            if len(raw) > max_line_bytes:
                raise DataValidationError("OKX order-book JSON line exceeds its limit")
            record = _event(raw, resource, dataset, event_number)
            instrument = str(record["instrument_id"])
            timestamp = record["event_time"]
            assert isinstance(timestamp, datetime)
            if instrument not in seen and record["action"] != "snapshot":
                raise DataValidationError(
                    "OKX order-book instrument must begin with a snapshot"
                )
            if instrument in previous and timestamp < previous[instrument]:
                raise DataValidationError("OKX order-book timestamps are out of order")
            seen.add(instrument)
            previous[instrument] = timestamp
            first = timestamp if first is None else min(first, timestamp)
            last = timestamp if last is None else max(last, timestamp)
            counts[instrument] = counts.get(instrument, 0) + 1
            pending.append(record)
            if len(pending) == chunk_events:
                flush()
        flush()
    finally:
        writer.close()
    if first is None or last is None:
        raise DataValidationError("OKX order-book archive cannot be empty")
    return sum(counts.values()), first, last, counts


def _partitions(
    resource: ArchiveObject,
    destination: Path,
    dataset: DatasetSpec,
    counts: Mapping[str, int],
) -> tuple[LogicalPartition, ...]:
    """Map every streamed instrument to its shared physical Parquet file."""
    start, end = _source_coverage(resource)
    return tuple(
        LogicalPartition(
            "okx",
            resource.key.product,
            resource.key.dataset,
            DataSubject("instrument", instrument),
            dataset.base_interval,
            start,
            end,
            destination,
            "instrument_id",
            instrument,
            rows,
            source_day=resource.key.period_start,
        )
        for instrument, rows in sorted(counts.items())
    )


def materialize_order_book(
    client: httpx.Client,
    resource: ArchiveObject,
    dataset: DatasetSpec,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    backoff: float,
    max_archive_bytes: int,
    max_member_bytes: int = 64 * 1024 * 1024 * 1024,
    max_line_bytes: int = 8 * 1024 * 1024,
    chunk_events: int = 16,
) -> MaterializedArchive:
    """Download and stream one verified OKX order-book archive atomically."""
    _positive_integer(max_member_bytes, "max_member_bytes")
    _positive_integer(max_line_bytes, "max_line_bytes")
    _positive_integer(chunk_events, "chunk_events")
    if dataset.name not in {"order_book_400", "order_book_5000"}:
        raise ValueError("order-book materialization requires an order-book dataset")
    if resource.integrity is None:
        raise DataValidationError("OKX archive has no integrity policy")
    start, end = _source_coverage(resource)
    legacy = Resource(
        resource.key.period_start,
        resource.url,
        None,
        integrity=resource.integrity,
        end_day=resource.key.period_end,
        coverage_start=start,
        coverage_end=end,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    with TemporaryDirectory(prefix="veldra-okx-book-") as directory:
        archive_path = Path(directory) / resource.key.remote_name
        digest = download(
            client,
            legacy,
            archive_path,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            max_bytes=max_archive_bytes,
        )
        try:
            with tarfile.open(archive_path, "r|gz") as archive:
                member = _member(archive, resource, max_member_bytes)
                source = archive.extractfile(member)
                if source is None:
                    raise DataValidationError("OKX order-book member cannot be read")
                rows, first, last, counts = _write(
                    source,
                    resource,
                    dataset,
                    partial,
                    chunk_events=chunk_events,
                    max_line_bytes=max_line_bytes,
                )
                if archive.next() is not None:
                    raise DataValidationError("OKX TAR contains more than one member")
        except DataValidationError:
            partial.unlink(missing_ok=True)
            raise
        except (tarfile.TarError, EOFError) as error:
            partial.unlink(missing_ok=True)
            raise DataValidationError("OKX order-book TAR/GZIP is invalid") from error
    partial.replace(destination)
    stat = destination.stat()
    materialization = Materialization(
        resource.key,
        destination,
        dataset.schema_version,
        rows,
        first,
        last,
        stat.st_size,
        local_mtime_ns=stat.st_mtime_ns,
        archive_revision=digest,
    )
    return MaterializedArchive(
        materialization, _partitions(resource, destination, dataset, counts)
    )
