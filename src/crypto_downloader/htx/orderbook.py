"""Safely stream HTX TAR/JSONL order-book events into Parquet."""

from datetime import UTC, datetime, timedelta
import json
import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import tarfile
from typing import IO
from urllib.parse import unquote, urlsplit

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from crypto_downloader.core.datasets import DatasetSpec
from crypto_downloader.core.download import download
from crypto_downloader.core.ingest import ArchiveError
from crypto_downloader.core.models import IngestedResource, Resource

type BookRow = tuple[datetime, int, str, str, int, float, float]

LOGGER = logging.getLogger(__name__)
BOOK_SCHEMA = pa.schema(
    [
        ("event_time", pa.timestamp("us", "UTC")),
        ("event_number", pa.int64()),
        ("action", pa.string()),
        ("side", pa.string()),
        ("level_number", pa.int64()),
        ("price", pa.float64()),
        ("quantity", pa.float64()),
    ]
)


def _positive_integer(value: object, name: str) -> int:
    """Return one positive integer setting.

    Args:
        value: The proposed setting.
        name: The setting name used in errors.

    Returns:
        The validated positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _member(
    archive: tarfile.TarFile, resource: Resource, max_bytes: int
) -> tarfile.TarInfo:
    """Return the single exact regular data member in an HTX TAR archive.

    Args:
        archive: The opened TAR/GZIP archive.
        resource: The source URL defining the expected member name.
        max_bytes: The maximum accepted uncompressed JSONL size.

    Returns:
        The validated TAR member metadata.
    """
    members = archive.getmembers()
    if len(members) != 1:
        raise ArchiveError("TAR must contain only one data member")
    member = members[0]
    if not member.isfile():
        raise ArchiveError("TAR member must be a regular file")
    archive_name = unquote(Path(urlsplit(resource.url).path).name)
    expected = archive_name.removesuffix(".tar.gz") + ".data"
    if member.name != expected or "/" in member.name or "\\" in member.name:
        raise ArchiveError("TAR must contain the exact expected data member")
    if member.size > max_bytes:
        raise ArchiveError("uncompressed JSONL size exceeds configured limit")
    return member


def _timestamp(value: object) -> datetime:
    """Convert one decimal epoch-microsecond value to UTC.

    Args:
        value: The source timestamp value.

    Returns:
        The exact UTC timestamp.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        micros = value
    elif isinstance(value, str) and value.isascii() and value.isdecimal():
        micros = int(value)
    else:
        raise ArchiveError("order-book timestamp must contain epoch microseconds")
    if not 100_000_000_000_000 <= micros < 100_000_000_000_000_000:
        raise ArchiveError("order-book timestamp has an invalid unit")
    try:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=micros)
    except OverflowError as error:
        raise ArchiveError(
            "order-book timestamp is outside the supported range"
        ) from error


def _number(value: object, name: str) -> float:
    """Return one finite numeric order-book value.

    Args:
        value: The source price or quantity.
        name: The field name used in errors.

    Returns:
        The finite floating-point value.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ArchiveError(f"order-book {name} must be numeric")
    try:
        number = float(value)
    except ValueError as error:
        raise ArchiveError(f"order-book {name} must be numeric") from error
    if not math.isfinite(number):
        raise ArchiveError(f"order-book {name} must be finite")
    return number


def _levels(value: object, side: str) -> list[tuple[int, float, float]]:
    """Validate and number one side of an order-book event.

    Args:
        value: The source list of price and quantity pairs.
        side: The canonical ``ask`` or ``bid`` label.

    Returns:
        Level number, price, and quantity tuples in source order.
    """
    if not isinstance(value, list):
        raise ArchiveError(f"order-book {side} levels must be a list")
    result: list[tuple[int, float, float]] = []
    for number, level in enumerate(value):
        if not isinstance(level, list) or len(level) != 2:
            raise ArchiveError("order-book level must contain price and quantity")
        price = _number(level[0], "price")
        quantity = _number(level[1], "quantity")
        if price <= 0:
            raise ArchiveError("order-book price must be positive")
        if quantity < 0:
            raise ArchiveError("order-book quantity must be nonnegative")
        result.append((number, price, quantity))
    return result


def _event_rows(
    value: object, resource: Resource, event_number: int
) -> tuple[datetime, list[BookRow]]:
    """Flatten one snapshot or update event into canonical level rows.

    Args:
        value: The decoded JSON value.
        resource: The resource defining symbol and time boundaries.
        event_number: The zero-based JSONL line number.

    Returns:
        The event timestamp and flattened ask/bid rows.
    """
    if not isinstance(value, dict):
        raise ArchiveError("order-book event must be a JSON object")
    symbol = value.get("instId")
    if not isinstance(symbol, str) or not symbol:
        raise ArchiveError("order-book event must contain instId")
    if resource.archive_symbol is not None and symbol != resource.archive_symbol:
        raise ArchiveError("order-book event symbol does not match its archive")
    action = value.get("action")
    if action not in {"snapshot", "update"}:
        raise ArchiveError("order-book action must be snapshot or update")
    event_time = _timestamp(value.get("ts"))
    start, end = resource.coverage
    if not start <= event_time < end:
        raise ArchiveError("order-book timestamp falls outside its source day")
    rows: list[BookRow] = []
    for side, field in (("ask", "asks"), ("bid", "bids")):
        rows.extend(
            (event_time, event_number, action, side, number, price, quantity)
            for number, price, quantity in _levels(value.get(field), side)
        )
    if not rows:
        raise ArchiveError("order-book event contains no levels")
    return event_time, rows


def _table(rows: list[BookRow]) -> pa.Table:
    """Convert canonical order-book tuples to a typed Arrow table.

    Args:
        rows: The flattened event levels.

    Returns:
        A table in canonical stored-column order.
    """
    columns = list(zip(*rows, strict=True))
    return pa.Table.from_arrays(
        [
            pa.array(column, type=field.type)
            for column, field in zip(columns, BOOK_SCHEMA)
        ],
        schema=BOOK_SCHEMA,
    )


def _write_events(
    source: IO[bytes],
    resource: Resource,
    destination: Path,
    chunk_rows: int,
    max_line_bytes: int,
) -> tuple[int, datetime, datetime]:
    """Stream JSON Lines into one temporary Parquet file.

    Args:
        source: The extracted TAR member stream.
        resource: The source symbol and time boundaries.
        destination: The temporary Parquet path.
        chunk_rows: The approximate flattened rows written per batch.
        max_line_bytes: The maximum accepted encoded JSON event size.

    Returns:
        Row count and first/last event timestamps.
    """
    writer: pq.ParquetWriter | None = None
    pending: list[BookRow] = []
    rows = 0
    first: datetime | None = None
    last: datetime | None = None
    previous: datetime | None = None

    def flush() -> None:
        """Write and clear accumulated canonical rows."""
        nonlocal writer, rows
        if not pending:
            return
        table = _table(pending)
        if writer is None:
            writer = pq.ParquetWriter(
                destination,
                BOOK_SCHEMA,
                compression="zstd",
                use_dictionary=["action", "side"],
            )
        writer.write_table(table)
        rows += len(pending)
        pending.clear()

    try:
        for event_number, raw_line in enumerate(source):
            if len(raw_line) > max_line_bytes:
                raise ArchiveError("order-book JSON line exceeds configured limit")
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ArchiveError(
                    f"order-book JSON line {event_number + 1} is invalid"
                ) from error
            event_time, event_rows = _event_rows(value, resource, event_number)
            if previous is not None and event_time < previous:
                raise ArchiveError("order-book event timestamps must not decrease")
            previous = event_time
            first = event_time if first is None else first
            last = event_time
            pending.extend(event_rows)
            if len(pending) >= chunk_rows:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    if rows == 0 or first is None or last is None:
        raise ArchiveError("order-book JSONL member cannot be empty")
    return rows, first, last


def ingest_order_book(
    client: httpx.Client,
    resource: Resource,
    dataset: DatasetSpec,
    destination: Path,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.5,
    chunk_rows: int = 200_000,
    max_archive_bytes: int = 2 * 1024 * 1024 * 1024,
    max_jsonl_bytes: int = 8 * 1024 * 1024 * 1024,
    max_json_line_bytes: int = 64 * 1024 * 1024,
) -> IngestedResource:
    """Download and safely convert one HTX order-book archive.

    Args:
        client: The HTTPX client used for source files.
        resource: The daily TAR/GZIP archive and checksum.
        dataset: The canonical order-book declaration.
        destination: The final Parquet path.
        timeout: The timeout for each HTTP attempt in seconds.
        retries: The retries after the first HTTP attempt.
        backoff: The initial exponential retry delay in seconds.
        chunk_rows: The approximate flattened rows written per batch.
        max_archive_bytes: The largest accepted compressed archive.
        max_jsonl_bytes: The largest accepted uncompressed JSONL member.
        max_json_line_bytes: The largest accepted encoded JSON event.

    Returns:
        Integrity, row-count, and timestamp metadata for the Parquet file.
    """
    if dataset.name != "order_book_updates":
        raise ValueError("order-book ingestion requires an order-book dataset")
    chunk_rows = _positive_integer(chunk_rows, "chunk_rows")
    max_archive_bytes = _positive_integer(max_archive_bytes, "max_archive_bytes")
    max_jsonl_bytes = _positive_integer(max_jsonl_bytes, "max_jsonl_bytes")
    max_json_line_bytes = _positive_integer(max_json_line_bytes, "max_json_line_bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    try:
        with TemporaryDirectory(prefix="crypto-downloader-") as directory:
            archive_path = Path(directory) / "source.tar.gz"
            archive_sha256 = download(
                client,
                resource,
                archive_path,
                timeout=timeout,
                retries=retries,
                backoff=backoff,
                max_bytes=max_archive_bytes,
            )
            try:
                with tarfile.open(archive_path, "r:gz") as archive:
                    member = _member(archive, resource, max_jsonl_bytes)
                    source = archive.extractfile(member)
                    if source is None:
                        raise ArchiveError("TAR data member cannot be read")
                    with source:
                        rows, first, last = _write_events(
                            source,
                            resource,
                            partial,
                            chunk_rows,
                            max_json_line_bytes,
                        )
            except (tarfile.TarError, OSError) as error:
                raise ArchiveError(
                    "source file is not a valid TAR/GZIP archive"
                ) from error
        partial.replace(destination)
        stat = destination.stat()
        metadata = IngestedResource(
            archive_sha256=archive_sha256,
            parquet_size=stat.st_size,
            parquet_mtime_ns=stat.st_mtime_ns,
            row_count=rows,
            first_timestamp=first,
            last_timestamp=last,
            timestamp_column=dataset.time_column,
            schema_version=dataset.schema_version,
        )
        LOGGER.info(
            "HTX order-book ingestion complete: day=%s rows=%d first=%s "
            "last=%s bytes=%d path=%s",
            resource.day,
            rows,
            first,
            last,
            stat.st_size,
            destination,
        )
        return metadata
    except BaseException:
        partial.unlink(missing_ok=True)
        LOGGER.exception(
            "HTX order-book ingestion failed: day=%s url=%s destination=%s",
            resource.day,
            resource.url,
            destination,
        )
        raise
