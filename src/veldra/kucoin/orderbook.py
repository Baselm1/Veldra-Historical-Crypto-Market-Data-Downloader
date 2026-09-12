"""Convert KuCoin level-50 JSON snapshots into nested Parquet rows."""

from collections.abc import Sequence
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import unquote, urlsplit
import zipfile

import duckdb
import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from veldra.core.datasets import DatasetSpec
from veldra.core.download import download
from veldra.core.ingest import ArchiveError
from veldra.core.models import DataValidationError, IngestedResource, Resource


def _positive_integer(value: object, name: str) -> int:
    """Return a positive integer setting or reject it."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _member(
    archive: zipfile.ZipFile, resource: Resource, max_bytes: int
) -> zipfile.ZipInfo:
    """Return the single safe CSV member from a KuCoin snapshot ZIP."""
    archive_name = unquote(Path(urlsplit(resource.url).path).name)
    expected = archive_name.removesuffix(".zip") + ".csv"
    members = archive.infolist()
    if (
        len(members) != 1
        or members[0].is_dir()
        or members[0].filename != expected
        or "/" in members[0].filename
        or "\\" in members[0].filename
    ):
        raise ArchiveError("ZIP must contain only the exact expected snapshot CSV")
    member = members[0]
    if member.flag_bits & 1:
        raise ArchiveError("encrypted ZIP members are not supported")
    if member.file_size > max_bytes:
        raise ArchiveError("uncompressed snapshot size exceeds configured limit")
    return member


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataValidationError("snapshot JSON contains a duplicate key")
        result[key] = value
    return result


def _number(value: object, name: str, *, positive: bool) -> float:
    """Return one finite source number with the declared sign constraint."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise DataValidationError(f"invalid snapshot {name}")
    try:
        number = float(value)
    except ValueError as error:
        raise DataValidationError(f"invalid snapshot {name}") from error
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise DataValidationError(f"invalid snapshot {name}")
    return number


def _levels(value: object, side: str, quantity_name: str) -> list[dict[str, float]]:
    """Validate and normalize one side of a level-50 order book."""
    if not isinstance(value, list) or len(value) > 50:
        raise DataValidationError(f"snapshot {side} must contain at most 50 levels")
    levels: list[dict[str, float]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 2:
            raise DataValidationError(f"snapshot {side} contains an invalid level")
        levels.append(
            {
                "price": _number(raw[0], "price", positive=True),
                quantity_name: _number(raw[1], "quantity", positive=False),
            }
        )
    prices = [level["price"] for level in levels]
    ordered = prices == sorted(prices, reverse=side == "bids")
    if not ordered or len(prices) != len(set(prices)):
        raise DataValidationError(f"snapshot {side} prices are not ordered uniquely")
    return levels


def _epoch_milliseconds(value: object) -> datetime:
    """Convert one exact epoch-millisecond value to a UTC timestamp."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError("invalid snapshot timestamp")
    if not 100_000_000_000 <= value < 100_000_000_000_000:
        raise DataValidationError("invalid snapshot timestamp unit")
    try:
        return datetime.fromtimestamp(value / 1_000, UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise DataValidationError("invalid snapshot timestamp") from error


def _event(text: bytes, dataset: DatasetSpec, resource: Resource) -> dict[str, object]:
    """Parse and validate one KuCoin snapshot JSON record."""
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataValidationError("snapshot contains invalid JSON") from error
    if not isinstance(value, dict):
        raise DataValidationError("snapshot JSON must contain an object")
    timestamp = _epoch_milliseconds(value.get("timestamp"))
    alternate = value.get("ts")
    if alternate is not None and _epoch_milliseconds(alternate) != timestamp:
        raise DataValidationError("snapshot timestamps disagree")
    start, end = resource.coverage
    if timestamp < start or timestamp >= end:
        raise DataValidationError("snapshot timestamp falls outside resource coverage")
    quantity_name = (
        "base_quantity" if dataset.product == "spot" else "contract_quantity"
    )
    result: dict[str, object] = {
        "event_time": timestamp,
        "bids": _levels(value.get("bids"), "bids", quantity_name),
        "asks": _levels(value.get("asks"), "asks", quantity_name),
    }
    if dataset.product != "spot":
        sequence = value.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise DataValidationError("invalid snapshot sequence")
        result["sequence"] = sequence
    return result


def _schema(dataset: DatasetSpec) -> pa.Schema:
    """Return the nested Arrow schema for one KuCoin product."""
    quantity_name = (
        "base_quantity" if dataset.product == "spot" else "contract_quantity"
    )
    level = pa.struct(
        (pa.field("price", pa.float64()), pa.field(quantity_name, pa.float64()))
    )
    fields = [pa.field("event_time", pa.timestamp("us", "UTC"))]
    if dataset.product != "spot":
        fields.append(pa.field("sequence", pa.int64()))
    fields.extend(
        (pa.field("bids", pa.list_(level)), pa.field("asks", pa.list_(level)))
    )
    return pa.schema(fields)


def _write_records(
    records: Sequence[dict[str, object]],
    schema: pa.Schema,
    writer: pq.ParquetWriter,
) -> None:
    """Write one validated snapshot batch to an unsorted Parquet file."""
    writer.write_table(pa.Table.from_pylist(records, schema=schema))


def _write_unsorted(
    source: Any,
    resource: Resource,
    dataset: DatasetSpec,
    destination: Path,
    chunk_rows: int,
    max_line_bytes: int,
) -> tuple[int, datetime, datetime]:
    """Stream snapshot JSON records into one temporary nested Parquet file."""
    if source.readline().strip() != b"data":
        raise ArchiveError("snapshot CSV must begin with the data header")
    schema = _schema(dataset)
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    records: list[dict[str, object]] = []
    rows = 0
    first: datetime | None = None
    last: datetime | None = None
    try:
        for line in source:
            if len(line) > max_line_bytes:
                raise ArchiveError("snapshot JSON line exceeds configured limit")
            if not line.strip():
                raise ArchiveError("snapshot CSV contains an empty record")
            record = _event(line, dataset, resource)
            timestamp = record["event_time"]
            assert isinstance(timestamp, datetime)
            first = timestamp if first is None else min(first, timestamp)
            last = timestamp if last is None else max(last, timestamp)
            records.append(record)
            rows += 1
            if len(records) == chunk_rows:
                _write_records(records, schema, writer)
                records.clear()
        if records:
            _write_records(records, schema, writer)
    finally:
        writer.close()
    if rows == 0 or first is None or last is None:
        raise ArchiveError("snapshot CSV cannot be empty")
    return rows, first, last


def _sort_parquet(source: Path, destination: Path, dataset: DatasetSpec) -> None:
    """Use DuckDB's external sort to order a potentially large snapshot day."""
    ordering = ", ".join(dataset.ordering_columns)
    with duckdb.connect() as connection:
        relation = connection.read_parquet(str(source)).order(ordering)
        relation.write_parquet(str(destination), compression="zstd")


def ingest_order_book(
    client: httpx.Client,
    resource: Resource,
    dataset: DatasetSpec,
    destination: Path,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.5,
    chunk_rows: int = 10_000,
    max_archive_bytes: int = 2 * 1024 * 1024 * 1024,
    max_snapshot_bytes: int = 16 * 1024 * 1024 * 1024,
    max_json_line_bytes: int = 1024 * 1024,
) -> IngestedResource:
    """Download and convert one verified KuCoin snapshot archive atomically.

    Args:
        client: The HTTPX client used for source files.
        resource: The daily archive and MD5 checksum URL.
        dataset: The nested Spot or Futures snapshot declaration.
        destination: The final Parquet path.
        timeout: The timeout for each HTTP attempt in seconds.
        retries: The retries after the first HTTP attempt.
        backoff: The initial exponential retry delay in seconds.
        chunk_rows: Snapshot rows written per temporary Parquet batch.
        max_archive_bytes: The largest accepted compressed ZIP.
        max_snapshot_bytes: The largest accepted uncompressed member.
        max_json_line_bytes: The largest accepted JSON snapshot record.

    Returns:
        Integrity, row-count, and timestamp metadata for the Parquet file.
    """
    if dataset.name != "order_book_snapshots":
        raise ValueError("snapshot ingestion requires an order-book dataset")
    chunk_rows = _positive_integer(chunk_rows, "chunk_rows")
    max_archive_bytes = _positive_integer(max_archive_bytes, "max_archive_bytes")
    max_snapshot_bytes = _positive_integer(max_snapshot_bytes, "max_snapshot_bytes")
    max_json_line_bytes = _positive_integer(max_json_line_bytes, "max_json_line_bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    try:
        with TemporaryDirectory(prefix="veldra-market-data-") as directory:
            temporary = Path(directory)
            archive_path = temporary / "source.zip"
            unsorted = temporary / "unsorted.parquet"
            archive_digest = download(
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
                    member = _member(archive, resource, max_snapshot_bytes)
                    with archive.open(member) as source:
                        rows, first, last = _write_unsorted(
                            source,
                            resource,
                            dataset,
                            unsorted,
                            chunk_rows,
                            max_json_line_bytes,
                        )
            except zipfile.BadZipFile as error:
                raise ArchiveError("source file is not a valid ZIP archive") from error
            _sort_parquet(unsorted, partial, dataset)
        partial.replace(destination)
        stat = destination.stat()
        return IngestedResource(
            archive_checksum=archive_digest,
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
        raise
