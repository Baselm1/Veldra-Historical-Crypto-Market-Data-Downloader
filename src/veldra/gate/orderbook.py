"""Ingest Gate hourly order-book archives into daily Parquet files."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any, Callable

import duckdb
import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from veldra.core.datasets import DatasetSpec
from veldra.core.download import download, head
from veldra.core.ingest import ArchiveError, ingest_gzip_archive
from veldra.core.models import IngestedResource, IntegritySpec, Resource
from veldra.gate.processing import normalize_chunk, validate_chunk

type HourIngester = Callable[..., IngestedResource]

_HOUR_URL = re.compile(r"^(?P<prefix>.+)(?P<hour>[0-9]{2})(?P<suffix>\.csv\.gz|\.gz)$")
_MD5 = re.compile(r"[0-9a-fA-F]{32}")


def _hour_url(url: str, hour: int) -> str:
    """Replace the final hour in one Gate archive URL."""
    match = _HOUR_URL.fullmatch(url)
    if match is None or not 0 <= hour <= 23:
        raise ValueError("Gate order-book URL or hour is invalid")
    return f"{match.group('prefix')}{hour:02d}{match.group('suffix')}"


def _integrity(response: httpx.Response) -> IntegritySpec:
    """Return Gate's plain ETag policy or Gzip-only validation fallback."""
    value = response.headers.get("ETag", "").strip().strip('"')
    if _MD5.fullmatch(value) is not None:
        return IntegritySpec("response_header", "md5", expected=value.lower())
    return IntegritySpec("archive_only")


def _hour_resource(
    client: httpx.Client,
    resource: Resource,
    hour: int,
    *,
    timeout: float,
    retries: int,
    backoff: float,
) -> Resource | None:
    """Return one existing hourly child with its available integrity metadata."""
    url = _hour_url(resource.url, hour)
    try:
        response = head(
            client,
            url,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
    except httpx.HTTPStatusError as error:
        if error.response.status_code in {404, 410}:
            return None
        raise
    return replace(resource, url=url, integrity=_integrity(response))


def _ingest_update_hour(
    client: httpx.Client,
    resource: Resource,
    dataset: DatasetSpec,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    backoff: float,
) -> IngestedResource:
    """Normalize one hourly Gate depth-update CSV into temporary Parquet."""
    return ingest_gzip_archive(
        client,
        resource,
        dataset,
        destination,
        normalizer=normalize_chunk,
        validator=validate_chunk,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
    )


def _timestamp(value: object, field: str) -> datetime:
    """Convert Gate second or millisecond JSON timestamps exactly to UTC."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ArchiveError(f"invalid Gate snapshot {field}")
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ArchiveError(f"invalid Gate snapshot {field}") from error
    if not number.is_finite() or number <= 0:
        raise ArchiveError(f"invalid Gate snapshot {field}")
    scale = 1_000 if number >= 100_000_000_000 else 1_000_000
    try:
        return datetime.fromtimestamp(int(number * scale) / 1_000_000, UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise ArchiveError(f"invalid Gate snapshot {field}") from error


def _level(value: object, *, spot: bool) -> dict[str, float]:
    """Normalize one Spot or Futures snapshot level."""
    if spot and isinstance(value, list) and len(value) == 2:
        price_value, quantity_value = value
    elif not spot and isinstance(value, dict):
        price_value, quantity_value = value.get("p"), value.get("s")
    else:
        raise ArchiveError("invalid Gate snapshot level")
    try:
        price = float(price_value)
        quantity = abs(float(quantity_value))
    except (TypeError, ValueError) as error:
        raise ArchiveError("invalid Gate snapshot level") from error
    if not math.isfinite(price) or price <= 0 or not math.isfinite(quantity):
        raise ArchiveError("invalid Gate snapshot level")
    return {"price": price, "quantity": quantity}


def _levels(value: object, *, spot: bool) -> list[dict[str, float]]:
    """Normalize one complete side of a Gate order-book snapshot."""
    if not isinstance(value, list):
        raise ArchiveError("invalid Gate snapshot levels")
    return [_level(level, spot=spot) for level in value]


def _snapshot_row(value: object, *, spot: bool) -> dict[str, Any]:
    """Normalize one Gate JSON object into a complete book state."""
    if not isinstance(value, dict):
        raise ArchiveError("invalid Gate snapshot row")
    identifier = value.get("id")
    if (
        isinstance(identifier, bool)
        or not isinstance(identifier, int)
        or identifier < 0
    ):
        raise ArchiveError("invalid Gate snapshot id")
    return {
        "event_time": _timestamp(value.get("current"), "current time"),
        "update_time": _timestamp(value.get("update"), "update time"),
        "update_id": identifier,
        "bids": _levels(value.get("bids"), spot=spot),
        "asks": _levels(value.get("asks"), spot=spot),
    }


def _snapshot_table(source: Path, dataset: DatasetSpec) -> pa.Table:
    """Read one compressed Gate JSON Lines snapshot archive."""
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            rows = [
                _snapshot_row(json.loads(line), spot=dataset.product == "spot")
                for line in stream
                if line.strip()
            ]
    except (
        gzip.BadGzipFile,
        EOFError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise ArchiveError("invalid Gate snapshot Gzip or JSON Lines") from error
    if not rows:
        raise ArchiveError("Gate snapshot archive contains no rows")
    level_type = pa.list_(
        pa.struct([pa.field("price", pa.float64()), pa.field("quantity", pa.float64())])
    )
    table = pa.table(
        {
            "event_time": pa.array(
                [row["event_time"] for row in rows], pa.timestamp("us", "UTC")
            ),
            "update_time": pa.array(
                [row["update_time"] for row in rows], pa.timestamp("us", "UTC")
            ),
            "update_id": pa.array([row["update_id"] for row in rows], pa.int64()),
            "bids": pa.array([row["bids"] for row in rows], level_type),
            "asks": pa.array([row["asks"] for row in rows], level_type),
        }
    )
    sort_keys = [(column, "ascending") for column in dataset.ordering_columns]
    indices = pa.compute.sort_indices(table, sort_keys=sort_keys)
    return table.take(indices)


def _ingest_snapshot_hour(
    client: httpx.Client,
    resource: Resource,
    dataset: DatasetSpec,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    backoff: float,
) -> IngestedResource:
    """Convert one verified Gate snapshot JSON Lines Gzip into Parquet."""
    archive = destination.with_suffix(".jsonl.gz")
    checksum = download(
        client,
        resource,
        archive,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        max_bytes=2 * 1024 * 1024 * 1024,
    )
    try:
        table = _snapshot_table(archive, dataset)
        first = table["event_time"][0].as_py()
        last = table["event_time"][-1].as_py()
        start, end = resource.coverage
        if first < start or last >= end:
            raise ArchiveError("timestamps fall outside the Gate snapshot day")
        pq.write_table(table, destination, compression="zstd")
        stat = destination.stat()
        return IngestedResource(
            checksum,
            stat.st_size,
            stat.st_mtime_ns,
            table.num_rows,
            first,
            last,
            dataset.time_column,
            dataset.schema_version,
        )
    finally:
        archive.unlink(missing_ok=True)


def _combine(
    paths: list[Path],
    metadata: list[IngestedResource],
    destination: Path,
    dataset: DatasetSpec,
) -> IngestedResource:
    """Merge hourly Parquet files into one globally ordered atomic daily file."""
    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    try:
        _validate_hourly_schemas(paths, dataset)
        _write_ordered_day(paths, partial, dataset.ordering_columns)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    stat = destination.stat()
    checksum = hashlib.sha256(
        "".join(item.archive_checksum for item in metadata).encode()
    ).hexdigest()
    return IngestedResource(
        checksum,
        stat.st_size,
        stat.st_mtime_ns,
        sum(item.row_count for item in metadata),
        min(item.first_timestamp for item in metadata),
        max(item.last_timestamp for item in metadata),
        dataset.time_column,
        dataset.schema_version,
    )


def _validate_hourly_schemas(paths: list[Path], dataset: DatasetSpec) -> None:
    """Confirm every hourly Parquet uses the declared canonical columns.

    Args:
        paths: The temporary hourly Parquet files.
        dataset: The order-book schema expected in every file.
    """
    if not paths:
        raise ArchiveError("Gate order-book day contains no hourly files")
    for path in paths:
        columns = tuple(pq.ParquetFile(path).schema_arrow.names)
        if columns != dataset.stored_columns:
            raise ArchiveError("hourly Gate Parquet schemas do not match")


def _write_ordered_day(
    paths: list[Path], destination: Path, ordering_columns: tuple[str, ...]
) -> None:
    """Use DuckDB's external sort to write one chronological daily Parquet.

    Args:
        paths: The temporary hourly Parquet files.
        destination: The temporary combined output path.
        ordering_columns: The deterministic canonical sort columns.
    """
    ordering = ", ".join(f'"{column}"' for column in ordering_columns)
    connection = duckdb.connect()
    try:
        connection.read_parquet([str(path) for path in paths]).order(
            ordering
        ).write_parquet(str(destination), compression="zstd")
    finally:
        connection.close()


def _children(
    client: httpx.Client,
    resource: Resource,
    *,
    timeout: float,
    retries: int,
    backoff: float,
) -> list[Resource]:
    """Discover every available hourly object inside one logical Gate day."""
    return [
        child
        for child in (
            _hour_resource(
                client,
                resource,
                hour,
                timeout=timeout,
                retries=retries,
                backoff=backoff,
            )
            for hour in range(24)
        )
        if child is not None
    ]


def ingest_order_book_day(
    client: httpx.Client,
    resource: Resource,
    dataset: DatasetSpec,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    backoff: float,
) -> IngestedResource:
    """Download available Gate hours and create one daily order-book Parquet.

    Args:
        client: The shared HTTP client.
        resource: The logical source day whose URL ends in hour zero.
        dataset: The update or snapshot schema.
        destination: The final daily Parquet path.
        timeout: The timeout for each HTTP request in seconds.
        retries: The retries after the first attempt.
        backoff: The initial exponential retry delay in seconds.

    Returns:
        Combined integrity, row count, timestamp, and file metadata.
    """
    if dataset.name not in {"order_book_updates", "order_book_snapshots"}:
        raise ValueError("Gate order-book ingestion requires an order-book dataset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="veldra-gate-depth-") as directory:
        children = _children(
            client,
            resource,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
        if not children:
            raise ArchiveError("Gate order-book day contains no hourly files")
        paths = [
            Path(directory) / f"{index:02d}.parquet" for index in range(len(children))
        ]
        worker: HourIngester = (
            _ingest_snapshot_hour
            if dataset.name == "order_book_snapshots"
            else _ingest_update_hour
        )
        with ThreadPoolExecutor(max_workers=min(8, len(children))) as pool:
            futures = [
                pool.submit(
                    worker,
                    client,
                    child,
                    dataset,
                    path,
                    timeout=timeout,
                    retries=retries,
                    backoff=backoff,
                )
                for child, path in zip(children, paths, strict=True)
            ]
            metadata = [future.result() for future in futures]
        return _combine(paths, metadata, destination, dataset)
