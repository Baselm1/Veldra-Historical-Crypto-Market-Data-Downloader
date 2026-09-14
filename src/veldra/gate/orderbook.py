"""Ingest Gate hourly order-book archives into daily Parquet files."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import httpx
import pyarrow.parquet as pq

from veldra.core.datasets import DatasetSpec
from veldra.core.download import head
from veldra.core.ingest import ArchiveError, ingest_gzip_archive
from veldra.core.models import IngestedResource, IntegritySpec, Resource
from veldra.gate.processing import normalize_chunk, validate_chunk

_HOUR_URL = re.compile(r"^(?P<prefix>.+)(?P<hour>[0-9]{2})(?P<suffix>\.csv\.gz|\.gz)$")
_MD5 = re.compile(r"[0-9a-fA-F]{32}")


def _hour_url(url: str, hour: int) -> str:
    """Replace the final hour in one Gate archive URL.

    Args:
        url: The logical day URL ending in hour zero.
        hour: The zero-based source hour.

    Returns:
        The exact hourly source URL.
    """
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


def _ingest_hour(
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


def _combine(
    paths: list[Path],
    metadata: list[IngestedResource],
    destination: Path,
    dataset: DatasetSpec,
) -> IngestedResource:
    """Append ordered hourly Parquet files into one atomic daily file."""
    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for path in paths:
            table = pq.read_table(path)
            if tuple(table.column_names) != dataset.stored_columns:
                raise ArchiveError("hourly Gate Parquet schemas do not match")
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
            writer.write_table(table)
        if writer is None:
            raise ArchiveError("Gate order-book day contains no hourly files")
        writer.close()
        writer = None
        partial.replace(destination)
    except BaseException:
        if writer is not None:
            writer.close()
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
    if dataset.name == "order_book_snapshots":
        raise NotImplementedError("Gate snapshot ingestion is not implemented yet")
    if dataset.name != "order_book_updates":
        raise ValueError("Gate order-book ingestion requires an order-book dataset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="veldra-gate-depth-") as directory:
        children = [
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
        if not children:
            raise ArchiveError("Gate order-book day contains no hourly files")
        paths = [
            Path(directory) / f"{index:02d}.parquet" for index in range(len(children))
        ]
        with ThreadPoolExecutor(max_workers=min(8, len(children))) as pool:
            futures = [
                pool.submit(
                    _ingest_hour,
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
