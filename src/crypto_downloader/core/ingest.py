"""Convert verified source ZIP archives into atomic Parquet files."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit
import zipfile

import httpx
from collections.abc import Callable
from datetime import date, datetime
from typing import Any
import pyarrow.csv as csv
import pyarrow as pa
import pyarrow.parquet as pq

from crypto_downloader.core.datasets import DatasetSpec
from crypto_downloader.core.download import download
from crypto_downloader.core.models import (
    DataValidationError,
    IngestedResource,
    Resource,
)

type Normalizer = Callable[[Any, DatasetSpec, float | None], Any]
type Validator = Callable[
    [Any, DatasetSpec, date, datetime | None, date | None], datetime
]

LOGGER = logging.getLogger(__name__)


class ArchiveError(DataValidationError):
    """Report an unsafe or malformed source archive."""


def _positive_integer(value: object, name: str) -> int:
    """Return a positive integer setting or reject it.

    Args:
        value: The proposed setting value.
        name: The setting name used in errors.

    Returns:
        The validated positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _member(
    archive: zipfile.ZipFile, resource: Resource, max_bytes: int
) -> zipfile.ZipInfo:
    """Return the single safe expected CSV member from an archive.

    Args:
        archive: The opened source ZIP file.
        resource: The source resource that determines the expected filename.
        max_bytes: The largest accepted uncompressed member.

    Returns:
        The validated CSV member metadata.
    """
    members = archive.infolist()
    archive_name = unquote(Path(urlsplit(resource.url).path).name)
    expected = archive_name.removesuffix(".zip") + ".csv"
    if (
        len(members) != 1
        or members[0].is_dir()
        or members[0].filename != expected
        or "/" in members[0].filename
        or "\\" in members[0].filename
        or not members[0].filename.endswith(".csv")
    ):
        raise ArchiveError("ZIP must contain only the exact expected CSV file")
    member = members[0]
    if member.flag_bits & 1:
        raise ArchiveError("encrypted ZIP members are not supported")
    if member.file_size > max_bytes:
        raise ArchiveError("uncompressed CSV size exceeds configured limit")
    return member


def _write_chunks(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    resource: Resource,
    dataset: DatasetSpec,
    partial: Path,
    chunk_rows: int,
    normalizer: Normalizer,
    validator: Validator,
) -> tuple[int, datetime, datetime]:
    """Stream a CSV through its source normalizer into one Parquet file.

    Args:
        archive: Opened ZIP archive.
        member: Verified CSV member.
        resource: Physical source archive and date bounds.
        dataset: Column and type declaration.
        partial: Temporary Parquet destination.
        chunk_rows: Maximum rows passed to validation at once.
        normalizer: Exchange-specific Arrow conversion function.
        validator: Exchange-specific Arrow validation function.

    Returns:
        Row count and first/last UTC timestamps.
    """
    writer: pq.ParquetWriter | None = None
    rows = 0
    first: datetime | None = None
    previous: datetime | None = None
    read_options = csv.ReadOptions(
        column_names=(
            list(dataset.source_columns) if dataset.csv_header == "absent" else None
        ),
        block_size=max(1024, min(chunk_rows * 128, 8 * 1024 * 1024)),
        use_threads=False,
    )
    convert_options = csv.ConvertOptions(
        column_types={name: pa.string() for name in dataset.source_columns},
        strings_can_be_null=True,
        null_values=[""],
    )
    try:
        with archive.open(member, "r") as source:
            reader = csv.open_csv(
                source, read_options=read_options, convert_options=convert_options
            )
            if tuple(reader.schema.names) != dataset.source_columns:
                raise ArchiveError(
                    "CSV does not match the expected source columns or field count"
                )
            for batch in reader:
                for offset in range(0, batch.num_rows, chunk_rows):
                    raw = pa.Table.from_batches([batch.slice(offset, chunk_rows)])
                    table = normalizer(raw, dataset, resource.contract_size)
                    previous = validator(
                        table,
                        dataset,
                        resource.day,
                        previous,
                        getattr(resource, "end_day", None),
                    )
                    if first is None:
                        first = table[dataset.time_column][0].as_py()
                    if writer is None:
                        writer = pq.ParquetWriter(
                            partial,
                            table.schema,
                            compression="zstd",
                            use_dictionary=False,
                        )
                    writer.write_table(table)
                    rows += table.num_rows
    except pa.ArrowException as error:
        raise ArchiveError(f"invalid or empty CSV: {error}") from error
    finally:
        if writer is not None:
            writer.close()
    if first is None or previous is None:
        raise ArchiveError("CSV cannot be empty")
    return rows, first, previous


def ingest_archive(
    client: httpx.Client,
    resource: Resource,
    dataset: DatasetSpec,
    destination: Path,
    *,
    normalizer: Normalizer,
    validator: Validator,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.5,
    chunk_rows: int = 200_000,
    max_archive_bytes: int = 2 * 1024 * 1024 * 1024,
    max_csv_bytes: int = 8 * 1024 * 1024 * 1024,
) -> IngestedResource:
    """Download and convert one verified daily archive.

    Args:
        client: The HTTPX client used to download source files.
        resource: The daily archive and checksum URLs.
        dataset: The schema used to interpret source rows.
        destination: The final Parquet path.
        normalizer: Exchange-specific Arrow conversion function.
        validator: Exchange-specific Arrow validation function.
        timeout: The timeout for each HTTP request in seconds.
        retries: The number of retries after the first HTTP attempt.
        backoff: The initial exponential retry delay in seconds.
        chunk_rows: The number of CSV rows normalized at once.
        max_archive_bytes: The largest accepted compressed archive.
        max_csv_bytes: The largest accepted uncompressed CSV member.

    Returns:
        Hashes, file metadata, row count, and timestamp bounds.
    """
    chunk_rows = _positive_integer(chunk_rows, "chunk_rows")
    max_archive_bytes = _positive_integer(max_archive_bytes, "max_archive_bytes")
    max_csv_bytes = _positive_integer(max_csv_bytes, "max_csv_bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    LOGGER.debug(
        "Archive ingestion started: day=%s url=%s destination=%s chunk_rows=%d",
        resource.day,
        resource.url,
        destination,
        chunk_rows,
    )

    try:
        with TemporaryDirectory(prefix="crypto-downloader-") as directory:
            archive_path = Path(directory) / "source.zip"
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
                with zipfile.ZipFile(archive_path) as archive:
                    member = _member(archive, resource, max_csv_bytes)
                    rows, first, last = _write_chunks(
                        archive,
                        member,
                        resource,
                        dataset,
                        partial,
                        chunk_rows,
                        normalizer,
                        validator,
                    )
            except zipfile.BadZipFile as error:
                raise ArchiveError("source file is not a valid ZIP archive") from error

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
            "Archive ingestion complete: day=%s rows=%d first=%s last=%s "
            "bytes=%d path=%s",
            resource.day,
            rows,
            metadata.first_timestamp,
            metadata.last_timestamp,
            stat.st_size,
            destination,
        )
        return metadata
    except BaseException:
        partial.unlink(missing_ok=True)
        LOGGER.exception(
            "Archive ingestion failed: day=%s url=%s destination=%s",
            resource.day,
            resource.url,
            destination,
        )
        raise
