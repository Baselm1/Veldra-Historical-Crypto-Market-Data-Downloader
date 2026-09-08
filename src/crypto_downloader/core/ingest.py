"""Convert verified source ZIP archives into atomic Parquet files."""

import csv as text_csv
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit
import zipfile

import httpx
from collections.abc import Callable, Iterable, Iterator
from datetime import date, datetime
from typing import Any
import pyarrow.csv as csv
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from crypto_downloader.core.datasets import CsvSchema, DatasetSpec
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


def _source_schema(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, dataset: DatasetSpec
) -> CsvSchema:
    """Select a declared CSV schema from the archive's first row.

    Args:
        archive: The verified ZIP archive.
        member: The verified single CSV member.
        dataset: The dataset declaring accepted source layouts.

    Returns:
        The structurally matching source CSV schema.
    """
    with archive.open(member, "r") as source:
        first_line = source.readline()
    try:
        fields = next(text_csv.reader([first_line.decode("utf-8-sig")]))
    except (UnicodeDecodeError, StopIteration, text_csv.Error) as error:
        raise ArchiveError("CSV has no readable first row") from error
    present = [
        schema
        for schema in dataset.csv_schemas
        if schema.header == "present" and tuple(fields) == schema.columns
    ]
    if present:
        return present[0]
    absent = [
        schema
        for schema in dataset.csv_schemas
        if schema.header == "absent" and len(fields) == len(schema.columns)
    ]
    if absent:
        return absent[0]
    raise ArchiveError("CSV does not match the expected source columns or field count")


def _timestamp_bounds(table: Any, column: str) -> tuple[datetime, datetime]:
    """Return the true minimum and maximum timestamp in one Arrow table.

    Args:
        table: The normalized Arrow table.
        column: The canonical timestamp column.

    Returns:
        The minimum and maximum timestamps present in the table.
    """
    bounds = pc.min_max(table[column]).as_py()
    first = bounds["min"]
    last = bounds["max"]
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        raise ArchiveError("normalized timestamps cannot be empty")
    return first, last


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
    schema = _source_schema(archive, member, dataset)
    try:
        with archive.open(member, "r") as source:
            reader = _csv_reader(source, schema, chunk_rows)
            tables = _normalized_tables(
                reader, dataset, resource, chunk_rows, normalizer
            )
            if dataset.sort_source_rows:
                return _write_sorted(tables, resource, dataset, partial, validator)
            return _write_ordered(tables, resource, dataset, partial, validator)
    except pa.ArrowException as error:
        raise ArchiveError(f"invalid or empty CSV: {error}") from error


def _csv_reader(source: Any, schema: CsvSchema, chunk_rows: int) -> Any:
    """Open one Arrow streaming reader for a selected source schema.

    Args:
        source: The open binary CSV stream.
        schema: The structurally selected source schema.
        chunk_rows: The normalization chunk size.

    Returns:
        A validated Arrow streaming CSV reader.
    """
    read_options = csv.ReadOptions(
        column_names=list(schema.columns) if schema.header == "absent" else None,
        block_size=max(1024, min(chunk_rows * 128, 8 * 1024 * 1024)),
        use_threads=False,
    )
    convert_options = csv.ConvertOptions(
        column_types={name: pa.string() for name in schema.columns},
        strings_can_be_null=True,
        null_values=[""],
    )
    reader = csv.open_csv(
        source, read_options=read_options, convert_options=convert_options
    )
    if tuple(reader.schema.names) != schema.columns:
        raise ArchiveError(
            "CSV does not match the expected source columns or field count"
        )
    return reader


def _normalized_tables(
    reader: Iterable[Any],
    dataset: DatasetSpec,
    resource: Resource,
    chunk_rows: int,
    normalizer: Normalizer,
) -> Iterator[Any]:
    """Yield normalized tables no larger than the configured chunk size.

    Args:
        reader: The Arrow record batches read from the CSV.
        dataset: The canonical dataset declaration.
        resource: The physical archive providing source context.
        chunk_rows: The largest normalized table size.
        normalizer: The exchange-specific Arrow conversion function.

    Yields:
        Canonical Arrow tables.
    """
    for batch in reader:
        for offset in range(0, batch.num_rows, chunk_rows):
            raw = pa.Table.from_batches([batch.slice(offset, chunk_rows)])
            yield normalizer(raw, dataset, resource.contract_size)


def _write_ordered(
    tables: Iterable[Any],
    resource: Resource,
    dataset: DatasetSpec,
    partial: Path,
    validator: Validator,
) -> tuple[int, datetime, datetime]:
    """Validate and stream already ordered canonical tables to Parquet.

    Args:
        tables: Canonical source-order tables.
        resource: The physical source archive.
        dataset: The canonical dataset declaration.
        partial: The temporary Parquet path.
        validator: The exchange-specific validation function.

    Returns:
        Row count and true minimum/maximum timestamps.
    """
    writer: pq.ParquetWriter | None = None
    rows = 0
    first: datetime | None = None
    last: datetime | None = None
    previous: datetime | None = None
    try:
        for table in tables:
            previous = validator(
                table,
                dataset,
                resource.day,
                previous,
                resource.end_day,
            )
            chunk_first, chunk_last = _timestamp_bounds(table, dataset.time_column)
            first = chunk_first if first is None else min(first, chunk_first)
            last = chunk_last if last is None else max(last, chunk_last)
            if writer is None:
                writer = pq.ParquetWriter(
                    partial,
                    table.schema,
                    compression="zstd",
                    use_dictionary=False,
                )
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if first is None or last is None or previous is None:
        raise ArchiveError("CSV cannot be empty")
    return rows, first, last


def _write_sorted(
    tables: Iterable[Any],
    resource: Resource,
    dataset: DatasetSpec,
    partial: Path,
    validator: Validator,
) -> tuple[int, datetime, datetime]:
    """Sort an unordered archive before validation and Parquet storage.

    Args:
        tables: Canonical tables in arbitrary source order.
        resource: The physical source archive.
        dataset: The canonical dataset declaration.
        partial: The temporary Parquet path.
        validator: The exchange-specific validation function.

    Returns:
        Row count and true minimum/maximum timestamps.
    """
    pending = list(tables)
    if not pending:
        raise ArchiveError("CSV cannot be empty")
    table = pa.concat_tables(pending).sort_by(
        [(column, "ascending") for column in dataset.ordering_columns]
    )
    validator(table, dataset, resource.day, None, resource.end_day)
    first, last = _timestamp_bounds(table, dataset.time_column)
    pq.write_table(
        table,
        partial,
        compression="zstd",
        use_dictionary=False,
    )
    return table.num_rows, first, last


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
