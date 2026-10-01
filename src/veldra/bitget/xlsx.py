"""Read and ingest verified Bitget ZIP/XLSX archives with strict limits."""

from collections.abc import Callable, Generator, Iterator
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any, cast
import xml.etree.ElementTree as ET
import zipfile

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from veldra.core.datasets import CsvSchema, DatasetSpec
from veldra.core.download import download
from veldra.core.ingest import ArchiveError
from veldra.core.models import IngestedResource, Resource

type Normalizer = Callable[[Any, DatasetSpec, float | None], Any]
type Validator = Callable[
    [Any, DatasetSpec, date, datetime | None, date | None], datetime
]

_CELL_REFERENCE = re.compile(r"([A-Z]+)[1-9][0-9]*")
_MAX_COMPRESSION_RATIO = 1_000


def _positive_integer(value: object, name: str) -> int:
    """Return one valid positive safety setting.

    Args:
        value: Proposed setting.
        name: Setting name used in validation errors.

    Returns:
        Validated positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _safe_member(member: zipfile.ZipInfo, limit: int, suffix: str) -> None:
    """Reject encrypted, oversized, suspicious, or wrongly typed members.

    Args:
        member: ZIP member metadata.
        limit: Maximum accepted expanded bytes.
        suffix: Required lowercase filename suffix.
    """
    if member.is_dir() or not member.filename.lower().endswith(suffix):
        raise ArchiveError(f"archive member must be a {suffix} file")
    if member.flag_bits & 1:
        raise ArchiveError("encrypted archive members are unsupported")
    if member.file_size < 1 or member.file_size > limit:
        raise ArchiveError("archive member size exceeds configured limit")
    compressed = max(member.compress_size, 1)
    if member.file_size / compressed > _MAX_COMPRESSION_RATIO:
        raise ArchiveError("archive member compression ratio is unsafe")


def _copy_member(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path, limit: int
) -> None:
    """Copy one verified ZIP member without extracting its path.

    Args:
        archive: Open source archive.
        member: Safe member selected by metadata.
        destination: Explicit temporary output path.
        limit: Maximum copied expanded bytes.
    """
    copied = 0
    with archive.open(member) as source, destination.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            copied += len(chunk)
            if copied > limit:
                raise ArchiveError("archive member size exceeds configured limit")
            output.write(chunk)


def extract_workbook(archive_path: Path, destination: Path, max_bytes: int) -> None:
    """Extract the sole safe workbook from one verified outer ZIP.

    Args:
        archive_path: Downloaded source ZIP.
        destination: Explicit temporary workbook path.
        max_bytes: Maximum expanded workbook size.
    """
    max_bytes = _positive_integer(max_bytes, "max_workbook_bytes")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) != 1:
                raise ArchiveError("Bitget archive must contain exactly one workbook")
            member = members[0]
            _safe_member(member, max_bytes, ".xlsx")
            _copy_member(archive, member, destination, max_bytes)
    except zipfile.BadZipFile as error:
        raise ArchiveError("source file is not a valid ZIP archive") from error


def _xml_members(
    archive: zipfile.ZipFile, max_xml_bytes: int
) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo | None]:
    """Return the sole worksheet and optional shared-string table.

    Args:
        archive: Open XLSX package.
        max_xml_bytes: Maximum expanded XML member size.

    Returns:
        Worksheet member and optional shared-string member.
    """
    sheets = [
        member
        for member in archive.infolist()
        if member.filename.startswith("xl/worksheets/")
        and member.filename.endswith(".xml")
        and not member.is_dir()
    ]
    if len(sheets) != 1:
        raise ArchiveError("Bitget workbook must contain exactly one worksheet")
    sheet = sheets[0]
    _safe_member(sheet, max_xml_bytes, ".xml")
    shared = (
        archive.getinfo("xl/sharedStrings.xml")
        if "xl/sharedStrings.xml" in archive.namelist()
        else None
    )
    if shared is not None:
        _safe_member(shared, max_xml_bytes, ".xml")
    return sheet, shared


def _tag(element: ET.Element) -> str:
    """Return an XML element's namespace-free local name.

    Args:
        element: Parsed XML node.

    Returns:
        Namespace-free element name.
    """
    return element.tag.rsplit("}", 1)[-1]


def _strings(archive: zipfile.ZipFile, member: zipfile.ZipInfo | None) -> list[str]:
    """Return optional XLSX shared strings in source order.

    Args:
        archive: Open XLSX package.
        member: Optional shared-string XML member.

    Returns:
        Decoded shared strings.
    """
    if member is None:
        return []
    values: list[str] = []
    with archive.open(member) as source:
        for _event, element in ET.iterparse(source, events=("end",)):
            if _tag(element) == "si":
                values.append(
                    "".join(
                        child.text or ""
                        for child in element.iter()
                        if _tag(child) == "t"
                    )
                )
                element.clear()
    return values


def _column(reference: object) -> int:
    """Return the zero-based column index from one XLSX cell reference.

    Args:
        reference: Cell reference such as ``A2`` or ``AA10``.

    Returns:
        Zero-based worksheet column index.
    """
    if not isinstance(reference, str):
        raise ArchiveError("worksheet cell reference is missing")
    match = _CELL_REFERENCE.fullmatch(reference)
    if match is None:
        raise ArchiveError("worksheet cell reference is invalid")
    number = 0
    for character in match.group(1):
        number = number * 26 + ord(character) - ord("A") + 1
    return number - 1


def _cell(cell: ET.Element, shared: list[str]) -> str | None:
    """Decode one XLSX cell as nullable source text.

    Args:
        cell: Parsed cell XML element.
        shared: Workbook shared-string table.

    Returns:
        Decoded text or ``None`` for a blank cell.
    """
    kind = cell.attrib.get("t")
    if kind == "inlineStr":
        return "".join(child.text or "" for child in cell.iter() if _tag(child) == "t")
    value = next((child.text for child in cell if _tag(child) == "v"), None)
    if value is None:
        return None
    if kind == "s":
        try:
            return shared[int(value)]
        except (ValueError, IndexError) as error:
            raise ArchiveError("worksheet shared-string index is invalid") from error
    if kind == "b":
        return "true" if value == "1" else "false"
    return value


def _worksheet_rows(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, shared: list[str]
) -> Iterator[list[str | None]]:
    """Yield sparse-safe worksheet rows without loading the sheet XML.

    Args:
        archive: Open XLSX package.
        member: Sole worksheet XML member.
        shared: Workbook shared-string table.

    Yields:
        Worksheet values aligned by their cell references.
    """
    with archive.open(member) as source:
        for _event, element in ET.iterparse(source, events=("end",)):
            if _tag(element) != "row":
                continue
            cells = [child for child in element if _tag(child) == "c"]
            values: list[str | None] = []
            for cell in cells:
                index = _column(cell.attrib.get("r"))
                if index < len(values):
                    raise ArchiveError("worksheet cells are not in column order")
                values.extend([None] * (index - len(values)))
                values.append(_cell(cell, shared))
            element.clear()
            yield values


def _schema(header: tuple[str | None, ...], dataset: DatasetSpec) -> CsvSchema:
    """Resolve a declared source schema from one workbook header.

    Args:
        header: First worksheet row.
        dataset: Dataset declaring accepted source layouts.

    Returns:
        Matching source schema.
    """
    if any(value is None for value in header):
        raise ArchiveError("worksheet header contains blank cells")
    names = tuple(cast(str, value) for value in header)
    for schema in dataset.csv_schemas:
        if schema.header == "present" and schema.columns == names:
            return schema
    raise ArchiveError("worksheet does not match a declared source schema")


def workbook_tables(
    workbook_path: Path,
    dataset: DatasetSpec,
    *,
    chunk_rows: int = 200_000,
    max_xml_bytes: int = 8 * 1024 * 1024 * 1024,
) -> Iterator[pa.Table]:
    """Yield source-text Arrow tables from one safe XLSX workbook.

    Args:
        workbook_path: Verified temporary XLSX path.
        dataset: Dataset declaring accepted headers.
        chunk_rows: Maximum records per yielded table.
        max_xml_bytes: Maximum expanded XML member size.

    Yields:
        Arrow source tables with a stable hidden physical row number.
    """
    chunk_rows = _positive_integer(chunk_rows, "chunk_rows")
    max_xml_bytes = _positive_integer(max_xml_bytes, "max_xml_bytes")
    try:
        with zipfile.ZipFile(workbook_path) as archive:
            sheet, shared_member = _xml_members(archive, max_xml_bytes)
            shared = _strings(archive, shared_member)
            rows = _worksheet_rows(archive, sheet, shared)
            try:
                header = tuple(next(rows))
            except StopIteration as error:
                raise ArchiveError("worksheet cannot be empty") from error
            schema = _schema(header, dataset)
            pending: list[list[str | None]] = []
            row_number = 0
            for row in rows:
                if len(row) > len(schema.columns):
                    raise ArchiveError("worksheet row has too many cells")
                row.extend([None] * (len(schema.columns) - len(row)))
                pending.append(row)
                if len(pending) == chunk_rows:
                    yield _table(schema.columns, pending, row_number)
                    row_number += len(pending)
                    pending = []
            if pending:
                yield _table(schema.columns, pending, row_number)
    except zipfile.BadZipFile as error:
        raise ArchiveError("archive member is not a valid XLSX workbook") from error
    except ET.ParseError as error:
        raise ArchiveError("workbook contains malformed XML") from error


def _table(
    columns: tuple[str, ...], rows: list[list[str | None]], offset: int
) -> pa.Table:
    """Build one source table from decoded worksheet rows.

    Args:
        columns: Declared source column names.
        rows: Decoded data rows.
        offset: Zero-based physical data-row offset.

    Returns:
        String-valued Arrow table with ``__row_number``.
    """
    values = {
        name: pa.array([row[index] for row in rows], type=pa.string())
        for index, name in enumerate(columns)
    }
    values["__row_number"] = pa.array(
        range(offset, offset + len(rows)), type=pa.int64()
    )
    return pa.table(values)


def _bounds(table: pa.Table, column: str) -> tuple[datetime, datetime]:
    """Return nonempty timestamp bounds from one canonical table.

    Args:
        table: Normalized canonical table.
        column: Primary UTC timestamp field.

    Returns:
        Minimum and maximum timestamps.
    """
    bounds = pc.min_max(table[column]).as_py()
    first = bounds["min"]
    last = bounds["max"]
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        raise ArchiveError("normalized timestamps cannot be empty")
    return first, last


def ingest_xlsx_archive(
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
    max_workbook_bytes: int = 8 * 1024 * 1024 * 1024,
    max_xml_bytes: int = 8 * 1024 * 1024 * 1024,
) -> IngestedResource:
    """Download, verify, normalize, and store one Bitget XLSX archive.

    Args:
        client: Shared HTTPX archive client.
        resource: Discovered daily archive.
        dataset: Canonical dataset declaration.
        destination: Final Parquet path.
        normalizer: Exchange-specific Arrow conversion.
        validator: Exchange-specific canonical validation.
        timeout: Per-attempt download timeout.
        retries: Retries following the first attempt.
        backoff: Initial retry delay.
        chunk_rows: Worksheet records normalized together.
        max_archive_bytes: Largest accepted outer ZIP.
        max_workbook_bytes: Largest accepted expanded XLSX.
        max_xml_bytes: Largest accepted expanded XLSX XML member.

    Returns:
        Source integrity and local Parquet metadata.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial.unlink(missing_ok=True)
    try:
        with TemporaryDirectory(prefix="veldra-bitget-") as directory:
            temporary = Path(directory)
            archive_path = temporary / "source.zip"
            workbook_path = temporary / "source.xlsx"
            digest = download(
                client,
                resource,
                archive_path,
                timeout=timeout,
                retries=retries,
                backoff=backoff,
                max_bytes=max_archive_bytes,
            )
            extract_workbook(archive_path, workbook_path, max_workbook_bytes)
            iterator = cast(
                Generator[pa.Table, None, None],
                workbook_tables(
                    workbook_path,
                    dataset,
                    chunk_rows=chunk_rows,
                    max_xml_bytes=max_xml_bytes,
                ),
            )
            with closing(iterator):
                tables = [
                    normalizer(table, dataset, resource.contract_size)
                    for table in iterator
                ]
            if not tables:
                raise ArchiveError("worksheet contains no data rows")
            table = pa.concat_tables(tables).sort_by(
                [(column, "ascending") for column in dataset.ordering_columns]
            )
            validator(table, dataset, resource.day, None, resource.end_day)
            first, last = _bounds(table, dataset.time_column)
            pq.write_table(table, partial, compression="zstd", use_dictionary=False)
        partial.replace(destination)
        stat = destination.stat()
        return IngestedResource(
            archive_checksum=digest,
            parquet_size=stat.st_size,
            parquet_mtime_ns=stat.st_mtime_ns,
            row_count=table.num_rows,
            first_timestamp=first,
            last_timestamp=last,
            timestamp_column=dataset.time_column,
            schema_version=dataset.schema_version,
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
