"""Stream deprecated OKX module 6 CSV/GZIP snapshots into Parquet."""

from collections.abc import Mapping, Sequence
import csv
from datetime import UTC, datetime, time, timedelta
import gzip
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import IO, cast

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

type LegacyLevel = dict[str, float | int]
type LegacyRecord = dict[str, object]
type LevelColumns = dict[str, dict[int, dict[str, int]]]

_LEVEL_COLUMN = re.compile(r"^(bid|ask)_(\d+)_(px|qty|ordCnt)$")


def _positive_integer(value: object, name: str) -> int:
    """Return one positive integer safety setting.

    Args:
        value: Candidate setting value.
        name: Setting name used in errors.

    Returns:
        Validated positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _coverage(resource: ArchiveObject) -> tuple[datetime, datetime]:
    """Return one module 6 UTC source-day range.

    Args:
        resource: Physical manifest object.

    Returns:
        Inclusive start and exclusive end timestamps.
    """
    start = datetime.combine(resource.key.period_start, time.min, UTC)
    end = datetime.combine(resource.key.period_end + timedelta(days=1), time.min, UTC)
    return start, end


def _line(source: IO[bytes], maximum: int, *, header: bool = False) -> list[str] | None:
    """Read and decode one bounded source CSV line.

    Args:
        source: Decompressed GZIP byte stream.
        maximum: Maximum bytes accepted for one record.
        header: Whether a UTF-8 byte-order mark is accepted.

    Returns:
        Parsed fields, or ``None`` at end of file.
    """
    raw = source.readline(maximum + 1)
    if not raw:
        return None
    if len(raw) > maximum:
        raise DataValidationError("OKX legacy order-book CSV line exceeds its limit")
    try:
        encoding = "utf-8-sig" if header else "utf-8"
        return next(csv.reader([raw.decode(encoding)], strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise DataValidationError("OKX legacy order-book CSV is invalid") from error


def _header(columns: Sequence[str]) -> tuple[LevelColumns, int, int, int]:
    """Parse a version-1 dynamic module 6 header.

    Args:
        columns: Source CSV column names.

    Returns:
        Level mappings plus time, exchange-time, and symbol positions.
    """
    if len(columns) != len(set(columns)):
        raise DataValidationError("OKX legacy order-book header contains duplicates")
    try:
        time_index = columns.index("timeMs")
        exchange_index = columns.index("exchTimeMs")
        symbol_index = columns.index("symbol")
    except ValueError as error:
        raise DataValidationError(
            "OKX legacy order-book header omits a required field"
        ) from error
    levels: LevelColumns = {"bid": {}, "ask": {}}
    allowed = {"timeMs", "exchTimeMs", "symbol"}
    for index, column in enumerate(columns):
        if column in allowed:
            continue
        matched = _LEVEL_COLUMN.fullmatch(column)
        if matched is None:
            raise DataValidationError(
                f"OKX legacy order-book header has unknown field {column!r}"
            )
        side, number_text, field = matched.groups()
        number = int(number_text)
        levels[side].setdefault(number, {})[field] = index
    bids = sorted(levels["bid"])
    asks = sorted(levels["ask"])
    if not bids or bids != asks or bids != list(range(1, len(bids) + 1)):
        raise DataValidationError("OKX legacy order-book levels are not contiguous")
    if len(bids) > 50 or any(
        set(fields) != {"px", "qty", "ordCnt"}
        for side in levels.values()
        for fields in side.values()
    ):
        raise DataValidationError("OKX legacy order-book level schema is invalid")
    return levels, time_index, exchange_index, symbol_index


def _timestamp(value: str, name: str) -> datetime:
    """Parse one epoch-millisecond snapshot timestamp.

    Args:
        value: Source timestamp text.
        name: Field name used in errors.

    Returns:
        UTC timestamp.
    """
    if not value.isascii() or not value.isdecimal():
        raise DataValidationError(f"OKX legacy {name} must be milliseconds")
    milliseconds = int(value)
    if not 100_000_000_000 <= milliseconds < 100_000_000_000_000:
        raise DataValidationError(f"OKX legacy {name} has an invalid unit")
    try:
        return datetime.fromtimestamp(milliseconds / 1_000, UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise DataValidationError(f"OKX legacy {name} is invalid") from error


def _number(value: str, name: str, *, positive: bool) -> float:
    """Parse one finite snapshot level number.

    Args:
        value: Source numeric text.
        name: Field name used in errors.
        positive: Whether zero is forbidden.

    Returns:
        Finite floating-point value.
    """
    try:
        result = float(value)
    except ValueError as error:
        raise DataValidationError(f"OKX legacy {name} must be numeric") from error
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise DataValidationError(f"OKX legacy {name} is invalid")
    return result


def _count(value: str) -> int:
    """Parse one nonnegative native order count.

    Args:
        value: Source order-count text.

    Returns:
        Nonnegative integer count.
    """
    if not value.isascii() or not value.isdecimal():
        raise DataValidationError("OKX legacy order count must be an integer")
    return int(value)


def _levels(
    row: Sequence[str], columns: Mapping[int, Mapping[str, int]], quantity: str
) -> list[LegacyLevel]:
    """Normalize populated source levels from one side of a snapshot.

    Args:
        row: Parsed source record.
        columns: Dynamic level-number field positions.
        quantity: Canonical quantity field name.

    Returns:
        Ordered populated levels.
    """
    result: list[LegacyLevel] = []
    empty_seen = False
    for number in sorted(columns):
        positions = columns[number]
        values = [row[positions[field]] for field in ("px", "qty", "ordCnt")]
        if not any(values):
            empty_seen = True
            continue
        if empty_seen or not all(values):
            raise DataValidationError("OKX legacy order-book level is incomplete")
        result.append(
            {
                "price": _number(values[0], "price", positive=True),
                quantity: _number(values[1], "quantity", positive=False),
                "order_count": _count(values[2]),
            }
        )
    return result


def _instrument(value: str, resource: ArchiveObject) -> str:
    """Normalize and scope-check one legacy `.OK` symbol.

    Args:
        value: Source symbol text.
        resource: Physical manifest scope.

    Returns:
        Native OKX instrument ID without the source suffix.
    """
    instrument = value.strip().removesuffix(".OK")
    if not instrument or not value.strip().endswith(".OK"):
        raise DataValidationError("OKX legacy symbol is invalid")
    subject = resource.key.subject
    matches = instrument == subject.value
    if subject.kind == "instrument_family":
        matches = instrument == f"{subject.value}-SWAP" or instrument.startswith(
            f"{subject.value}-"
        )
    if not matches:
        raise DataValidationError("OKX legacy symbol does not match its archive")
    return instrument


def _record(
    row: Sequence[str],
    header: tuple[LevelColumns, int, int, int],
    resource: ArchiveObject,
    dataset: DatasetSpec,
    event_number: int,
) -> LegacyRecord:
    """Normalize one dynamic legacy snapshot row.

    Args:
        row: Parsed source record.
        header: Validated dynamic header mapping.
        resource: Physical manifest scope.
        dataset: Product-specific snapshot declaration.
        event_number: Stable physical row sequence.

    Returns:
        Canonical nested snapshot record.
    """
    levels, time_index, exchange_index, symbol_index = header
    event_time = _timestamp(row[time_index], "timeMs")
    exchange_time = _timestamp(row[exchange_index], "exchTimeMs")
    start, end = _coverage(resource)
    if not start <= event_time < end or not start <= exchange_time < end:
        raise DataValidationError("OKX legacy timestamp is outside its source day")
    quantity = "base_quantity" if dataset.product == "spot" else "contract_quantity"
    return {
        "instrument_id": _instrument(row[symbol_index], resource),
        "event_time": event_time,
        "exchange_time": exchange_time,
        "event_number": event_number,
        "bids": _levels(row, levels["bid"], quantity),
        "asks": _levels(row, levels["ask"], quantity),
    }


def _schema(dataset: DatasetSpec) -> pa.Schema:
    """Return the product-specific version-1 Arrow schema.

    Args:
        dataset: Legacy snapshot declaration.

    Returns:
        Nested immutable Arrow schema.
    """
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
            pa.field("exchange_time", pa.timestamp("us", "UTC")),
            pa.field("event_number", pa.int64()),
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
    chunk_rows: int,
    max_line_bytes: int,
) -> tuple[int, datetime, datetime, dict[str, int]]:
    """Stream one dynamic CSV through bounded Arrow row groups.

    Args:
        source: Decompressed GZIP byte stream.
        resource: Physical manifest object.
        dataset: Product-specific snapshot declaration.
        destination: Partial Parquet output path.
        chunk_rows: Maximum snapshots held before a write.
        max_line_bytes: Maximum decompressed bytes per CSV record.

    Returns:
        Row count, timestamp bounds, and per-instrument counts.
    """
    columns = _line(source, max_line_bytes, header=True)
    if columns is None:
        raise DataValidationError("OKX legacy order-book CSV cannot be empty")
    parsed_header = _header(columns)
    schema = _schema(dataset)
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    pending: list[LegacyRecord] = []
    counts: dict[str, int] = {}
    first: datetime | None = None
    last: datetime | None = None

    def flush() -> None:
        """Write and release one bounded snapshot batch."""
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=schema))
            pending.clear()

    try:
        event_number = 0
        while (row := _line(source, max_line_bytes)) is not None:
            if len(row) != len(columns):
                raise DataValidationError("OKX legacy order-book row shape is invalid")
            record = _record(row, parsed_header, resource, dataset, event_number)
            timestamp = record["event_time"]
            assert isinstance(timestamp, datetime)
            instrument = str(record["instrument_id"])
            first = timestamp if first is None else min(first, timestamp)
            last = timestamp if last is None else max(last, timestamp)
            counts[instrument] = counts.get(instrument, 0) + 1
            pending.append(record)
            event_number += 1
            if len(pending) == chunk_rows:
                flush()
        flush()
    finally:
        writer.close()
    if first is None or last is None:
        raise DataValidationError("OKX legacy order-book CSV has no snapshots")
    return sum(counts.values()), first, last, counts


def _partitions(
    resource: ArchiveObject,
    destination: Path,
    dataset: DatasetSpec,
    counts: Mapping[str, int],
) -> tuple[LogicalPartition, ...]:
    """Map streamed legacy symbols to their shared materialization.

    Args:
        resource: Physical manifest object.
        destination: Published Parquet path.
        dataset: Legacy snapshot declaration.
        counts: Per-instrument row counts.

    Returns:
        Exact instrument partitions and optional family partition.
    """
    start, end = _coverage(resource)
    values = [
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
    ]
    if resource.key.remote_scope_kind == "instrument_family":
        values.append(
            LogicalPartition(
                "okx",
                resource.key.product,
                resource.key.dataset,
                resource.key.subject,
                dataset.base_interval,
                start,
                end,
                destination,
                None,
                None,
                sum(counts.values()),
                source_day=resource.key.period_start,
            )
        )
    return tuple(values)


def materialize_legacy_order_book(
    client: httpx.Client,
    resource: ArchiveObject,
    dataset: DatasetSpec,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    backoff: float,
    max_archive_bytes: int,
    chunk_rows: int = 1_000,
    max_line_bytes: int = 128 * 1024,
) -> MaterializedArchive:
    """Download and stream one verified module 6 snapshot archive.

    Args:
        client: Shared HTTP connection pool.
        resource: Physical manifest object.
        dataset: Explicit legacy snapshot declaration.
        destination: Final Parquet path.
        timeout: Per-attempt archive timeout.
        retries: Retries after the first attempt.
        backoff: Initial retry delay.
        max_archive_bytes: Maximum compressed source size.
        chunk_rows: Maximum snapshots held per Arrow write.
        max_line_bytes: Maximum decompressed bytes per CSV row.

    Returns:
        Materialization and queryable logical partitions.
    """
    _positive_integer(chunk_rows, "chunk_rows")
    _positive_integer(max_line_bytes, "max_line_bytes")
    if dataset.name != "legacy_order_book_50":
        raise ValueError("legacy materialization requires the module 6 dataset")
    if not resource.key.remote_name.endswith(".csv.gz"):
        raise DataValidationError("OKX legacy archive must be CSV/GZIP")
    if resource.integrity is None:
        raise DataValidationError("OKX archive has no integrity policy")
    start, end = _coverage(resource)
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
    with TemporaryDirectory(prefix="veldra-okx-legacy-book-") as directory:
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
            with gzip.open(archive_path, "rb") as source:
                rows, first, last, counts = _write(
                    cast(IO[bytes], source),
                    resource,
                    dataset,
                    partial,
                    chunk_rows=chunk_rows,
                    max_line_bytes=max_line_bytes,
                )
        except DataValidationError:
            partial.unlink(missing_ok=True)
            raise
        except (gzip.BadGzipFile, EOFError, OSError) as error:
            partial.unlink(missing_ok=True)
            raise DataValidationError("OKX legacy GZIP archive is invalid") from error
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
