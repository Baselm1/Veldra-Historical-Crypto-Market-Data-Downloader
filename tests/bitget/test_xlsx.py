"""Test safe streaming ingestion of Bitget's nested ZIP/XLSX files."""

from datetime import UTC, date, datetime
import hashlib
import io
from pathlib import Path
import zipfile

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from veldra.bitget.datasets import get_dataset
from veldra.bitget.xlsx import extract_workbook, ingest_xlsx_archive, workbook_tables
from veldra.core.ingest import ArchiveError
from veldra.core.models import IntegritySpec, Resource


def workbook(
    rows: list[list[str]], *, shared: bool = False, sheet_name: str = "sheet1.xml"
) -> bytes:
    """Return one minimal single-sheet XLSX package."""
    strings: list[str] = []
    encoded_rows: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        cells: list[str] = []
        for index, value in enumerate(row):
            column = chr(ord("A") + index)
            if shared:
                strings.append(value)
                content = (
                    f'<c r="{column}{row_number}" t="s"><v>{len(strings)-1}</v></c>'
                )
            else:
                content = f'<c r="{column}{row_number}" t="inlineStr"><is><t>{value}</t></is></c>'
            cells.append(content)
        encoded_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(encoded_rows)}</sheetData></worksheet>'
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"xl/worksheets/{sheet_name}", sheet)
        if shared:
            values = "".join(f"<si><t>{value}</t></si>" for value in strings)
            archive.writestr(
                "xl/sharedStrings.xml",
                '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f"{values}</sst>",
            )
    return output.getvalue()


def outer(payload: bytes, *, name: str = "data.xlsx", extra: bool = False) -> bytes:
    """Return one representative outer Bitget ZIP."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
        if extra:
            archive.writestr("extra.xlsx", payload)
    return output.getvalue()


def malformed_workbook() -> bytes:
    """Return one XLSX ZIP containing malformed worksheet XML."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", "<worksheet><sheetData>")
    return output.getvalue()


def kline_rows() -> list[list[str]]:
    """Return a declared header and two out-of-order source rows."""
    return [
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "basevolume",
            "usdtvolume",
        ],
        ["1735660860", "2", "3", "1", "2.5", "4", "10"],
        ["1735660800", "1", "2", "0.5", "2", "3", "6"],
    ]


def test_workbook_tables_stream_chunks_and_physical_row_numbers(tmp_path: Path) -> None:
    """Decode inline strings into bounded Arrow tables."""
    path = tmp_path / "book.xlsx"
    path.write_bytes(workbook(kline_rows()))
    tables = list(workbook_tables(path, get_dataset("spot", "klines"), chunk_rows=1))
    assert len(tables) == 2
    assert tables[0].column_names[-1] == "__row_number"
    assert tables[0]["timestamp"].to_pylist() == ["1735660860"]
    assert tables[1]["__row_number"].to_pylist() == [1]


def test_shared_strings_are_supported(tmp_path: Path) -> None:
    """Decode workbooks that use the standard shared-string table."""
    path = tmp_path / "book.xlsx"
    path.write_bytes(workbook(kline_rows(), shared=True))
    table = next(workbook_tables(path, get_dataset("spot", "klines")))
    assert table["open"].to_pylist() == ["2", "1"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-a-zip", "valid ZIP"),
        (outer(b"not-xlsx", name="data.csv"), ".xlsx"),
        (outer(workbook(kline_rows()), extra=True), "exactly one"),
    ],
    ids=("invalid", "wrong-member", "ambiguous"),
)
def test_outer_archives_fail_closed(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    """Reject malformed, wrongly typed, or ambiguous outer archives."""
    source = tmp_path / "source.zip"
    source.write_bytes(payload)
    with pytest.raises(ArchiveError, match=message):
        extract_workbook(source, tmp_path / "out.xlsx", 1024 * 1024)


def test_outer_archive_respects_expanded_size_limit(tmp_path: Path) -> None:
    """Reject a workbook larger than the configured safety bound."""
    source = tmp_path / "source.zip"
    source.write_bytes(outer(workbook(kline_rows())))
    with pytest.raises(ArchiveError, match="size"):
        extract_workbook(source, tmp_path / "out.xlsx", 10)


@pytest.mark.parametrize(
    "payload",
    [
        b"invalid",
        malformed_workbook(),
        workbook([["wrong"], ["value"]]),
        workbook([kline_rows()[0], ["1"] * 8]),
    ],
    ids=("invalid", "malformed-xml", "bad-header", "wide-row"),
)
def test_malformed_workbooks_are_rejected(tmp_path: Path, payload: bytes) -> None:
    """Reject invalid ZIPs, headers, XML packages, and wide rows."""
    path = tmp_path / "book.xlsx"
    path.write_bytes(payload)
    with pytest.raises(ArchiveError):
        list(workbook_tables(path, get_dataset("spot", "klines")))


def test_empty_worksheet_is_rejected(tmp_path: Path) -> None:
    """Reject a syntactically valid worksheet without a header."""
    path = tmp_path / "book.xlsx"
    path.write_bytes(workbook([]))
    with pytest.raises(ArchiveError, match="empty"):
        list(workbook_tables(path, get_dataset("spot", "klines")))


def test_ingestion_verifies_etag_sorts_and_writes_atomically(tmp_path: Path) -> None:
    """Verify the outer ZIP and store normalized rows in deterministic order."""
    payload = outer(workbook(kline_rows()))
    digest = hashlib.md5(payload).hexdigest()

    def response(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, headers={"ETag": f'"{digest}"'})

    resource = Resource(
        date(2025, 1, 1),
        "https://img.bitgetimg.com/file.zip",
        None,
        integrity=IntegritySpec("response_header", "md5"),
    )
    dataset = get_dataset("spot", "klines")

    def normalize(table: pa.Table, *_args: object) -> pa.Table:
        """Convert representative source text into canonical fields."""
        return pa.table(
            {
                "open_time": pa.array(
                    [
                        datetime.fromtimestamp(int(value), UTC)
                        for value in table["timestamp"].to_pylist()
                    ],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                **{
                    name: pc.cast(table[source], pa.float64())
                    for name, source in {
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "base_volume": "basevolume",
                        "quote_volume": "usdtvolume",
                    }.items()
                },
            }
        )

    def validate(
        table: pa.Table,
        *_args: object,
    ) -> datetime:
        """Return the final normalized timestamp."""
        value = table["open_time"][-1].as_py()
        assert isinstance(value, datetime)
        return value

    destination = tmp_path / "result.parquet"
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        result = ingest_xlsx_archive(
            client,
            resource,
            dataset,
            destination,
            normalizer=normalize,
            validator=validate,
        )
    table = pq.read_table(destination)
    assert result.archive_checksum == digest
    assert result.row_count == 2
    assert table["open_time"].to_pylist() == [
        datetime(2024, 12, 31, 16, tzinfo=UTC),
        datetime(2024, 12, 31, 16, 1, tzinfo=UTC),
    ]


def test_ingestion_removes_partial_output_after_failure(tmp_path: Path) -> None:
    """Leave no visible Parquet when workbook validation fails."""
    payload = outer(workbook(kline_rows()))
    digest = hashlib.md5(payload).hexdigest()
    resource = Resource(
        date(2025, 1, 1),
        "https://img.bitgetimg.com/file.zip",
        None,
        integrity=IntegritySpec("response_header", "md5"),
    )
    destination = tmp_path / "result.parquet"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=payload, headers={"ETag": digest}
            )
        )
    ) as client:
        with pytest.raises(RuntimeError, match="failed"):
            ingest_xlsx_archive(
                client,
                resource,
                get_dataset("spot", "klines"),
                destination,
                normalizer=lambda table, dataset, size: (_ for _ in ()).throw(
                    RuntimeError("failed")
                ),
                validator=lambda *args: datetime.now(UTC),
            )
    assert not destination.exists()
    assert not destination.with_name("result.parquet.part").exists()
