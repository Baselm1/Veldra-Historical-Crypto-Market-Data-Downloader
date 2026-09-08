"""Test HTX Spot Kline schemas, processing, facade, and ingestion."""

from datetime import UTC, date, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import zipfile

from collections.abc import Sequence
import duckdb
import httpx
import pandas as pd
import pyarrow as pa
import pytest

from crypto_downloader.core.models import DataValidationError, Resource
from crypto_downloader.core.query import query_parquet
from crypto_downloader.htx.connector import HTXConnector
from crypto_downloader.htx.datasets import (
    NEW_KLINE_COLUMNS,
    OLD_KLINE_COLUMNS,
    SPOT_KLINES,
    get_dataset,
)
from crypto_downloader.htx.facade import HTX
from crypto_downloader.htx.processing import normalize_chunk, validate_chunk


def raw_table(columns: tuple[str, ...], rows: Sequence[tuple[object, ...]]) -> pa.Table:
    """Build a string-valued source table.

    Args:
        columns: The exact source column names.
        rows: Source values in column order.

    Returns:
        An Arrow table shaped like CSV ingestion output.
    """
    return pa.table(
        {
            column: pa.array([str(row[index]) for row in rows], type=pa.string())
            for index, column in enumerate(columns)
        }
    )


@pytest.mark.parametrize(
    ("columns", "row", "expected_base", "expected_quote"),
    [
        (
            OLD_KLINE_COLUMNS,
            (1735660800, 95432.42, 95331.58, 95451.64, 95331.58, 996761.61, 10.44),
            10.44,
            996761.61,
        ),
        (
            NEW_KLINE_COLUMNS,
            (
                "BTC-USDT",
                78671.96,
                78728.52,
                78641.25,
                78726.96,
                1.547,
                121749.9,
                121749.8,
                1788795720,
            ),
            1.547,
            121749.8,
        ),
    ],
)
def test_spot_kline_variants_normalize_to_one_schema(
    columns: tuple[str, ...],
    row: tuple[object, ...],
    expected_base: float,
    expected_quote: float,
) -> None:
    """Confirm old and new layouts preserve explicit base and quote units.

    Args:
        columns: The source layout under test.
        row: One representative source row.
        expected_base: The canonical base volume.
        expected_quote: The canonical quote volume.
    """
    result = normalize_chunk(raw_table(columns, [row]), SPOT_KLINES)

    assert result.column_names == list(SPOT_KLINES.stored_columns)
    assert result["base_volume"][0].as_py() == pytest.approx(expected_base)
    assert result["quote_volume"][0].as_py() == pytest.approx(expected_quote)
    assert result["open_time"][0].as_py().utcoffset() == UTC.utcoffset(None)


def test_spot_kline_validation_uses_utc_plus_eight_source_days() -> None:
    """Confirm a source day begins at 16:00 UTC on the preceding date."""
    rows = [
        (1735660860, 2, 2, 3, 1, 20, 10),
        (1735660800, 2, 2, 3, 1, 20, 10),
    ]
    normalized = normalize_chunk(raw_table(OLD_KLINE_COLUMNS, rows), SPOT_KLINES)
    sorted_table = normalized.sort_by([("open_time", "ascending")])

    last = validate_chunk(sorted_table, SPOT_KLINES, date(2025, 1, 1))

    assert sorted_table["open_time"][0].as_py() == datetime(
        2024, 12, 31, 16, tzinfo=UTC
    )
    assert last == datetime(2024, 12, 31, 16, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(1735660801, 2, 2, 3, 1, 20, 10)], "aligned"),
        ([(1735574399, 2, 2, 3, 1, 20, 10)], "outside"),
        ([(1735660800, 0, 2, 3, 1, 20, 10)], "positive"),
        ([(1735660800, 4, 2, 3, 1, 20, 10)], "high"),
        ([(1735660800, 2, 2, 3, 4, 20, 10)], "high"),
        ([(1735660800, 2, 2, 3, 1, -1, 10)], "nonnegative"),
    ],
)
def test_spot_kline_validation_rejects_bad_source_values(
    rows: list[tuple[object, ...]], message: str
) -> None:
    """Confirm malformed prices, times, and quantities fail clearly.

    Args:
        rows: The source rows under test.
        message: Text identifying the expected validation failure.
    """
    table = normalize_chunk(raw_table(OLD_KLINE_COLUMNS, rows), SPOT_KLINES)
    with pytest.raises(DataValidationError, match=message):
        validate_chunk(table, SPOT_KLINES, date(2025, 1, 1))


def test_dataset_resolution_accepts_only_implemented_spot_one_minute_storage() -> None:
    """Confirm the registry distinguishes exposed and future HTX datasets."""
    assert get_dataset("spot", "klines") is SPOT_KLINES
    with pytest.raises(ValueError, match="currently stores"):
        get_dataset("spot", "klines", kline_base_interval="5m")
    with pytest.raises(ValueError, match="unsupported dataset"):
        get_dataset("spot", "trades")


def archive_bytes(filename: str, csv_text: str) -> bytes:
    """Create one in-memory single-member ZIP archive.

    Args:
        filename: The expected CSV member name.
        csv_text: The UTF-8 CSV contents.

    Returns:
        Complete ZIP bytes.
    """
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, csv_text)
    return output.getvalue()


@pytest.mark.parametrize("header", [False, True])
def test_connector_ingests_old_headerless_and_header_bearing_klines(
    tmp_path: Path, header: bool
) -> None:
    """Confirm verified old CSV forms become sorted canonical Parquet.

    Args:
        tmp_path: The isolated cache directory.
        header: Whether the old source CSV contains a header row.
    """
    basename = "BTCUSDT-1min-2025-01-01.zip"
    lines = [
        "1735660860,2,2,3,1,20,10",
        "1735660800,2,2,3,1,20,10",
    ]
    if header:
        lines.insert(0, ",".join(OLD_KLINE_COLUMNS))
    body = archive_bytes(basename.removesuffix(".zip") + ".csv", "\n".join(lines))
    digest = sha256(body).hexdigest()
    resource = Resource(
        date(2025, 1, 1),
        f"https://example.test/{basename}",
        "https://example.test/source.CHECKSUM",
        coverage_start=datetime(2024, 12, 31, 16, tzinfo=UTC),
        coverage_end=datetime(2025, 1, 1, 16, tzinfo=UTC),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a matching sidecar or archive."""
        if request.url.path.endswith("CHECKSUM"):
            return httpx.Response(200, text=f"{digest}  {basename}\n")
        return httpx.Response(200, content=body)

    destination = tmp_path / "day.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        metadata = HTXConnector(retries=0).ingest(
            client, resource, SPOT_KLINES, destination
        )

    frame = pd.read_parquet(destination)
    assert metadata.row_count == 2
    assert frame["open_time"].tolist() == sorted(frame["open_time"].tolist())
    with duckdb.connect() as connection:
        queried = query_parquet(
            connection,
            [destination],
            SPOT_KLINES,
            datetime(2024, 12, 31, 16, tzinfo=UTC),
            datetime(2024, 12, 31, 16, 2, tzinfo=UTC),
            SPOT_KLINES.resolve_columns(None),
            interval="1m",
            gap_policy="keep",
        )
    assert len(queried) == 2


def test_htx_facade_constructs_lazily_and_delegates_spot_klines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm the public facade remains declarative and preserves input shape.

    Args:
        tmp_path: The isolated data directory.
        monkeypatch: Pytest's attribute replacement helper.
    """
    service = HTX(tmp_path, progress=False, earliest_date="all", max_workers=5)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    expected = pd.DataFrame({"open": [1.0]})

    def get_data(*args: object, **kwargs: object) -> pd.DataFrame:
        """Record one delegated engine request."""
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(service._downloader, "get_data", get_data)
    actual = service.get_klines(
        "BTCUSDT",
        "2025-01-01",
        "2025-01-02",
        interval="1h",
        columns=["open"],
        gap_policy="keep",
    )

    assert actual is expected
    assert service.data_dir == tmp_path.resolve()
    assert service.earliest_date is None
    assert service.kline_base_interval == "1m"
    assert service.max_workers == 5
    assert calls[0][1] == {
        "product": "spot",
        "dataset": "klines",
        "interval": "1h",
        "desired_columns": ["open"],
        "refresh": False,
        "offline": False,
        "gap_policy": "keep",
        "progress": False,
    }
