"""Exercise physical monthly ingestion, fallback, and mixed-cache queries."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
import hashlib
from pathlib import Path
from urllib.parse import unquote

import duckdb
import httpx
import pandas as pd
import pytest

from crypto_downloader.core.cache import cache_resources
from crypto_downloader.core.catalog import Catalog
from crypto_downloader.core.models import Resource, ResourceKey
from crypto_downloader.core.planner import plan_archives, select_archives
from crypto_downloader.core.query import query_parquet
from crypto_downloader.core.ingest import ingest_archive
from crypto_downloader.binance.connector import BinanceConnector
from crypto_downloader.binance.datasets import DATASETS, SPOT_KLINES
from crypto_downloader.binance.processing import normalize_chunk, validate_chunk
from test_ingest import archive_bytes
from test_spot_kline_pipeline import listing

KEY = ResourceKey("binance", "spot", "klines", "BTCUSDT", "1m")
FIRST = date(2024, 1, 1)
LAST = date(2024, 1, 31)
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 2, 2, tzinfo=UTC)
FIXTURES = Path(__file__).parent / "fixtures"


class Portal:
    """Serve daily and monthly archives through real connector and HTTP paths."""

    def __init__(
        self,
        monthly: bool = True,
        corrupt: bool = False,
        missing_day: date | None = None,
    ) -> None:
        """Configure archive publication, corruption and an optional missing day."""
        self.downloads: list[str] = []
        self.listings: list[str] = []
        self.objects: dict[str, bytes] = {}
        template = (
            (FIXTURES / "binance_spot_klines_2024-01-01.csv")
            .read_text()
            .strip()
            .splitlines()
        )
        all_rows = []
        for i in range(32):
            day = FIRST + timedelta(days=i)
            rows = []
            for raw in template:
                values = raw.split(",")
                values[0] = str(int(values[0]) + i * 86400000)
                values[6] = str(int(values[6]) + i * 86400000)
                rows.append(",".join(values))
            if i < 31 and day != missing_day:
                all_rows.extend(rows)
            if day != missing_day:
                name = f"BTCUSDT-1m-{day}.csv"
                key = f"data/spot/daily/klines/BTCUSDT/1m/{name[:-4]}.zip"
                self.objects[key] = archive_bytes(
                    ("\n".join(rows) + "\n").encode(), names=(name,)
                )
        if monthly:
            name = "BTCUSDT-1m-2024-01.csv"
            self.objects[f"data/spot/monthly/klines/BTCUSDT/1m/{name[:-4]}.zip"] = (
                b"bad zip"
                if corrupt
                else archive_bytes(("\n".join(all_rows) + "\n").encode(), names=(name,))
            )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Serve XML discovery, checksums and archive payloads with request counts."""
        if request.url.host.startswith("s3-"):
            prefix = request.url.params["prefix"]
            self.listings.append(prefix)
            marker = request.url.params.get("marker", "")
            keys = tuple(
                key
                for key in sorted(self.objects)
                if key.startswith(prefix) and key > marker
            )
            return httpx.Response(200, text=listing(keys=keys))
        key = unquote(request.url.path).lstrip("/")
        checksum = key.endswith(".CHECKSUM")
        key = key.removesuffix(".CHECKSUM")
        if key not in self.objects:
            return httpx.Response(404)
        if checksum:
            return httpx.Response(
                200,
                text=f"{hashlib.sha256(self.objects[key]).hexdigest()}  {Path(key).name}",
            )
        self.downloads.append(key)
        return httpx.Response(200, content=self.objects[key])


def test_month_and_daily_tail_are_cached_once_and_filtered_exactly(
    tmp_path: Path,
) -> None:
    """Store one monthly Parquet plus a daily tail and reuse both for subranges."""
    portal = Portal()
    with (
        duckdb.connect() as connection,
        httpx.Client(transport=httpx.MockTransport(portal)) as client,
    ):
        catalog = Catalog(connection)
        connector = BinanceConnector(retries=0)
        resources = plan_archives(
            connector, catalog, client, KEY, START, END, dataset=SPOT_KLINES
        )
        assert [(r.day, r.last_day) for r in resources] == [
            (FIRST, LAST),
            (date(2024, 2, 1), date(2024, 2, 1)),
        ]
        coverage = cache_resources(
            connector, catalog, client, KEY, SPOT_KLINES, resources, tmp_path
        )
        assert not coverage.problems
        assert sorted(p.name for p in coverage.paths) == [
            "2024-01.parquet",
            "2024-02-01.parquet",
        ]
        assert len(portal.downloads) == 2
        sub_start = START + timedelta(days=14)
        sub_end = sub_start + timedelta(minutes=1)
        again = plan_archives(
            connector,
            catalog,
            client,
            KEY,
            sub_start,
            sub_end,
            dataset=SPOT_KLINES,
            offline=True,
        )
        cached = cache_resources(
            connector, catalog, client, KEY, SPOT_KLINES, again, tmp_path, offline=True
        )
        frame = query_parquet(
            connection,
            cached.paths,
            SPOT_KLINES,
            sub_start,
            sub_end,
            {"open_time": "open_time", "close": "close"},
            gap_policy="keep",
        )
        assert len(frame) == 1
        assert frame.iloc[0].open_time == sub_start
        assert len(portal.downloads) == 2
        assert connection.execute(
            "select count(*) from resources where cadence='monthly'"
        ).fetchone() == (1,)


@pytest.mark.parametrize("corrupt", [False, True])
def test_absent_or_corrupt_monthly_archives_fall_back_to_daily(
    tmp_path: Path, corrupt: bool
) -> None:
    """Retain available daily files and report an isolated missing fallback date."""
    portal = Portal(monthly=corrupt, corrupt=corrupt, missing_day=date(2024, 1, 5))
    with (
        duckdb.connect() as connection,
        httpx.Client(transport=httpx.MockTransport(portal)) as client,
    ):
        catalog = Catalog(connection)
        source = BinanceConnector(retries=0)
        resources = plan_archives(
            source, catalog, client, KEY, START, END, dataset=SPOT_KLINES
        )
        coverage = cache_resources(
            source, catalog, client, KEY, SPOT_KLINES, resources, tmp_path
        )
        assert len(coverage.paths) == 31
        if corrupt:
            assert any(p.date == date(2024, 1, 5) for p in coverage.problems)
            assert coverage.warnings[0].code == "monthly_fallback"
            assert (
                catalog.resources(replace(KEY, cadence="monthly"), FIRST, LAST)[
                    0
                ].status
                == "failed"
            )
        before = len(portal.downloads)
        resources = plan_archives(
            source, catalog, client, KEY, START, END, dataset=SPOT_KLINES
        )
        cache_resources(source, catalog, client, KEY, SPOT_KLINES, resources, tmp_path)
        assert len(portal.downloads) == before


@pytest.mark.parametrize(
    "key", [key for key in DATASETS if key[1] not in {"metrics", "book_depth"}]
)
def test_each_monthly_dataset_uses_one_streamed_parquet(
    tmp_path: Path, key: tuple[str, str]
) -> None:
    """Verify every declared monthly family through SHA-256, parsing and DuckDB."""
    dataset = DATASETS[key]
    raw = (FIXTURES / f"binance_{key[0]}_{key[1]}_2024-01-01.csv").read_bytes()
    name = "sample-2024-01.zip"
    payload = archive_bytes(raw, names=("sample-2024-01.csv",))

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a real checksum sidecar and a representative monthly ZIP."""
        return (
            httpx.Response(200, text=f"{hashlib.sha256(payload).hexdigest()}  {name}")
            if str(request.url).endswith(".CHECKSUM")
            else httpx.Response(200, content=payload)
        )

    resource = Resource(
        FIRST,
        f"https://example/{name}",
        f"https://example/{name}.CHECKSUM",
        end_day=LAST,
        cadence="monthly",
        contract_size=100.0,
    )
    path = tmp_path / "2024-01.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        metadata = ingest_archive(
            client,
            resource,
            dataset,
            path,
            normalizer=normalize_chunk,
            validator=validate_chunk,
            chunk_rows=1,
        )
    assert metadata.row_count > 0
    frame = pd.read_parquet(path)
    assert tuple(frame.columns) == dataset.stored_columns
    assert len(list(tmp_path.glob("*.parquet"))) == 1


def test_monthly_missing_day_is_reported_and_never_filled(tmp_path: Path) -> None:
    """An archive's calendar span must not hide a completely absent source day."""
    from crypto_downloader.core.pair import _populate_cached_query, _result
    from crypto_downloader.core.reporting import Reporter
    from crypto_downloader.core.request import Request

    portal = Portal(missing_day=date(2024, 1, 5))
    request = Request.parse("BTCUSDT", START, END).resolve_dataset(SPOT_KLINES)
    result = _result("BTCUSDT", request, SPOT_KLINES)
    with (
        duckdb.connect() as connection,
        httpx.Client(transport=httpx.MockTransport(portal)) as client,
    ):
        catalog = Catalog(connection)
        source = BinanceConnector(retries=0)
        resources = plan_archives(
            source, catalog, client, KEY, START, END, dataset=SPOT_KLINES
        )
        coverage = cache_resources(
            source, catalog, client, KEY, SPOT_KLINES, resources, tmp_path
        )
        error = _populate_cached_query(
            result,
            catalog,
            KEY,
            coverage.paths,
            SPOT_KLINES,
            request,
            (START, END),
            "binance",
            Reporter(False),
        )
        assert error is None
        assert any(p.date == date(2024, 1, 5) for p in result.problems)
        assert date(2024, 1, 5) not in set(result.data.open_time.dt.date)
        assert not result.complete


def test_existing_daily_cache_does_not_get_downloaded_again(tmp_path: Path) -> None:
    """A later full-month request keeps an already cached daily partition."""
    portal = Portal()
    with (
        duckdb.connect() as connection,
        httpx.Client(transport=httpx.MockTransport(portal)) as client,
    ):
        catalog = Catalog(connection)
        source = BinanceConnector(retries=0)
        first = plan_archives(
            source,
            catalog,
            client,
            KEY,
            START,
            START + timedelta(days=1),
            dataset=SPOT_KLINES,
        )
        cache_resources(source, catalog, client, KEY, SPOT_KLINES, first, tmp_path)
        month = plan_archives(
            source, catalog, client, KEY, START, END, dataset=SPOT_KLINES
        )
        assert len(month) == 32
        assert all(r.cadence == "daily" for r in month)
        coverage = cache_resources(
            source, catalog, client, KEY, SPOT_KLINES, month, tmp_path
        )
        assert len(portal.downloads) == len(coverage.paths) == 32
        assert len(set(portal.downloads)) == 32


def test_refresh_keeps_valid_monthly_file_and_limited_fallback(tmp_path: Path) -> None:
    """Refresh compares checksums once and recovery honors a narrow request."""
    portal = Portal()
    with (
        duckdb.connect() as connection,
        httpx.Client(transport=httpx.MockTransport(portal)) as client,
    ):
        catalog = Catalog(connection)
        source = BinanceConnector(retries=0)
        resources = plan_archives(
            source, catalog, client, KEY, START, END, dataset=SPOT_KLINES
        )
        cache_resources(source, catalog, client, KEY, SPOT_KLINES, resources, tmp_path)
        resources = plan_archives(
            source, catalog, client, KEY, START, END, dataset=SPOT_KLINES, refresh=True
        )
        cache_resources(
            source, catalog, client, KEY, SPOT_KLINES, resources, tmp_path, refresh=True
        )
        assert len(portal.downloads) == 2
        monthly = next(r for r in resources if r.cadence == "monthly")
        # Change the source archive while keeping its checksum consistent.
        portal.objects[next(key for key in portal.objects if "/monthly/" in key)] = (
            b"invalid zip"
        )
        subset = cache_resources(
            source,
            catalog,
            client,
            KEY,
            SPOT_KLINES,
            [monthly],
            tmp_path,
            refresh=True,
            requested_range=(date(2024, 1, 15), date(2024, 1, 15)),
        )
        assert not subset.problems
        assert [path.name for path in subset.paths] == ["2024-01-15.parquet"]
        assert len(portal.downloads) == 4


def test_bad_month_name_is_ignored() -> None:
    """An unrelated or malformed bucket filename does not abort discovery."""
    portal = Portal()
    portal.objects["data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-99.zip"] = b""
    with httpx.Client(transport=httpx.MockTransport(portal)) as client:
        resources = BinanceConnector().resources(
            client, replace(KEY, cadence="monthly"), FIRST, LAST
        )
    assert len(resources) == 1
