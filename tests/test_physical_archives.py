"""Test catalog isolation and selection of physical archive ranges."""

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
import duckdb
from veldra.core.catalog import Catalog
from veldra.core.models import Resource, ResourceKey, IngestedResource

KEY = ResourceKey("binance", "spot", "klines", "BTCUSDT", "1m")


def test_monthly_and_daily_archives_on_same_day_are_distinct(tmp_path: Path) -> None:
    """Persist one month once and keep an independent daily file on its first day."""
    with duckdb.connect() as connection:
        catalog = Catalog(connection)
        day = date(2024, 1, 1)
        daily = Resource(day, "https://example/d.zip", "https://example/d.zip.CHECKSUM")
        monthly = replace(
            daily,
            url="https://example/m.zip",
            end_day=date(2024, 1, 31),
            cadence="monthly",
        )
        month_key = replace(KEY, cadence="monthly")
        catalog.save_discovery(KEY, day, day, [daily])
        catalog.save_discovery(month_key, day, date(2024, 1, 31), [monthly])
        assert catalog.resources(KEY, day, day) == [daily]
        assert catalog.resources(month_key, date(2024, 1, 15), date(2024, 1, 16)) == [
            monthly
        ]
        assert connection.execute("select count(*) from resources").fetchone() == (2,)
        stamp = datetime(2024, 1, 1, tzinfo=UTC)
        meta = IngestedResource("a" * 64, 1, 1, 1, stamp, stamp)
        catalog.mark_ready(month_key, day, tmp_path / "2024-01.parquet", meta)
        assert catalog.resources(KEY, day, day)[0].status == "discovered"
        assert catalog.resources(month_key, day, day)[0].status == "ready"


def test_legacy_resource_key_migrates_without_losing_cache(tmp_path: Path) -> None:
    """Keep existing daily paths and exact scan ranges through repeated migration."""
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE resources(source VARCHAR,product VARCHAR,dataset VARCHAR,symbol VARCHAR,interval VARCHAR,day DATE,url VARCHAR,checksum_url VARCHAR,status VARCHAR,archive_sha256 VARCHAR,parquet_path VARCHAR,parquet_size BIGINT,parquet_mtime_ns BIGINT,row_count BIGINT,first_timestamp TIMESTAMP,last_timestamp TIMESTAMP,error VARCHAR,last_attempt_at TIMESTAMP, PRIMARY KEY(source,product,dataset,symbol,interval,day))"
        )
        connection.execute(
            "INSERT INTO resources VALUES ('binance','spot','klines','BTCUSDT','1m','2024-01-01','url','sum','ready','hash',?,10,20,1440,NULL,NULL,NULL,NULL)",
            [str(tmp_path / "cached.parquet")],
        )
        connection.execute(
            "CREATE TABLE discoveries(source VARCHAR,product VARCHAR,dataset VARCHAR,symbol VARCHAR,interval VARCHAR,start_day DATE,end_day DATE,scanned_at TIMESTAMP)"
        )
        connection.execute(
            "INSERT INTO discoveries VALUES('binance','spot','klines','BTCUSDT','1m','2024-01-01','2024-01-07',now())"
        )
        Catalog(connection)
        catalog = Catalog(connection)
        cached = catalog.resources(KEY, date(2024, 1, 1), date(2024, 1, 1))[0]
        assert cached.parquet_path == tmp_path / "cached.parquet"
        assert cached.status == "ready"
        assert catalog.discovery_ranges(KEY) == [(date(2024, 1, 1), date(2024, 1, 7))]
        assert "discoveries" not in {
            r[0] for r in connection.execute("show tables").fetchall()
        }
