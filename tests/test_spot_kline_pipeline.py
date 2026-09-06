"""Test the first imported Binance Spot kline workflow."""

from datetime import UTC, date, datetime
import hashlib
from io import BytesIO
import logging
import os
from pathlib import Path
import zipfile

import httpx
import pandas as pd
import pytest

import crypto_downloader as crypto
import crypto_downloader.pair as pair_module
from crypto_downloader.cache import parquet_path, valid_cached_path
from crypto_downloader.catalog import open_catalog
from crypto_downloader.discovery import _validate_resources, requested_days
from crypto_downloader.downloader import Downloader
from crypto_downloader.models import Resource, ResourceKey, Result
from crypto_downloader.sources.binance import Binance

FIXTURES = Path(__file__).parent / "fixtures"
DAY = date(2024, 1, 1)
KEY = ResourceKey("binance", "spot", "klines", "BTCUSDT", "1m")
ARCHIVE_NAME = "BTCUSDT-1m-2024-01-01.zip"
OBJECT_KEY = f"data/spot/daily/klines/BTCUSDT/1m/{ARCHIVE_NAME}"


def archive_bytes() -> bytes:
    """Build one realistic daily Binance archive.

    Returns:
        ZIP bytes containing the pre-2025 Spot kline fixture.
    """
    output = BytesIO()
    content = (FIXTURES / "binance_spot_klines_2024-01-01.csv").read_bytes()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(ARCHIVE_NAME.removesuffix(".zip") + ".csv", content)
    return output.getvalue()


def listing(*, keys: tuple[str, ...] = (), prefixes: tuple[str, ...] = ()) -> str:
    """Build one complete S3-style XML listing.

    Args:
        keys: Object keys included in the response.
        prefixes: Folder prefixes included in the response.

    Returns:
        A complete bucket listing document.
    """
    objects = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    folders = "".join(
        f"<CommonPrefixes><Prefix>{prefix}</Prefix></CommonPrefixes>"
        for prefix in prefixes
    )
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>false</IsTruncated>{objects}{folders}</ListBucketResult>"
    )


class BinanceServer:
    """Serve deterministic Binance metadata and archive responses."""

    def __init__(
        self,
        *,
        resource_available: bool = True,
        checksum_valid: bool = True,
    ) -> None:
        """Configure resource availability and integrity.

        Args:
            resource_available: Whether discovery should list the daily archive.
            checksum_valid: Whether the sidecar should match the archive.
        """
        self.resource_available = resource_available
        self.checksum_valid = checksum_valid
        self.payload = archive_bytes()
        self.archive_requests = 0
        self.resource_requests = 0
        self.market_requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return the mocked response for one Binance request.

        Args:
            request: The outgoing HTTPX request.

        Returns:
            The configured metadata, sidecar, or archive response.
        """
        host = request.url.host
        if host == "api.binance.com":
            self.market_requests += 1
            return httpx.Response(
                200,
                json={
                    "symbols": [
                        {
                            "symbol": "BTCUSDT",
                            "baseAsset": "BTC",
                            "quoteAsset": "USDT",
                            "status": "TRADING",
                        }
                    ]
                },
            )
        if host == "s3-ap-northeast-1.amazonaws.com":
            if request.url.params.get("delimiter") == "/":
                return httpx.Response(
                    200,
                    text=listing(prefixes=("data/spot/daily/klines/BTCUSDT/",)),
                )
            self.resource_requests += 1
            keys = (OBJECT_KEY,) if self.resource_available else ()
            return httpx.Response(200, text=listing(keys=keys))
        if str(request.url).endswith(".CHECKSUM"):
            digest = hashlib.sha256(self.payload).hexdigest()
            if not self.checksum_valid:
                digest = "0" * 64
            return httpx.Response(200, text=f"{digest}  {ARCHIVE_NAME}\n")
        if host == "data.binance.vision":
            self.archive_requests += 1
            return httpx.Response(200, content=self.payload)
        raise AssertionError(f"unexpected request: {request.url}")


def downloader(tmp_path: Path, server: BinanceServer) -> Downloader:
    """Create a downloader connected to one mocked Binance server.

    Args:
        tmp_path: The isolated data directory.
        server: The mocked Binance responder.

    Returns:
        A configured downloader service.
    """
    return Downloader(
        tmp_path,
        source=Binance(retries=0),
        transport=httpx.MockTransport(server),
    )


def test_requested_days_uses_the_exclusive_end_boundary() -> None:
    """Confirm exact timestamp ranges map to the necessary daily archives."""
    assert requested_days(
        datetime(2024, 1, 1, 12, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    ) == (date(2024, 1, 1), date(2024, 1, 1))
    assert requested_days(
        datetime(2024, 1, 1, 12, tzinfo=UTC),
        datetime(2024, 1, 2, 0, 0, 0, 1, tzinfo=UTC),
    ) == (date(2024, 1, 1), date(2024, 1, 2))


def test_requested_days_rejects_an_empty_or_reversed_range() -> None:
    """Confirm discovery cannot plan days for an invalid range."""
    with pytest.raises(ValueError, match="range"):
        requested_days(
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    "resources",
    [
        [
            Resource(DAY, "one", "one.checksum"),
            Resource(DAY, "two", "two.checksum"),
        ],
        [Resource(date(2024, 1, 2), "one", "one.checksum")],
    ],
)
def test_discovery_rejects_invalid_source_resource_sets(
    resources: list[Resource],
) -> None:
    """Confirm duplicate and out-of-range source results cannot be cataloged.

    Args:
        resources: The invalid source resource set.
    """
    with pytest.raises(ValueError, match="resource"):
        _validate_resources(resources, DAY, DAY)


def test_parquet_path_is_deterministic_and_partitioned(tmp_path: Path) -> None:
    """Confirm cache paths include every resource identity component."""
    assert parquet_path(tmp_path, KEY, DAY) == (
        tmp_path
        / "parquet"
        / "binance"
        / "spot"
        / "klines"
        / "BTCUSDT"
        / "1m"
        / "2024-01-01.parquet"
    )


def test_valid_cached_path_checks_ready_size_and_modification_time(
    tmp_path: Path,
) -> None:
    """Confirm inexpensive catalog metadata protects cache reuse."""
    path = tmp_path / "one.parquet"
    path.write_bytes(b"cached")
    stat = path.stat()
    ready = Resource(
        DAY,
        "archive",
        "checksum",
        status="ready",
        parquet_path=path,
        parquet_size=stat.st_size,
        parquet_mtime_ns=stat.st_mtime_ns,
    )

    assert valid_cached_path(ready) == path
    assert valid_cached_path(Resource(DAY, "archive", "checksum")) is None
    assert (
        valid_cached_path(
            Resource(
                DAY,
                "archive",
                "checksum",
                status="ready",
                parquet_path=path,
                parquet_size=stat.st_size + 1,
                parquet_mtime_ns=stat.st_mtime_ns,
            )
        )
        is None
    )
    path.unlink()
    assert valid_cached_path(ready) is None


def test_downloader_completes_and_reuses_one_spot_kline_day(
    tmp_path: Path,
) -> None:
    """Confirm the full imported workflow downloads once and then uses Parquet."""
    server = BinanceServer()
    service = downloader(tmp_path, server)

    first = service.get_results(
        "BTCUSDT",
        "2024-01-01",
        "2024-01-01",
        desired_columns={"open_time": "time", "close": "price"},
    )
    second = service.get_results(
        "BTCUSDT",
        "2024-01-01",
        "2024-01-01",
        desired_columns={"open_time": "time", "close": "price"},
    )

    assert isinstance(first, Result)
    assert isinstance(second, Result)
    assert first.complete
    assert first.pair == "BTCUSDT"
    assert first.requested_range == (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    )
    assert first.used_range == first.requested_range
    assert first.available_range is not None
    assert first.available_range[0] == first.requested_range[0]
    assert first.available_range[1].date() == datetime.now(UTC).date()
    assert list(first.data.columns) == ["time", "price"]
    assert first.data["price"].tolist() == [42298.61, 42320.0]
    pd.testing.assert_frame_equal(first.data, second.data)
    assert server.archive_requests == 1
    assert server.resource_requests == 2
    assert server.market_requests == 1

    expected = parquet_path(tmp_path, KEY, DAY)
    assert expected.is_file()
    with open_catalog(tmp_path / "catalog.duckdb") as catalog:
        markets = catalog.markets("binance", "spot")
        resource = catalog.resources(KEY, DAY, DAY)[0]
    assert markets[0].symbol == "BTCUSDT"
    assert resource.status == "ready"
    assert resource.parquet_path == expected


def test_public_get_data_returns_a_dataframe_with_its_report(tmp_path: Path) -> None:
    """Confirm the convenience function exposes the intended imported API."""
    server = BinanceServer()

    frame = crypto.get_data(
        "BTCUSDT",
        "2024-01-01",
        "2024-01-01",
        data_dir=tmp_path,
        desired_columns=["open_time", "close"],
        source=Binance(retries=0),
        transport=httpx.MockTransport(server),
    )

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert frame.attrs["download"]["pair"] == "BTCUSDT"
    assert frame.attrs["download"]["complete"] is True


def test_public_get_results_returns_the_structured_result(tmp_path: Path) -> None:
    """Confirm the convenience report API does not discard diagnostics."""
    server = BinanceServer()

    result = crypto.get_results(
        "BTCUSDT",
        "2024-01-01",
        "2024-01-01",
        data_dir=tmp_path,
        source=Binance(retries=0),
        transport=httpx.MockTransport(server),
    )

    assert isinstance(result, Result)
    assert result.complete


def test_dataframe_list_preserves_requested_pair_order(tmp_path: Path) -> None:
    """Confirm the public service retains list-shaped input and output."""
    server = BinanceServer()

    frames = downloader(tmp_path, server).get_data(
        ["BTCUSDT", "BTCUSDT"], "2024-01-01", "2024-01-01"
    )

    assert isinstance(frames, list)
    assert len(frames) == 2
    assert all(frame.attrs["download"]["pair"] == "BTCUSDT" for frame in frames)
    assert server.archive_requests == 1


def test_unknown_pair_returns_an_error_without_resource_requests(
    tmp_path: Path,
) -> None:
    """Confirm a pair-specific lookup failure remains a normal result."""
    server = BinanceServer()

    result = downloader(tmp_path, server).get_results(
        "NOTREALPAIR", "2024-01-01", "2024-01-01"
    )

    assert isinstance(result, Result)
    assert result.data.empty
    assert [error.code for error in result.errors] == ["unknown_pair"]
    assert not result.complete
    assert server.resource_requests == 0
    assert server.archive_requests == 0


def test_unavailable_daily_resource_returns_a_problem(tmp_path: Path) -> None:
    """Confirm a valid empty listing is reported instead of fabricated data."""
    server = BinanceServer(resource_available=False)

    result = downloader(tmp_path, server).get_results(
        "BTCUSDT", "2024-01-01", "2024-01-01"
    )

    assert isinstance(result, Result)
    assert result.data.empty
    assert result.problems == []
    assert [error.code for error in result.errors] == ["no_availability"]
    assert not result.complete
    assert server.archive_requests == 0


def test_failed_archive_is_recorded_without_raising_for_the_pair(
    tmp_path: Path,
) -> None:
    """Confirm an ingestion failure becomes catalog state and a result problem."""
    server = BinanceServer(checksum_valid=False)

    result = downloader(tmp_path, server).get_results(
        "BTCUSDT", "2024-01-01", "2024-01-01"
    )

    assert isinstance(result, Result)
    assert [problem.code for problem in result.problems] == ["resource_failed"]
    assert "SHA-256" in result.problems[0].message
    assert result.data.empty
    with open_catalog(tmp_path / "catalog.duckdb") as catalog:
        resource = catalog.resources(KEY, DAY, DAY)[0]
    assert resource.status == "failed"
    assert "SHA-256" in (resource.error or "")


def test_malformed_discovery_returns_a_pair_error(tmp_path: Path) -> None:
    """Confirm source-listing errors do not produce false unavailability."""
    server = BinanceServer()

    def malformed(request: httpx.Request) -> httpx.Response:
        """Serve valid markets followed by malformed resource XML."""
        if (
            request.url.host == "s3-ap-northeast-1.amazonaws.com"
            and request.url.params.get("delimiter") is None
        ):
            return httpx.Response(200, text="not XML")
        return server(request)

    service = Downloader(
        tmp_path,
        source=Binance(retries=0),
        transport=httpx.MockTransport(malformed),
    )
    result = service.get_results("BTCUSDT", "2024-01-01", "2024-01-01")

    assert isinstance(result, Result)
    assert [error.code for error in result.errors] == ["discovery_failed"]
    assert result.problems == []
    assert result.data.empty


def test_corrupt_reused_parquet_is_detected_and_rebuilt(tmp_path: Path) -> None:
    """Confirm an unreadable reused file is replaced from its verified archive.

    Args:
        tmp_path: The isolated downloader directory.
    """
    server = BinanceServer()
    service = downloader(tmp_path, server)
    first = service.get_results("BTCUSDT", "2024-01-01", "2024-01-01")
    assert isinstance(first, Result)
    path = parquet_path(tmp_path, KEY, DAY)
    stat = path.stat()
    path.write_bytes(b"x" * stat.st_size)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    result = service.get_results("BTCUSDT", "2024-01-01", "2024-01-01")

    assert isinstance(result, Result)
    assert result.complete
    assert len(result.data) == 2
    assert server.archive_requests == 2


def test_query_failure_remains_an_isolated_result_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm an unexpected DuckDB failure does not escape the pair result.

    Args:
        tmp_path: The isolated downloader directory.
        monkeypatch: The fixture replacing the Parquet query operation.
    """
    server = BinanceServer()

    def fail_query(*args: object, **kwargs: object) -> pd.DataFrame:
        """Raise one representative query failure.

        Args:
            args: Unused positional query arguments.
            kwargs: Unused keyword query arguments.

        Returns:
            This function always raises instead of returning a frame.
        """
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(pair_module, "query_parquet", fail_query)
    result = downloader(tmp_path, server).get_results(
        "BTCUSDT", "2024-01-01", "2024-01-01"
    )

    assert isinstance(result, Result)
    assert [error.code for error in result.errors] == ["query_failed"]
    assert result.data.empty


def test_offline_pipeline_reuses_markets_listings_and_parquet(tmp_path: Path) -> None:
    """Confirm a fully cached request performs no source HTTP requests offline.

    Args:
        tmp_path: The isolated downloader directory.
    """
    server = BinanceServer()
    service = downloader(tmp_path, server)
    online = service.get_results("BTCUSDT", "2024-01-01", "2024-01-01")
    counts = (
        server.market_requests,
        server.resource_requests,
        server.archive_requests,
    )

    offline = service.get_results("BTCUSDT", "2024-01-01", "2024-01-01", offline=True)

    assert isinstance(online, Result)
    assert isinstance(offline, Result)
    assert offline.complete
    pd.testing.assert_frame_equal(offline.data, online.data)
    assert counts == (
        server.market_requests,
        server.resource_requests,
        server.archive_requests,
    )


def test_offline_pipeline_requires_cached_market_metadata(tmp_path: Path) -> None:
    """Confirm offline mode explains why an unused data directory cannot run.

    Args:
        tmp_path: The isolated downloader directory.
    """
    server = BinanceServer()

    with pytest.raises(RuntimeError, match="cached market metadata"):
        downloader(tmp_path, server).get_results(
            "BTCUSDT", "2024-01-01", "2024-01-01", offline=True
        )

    assert server.market_requests == 0
    assert server.resource_requests == 0
    assert server.archive_requests == 0


def test_offline_pipeline_reports_a_missing_cached_partition(tmp_path: Path) -> None:
    """Confirm offline mode does not repair a deleted local partition.

    Args:
        tmp_path: The isolated downloader directory.
    """
    server = BinanceServer()
    service = downloader(tmp_path, server)
    service.get_results("BTCUSDT", "2024-01-01", "2024-01-01")
    parquet_path(tmp_path, KEY, DAY).unlink()
    archive_requests = server.archive_requests

    result = service.get_results("BTCUSDT", "2024-01-01", "2024-01-01", offline=True)

    assert isinstance(result, Result)
    assert [problem.code for problem in result.problems] == ["offline_missing"]
    assert result.data.empty
    assert server.archive_requests == archive_requests


def test_non_base_interval_is_resampled_through_the_complete_pipeline(
    tmp_path: Path,
) -> None:
    """Confirm the public pipeline returns aggregated rather than base rows.

    Args:
        tmp_path: The isolated downloader directory.
    """
    server = BinanceServer()

    result = downloader(tmp_path, server).get_results(
        "BTCUSDT", "2024-01-01", "2024-01-01", interval="1h"
    )

    assert isinstance(result, Result)
    assert len(result.data) == 1
    assert result.data.loc[0, "open"] == 42283.58
    assert result.data.loc[0, "high"] == 42320.00
    assert result.data.loc[0, "low"] == 42261.02
    assert result.data.loc[0, "close"] == 42320.00
    assert result.data.loc[0, "volume"] == pytest.approx(57.09503)
    assert result.data.loc[0, "trade_count"] == 2040
    assert server.resource_requests == 1
    assert server.archive_requests == 1


def test_source_product_mismatch_is_rejected_before_network_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm dataset support and source support must agree."""
    server = BinanceServer()
    source = Binance(retries=0)
    monkeypatch.setattr(source, "products", ("um",))

    with pytest.raises(ValueError, match="unsupported product"):
        Downloader(
            tmp_path,
            source=source,
            transport=httpx.MockTransport(server),
        ).get_results("BTCUSDT", "2024-01-01", "2024-01-01")

    assert server.resource_requests == 0


def test_downloader_defaults_to_the_binance_source(tmp_path: Path) -> None:
    """Confirm callers do not need to construct the default strategy."""
    service = Downloader(tmp_path)

    assert isinstance(service.source, Binance)
    assert service.earliest_date == date(2020, 1, 1)
    assert service.max_workers == 32


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("max_workers", 0),
        ("max_workers", True),
        ("discovery_tail_days", -1),
        ("discovery_tail_days", 1.5),
        ("market_refresh_hours", 0),
        ("market_refresh_hours", True),
    ],
)
def test_downloader_rejects_invalid_durability_settings(
    tmp_path: Path, setting: str, value: object
) -> None:
    """Confirm concurrency and tail settings must be positive integers.

    Args:
        tmp_path: The isolated downloader directory.
        setting: The invalid constructor setting.
        value: The proposed invalid value.
    """
    with pytest.raises((TypeError, ValueError), match=setting):
        Downloader(tmp_path, **{setting: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(("option", "value"), [("refresh", 1), ("offline", "yes")])
def test_downloader_rejects_nonboolean_durability_options(
    tmp_path: Path, option: str, value: object
) -> None:
    """Confirm refresh and offline request flags require actual Booleans.

    Args:
        tmp_path: The isolated downloader directory.
        option: The invalid request option.
        value: The proposed invalid value.
    """
    server = BinanceServer()
    with pytest.raises(TypeError, match=option):
        if option == "refresh":
            downloader(tmp_path, server).get_results(
                "BTCUSDT", "2024-01-01", "2024-01-01", refresh=value  # type: ignore[arg-type]
            )
        else:
            downloader(tmp_path, server).get_results(
                "BTCUSDT", "2024-01-01", "2024-01-01", offline=value  # type: ignore[arg-type]
            )


def test_progress_false_silences_all_rich_pipeline_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Confirm imported callers can completely disable interactive output.

    Args:
        tmp_path: The isolated downloader directory.
        capsys: Pytest's captured output streams.
    """
    server = BinanceServer()

    downloader(tmp_path, server).get_results(
        "BTCUSDT", "2024-01-01", "2024-01-01", progress=False
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_enabled_progress_explains_the_complete_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Confirm an interactive request reports each useful stage.

    Args:
        tmp_path: The isolated downloader directory.
        capsys: Pytest's captured output streams.
    """
    server = BinanceServer()

    downloader(tmp_path, server).get_results("BTCUSDT", "2024-01-01", "2024-01-01")

    output = capsys.readouterr().err
    assert "INFO Binance spot klines: 1 pair" in output
    assert "OK Markets refreshed: 1 pair (1 TRADING)" in output
    assert "INFO BTCUSDT: matched BTC/USDT; status TRADING" in output
    assert "INFO BTCUSDT: found 1 daily file" in output
    assert "INFO BTCUSDT: 1 daily file | 0 cached, 1 to download" in output
    assert "BTCUSDT 2024-01-01 cached" in output
    assert "OK BTCUSDT: returned 2 rows" in output


def test_cached_pipeline_reports_that_no_download_is_needed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Confirm repeat requests clearly identify reusable local files.

    Args:
        tmp_path: The isolated downloader directory.
        capsys: Pytest's captured output streams.
    """
    server = BinanceServer()
    service = downloader(tmp_path, server)
    service.get_results("BTCUSDT", "2024-01-01", "2024-01-01", progress=False)
    capsys.readouterr()

    service.get_results("BTCUSDT", "2024-01-01", "2024-01-01")

    output = capsys.readouterr().err
    assert "INFO BTCUSDT: 1 daily file | 1 cached, 0 to download" in output
    assert "OK BTCUSDT: returned 2 rows" in output


def test_pipeline_emits_standard_logs_without_configuring_root_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Confirm applications can capture library diagnostics with their own setup.

    Args:
        tmp_path: The isolated downloader directory.
        caplog: Pytest's captured standard log records.
    """
    root = logging.getLogger()
    handlers = tuple(root.handlers)
    level = root.level
    server = BinanceServer()

    with caplog.at_level(logging.DEBUG, logger="crypto_downloader"):
        downloader(tmp_path, server).get_results(
            "BTCUSDT", "2024-01-01", "2024-01-01", progress=False
        )

    assert tuple(root.handlers) == handlers
    assert root.level == level
    assert any("Request started" in message for message in caplog.messages)
    assert any("Market snapshot refreshed" in message for message in caplog.messages)
    assert any("Resource discovery complete" in message for message in caplog.messages)
    assert any("Daily resource cached" in message for message in caplog.messages)
    assert any("Parquet query complete" in message for message in caplog.messages)
    assert any("Request complete" in message for message in caplog.messages)


@pytest.mark.parametrize("progress", [None, 1, "yes"])
def test_pipeline_rejects_non_boolean_progress_values_before_network_access(
    tmp_path: Path, progress: object
) -> None:
    """Confirm interactive output requires an explicit Boolean switch.

    Args:
        tmp_path: The isolated downloader directory.
        progress: The invalid progress option.
    """
    server = BinanceServer()

    with pytest.raises(TypeError, match="progress"):
        downloader(tmp_path, server).get_results(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-01",
            progress=progress,  # type: ignore[arg-type]
        )

    assert server.market_requests == 0


def test_refresh_and_offline_cannot_be_requested_together(tmp_path: Path) -> None:
    """Confirm contradictory source-access modes fail before network access.

    Args:
        tmp_path: The isolated downloader directory.
    """
    server = BinanceServer()
    with pytest.raises(ValueError, match="refresh and offline"):
        downloader(tmp_path, server).get_results(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-01",
            refresh=True,
            offline=True,
        )
    assert server.market_requests == 0


@pytest.mark.parametrize("data_dir", ["", "   ", 1, None])
def test_downloader_rejects_invalid_data_directories(data_dir: object) -> None:
    """Confirm local storage must be a nonempty path value.

    Args:
        data_dir: The invalid local storage value.
    """
    with pytest.raises((TypeError, ValueError), match="data_dir"):
        Downloader(data_dir)  # type: ignore[arg-type]
