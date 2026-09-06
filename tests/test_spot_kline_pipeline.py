"""Test the first imported Binance Spot kline workflow."""

from datetime import UTC, date, datetime
import hashlib
from io import BytesIO
import os
from pathlib import Path
import zipfile

import httpx
import pandas as pd
import pytest

import crypto_downloader as crypto
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

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return the mocked response for one Binance request.

        Args:
            request: The outgoing HTTPX request.

        Returns:
            The configured metadata, sidecar, or archive response.
        """
        host = request.url.host
        if host == "api.binance.com":
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
    assert first.available_range == first.requested_range
    assert list(first.data.columns) == ["time", "price"]
    assert first.data["price"].tolist() == [42298.61, 42320.0]
    pd.testing.assert_frame_equal(first.data, second.data)
    assert server.archive_requests == 1
    assert server.resource_requests == 2

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
    assert [problem.code for problem in result.problems] == ["resource_unavailable"]
    assert result.errors == []
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


def test_corrupt_reused_parquet_returns_a_query_error(tmp_path: Path) -> None:
    """Confirm an unreadable reused file remains a structured pair failure."""
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
    assert [error.code for error in result.errors] == ["query_failed"]
    assert result.data.empty


def test_non_base_interval_is_rejected_before_network_access(
    tmp_path: Path,
) -> None:
    """Confirm Phase 9 does not silently return one-minute rows as hourly data."""
    server = BinanceServer()

    with pytest.raises(ValueError, match="resampling"):
        downloader(tmp_path, server).get_results(
            "BTCUSDT", "2024-01-01", "2024-01-01", interval="1h"
        )

    assert server.resource_requests == 0
    assert server.archive_requests == 0


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


@pytest.mark.parametrize("data_dir", ["", "   ", 1, None])
def test_downloader_rejects_invalid_data_directories(data_dir: object) -> None:
    """Confirm local storage must be a nonempty path value.

    Args:
        data_dir: The invalid local storage value.
    """
    with pytest.raises((TypeError, ValueError), match="data_dir"):
        Downloader(data_dir)  # type: ignore[arg-type]
