"""Test OKX Spot Kline normalization, caching, and public retrieval."""

from __future__ import annotations

from base64 import b64encode
from datetime import UTC, date, datetime
from hashlib import md5
from io import BytesIO
from pathlib import Path
import zipfile

import httpx
import pandas as pd
import pytest

from veldra import OKX
from veldra.core.models import (
    ArchiveKey,
    ArchiveObject,
    DataValidationError,
    Gap,
    IntegritySpec,
    Market,
    MissingCandlesError,
    Result,
)
from veldra.okx.client import OKXResponseError
from veldra.okx.datasets import KLINE_SOURCE_COLUMNS, SPOT_KLINES, get_dataset
from veldra.okx.processing import OKXArchiveProvider, normalize_klines


def source_frame(**changes: object) -> pd.DataFrame:
    """Build two valid source Klines with optional column replacement.

    Args:
        changes: Source columns replacing fixture values.

    Returns:
        Mutable OKX source frame.
    """
    values: dict[str, object] = {
        "instrument_name": ["BTC-USDT", "BTC-USDT"],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "vol": [2.0, 3.0],
        "vol_ccy": [201.0, 306.0],
        "vol_quote": [201.0, 306.0],
        "open_time": [1735689600000, 1735689660000],
        "confirm": [1, 1],
    }
    values.update(changes)
    return pd.DataFrame(values, columns=KLINE_SOURCE_COLUMNS)


def archive_bytes(name: str, frame: pd.DataFrame) -> bytes:
    """Create one in-memory source ZIP.

    Args:
        name: Remote ZIP filename.
        frame: CSV rows stored in the archive.

    Returns:
        Complete ZIP bytes.
    """
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name.removesuffix(".zip") + ".csv", frame.to_csv(index=False))
    return output.getvalue()


def physical(name: str = "BTC-USDT-candlesticks-2025-01-01.zip") -> ArchiveObject:
    """Build one physical Spot Kline archive fixture.

    Args:
        name: Remote archive filename.

    Returns:
        Discovered OKX archive object.
    """
    key = ArchiveKey(
        "okx",
        "spot",
        "klines",
        "module_2",
        "instrument",
        "BTC-USDT",
        "daily",
        date(2025, 1, 1),
        date(2025, 1, 1),
        name,
    )
    return ArchiveObject(
        key,
        f"https://files.test/{name}",
        integrity=IntegritySpec("response_header", algorithm="md5"),
    )


def test_spot_kline_schema_and_normalization_are_canonical() -> None:
    """Confirm module 2 rows retain precise canonical units and UTC times."""
    dataset = get_dataset("spot", "klines")
    frame = normalize_klines(source_frame(), dataset)
    assert list(frame.columns) == ["instrument_id", *dataset.stored_columns]
    assert frame["base_volume"].tolist() == [2.0, 3.0]
    assert frame["quote_volume"].tolist() == [201.0, 306.0]
    assert str(frame["open_time"].dtype) == "datetime64[ms, UTC]"
    assert dataset.resolve_interval("1h") == "1h"


def test_exact_duplicate_klines_are_removed_but_conflicts_fail() -> None:
    """Confirm duplicate event keys cannot silently select conflicting prices."""
    raw = source_frame()
    exact = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    assert len(normalize_klines(exact, SPOT_KLINES)) == 2
    conflict = raw.copy()
    conflict.loc[1, "open_time"] = conflict.loc[0, "open_time"]
    with pytest.raises(DataValidationError, match="conflicting"):
        normalize_klines(conflict, SPOT_KLINES)


@pytest.mark.parametrize(
    "changes",
    [
        {"confirm": [1, 0]},
        {"open": [0, 1]},
        {"vol": [-1, 1]},
        {"high": [98, 103]},
        {"open_time": [1735689600001, 1735689660000]},
        {"instrument_name": ["", "BTC-USDT"]},
        {"close": ["bad", "102"]},
    ],
)
def test_invalid_spot_kline_values_fail_visibly(changes: dict[str, object]) -> None:
    """Confirm malformed source values never become queryable Parquet.

    Args:
        changes: Invalid fixture column replacement.
    """
    with pytest.raises(DataValidationError):
        normalize_klines(source_frame(**changes), SPOT_KLINES)


def test_archive_provider_verifies_and_partitions_shared_klines(tmp_path: Path) -> None:
    """Confirm one physical file produces one predicate per native instrument."""
    item = physical()
    raw = pd.concat(
        [
            source_frame(),
            source_frame(
                instrument_name=["ETH-USDT", "ETH-USDT"],
                open_time=[1735689600000, 1735689660000],
            ),
        ],
        ignore_index=True,
    )
    content = archive_bytes(item.key.remote_name, raw)

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one integrity-bearing archive response."""
        return httpx.Response(
            200,
            content=content,
            headers={"Content-MD5": b64encode(md5(content).digest()).decode()},
            request=request,
        )

    destination = tmp_path / "source.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        value = OKXArchiveProvider(client, retries=0).materialize(item, destination)
    assert destination.is_file()
    assert value.materialization.row_count == 4
    assert [partition.subject.value for partition in value.partitions] == [
        "BTC-USDT",
        "ETH-USDT",
    ]
    assert all(
        partition.materialization_path == destination for partition in value.partitions
    )


class OKXFixture:
    """Serve current instruments, manifests, and two source-day ZIPs."""

    def __init__(self) -> None:
        """Create request counters for online and offline assertions."""
        self.manifests = 0
        self.files = 0

    @staticmethod
    def _instrument() -> dict[str, object]:
        """Return one current Spot instrument row."""
        return {
            "instType": "SPOT",
            "instId": "BTC-USDT",
            "instFamily": "",
            "baseCcy": "BTC",
            "quoteCcy": "USDT",
            "settleCcy": "",
            "ctType": "",
            "ctVal": "",
            "ctMult": "",
            "ctValCcy": "",
            "state": "live",
            "ruleType": "normal",
            "listTime": "1609459200000",
            "expTime": "",
            "stk": "",
            "optType": "",
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return a valid response for each request in the public workflow.

        Args:
            request: HTTP request made by the facade.

        Returns:
            Current API envelope, manifest, or archive bytes.
        """
        if request.url.path.endswith("/instruments"):
            return httpx.Response(
                200,
                json={"code": "0", "msg": "", "data": [self._instrument()]},
                request=request,
            )
        if request.url.path.endswith("/market-data-history"):
            self.manifests += 1
            details = []
            for source_day, timestamp in (
                ("2025-01-01", 1735689600000),
                ("2025-01-02", 1735776000000),
            ):
                name = f"BTC-USDT-candlesticks-{source_day}.zip"
                details.append(
                    {
                        "dataTs": str(timestamp),
                        "filename": name,
                        "sizeMB": "0.01",
                        "url": f"https://files.test/{name}",
                    }
                )
            group = {
                "instId": "BTC-USDT",
                "instFamily": "",
                "groupDetails": details,
            }
            return httpx.Response(
                200,
                json={
                    "code": "0",
                    "msg": "",
                    "data": [{"dateAggrType": "daily", "details": [group]}],
                },
                request=request,
            )
        if request.url.host == "files.test":
            self.files += 1
            name = Path(request.url.path).name
            day = "01" if "2025-01-01" in name else "02"
            timestamp = 1735689600000 if day == "01" else 1735776000000
            content = archive_bytes(
                name,
                source_frame(
                    open_time=[timestamp, timestamp + 60_000],
                    open=[100, 101],
                    high=[102, 103],
                    low=[99, 100],
                    close=[101, 102],
                ),
            )
            return httpx.Response(
                200,
                content=content,
                headers={"Content-MD5": b64encode(md5(content).digest()).decode()},
                request=request,
            )
        raise AssertionError(f"unexpected request {request.url}")


@pytest.mark.parametrize(
    ("arguments", "exception"),
    [
        ({"market_refresh_hours": True}, TypeError),
        ({"market_refresh_hours": float("nan")}, ValueError),
        ({"market_refresh_hours": float("inf")}, ValueError),
        ({"timeout": 0}, ValueError),
        ({"timeout": float("nan")}, ValueError),
        ({"retries": True}, TypeError),
        ({"retries": -1}, ValueError),
        ({"backoff": -0.1}, ValueError),
        ({"backoff": float("inf")}, ValueError),
        ({"earliest_date": "2999-01-01"}, ValueError),
    ],
)
def test_okx_constructor_rejects_invalid_runtime_settings(
    tmp_path: Path, arguments: dict[str, object], exception: type[Exception]
) -> None:
    """Confirm malformed runtime settings fail before disk or network access."""
    with pytest.raises(exception):
        OKX(tmp_path, progress=False, **arguments)  # type: ignore[arg-type]


def test_public_spot_klines_cache_query_and_offline_reuse(tmp_path: Path) -> None:
    """Confirm a cold request becomes an exact reusable DataFrame."""
    fixture = OKXFixture()
    service = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    frame = service.get_klines(
        "BTC-USDT", "2025-01-01", "2025-01-01", gap_policy="keep"
    )
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert frame["open_time"].min() == pd.Timestamp("2025-01-01", tz="UTC")
    assert frame.attrs["download"]["complete"] is True
    assert fixture.files == 2

    cached = service.get_klines(
        "BTC-USDT",
        "2025-01-01",
        "2025-01-01",
        gap_policy="keep",
        offline=True,
    )
    assert isinstance(cached, pd.DataFrame)
    pd.testing.assert_frame_equal(frame, cached)
    assert fixture.files == 2


def test_partial_only_resampling_is_a_complete_empty_okx_result(tmp_path: Path) -> None:
    """Confirm exact sub-bucket requests are not mislabeled as missing data."""
    fixture = OKXFixture()
    service = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    start = datetime(2025, 1, 1, 0, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 0, 2, tzinfo=UTC)

    frame = service.get_klines("BTC-USDT", start, end, interval="5m", gap_policy="keep")

    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    report = frame.attrs["download"]
    assert report["complete"] is True
    assert report["used_range"] == [start.isoformat(), end.isoformat()]
    assert [warning["code"] for warning in report["warnings"]] == [
        "partial_buckets_trimmed"
    ]
    assert report["problems"] == []


def test_public_multi_request_is_ordered_and_unknown_pair_isolated(
    tmp_path: Path,
) -> None:
    """Confirm one typo does not stop a valid instrument in the same request."""
    fixture = OKXFixture()
    service = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    frames = service.get_klines(
        ["BTCSUDT", "BTC-USDT"],
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 2, tzinfo=UTC),
        gap_policy="keep",
    )
    assert isinstance(frames, list)
    assert frames[0].empty
    assert frames[0].attrs["download"]["errors"][0]["code"] == "unknown_pair"
    assert frames[0].attrs["download"]["errors"][0]["suggestions"] == ["BTC-USDT"]
    assert len(frames[1]) == 2


def test_public_multi_request_isolates_one_source_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm a manifest failure becomes one error without stopping siblings."""
    api = OKX(tmp_path, earliest_date="all", retries=0, progress=False)
    markets = [
        Market(
            name,
            name.replace("-", ""),
            base_asset=name.split("-")[0],
            quote_asset="USDT",
            status="live",
            source="okx",
            product="spot",
            active=True,
        )
        for name in ("BTC-USDT", "ETH-USDT")
    ]
    monkeypatch.setattr(api._service, "_markets", lambda *args, **kwargs: markets)

    def retrieve(*args: object, **kwargs: object) -> Result:
        """Fail BTC and return a completed ETH result."""
        market = args[2]
        request = args[5]
        assert isinstance(market, Market)
        if market.symbol == "BTC-USDT":
            raise OKXResponseError("50011", "source busy", retryable=True)
        return Result(
            market.symbol,
            pd.DataFrame({"open_time": pd.Series(dtype="datetime64[us, UTC]")}),
            (request.start, request.end),  # type: ignore[attr-defined]
            used_range=(request.start, request.end),  # type: ignore[attr-defined]
            source="okx",
            product="spot",
            dataset="klines",
        )

    monkeypatch.setattr(api._service, "_pair", retrieve)
    frames = api.get_klines(
        ["BTC-USDT", "ETH-USDT"], "2025-01-01", "2025-01-01", gap_policy="keep"
    )
    assert isinstance(frames, list)
    assert frames[0].attrs["download"]["errors"][0]["code"] == "source_failed"
    assert frames[1].attrs["download"]["complete"] is True


def test_strict_gap_failure_is_not_converted_to_a_source_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm the documented strict gap policy still raises its typed error."""
    api = OKX(tmp_path, earliest_date="all", retries=0, progress=False)
    market = Market(
        "BTC-USDT",
        "BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        status="live",
        source="okx",
        product="spot",
        active=True,
    )
    monkeypatch.setattr(api._service, "_markets", lambda *args, **kwargs: [market])

    def strict_failure(*args: object, **kwargs: object) -> Result:
        """Raise the same typed failure produced by strict Kline querying."""
        raise MissingCandlesError(
            "BTC-USDT",
            [
                Gap(
                    datetime(2025, 1, 1, tzinfo=UTC),
                    datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
                    1,
                )
            ],
        )

    monkeypatch.setattr(api._service, "_pair", strict_failure)
    with pytest.raises(MissingCandlesError):
        api.get_klines("BTC-USDT", "2025-01-01", "2025-01-01", gap_policy="raise")


def test_corrupted_okx_parquet_is_rebuilt_once_online(tmp_path: Path) -> None:
    """Confirm an unreadable shared cache file is removed and downloaded again."""
    fixture = OKXFixture()
    api = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    first = api.get_klines("BTC-USDT", "2025-01-01", "2025-01-01", gap_policy="keep")
    assert isinstance(first, pd.DataFrame)
    parquet = next((tmp_path / "okx").rglob("*.parquet"))
    parquet.write_bytes(b"not parquet")

    rebuilt = api.get_klines("BTC-USDT", "2025-01-01", "2025-01-01", gap_policy="keep")
    assert isinstance(rebuilt, pd.DataFrame)
    assert len(rebuilt) == 2
    assert rebuilt.attrs["download"]["complete"] is True
    assert fixture.files == 4


def test_corrupted_okx_parquet_is_reported_cleanly_offline(tmp_path: Path) -> None:
    """Confirm offline cache corruption returns a structured recoverable error."""
    fixture = OKXFixture()
    api = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    api.get_klines("BTC-USDT", "2025-01-01", "2025-01-01", gap_policy="keep")
    parquet = next((tmp_path / "okx").rglob("*.parquet"))
    parquet.write_bytes(b"not parquet")

    result = api.get_klines(
        "BTC-USDT",
        "2025-01-01",
        "2025-01-01",
        gap_policy="keep",
        offline=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert result.attrs["download"]["errors"][0]["code"] == "query_failed"
    assert not parquet.exists()


def test_okx_archive_materialization_does_not_swallow_interrupts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm Ctrl+C escapes archive workers instead of becoming a data gap."""
    fixture = OKXFixture()
    api = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )

    def interrupted(*args: object, **kwargs: object) -> object:
        """Simulate an interrupt raised while awaiting archive materialization."""
        raise KeyboardInterrupt()

    monkeypatch.setattr(OKXArchiveProvider, "materialize", interrupted)
    with pytest.raises(KeyboardInterrupt):
        api.get_klines("BTC-USDT", "2025-01-01", "2025-01-01", gap_policy="keep")
