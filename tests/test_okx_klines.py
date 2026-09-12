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
    IntegritySpec,
)
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
