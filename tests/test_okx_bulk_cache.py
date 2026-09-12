"""Test explicit OKX all-market cache plans and reports."""

from base64 import b64encode
from datetime import UTC, datetime
from hashlib import md5
from io import BytesIO
from pathlib import Path
import zipfile

import httpx
import pandas as pd
import pytest

from veldra import OKX
from veldra.okx.datasets import KLINE_SOURCE_COLUMNS
from veldra.okx.reports import CacheReport


def archive_bytes(name: str) -> bytes:
    """Return one tiny all-Spot Kline archive containing two instruments."""
    frame = pd.DataFrame(
        [
            ["BTC-USDT", 100, 102, 99, 101, 2, 201, 201, 1735689600000, 1],
            ["ETH-USDT", 10, 12, 9, 11, 3, 33, 33, 1735689600000, 1],
        ],
        columns=KLINE_SOURCE_COLUMNS,
    )
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name.removesuffix(".zip") + ".csv", frame.to_csv(index=False))
    return output.getvalue()


class BulkFixture:
    """Serve current Spot metadata and one all-market source archive."""

    def __init__(self) -> None:
        """Create manifest and file counters."""
        self.manifests = 0
        self.files = 0

    @staticmethod
    def _instrument(symbol: str) -> dict[str, object]:
        """Return one complete current Spot instrument row."""
        base, quote = symbol.split("-")
        return {
            "instType": "SPOT",
            "instId": symbol,
            "instFamily": "",
            "baseCcy": base,
            "quoteCcy": quote,
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
        """Return one response for each bulk-cache operation."""
        if request.url.path.endswith("/instruments"):
            rows = [self._instrument("BTC-USDT"), self._instrument("ETH-USDT")]
            return httpx.Response(
                200, json={"code": "0", "msg": "", "data": rows}, request=request
            )
        if request.url.path.endswith("/market-data-history"):
            self.manifests += 1
            assert request.url.params["instIdList"] == "ANY"
            name = "allspot-candlesticks-2025-01-01.zip"
            group = {
                "instId": "",
                "instFamily": "",
                "groupDetails": [
                    {
                        "dataTs": "1735689600000",
                        "filename": name,
                        "sizeMB": "0.01",
                        "url": f"https://files.test/{name}",
                    }
                ],
            }
            return httpx.Response(
                200,
                json={"code": "0", "msg": "", "data": [{"details": [group]}]},
                request=request,
            )
        if request.url.host == "files.test":
            self.files += 1
            name = Path(request.url.path).name
            content = archive_bytes(name)
            return httpx.Response(
                200,
                content=content,
                headers={"Content-MD5": b64encode(md5(content).digest()).decode()},
                request=request,
            )
        raise AssertionError(f"unexpected request {request.url}")


def range_values() -> tuple[datetime, datetime]:
    """Return a two-minute exact test range within one source day."""
    return (
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 2, tzinfo=UTC),
    )


def test_bulk_cache_dry_cold_warm_offline_and_narrow_reuse(tmp_path: Path) -> None:
    """Confirm one shared file is planned once, cached once, and queried narrowly."""
    fixture = BulkFixture()
    api = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    start, end = range_values()
    dry = api.cache_klines(start, end, dry_run=True)
    assert dry.remote_files == 1 and dry.remote_bytes > 0
    assert dry.downloaded_files == 0 and not dry.complete
    assert fixture.files == 0

    cold = api.cache_klines(start, end)
    assert cold.downloaded_files == 1 and cold.remote_files == 0
    assert cold.logical_subjects == 2 and cold.cached_rows == 2
    assert cold.complete and cold.local_bytes > 0
    assert fixture.files == 1

    warm = api.cache_klines(start, end)
    assert warm.cached_files == 1 and warm.downloaded_files == 0
    assert warm.complete and fixture.files == 1

    offline = api.cache_klines(start, end, offline=True)
    assert offline.offline and offline.cached_files == 1 and offline.complete

    frame = api.get_klines("BTC-USDT", start, end, gap_policy="keep", offline=True)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1 and frame.loc[0, "close"] == 101


def test_bulk_facade_rejects_non_all_subjects(tmp_path: Path) -> None:
    """Confirm cache-only wrappers cannot masquerade as narrow retrieval."""
    api = OKX(tmp_path, progress=False)
    with pytest.raises(ValueError, match="must be 'all'"):
        api.cache_trades("2025-01-01", "2025-01-02", instruments="BTC-USDT")  # type: ignore[arg-type]


def test_cache_report_validates_shape_and_completion() -> None:
    """Confirm reports reject negative counters and empty caches are incomplete."""
    start, end = range_values()
    report = CacheReport(
        "okx",
        "spot",
        "klines",
        (start, end),
        "bulk",
        "empty",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        False,
        True,
    )
    assert not report.complete
    with pytest.raises(ValueError, match="negative"):
        CacheReport(
            "okx",
            "spot",
            "klines",
            (start, end),
            "bulk",
            "bad",
            -1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            False,
            False,
        )
