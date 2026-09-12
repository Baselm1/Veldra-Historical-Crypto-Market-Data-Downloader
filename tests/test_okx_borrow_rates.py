"""Test currency-scoped OKX margin borrowing archives."""

from base64 import b64encode
from datetime import UTC, datetime
from hashlib import md5
from io import BytesIO
from pathlib import Path
import zipfile

import httpx
import pandas as pd

from veldra import OKX
from veldra.okx.datasets import BORROW_RATE_SOURCE_COLUMNS


def borrow_archive(name: str) -> bytes:
    """Return one monthly borrowing archive for its named currency."""
    currency = name.split("-", 1)[0]
    rows = (
        [
            ["BTC", "0.0001", "1735689600000"],
            ["BTC", "0.0003", "1735693200000"],
        ]
        if currency == "BTC"
        else [["USDT", "0.0002", "1735689600000"]]
    )
    frame = pd.DataFrame(
        rows,
        columns=BORROW_RATE_SOURCE_COLUMNS,
    )
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name.removesuffix(".zip") + ".csv", frame.to_csv(index=False))
    return output.getvalue()


class BorrowFixture:
    """Serve one currency-scoped borrowing manifest and archive."""

    def __init__(self) -> None:
        """Create a physical archive counter."""
        self.files = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return the manifest or borrowing archive response."""
        if request.url.path.endswith("/market-data-history"):
            assert request.url.params["ccyList"] in {"BTC", "USDT"}
            assert request.url.params["dateAggrType"] == "monthly"
            currency = request.url.params["ccyList"]
            name = f"{currency}-borrowrates-2025-01.zip"
            group = {
                "ccy": currency,
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
            content = borrow_archive(Path(request.url.path).name)
            return httpx.Response(
                200,
                content=content,
                headers={"Content-MD5": b64encode(md5(content).digest()).decode()},
                request=request,
            )
        raise AssertionError(f"unexpected request {request.url}")


def test_borrow_rates_preserve_currency_scope_and_reuse_cache(tmp_path: Path) -> None:
    """Confirm currency archives become separately queryable logical rows."""
    fixture = BorrowFixture()
    api = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 2, tzinfo=UTC)
    frames = api.get_borrow_rates(["btc", "USDT"], start, end)
    assert isinstance(frames, list)
    assert frames[0]["borrow_rate"].tolist() == [0.0001, 0.0003]
    assert frames[1]["borrow_rate"].tolist() == [0.0002]
    assert frames[0].attrs["download"]["pair"] == "BTC"

    cached = api.get_borrow_rates("BTC", start, end, offline=True)
    assert isinstance(cached, pd.DataFrame)
    assert len(cached) == 2
    assert fixture.files == 2


def test_borrow_rates_honor_the_configured_history_floor(tmp_path: Path) -> None:
    """Confirm a wholly excluded borrowing range performs no source request."""
    fixture = BorrowFixture()
    api = OKX(
        tmp_path,
        earliest_date="2025-01-02",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    frame = api.get_borrow_rates("BTC", "2025-01-01", "2025-01-01")
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    assert frame.attrs["download"]["warnings"][0]["code"] == "configured_start"
    assert fixture.files == 0
