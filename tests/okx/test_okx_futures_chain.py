"""Test exact and family-wide OKX dated Futures history."""

from base64 import b64encode
from datetime import UTC, datetime
from hashlib import md5
from io import BytesIO
from pathlib import Path
import zipfile

import httpx
import pandas as pd

from veldra import OKX
from veldra.okx.datasets import KLINE_SOURCE_COLUMNS


def archive_bytes(name: str) -> bytes:
    """Return one family archive with expired and current Futures Klines."""
    rows: list[list[object]] = []
    for instrument, price in (("BTC-USD-250103", 100), ("BTC-USD-250131", 200)):
        rows.extend(
            [
                [
                    instrument,
                    price,
                    price + 2,
                    price - 1,
                    price + 1,
                    2,
                    0.2,
                    20,
                    1735689600000,
                    1,
                ],
                [
                    instrument,
                    price + 1,
                    price + 3,
                    price,
                    price + 2,
                    3,
                    0.3,
                    30,
                    1735689660000,
                    1,
                ],
            ]
        )
    frame = pd.DataFrame(rows, columns=KLINE_SOURCE_COLUMNS)
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name.removesuffix(".zip") + ".csv", frame.to_csv(index=False))
    return output.getvalue()


class FuturesFixture:
    """Serve one current contract and one family archive containing two contracts."""

    def __init__(self) -> None:
        """Create a physical download counter."""
        self.files = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return current metadata, a family manifest, or its archive."""
        if request.url.path.endswith("/instruments"):
            row = {
                "instType": "FUTURES",
                "instId": "BTC-USD-250131",
                "instFamily": "BTC-USD",
                "baseCcy": "BTC",
                "quoteCcy": "USD",
                "settleCcy": "BTC",
                "ctType": "inverse",
                "ctVal": "100",
                "ctMult": "1",
                "ctValCcy": "USD",
                "state": "live",
                "ruleType": "normal",
                "listTime": "1609459200000",
                "expTime": "1738281600000",
                "stk": "",
                "optType": "",
            }
            return httpx.Response(
                200, json={"code": "0", "msg": "", "data": [row]}, request=request
            )
        if request.url.path.endswith("/market-data-history"):
            assert request.url.params["instFamilyList"] == "BTC-USD"
            name = "BTC-USD-futureschain-candlesticks-2025-01-01.zip"
            group = {
                "instFamily": "BTC-USD",
                "instId": "",
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


def test_expired_exact_contract_and_whole_chain_share_one_file(tmp_path: Path) -> None:
    """Confirm exact and chain queries reuse one family materialization."""
    fixture = FuturesFixture()
    api = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 0, 2, tzinfo=UTC)
    exact = api.get_klines(
        "BTC-USD-250103",
        start,
        end,
        product="inverse_futures",
        gap_policy="keep",
    )
    assert isinstance(exact, pd.DataFrame)
    assert len(exact) == 2 and exact.loc[0, "open"] == 100
    assert exact.attrs["download"]["complete"] is True

    chain = api.get_futures_chain_klines(
        "BTC-USD",
        start,
        end,
        product="inverse_futures",
        interval="1h",
        offline=True,
    )
    assert len(chain) == 2
    assert chain["instrument_id"].tolist() == ["BTC-USD-250103", "BTC-USD-250131"]
    assert chain["contract_volume"].tolist() == [5.0, 5.0]
    assert fixture.files == 1


def test_chain_style_filter_and_product_family_validation(tmp_path: Path) -> None:
    """Confirm family and style filters remain explicit and product-safe."""
    fixture = FuturesFixture()
    api = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 0, 2, tzinfo=UTC)
    frame = api.get_futures_chain_klines(
        "BTC-USD",
        start,
        end,
        product="inverse_futures",
        contract_style="normal",
    )
    assert len(frame) == 4
    try:
        api.get_futures_chain_klines("BTC-USDT", start, end, product="inverse_futures")
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched Futures family was accepted")
