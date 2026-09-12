"""Test exact and family-wide OKX Options archive history."""

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
from veldra.okx.chain import OptionChainFilter
from veldra.okx.datasets import KLINE_SOURCE_COLUMNS, get_dataset
from veldra.okx.identities import historical_option


def option_archive(name: str) -> bytes:
    """Return one family archive containing three expired Option contracts."""
    rows: list[list[object]] = []
    contracts = (
        ("BTC-USD-250103-90000-C", 100),
        ("BTC-USD-250103-90000-P", 200),
        ("BTC-USD-250103-100000-P", 300),
    )
    for instrument, price in contracts:
        rows.append(
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
            ]
        )
    frame = pd.DataFrame(rows, columns=KLINE_SOURCE_COLUMNS)
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name.removesuffix(".zip") + ".csv", frame.to_csv(index=False))
    return output.getvalue()


class OptionsFixture:
    """Serve current Options metadata and one expired-chain archive."""

    def __init__(self) -> None:
        """Create a physical archive counter."""
        self.files = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return one public API or archive response."""
        if request.url.path.endswith("/underlying"):
            return httpx.Response(
                200,
                json={"code": "0", "msg": "", "data": [["BTC-USD"]]},
                request=request,
            )
        if request.url.path.endswith("/instruments"):
            row = {
                "instType": "OPTION",
                "instId": "BTC-USD-260327-100000-C",
                "instFamily": "BTC-USD",
                "baseCcy": "",
                "quoteCcy": "",
                "settleCcy": "BTC",
                "ctType": "",
                "ctVal": "0.01",
                "ctMult": "1",
                "ctValCcy": "BTC",
                "state": "live",
                "ruleType": "normal",
                "listTime": "1735689600000",
                "expTime": "1774569600000",
                "stk": "100000",
                "optType": "C",
            }
            return httpx.Response(
                200, json={"code": "0", "msg": "", "data": [row]}, request=request
            )
        if request.url.path.endswith("/market-data-history"):
            assert request.url.params["instFamilyList"] == "BTC-USD"
            name = "BTC-USD-option-candlesticks-2025-01-01.zip"
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
            content = option_archive(Path(request.url.path).name)
            return httpx.Response(
                200,
                content=content,
                headers={"Content-MD5": b64encode(md5(content).digest()).decode()},
                request=request,
            )
        raise AssertionError(f"unexpected request {request.url}")


def test_exact_expired_option_and_filtered_chain_share_archive(tmp_path: Path) -> None:
    """Confirm exact and filtered chain queries share one family file."""
    fixture = OptionsFixture()
    api = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 0, 1, tzinfo=UTC)
    exact = api.get_klines(
        "BTC-USD-250103-90000-C",
        start,
        end,
        product="options",
        gap_policy="keep",
    )
    assert isinstance(exact, pd.DataFrame)
    assert exact["open"].tolist() == [100]

    chain = api.get_option_chain_klines(
        "BTC-USD",
        start,
        end,
        expiry="2025-01-03",
        strike_min=95000,
        option_type="put",
        offline=True,
    )
    assert chain["instrument_id"].tolist() == ["BTC-USD-250103-100000-P"]
    assert fixture.files == 1


def test_option_declarations_and_archive_identity() -> None:
    """Confirm Option datasets and expired identities retain native units."""
    klines = get_dataset("options", "klines")
    trades = get_dataset("options", "trades")
    order_book = get_dataset("options", "order_book_400")
    assert klines.stored_columns[-3:] == (
        "contract_volume",
        "base_volume",
        "quote_volume",
    )
    assert "contract_quantity" in trades.stored_columns
    assert order_book.max_concurrency == 4
    identity = historical_option("btc-usd-250103-90000-p")
    assert identity.family == "BTC-USD"
    assert identity.expiry == date(2025, 1, 3)
    assert identity.strike == 90000
    assert identity.option_type == "P"
    assert identity.provenance == "archive_identity"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"strike_min": -1}, "strike_min"),
        ({"strike_max": -1}, "strike_max"),
        ({"strike_min": 2, "strike_max": 1}, "exceed"),
        ({"option_type": "X"}, "option_type"),
    ],
)
def test_option_chain_filter_rejects_invalid_values(
    arguments: dict[str, object], message: str
) -> None:
    """Confirm Option query filters reject malformed constraints.

    Args:
        arguments: Invalid filter fields.
        message: Expected error text.
    """
    with pytest.raises(ValueError, match=message):
        OptionChainFilter(**arguments)  # type: ignore[arg-type]


def test_option_facade_rejects_wrong_filter_types_and_family(tmp_path: Path) -> None:
    """Confirm public Option filters fail before source access."""
    api = OKX(tmp_path, progress=False)
    with pytest.raises(TypeError, match="strike_min"):
        api.get_option_chain_klines(
            "BTC-USD", "2025-01-01", "2025-01-02", strike_min="1"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="option_type"):
        api.get_option_chain_trades(
            "BTC-USD", "2025-01-01", "2025-01-02", option_type="x"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="does not match"):
        api.get_option_chain_klines("BTC-USDC", "2025-01-01", "2025-01-02")
