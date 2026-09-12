"""Test cached OKX public historical REST datasets."""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from veldra import OKX
from veldra.okx.client import OKXResponseError
from veldra.okx.rest import REST_SPECS, normalize_rest


class RESTFixture:
    """Serve representative rows for every supplemental public endpoint."""

    def __init__(self) -> None:
        """Create request counters by endpoint path."""
        self.calls: dict[str, int] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return one valid historical endpoint envelope."""
        path = request.url.path
        self.calls[path] = self.calls.get(path, 0) + 1
        if path.endswith("index-candles") or path.endswith("mark-price-candles"):
            data: list[object] = [
                ["1735689600000", "100", "102", "99", "101", "1"],
                ["1735689660000", "101", "103", "100", "102", "1"],
            ]
        elif path.endswith("premium-history"):
            data = [
                {"instId": "BTC-USDT-SWAP", "premium": "0.001", "ts": "1735689600000"}
            ]
        elif path.endswith("funding-rate-history"):
            data = [
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingRate": "0.0001",
                    "realizedRate": "0.00009",
                    "fundingTime": "1735689600000",
                    "formulaType": "withRate",
                    "method": "current_period",
                }
            ]
        elif path.endswith("settlement-history"):
            data = [
                {
                    "ts": "1735689600000",
                    "details": [{"instId": "BTC-USD-250103", "settlePx": "100"}],
                }
            ]
        elif path.endswith("delivery-exercise-history"):
            data = [
                {
                    "ts": "1735689600000",
                    "details": [
                        {
                            "insId": "BTC-USD-250103-100000-P",
                            "px": "0.01",
                            "type": "exercised",
                        }
                    ],
                }
            ]
        elif path.endswith("taker-volume"):
            data = [["1735689600000", "20", "30"]]
        elif path.endswith("long-short-account-ratio"):
            data = [["1735689600000", "1.2"]]
        elif path.endswith("open-interest-volume"):
            data = [["1735689600000", "1000", "2000"]]
        else:
            raise AssertionError(f"unexpected request {request.url}")
        return httpx.Response(
            200, json={"code": "0", "msg": "", "data": data}, request=request
        )


def configured(tmp_path: Path, fixture: RESTFixture) -> OKX:
    """Return an isolated facade using the REST fixture.

    Args:
        tmp_path: Temporary storage root.
        fixture: Source response handler.

    Returns:
        Configured OKX facade.
    """
    return OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )


def test_price_rate_and_lifecycle_history_is_cached(tmp_path: Path) -> None:
    """Confirm public REST methods normalize and cache their native shapes."""
    fixture = RESTFixture()
    api = configured(tmp_path, fixture)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 0, 2, tzinfo=UTC)

    index = api.get_index_price_klines("BTC-USD", start, end)
    mark = api.get_mark_price_klines("BTC-USDT-SWAP", start, end, product="linear_swap")
    premium = api.get_premium_history(
        "BTC-USDT-SWAP", start, end, product="linear_swap"
    )
    funding = api.get_recent_funding_rates(
        "BTC-USDT-SWAP", start, end, product="linear_swap"
    )
    settlements = api.get_settlement_history(
        "BTC-USD", start, end, product="inverse_futures"
    )
    exercises = api.get_delivery_exercise_history(
        "BTC-USD", start, end, product="options"
    )

    assert index["close"].tolist() == [101.0, 102.0]
    assert mark["open"].tolist() == [100.0, 101.0]
    assert premium["premium"].tolist() == [0.001]
    assert funding["realized_rate"].tolist() == [0.00009]
    assert settlements["event_type"].tolist() == ["settlement"]
    assert exercises["event_type"].tolist() == ["exercised"]

    cached = api.get_index_price_klines("BTC-USD", start, end, offline=True)
    assert len(cached) == 2
    assert fixture.calls["/api/v5/market/history-index-candles"] == 1


def test_trading_statistics_keep_distinct_schemas(tmp_path: Path) -> None:
    """Confirm each analytical endpoint retains its declared quantities."""
    fixture = RESTFixture()
    api = configured(tmp_path, fixture)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 0, 1, tzinfo=UTC)

    interest = api.get_open_interest_history("BTC", start, end)
    taker = api.get_taker_volume("BTC", start, end, market="spot")
    ratio = api.get_long_short_ratio("BTC", start, end)
    options = api.get_option_interest_volume("BTC", start, end)

    assert interest[["open_interest", "volume"]].iloc[0].tolist() == [1000, 2000]
    assert taker[["sell_volume", "buy_volume"]].iloc[0].tolist() == [20, 30]
    assert ratio["ratio"].tolist() == [1.2]
    assert options[["open_interest", "volume"]].iloc[0].tolist() == [1000, 2000]


@pytest.mark.parametrize(
    "call",
    [
        lambda api: api.get_index_price_klines(
            "BTC-USD", "2025-01-01", "2025-01-02", interval="2d"
        ),
        lambda api: api.get_open_interest_history(
            "BTC", "2025-01-01", "2025-01-02", period="2h"
        ),
        lambda api: api.get_taker_volume(
            "BTC", "2025-01-01", "2025-01-02", market="future"
        ),
        lambda api: api.get_option_interest_volume(
            "BTC", "2025-01-01", "2025-01-02", period="1h"
        ),
    ],
)
def test_rest_facade_rejects_unsupported_enums(
    tmp_path: Path, call: Callable[[OKX], object]
) -> None:
    """Confirm REST facade enums fail before network access.

    Args:
        tmp_path: Temporary storage root.
        call: Invalid facade invocation.
    """
    api = OKX(tmp_path, progress=False)
    with pytest.raises(ValueError):
        call(api)


def test_rest_normalizer_rejects_malformed_rows() -> None:
    """Confirm malformed source shapes never reach the cache."""
    with pytest.raises(OKXResponseError, match="shape"):
        normalize_rest(REST_SPECS["index_price_klines"], [["1"]])
    with pytest.raises(OKXResponseError, match="objects"):
        normalize_rest(REST_SPECS["premium_history"], [["not", "an", "object"]])
    with pytest.raises(OKXResponseError, match="details"):
        normalize_rest(REST_SPECS["settlements"], [{"ts": "1", "details": {}}])


def test_rest_pagination_stops_at_requested_start_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm descending pages stop at the range boundary and cache exactly."""
    monkeypatch.setitem(
        REST_SPECS,
        "premium_history",
        replace(REST_SPECS["premium_history"], page_size=2),
    )
    calls = 0

    def pages(request: httpx.Request) -> httpx.Response:
        """Return two descending premium pages."""
        nonlocal calls
        calls += 1
        after = int(request.url.params["after"])
        timestamps = (
            ["1735689720000", "1735689660000"]
            if after > 1735689660000
            else ["1735689600000"]
        )
        data = [
            {"instId": "BTC-USDT-SWAP", "premium": "0.001", "ts": timestamp}
            for timestamp in timestamps
        ]
        return httpx.Response(
            200, json={"code": "0", "msg": "", "data": data}, request=request
        )

    api = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(pages),
    )
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 0, 3, tzinfo=UTC)
    frame = api.get_premium_history("BTC-USDT-SWAP", start, end, product="linear_swap")
    assert len(frame) == 3 and calls == 2
    cached = api.get_premium_history(
        "BTC-USDT-SWAP", start, end, product="linear_swap", offline=True
    )
    assert len(cached) == 3 and calls == 2


def test_empty_rest_range_is_cached_as_an_empty_typed_frame(tmp_path: Path) -> None:
    """Confirm valid empty success responses remain queryable offline."""
    calls = 0

    def empty(request: httpx.Request) -> httpx.Response:
        """Return a valid empty OKX envelope."""
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"code": "0", "msg": "", "data": []}, request=request
        )

    api = OKX(
        tmp_path,
        retries=0,
        progress=False,
        transport=httpx.MockTransport(empty),
    )
    first = api.get_index_price_klines("NEW-USD", "2025-01-01", "2025-01-02")
    second = api.get_index_price_klines(
        "NEW-USD", "2025-01-01", "2025-01-02", offline=True
    )
    assert first.empty and second.empty and calls == 1
    assert str(second["open_time"].dtype) == "datetime64[us, UTC]"
