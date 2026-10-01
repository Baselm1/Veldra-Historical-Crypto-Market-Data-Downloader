"""Test Bitget reference-market history."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import pytest

from veldra.bitget.client import BitgetResponseError
from veldra.bitget.reference import BitgetReferenceService, candle_frame, funding_frame


class Client:
    """Return deterministic endpoint values and record requests."""

    def __init__(self, values: list[object]) -> None:
        """Store queued response data."""
        self.values = values
        self.calls: list[tuple[str, str, Mapping[str, str]]] = []

    def request(
        self, _method: str, url: str, *, policy_key: str, params: Mapping[str, str]
    ) -> object:
        """Return the next queued data value."""
        self.calls.append((url, policy_key, params))
        return self.values.pop(0)


def test_candle_frame_is_canonical_sorted_and_deduplicated() -> None:
    """Array candles become exact canonical DataFrame rows."""
    frame = candle_frame(
        [
            ["1735689660000", "2", "3", "1", "2.5", "4", "10"],
            ["1735689600000", "1", "2", ".5", "1.5", "3", "5"],
        ]
    )
    assert list(frame.columns) == [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
    ]
    assert frame.open_time.is_monotonic_increasing


def test_reference_service_sends_candle_type_and_exact_bounds() -> None:
    """Reference types and half-open bounds map to the unified endpoint."""
    client = Client([[["1735689600000", "1", "2", ".5", "1.5", "3", "5"]]])
    service = BitgetReferenceService(client)  # type: ignore[arg-type]
    start = datetime(2025, 1, 1, tzinfo=UTC)
    result = service.candles(
        "BTCUSDT",
        "usdt_futures",
        "mark_price_klines",
        "1m",
        start,
        start + timedelta(minutes=100),
    )
    assert len(result) == 1
    assert client.calls[0][0].endswith("/api/v3/market/history-candles")
    assert client.calls[0][2]["type"] == "mark"
    assert client.calls[0][2]["limit"] == "100"
    assert client.calls[0][1] == "history_candles"


def test_reference_service_pages_one_minute_history_in_hundred_row_windows() -> None:
    """A full one-minute day uses bounded historical endpoint requests."""
    client = Client([[] for _ in range(15)])
    service = BitgetReferenceService(client)  # type: ignore[arg-type]
    start = datetime(2025, 1, 1, tzinfo=UTC)
    result = service.candles(
        "BTCUSDT",
        "usdt_futures",
        "index_price_klines",
        "1m",
        start,
        start + timedelta(days=1),
    )
    assert result.empty
    assert len(client.calls) == 15


def test_funding_pages_until_short_page_and_filters_range() -> None:
    """Funding pagination terminates and returns exact requested timestamps."""
    rows = [
        {"fundingRateTimestamp": str(1735689600000 + index), "fundingRate": "0.001"}
        for index in range(100)
    ]
    client = Client([{"resultList": rows}, {"resultList": []}])
    service = BitgetReferenceService(client)  # type: ignore[arg-type]
    start = datetime(2025, 1, 1, tzinfo=UTC)
    result = service.funding("BTCUSDT", "usdt_futures", start, start.replace(day=2))
    assert len(result) == 100
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "value", [[["bad"]], [{"fundingRateTimestamp": "x", "fundingRate": "1"}]]
)
def test_malformed_reference_rows_fail_closed(value: list[object]) -> None:
    """Invalid row shapes and timestamps cannot enter returned frames."""
    with pytest.raises(BitgetResponseError):
        if isinstance(value[0], list):
            candle_frame(value)
        else:
            funding_frame(value)


def test_empty_reference_frames_have_stable_dtypes() -> None:
    """Empty endpoint results still expose usable UTC schemas."""
    assert str(candle_frame([]).open_time.dtype) == "datetime64[us, UTC]"
    assert str(funding_frame([]).funding_time.dtype) == "datetime64[us, UTC]"
