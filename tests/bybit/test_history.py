"""Test Bybit native Kline history normalization and pagination."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from veldra.bybit.client import BybitResponseError
from veldra.bybit.history import BybitHistory, kline_frame, object_frame


class Client:
    """Return configured V5 pages while retaining request parameters."""

    def __init__(self, pages: list[object]) -> None:
        """Store source pages in call order."""
        self.pages = pages
        self.calls: list[tuple[str, dict[str, str]]] = []

    def v5(self, path: str, params: Mapping[str, str] | None = None) -> object:
        """Return the next configured page."""
        assert params is not None
        self.calls.append((path, dict(params)))
        return self.pages.pop(0)


def _row(timestamp: int, *, volumes: bool = True) -> list[object]:
    """Return one representative native Kline array."""
    values: list[object] = [str(timestamp), "100", "110", "90", "105"]
    if volumes:
        values.extend(("2", "205"))
    return values


def test_market_klines_preserve_product_specific_quantity_units() -> None:
    """Name inverse contract and base quantities without false quote units."""
    spot = kline_frame([_row(1735689600000)], "spot", "klines")
    inverse = kline_frame([_row(1735689600000)], "inverse", "klines")
    assert spot.columns.tolist()[-2:] == ["base_volume", "quote_volume"]
    assert inverse.columns.tolist()[-2:] == ["contract_volume", "base_volume"]
    assert inverse.iloc[0, -2:].tolist() == [2.0, 205.0]


def test_reference_klines_are_ohlc_only() -> None:
    """Do not invent volume fields absent from reference-price endpoints."""
    frame = kline_frame(
        [_row(1735689600000, volumes=False)], "linear", "mark_price_klines"
    )
    assert frame.columns.tolist() == ["open_time", "open", "high", "low", "close"]


def test_kline_history_pages_backward_and_filters_exact_range() -> None:
    """Advance descending pages without duplicating an inclusive cursor."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    first = [_row(1735690600000 - index) for index in range(1000)]
    second = [_row(1735689600000), _row(1735689599999)]
    client = Client([{"list": first}, {"list": second}])
    frame = BybitHistory(client).klines(
        "BTCUSDT", "spot", "klines", "1m", start, start + timedelta(days=1)
    )
    assert not frame.empty
    assert (frame.open_time >= start).all()
    assert int(client.calls[1][1]["end"]) < int(client.calls[0][1]["end"])
    assert client.calls[0][1]["category"] == "spot"


@pytest.mark.parametrize(
    "rows",
    [
        [["bad", "1", "2", "0", "1", "1", "1"]],
        [["1735689600000", "100", "90", "95", "105", "1", "1"]],
        [["1735689600000", "100", "110", "90", "105", "-1", "1"]],
        [["1735689600000"]],
    ],
)
def test_invalid_kline_rows_fail_closed(rows: list[object]) -> None:
    """Reject bad units, OHLC bounds, quantities, and row shapes."""
    with pytest.raises(BybitResponseError):
        kline_frame(rows, "spot", "klines")


def test_empty_kline_frame_keeps_typed_schema() -> None:
    """Return usable UTC and numeric dtypes when the source has no rows."""
    frame = kline_frame([], "spot", "klines")
    assert str(frame.open_time.dtype) == "datetime64[us, UTC]"
    assert frame.base_volume.dtype == "float64"


def test_empty_volatility_frame_keeps_integer_period() -> None:
    """Keep the declared period type when volatility history is absent."""
    frame = object_frame([], "historical_volatility")
    assert frame.period.dtype == "int64"
    assert str(frame.event_time.dtype) == "datetime64[us, UTC]"


def test_history_validates_ranges_limits_and_result_shape() -> None:
    """Reject invalid controls and malformed endpoint envelopes."""
    now = datetime.now(UTC)
    history = BybitHistory(Client([]))
    with pytest.raises(ValueError, match="end after"):
        history.klines("BTCUSDT", "spot", "klines", "1m", now, now)
    with pytest.raises(ValueError, match="positive"):
        history.klines(
            "BTCUSDT", "spot", "klines", "1m", now, now + timedelta(1), max_pages=0
        )
    with pytest.raises(BybitResponseError, match="result"):
        BybitHistory(Client([[]])).klines(
            "BTCUSDT", "spot", "klines", "1m", now, now + timedelta(1)
        )


def test_history_rejects_naive_datetime_boundaries() -> None:
    """Do not interpret timezone-naive caller values implicitly."""
    with pytest.raises(ValueError, match="timezone-aware"):
        BybitHistory(Client([])).klines(
            "BTCUSDT",
            "spot",
            "klines",
            "1m",
            datetime(2025, 1, 1),
            datetime(2025, 1, 2),
        )


def test_kline_frame_sorts_stably_and_deduplicates() -> None:
    """Return ascending unique timestamps from reverse source pages."""
    later = _row(1735689660000)
    frame = kline_frame([later, _row(1735689600000), later], "spot", "klines")
    assert frame.open_time.is_monotonic_increasing
    assert frame.open_time.is_unique
    assert isinstance(frame, pd.DataFrame)


@pytest.mark.parametrize(
    ("dataset", "row", "columns"),
    [
        (
            "funding_rates",
            {"fundingRateTimestamp": "1735689600000", "fundingRate": "-0.0001"},
            ["funding_time", "funding_rate"],
        ),
        (
            "open_interest",
            {
                "timestamp": "1735689600000",
                "openInterest": "100",
                "singleOpenInterest": "50",
            },
            ["event_time", "open_interest", "single_open_interest"],
        ),
        (
            "long_short_ratios",
            {"timestamp": "1735689600000", "buyRatio": "0.4", "sellRatio": "0.6"},
            ["event_time", "buy_ratio", "sell_ratio"],
        ),
        (
            "historical_volatility",
            {"time": "1735689600000", "period": 30, "value": "0.5"},
            ["event_time", "period", "volatility"],
        ),
        (
            "delivery_prices",
            {"deliveryTime": "1735689600000", "deliveryPrice": "95000"},
            ["delivery_time", "delivery_price"],
        ),
    ],
)
def test_object_histories_have_canonical_typed_columns(
    dataset: str, row: dict[str, object], columns: list[str]
) -> None:
    """Normalize each object-shaped public history consistently."""
    frame = object_frame([row], dataset)
    assert frame.columns.tolist() == columns
    assert str(frame.iloc[0, 0].tzinfo) == "UTC"


def test_old_open_interest_rows_allow_missing_single_side_value() -> None:
    """Preserve older history predating Bybit's single-side field."""
    frame = object_frame(
        [{"timestamp": "1735689600000", "openInterest": "100"}],
        "open_interest",
    )
    assert pd.isna(frame.single_open_interest.iloc[0])


def test_funding_history_pages_by_oldest_timestamp() -> None:
    """Page backward through the non-cursor funding endpoint."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [
        {
            "fundingRateTimestamp": str(1735700000000 - index),
            "fundingRate": "0.0001",
        }
        for index in range(200)
    ]
    client = Client([{"list": rows}, {"list": []}])
    frame = BybitHistory(client).funding_rates(
        "BTCUSDT", "linear", start, start + timedelta(days=1)
    )
    assert len(client.calls) == 2
    assert frame.funding_time.is_monotonic_increasing


@pytest.mark.parametrize("dataset", ["open_interest", "long_short_ratios"])
def test_position_histories_follow_source_cursors(dataset: str) -> None:
    """Use the endpoint cursor while retaining one exact range."""
    field = (
        {"openInterest": "100", "singleOpenInterest": "50"}
        if dataset == "open_interest"
        else {"buyRatio": "0.4", "sellRatio": "0.6"}
    )
    row = {"timestamp": "1735689600000", **field}
    client = Client(
        [
            {"list": [row], "nextPageCursor": "next"},
            {"list": [], "nextPageCursor": ""},
        ]
    )
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = BybitHistory(client).positions(
        "BTCUSDT", "linear", dataset, "1h", start, start + timedelta(days=1)
    )
    assert len(frame) == 1
    assert client.calls[1][1]["cursor"] == "next"


def test_volatility_splits_requests_at_thirty_days() -> None:
    """Respect Bybit's maximum Option volatility request window."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    client = Client([[], []])
    frame = BybitHistory(client).volatility(
        "BTC", 30, start, start + timedelta(days=31)
    )
    assert frame.empty
    assert len(client.calls) == 2


def test_delivery_prices_filter_exact_range() -> None:
    """Filter cursor-based delivery histories by timestamp."""
    row = {"deliveryTime": "1735689600000", "deliveryPrice": "95000"}
    client = Client([{"list": [row], "nextPageCursor": ""}])
    start = datetime(2025, 1, 1, tzinfo=UTC)
    frame = BybitHistory(client).delivery_prices(
        "BTC-27DEC24-100000-C", "options", start, start + timedelta(days=1)
    )
    assert len(frame) == 1


def test_history_rejects_cursor_loops_and_invalid_controls() -> None:
    """Fail safely on repeated cursors, periods, and source envelopes."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    looping = Client(
        [
            {"list": [], "nextPageCursor": "same"},
            {"list": [], "nextPageCursor": "same"},
        ]
    )
    with pytest.raises(BybitResponseError, match="repeated"):
        BybitHistory(looping).positions(
            "BTCUSDT",
            "linear",
            "open_interest",
            "1h",
            start,
            start + timedelta(days=1),
        )
    with pytest.raises(ValueError, match="period"):
        BybitHistory(Client([])).positions(
            "BTCUSDT",
            "linear",
            "open_interest",
            "2h",
            start,
            start + timedelta(days=1),
        )
    with pytest.raises(ValueError, match="volatility period"):
        BybitHistory(Client([])).volatility("BTC", 10, start, start + timedelta(days=1))


@pytest.mark.parametrize(
    ("rows", "dataset"),
    [
        ([{"time": True, "period": 30, "value": "0.5"}], "historical_volatility"),
        (
            [{"time": "1735689600000", "period": -1, "value": "0.5"}],
            "historical_volatility",
        ),
        (["not-an-object"], "funding_rates"),
    ],
)
def test_invalid_object_history_values_fail_closed(
    rows: list[object], dataset: str
) -> None:
    """Reject malformed source objects, timestamps, and periods."""
    with pytest.raises(BybitResponseError):
        object_frame(rows, dataset)
