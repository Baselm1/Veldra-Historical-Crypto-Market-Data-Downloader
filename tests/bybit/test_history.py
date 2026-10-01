"""Test Bybit native Kline history normalization and pagination."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from veldra.bybit.client import BybitResponseError
from veldra.bybit.history import BybitHistory, kline_frame


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
