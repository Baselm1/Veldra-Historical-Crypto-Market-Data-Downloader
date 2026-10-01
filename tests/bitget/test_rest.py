"""Test persistent local storage for Bitget REST histories."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import duckdb
import pandas as pd
import pytest

from veldra.bitget.rest import BitgetRESTCache
from veldra.core.catalog import Catalog
from veldra.core.subjects import DataSubject


def _cache(tmp_path: Path) -> BitgetRESTCache:
    """Return a cache backed by an isolated in-memory catalog."""
    return BitgetRESTCache(Catalog(duckdb.connect()), tmp_path)


def _range() -> tuple[datetime, datetime]:
    """Return one deterministic UTC request range."""
    return (
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    )


def _candles() -> pd.DataFrame:
    """Return canonical reference candles inside the test range."""
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T12:00:00Z"]
            ).astype("datetime64[us, UTC]"),
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "base_volume": [0.0, 0.0],
            "quote_volume": [0.0, 0.0],
        }
    )


def _funding() -> pd.DataFrame:
    """Return canonical funding rows inside the test range."""
    return pd.DataFrame(
        {
            "funding_time": pd.to_datetime(["2025-01-01T08:00:00Z"]).astype(
                "datetime64[us, UTC]"
            ),
            "funding_rate": [0.0001],
        }
    )


def test_covering_range_is_fetched_once_then_queried_from_parquet(
    tmp_path: Path,
) -> None:
    """A stored enclosing range satisfies later exact and smaller requests."""
    cache = _cache(tmp_path)
    start, end = _range()
    fetch = Mock(return_value=_candles())

    first = cache.get(
        "mark_price_klines",
        DataSubject("instrument", "BTCUSDT"),
        start,
        end,
        product="usdt_futures",
        interval="1m",
        fetch=fetch,
    )
    second = cache.get(
        "mark_price_klines",
        DataSubject("instrument", "BTCUSDT"),
        start,
        datetime(2025, 1, 1, 1, tzinfo=UTC),
        product="usdt_futures",
        interval="1m",
        fetch=fetch,
    )

    assert fetch.call_count == 1
    assert len(first) == 2
    assert second.open_time.tolist() == [pd.Timestamp("2025-01-01T00:00:00Z")]
    assert list(tmp_path.rglob("*.parquet"))


def test_empty_range_is_persisted_and_not_requested_again(tmp_path: Path) -> None:
    """A confirmed empty response remains a valid locally known range."""
    cache = _cache(tmp_path)
    start, end = _range()
    empty = _candles().iloc[0:0]
    fetch = Mock(return_value=empty)

    assert cache.get(
        "mark_price_klines",
        DataSubject("instrument", "BTCUSDT"),
        start,
        end,
        product="usdt_futures",
        interval="1m",
        fetch=fetch,
    ).empty
    assert cache.get(
        "mark_price_klines",
        DataSubject("instrument", "BTCUSDT"),
        start,
        end,
        product="usdt_futures",
        interval="1m",
        fetch=fetch,
    ).empty
    assert fetch.call_count == 1


def test_event_dataset_uses_the_same_presence_based_storage(tmp_path: Path) -> None:
    """Funding history is also stored once without a synthetic interval."""
    cache = _cache(tmp_path)
    start, end = _range()
    fetch = Mock(return_value=_funding())
    subject = DataSubject("instrument", "BTCUSDT")

    first = cache.get(
        "funding_rates",
        subject,
        start,
        end,
        product="usdt_futures",
        fetch=fetch,
    )
    second = cache.get(
        "funding_rates",
        subject,
        start,
        end,
        product="usdt_futures",
        fetch=fetch,
    )

    assert fetch.call_count == 1
    pd.testing.assert_frame_equal(first, second)


def test_unreadable_local_file_is_replaced_from_source(tmp_path: Path) -> None:
    """An unreadable materialization is rebuilt instead of being returned."""
    cache = _cache(tmp_path)
    start, end = _range()
    fetch = Mock(return_value=_candles())
    cache.get(
        "mark_price_klines",
        DataSubject("instrument", "BTCUSDT"),
        start,
        end,
        product="usdt_futures",
        interval="1m",
        fetch=fetch,
    )
    path = next(tmp_path.rglob("*.parquet"))
    path.write_bytes(b"not parquet")

    rebuilt = cache.get(
        "mark_price_klines",
        DataSubject("instrument", "BTCUSDT"),
        start,
        end,
        product="usdt_futures",
        interval="1m",
        fetch=fetch,
    )

    assert fetch.call_count == 2
    assert len(rebuilt) == 2


@pytest.mark.parametrize("mutation", ["columns", "range", "order"])
def test_source_frame_must_be_safe_to_publish(tmp_path: Path, mutation: str) -> None:
    """Malformed, out-of-range, and unsorted responses never enter storage."""
    cache = _cache(tmp_path)
    start, end = _range()
    frame = _candles()
    if mutation == "columns":
        frame = frame.drop(columns="close")
    elif mutation == "range":
        frame.loc[0, "open_time"] = pd.Timestamp("2024-12-31T23:59:00Z")
    else:
        frame = frame.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="Bitget REST response"):
        cache.get(
            "mark_price_klines",
            DataSubject("instrument", "BTCUSDT"),
            start,
            end,
            product="usdt_futures",
            interval="1m",
            fetch=lambda: frame,
        )
    assert not list(tmp_path.rglob("*.parquet"))


def test_request_range_must_be_aware_and_ordered(tmp_path: Path) -> None:
    """The storage boundary rejects naive or reversed time ranges."""
    cache = _cache(tmp_path)
    start, end = _range()
    subject = DataSubject("instrument", "BTCUSDT")
    with pytest.raises(ValueError, match="valid aware"):
        cache.get(
            "funding_rates",
            subject,
            start.replace(tzinfo=None),
            end,
            product="usdt_futures",
            fetch=lambda: pd.DataFrame(),
        )
    with pytest.raises(ValueError, match="valid aware"):
        cache.get(
            "funding_rates",
            subject,
            end,
            start,
            product="usdt_futures",
            fetch=lambda: pd.DataFrame(),
        )
