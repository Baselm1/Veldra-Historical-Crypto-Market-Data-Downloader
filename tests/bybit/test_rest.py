"""Test reusable Bybit REST response materializations."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from veldra.bybit.rest import BybitRESTCache
from veldra.core.catalog import Catalog
from veldra.core.subjects import DataSubject


def _cache(tmp_path: Path) -> BybitRESTCache:
    """Return an isolated in-memory catalog and on-disk cache."""
    return BybitRESTCache(Catalog(duckdb.connect()), tmp_path)


def _frame() -> pd.DataFrame:
    """Return two canonical Spot candles."""
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:01:00Z"], utc=True
            ).astype("datetime64[us, UTC]"),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "base_volume": [1.0, 2.0],
            "quote_volume": [101.0, 204.0],
        }
    )


def test_cache_reuses_an_enclosing_range_offline(tmp_path: Path) -> None:
    """Let one larger REST materialization answer a smaller later request."""
    cache = _cache(tmp_path)
    subject = DataSubject("instrument", "BTCUSDT")
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 0, 2, tzinfo=UTC)
    calls = 0

    def fetch() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return _frame()

    first = cache.get(
        "klines",
        subject,
        start,
        end,
        product="spot",
        interval="1m",
        fetch=fetch,
    )
    subset = cache.get(
        "klines",
        subject,
        start,
        datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
        product="spot",
        interval="1m",
        fetch=fetch,
        offline=True,
    )
    assert len(first) == 2
    assert subset["close"].tolist() == [101.0]
    assert calls == 1


def test_refresh_replaces_an_existing_response(tmp_path: Path) -> None:
    """Honor an explicit online refresh even for immutable history."""
    cache = _cache(tmp_path)
    subject = DataSubject("instrument", "BTCUSDT")
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 0, 2, tzinfo=UTC)
    calls = 0

    def fetch() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        frame = _frame()
        frame["close"] += calls
        return frame

    cache.get("klines", subject, start, end, product="spot", interval="1m", fetch=fetch)
    refreshed = cache.get(
        "klines",
        subject,
        start,
        end,
        product="spot",
        interval="1m",
        fetch=fetch,
        refresh=True,
    )
    assert refreshed["close"].tolist() == [103.0, 104.0]
    assert calls == 2


def test_corrupt_cache_is_rebuilt_online_and_rejected_offline(tmp_path: Path) -> None:
    """Never return a damaged local Parquet materialization."""
    cache = _cache(tmp_path)
    subject = DataSubject("instrument", "BTCUSDT")
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, 0, 2, tzinfo=UTC)
    cache.get(
        "klines",
        subject,
        start,
        end,
        product="spot",
        interval="1m",
        fetch=_frame,
    )
    partition = cache.catalog.partitions_between(
        "bybit", "spot", "klines", subject, "1m", start, end
    )[0]
    partition.materialization_path.write_text("not parquet", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        cache.get(
            "klines",
            subject,
            start,
            end,
            product="spot",
            interval="1m",
            fetch=_frame,
            offline=True,
        )
    rebuilt = cache.get(
        "klines",
        subject,
        start,
        end,
        product="spot",
        interval="1m",
        fetch=_frame,
    )
    assert len(rebuilt) == 2


def test_offline_cache_miss_is_explicit(tmp_path: Path) -> None:
    """Report rather than silently contacting Bybit in offline mode."""
    with pytest.raises(RuntimeError, match="cached covering"):
        _cache(tmp_path).get(
            "klines",
            DataSubject("instrument", "BTCUSDT"),
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
            product="spot",
            interval="1m",
            fetch=_frame,
            offline=True,
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda frame: frame.rename(columns={"close": "closing"}),
        lambda frame: frame.iloc[::-1],
        lambda frame: frame.assign(
            open_time=pd.to_datetime(
                ["2024-12-31T23:59:00Z", "2025-01-01T00:01:00Z"], utc=True
            )
        ),
    ],
)
def test_cache_rejects_unsafe_source_frames(
    tmp_path: Path, change: Callable[[pd.DataFrame], pd.DataFrame]
) -> None:
    """Reject wrong schemas, ordering, and out-of-range source rows."""
    cache = _cache(tmp_path)
    with pytest.raises(ValueError):
        cache.get(
            "klines",
            DataSubject("instrument", "BTCUSDT"),
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, 0, 2, tzinfo=UTC),
            product="spot",
            interval="1m",
            fetch=lambda: change(_frame()),
        )


def test_cache_rejects_invalid_configuration_and_ranges(tmp_path: Path) -> None:
    """Validate cache lifetimes and aware increasing request ranges."""
    with pytest.raises(ValueError, match="positive"):
        BybitRESTCache(Catalog(duckdb.connect()), tmp_path, mutable_hours=0)
    cache = _cache(tmp_path)
    with pytest.raises(ValueError, match="aware"):
        cache.get(
            "klines",
            DataSubject("instrument", "BTCUSDT"),
            datetime(2025, 1, 2),
            datetime(2025, 1, 1),
            product="spot",
            interval="1m",
            fetch=_frame,
        )
