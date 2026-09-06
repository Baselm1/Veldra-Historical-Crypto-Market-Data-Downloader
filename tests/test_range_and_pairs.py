"""Test pair resolution and independent availability cleanup."""

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import httpx
import pandas as pd
import pytest

import crypto_downloader.downloader as downloader_module
from crypto_downloader.datasets import DatasetSpec
from crypto_downloader.downloader import Downloader
from crypto_downloader.models import (
    IngestedResource,
    Market,
    Resource,
    ResourceKey,
    Result,
)
from crypto_downloader.processing import file_sha256
from crypto_downloader.request import normalize_pair

TODAY = date(2025, 1, 5)


class RangeSource:
    """Provide deterministic markets and daily files without network access."""

    code: str = "range_source"
    products: tuple[str, ...] = ("spot",)
    active_statuses: frozenset[str] = frozenset({"TRADING"})

    def __init__(
        self,
        markets: list[Market],
        days: dict[str, list[date]],
    ) -> None:
        """Store market metadata and available archive days.

        Args:
            markets: The complete market snapshot returned to the downloader.
            days: Available daily resources indexed by native symbol.
        """
        self.market_rows = markets
        self.days = days
        self.resource_calls: list[tuple[str, date, date]] = []

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return the configured market snapshot.

        Args:
            client: The unused HTTPX client.
            product: The requested source product.

        Returns:
            A copy of the configured markets.
        """
        return self.market_rows.copy()

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return configured resources inside an inclusive range.

        Args:
            client: The unused HTTPX client.
            key: The requested source dataset identity.
            start_day: The first archive day to include.
            end_day: The last archive day to include.

        Returns:
            Available daily resources ordered by date.
        """
        self.resource_calls.append((key.symbol, start_day, end_day))
        return [
            Resource(day, f"memory://{key.symbol}/{day}.zip", "memory://checksum")
            for day in self.days.get(key.symbol, [])
            if start_day <= day <= end_day
        ]

    def ingest(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Write one canonical candle to the requested Parquet path.

        Args:
            client: The unused HTTPX client.
            resource: The daily resource represented by the row.
            dataset: The Spot kline schema used by the downloader.
            destination: The final Parquet path.

        Returns:
            Integrity metadata for the generated test file.
        """
        opened = datetime.combine(resource.day, time.min, UTC)
        frame = pd.DataFrame(
            {
                "open_time": pd.Series([opened], dtype="datetime64[us, UTC]"),
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [10.0],
                "close_time": pd.Series(
                    [opened + timedelta(seconds=59, microseconds=999999)],
                    dtype="datetime64[us, UTC]",
                ),
                "quote_volume": [1000.0],
                "trade_count": pd.Series([10], dtype="int64"),
                "taker_buy_base_volume": [4.0],
                "taker_buy_quote_volume": [400.0],
            }
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
        stat = destination.stat()
        digest = file_sha256(destination)
        return IngestedResource(
            archive_sha256="a" * 64,
            parquet_sha256=digest,
            parquet_size=stat.st_size,
            parquet_mtime_ns=stat.st_mtime_ns,
            row_count=1,
            first_timestamp=opened,
            last_timestamp=opened,
        )


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the UTC yesterday boundary deterministic for every test."""
    monkeypatch.setattr(downloader_module, "utc_today", lambda: TODAY, raising=False)


def market(symbol: str, status: str = "TRADING") -> Market:
    """Create one normalized USDT test market.

    Args:
        symbol: The native source symbol.
        status: The source-native market status.

    Returns:
        A market with representative asset metadata.
    """
    return Market(symbol, normalize_pair(symbol), "BTC", "USDT", status)


def service(tmp_path: Path, source: RangeSource) -> Downloader:
    """Create a downloader using deterministic availability.

    Args:
        tmp_path: The isolated data directory.
        source: The source strategy used by the test.

    Returns:
        A downloader with the 2020 history boundary.
    """
    return Downloader(tmp_path, source=source, earliest_date=date(2020, 1, 1))


def one_result(value: Result | list[Result]) -> Result:
    """Require and return a single-pair result.

    Args:
        value: The downloader return value to narrow.

    Returns:
        The single structured result.
    """
    assert isinstance(value, Result)
    return value


def test_normalized_pair_spelling_resolves_without_changing_the_source_symbol(
    tmp_path: Path,
) -> None:
    """Confirm separators and case are ignored only during pair comparison."""
    source = RangeSource([market("BTCUSDT")], {"BTCUSDT": [date(2024, 1, 1)]})

    result = one_result(
        service(tmp_path, source).get_results("btc-usdt", "2024-01-01", "2024-01-01")
    )

    assert result.pair == "BTCUSDT"
    assert len(result.data) == 1
    assert result.errors == []


def test_unknown_pair_returns_ranked_suggestions_without_substitution(
    tmp_path: Path,
) -> None:
    """Confirm a typo remains an error while suggesting likely native symbols."""
    source = RangeSource([market("BTCUSDT"), market("ETHUSDT"), market("BTCUSDC")], {})

    result = one_result(
        service(tmp_path, source).get_results("BTCSUDT", "2024-01-01", "2024-01-01")
    )

    assert [error.code for error in result.errors] == ["unknown_pair"]
    assert result.errors[0].suggestions[0] == "BTCUSDT"
    assert result.pair == "BTCSUDT"
    assert source.resource_calls == []


def test_ambiguous_normalized_pair_lists_exact_candidates(tmp_path: Path) -> None:
    """Confirm normalized collisions require the caller to choose a native symbol."""
    source = RangeSource([market("BTC-USDT"), market("BTC_USDT")], {})

    result = one_result(
        service(tmp_path, source).get_results("btcusdt", "2024-01-01", "2024-01-01")
    )

    assert [error.code for error in result.errors] == ["ambiguous_pair"]
    assert result.errors[0].suggestions == ("BTC-USDT", "BTC_USDT")
    assert source.resource_calls == []


def test_exact_native_symbol_wins_over_a_normalized_collision(tmp_path: Path) -> None:
    """Confirm callers can disambiguate by supplying the exact source symbol."""
    source = RangeSource(
        [market("BTC-USDT"), market("BTC_USDT")],
        {"BTC-USDT": [date(2024, 1, 1)]},
    )

    result = one_result(
        service(tmp_path, source).get_results("BTC-USDT", "2024-01-01", "2024-01-01")
    )

    assert result.pair == "BTC-USDT"
    assert result.errors == []
    assert len(result.data) == 1


def test_start_is_trimmed_to_global_and_pair_availability(tmp_path: Path) -> None:
    """Confirm requests cannot precede 2020 or a pair's first archive."""
    source = RangeSource(
        [market("BTCUSDT")],
        {"BTCUSDT": [date(2019, 1, 1), date(2020, 1, 2)]},
    )

    result = one_result(
        service(tmp_path, source).get_results("BTCUSDT", "2019-01-01", "2020-01-02")
    )

    assert result.available_range == (
        datetime(2020, 1, 2, tzinfo=UTC),
        datetime(2025, 1, 5, tzinfo=UTC),
    )
    assert result.used_range == (
        datetime(2020, 1, 2, tzinfo=UTC),
        datetime(2020, 1, 3, tzinfo=UTC),
    )
    assert [warning.code for warning in result.warnings] == ["start_trimmed"]
    assert "BTCUSDT" in result.warnings[0].message
    assert "2020-01-02" in result.warnings[0].message
    assert source.resource_calls == [("BTCUSDT", date(2020, 1, 1), date(2025, 1, 4))]


def test_active_market_end_is_trimmed_to_yesterday_boundary(tmp_path: Path) -> None:
    """Confirm active markets never request today's incomplete daily archive."""
    days = [date(2025, 1, day) for day in (2, 3, 4)]
    source = RangeSource([market("BTCUSDT")], {"BTCUSDT": days})

    result = one_result(
        service(tmp_path, source).get_results("BTCUSDT", "2025-01-02", "2025-01-10")
    )

    assert result.available_range == (
        datetime(2025, 1, 2, tzinfo=UTC),
        datetime(2025, 1, 5, tzinfo=UTC),
    )
    assert result.used_range == result.available_range
    assert [warning.code for warning in result.warnings] == ["end_trimmed"]
    assert len(result.data) == 3
    assert result.complete


def test_inactive_market_end_is_trimmed_to_its_last_known_archive(
    tmp_path: Path,
) -> None:
    """Confirm delisted markets stop at their own final available day."""
    source = RangeSource(
        [market("BTCUSDT", "BREAK")],
        {"BTCUSDT": [date(2024, 1, 2), date(2024, 1, 3)]},
    )

    result = one_result(
        service(tmp_path, source).get_results("BTCUSDT", "2024-01-02", "2025-01-01")
    )

    assert result.available_range == (
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 4, tzinfo=UTC),
    )
    assert result.used_range == result.available_range
    assert [warning.code for warning in result.warnings] == ["end_trimmed"]
    assert len(result.data) == 2
    assert result.complete


def test_both_request_edges_can_be_trimmed_independently(tmp_path: Path) -> None:
    """Confirm one request can receive both boundary warnings."""
    source = RangeSource(
        [market("BTCUSDT", "BREAK")],
        {"BTCUSDT": [date(2022, 2, 2), date(2022, 2, 3)]},
    )

    result = one_result(
        service(tmp_path, source).get_results("BTCUSDT", "2019-01-01", "2030-01-01")
    )

    assert result.used_range == (
        datetime(2022, 2, 2, tzinfo=UTC),
        datetime(2022, 2, 4, tzinfo=UTC),
    )
    assert [warning.code for warning in result.warnings] == [
        "start_trimmed",
        "end_trimmed",
    ]
    assert result.complete


@pytest.mark.parametrize(
    ("start", "end"),
    [("2020-01-01", "2020-01-01"), ("2024-01-03", "2024-01-04")],
)
def test_request_entirely_outside_inactive_availability_has_no_overlap(
    tmp_path: Path, start: str, end: str
) -> None:
    """Confirm ranges before or after a delisted market return no fabricated rows.

    Args:
        tmp_path: The isolated data directory.
        start: The requested first date.
        end: The requested final date.
    """
    source = RangeSource([market("BTCUSDT", "BREAK")], {"BTCUSDT": [date(2024, 1, 2)]})

    result = one_result(service(tmp_path, source).get_results("BTCUSDT", start, end))

    assert result.used_range is None
    assert result.data.empty
    assert result.warnings[-1].code == "no_overlap"
    assert not result.complete


def test_known_market_without_archives_returns_no_availability_error(
    tmp_path: Path,
) -> None:
    """Confirm absence of all source history differs from a missing requested day."""
    source = RangeSource([market("BTCUSDT")], {})

    result = one_result(
        service(tmp_path, source).get_results("BTCUSDT", "2024-01-01", "2024-01-01")
    )

    assert [error.code for error in result.errors] == ["no_availability"]
    assert result.problems == []
    assert result.available_range is None


def test_multiple_pairs_keep_order_ranges_and_independent_failures(
    tmp_path: Path,
) -> None:
    """Confirm one invalid pair does not cancel other ordered pair results."""
    source = RangeSource(
        [market("BTCUSDT"), Market("ETHUSDT", "ETHUSDT", "ETH", "USDT", "BREAK")],
        {
            "BTCUSDT": [date(2024, 1, 1)],
            "ETHUSDT": [date(2024, 1, 2)],
        },
    )

    results = service(tmp_path, source).get_results(
        ["ETHUSDT", "NOTREAL", "BTCUSDT"],
        "2024-01-01",
        "2024-01-02",
    )

    assert isinstance(results, list)
    assert [result.pair for result in results] == [
        "ETHUSDT",
        "NOTREAL",
        "BTCUSDT",
    ]
    assert results[0].used_range == (
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    )
    assert [error.code for error in results[1].errors] == ["unknown_pair"]
    assert results[2].used_range == (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    )
    assert [problem.date for problem in results[2].problems] == [date(2024, 1, 2)]


def test_pair_status_controls_incremental_rediscovery(tmp_path: Path) -> None:
    """Confirm active pairs rescan their tail while inactive pairs reuse discovery.

    Args:
        tmp_path: The isolated downloader directory.
    """
    source = RangeSource(
        [market("BTCUSDT"), Market("ETHUSDT", "ETHUSDT", "ETH", "USDT", "BREAK")],
        {
            "BTCUSDT": [date(2024, 1, 1)],
            "ETHUSDT": [date(2024, 1, 1)],
        },
    )
    downloader = service(tmp_path, source)
    downloader.get_results(["BTCUSDT", "ETHUSDT"], "2024-01-01", "2024-01-01")
    source.resource_calls.clear()

    downloader.get_results(["BTCUSDT", "ETHUSDT"], "2024-01-01", "2024-01-01")

    assert source.resource_calls == [("BTCUSDT", date(2024, 12, 29), date(2025, 1, 4))]


@pytest.mark.parametrize(
    "earliest",
    [
        date(2017, 12, 31),
        "not-a-date",
        datetime(2020, 1, 1, 1, tzinfo=UTC),
        TODAY,
    ],
)
def test_invalid_earliest_history_boundary_is_rejected(
    tmp_path: Path, earliest: object
) -> None:
    """Confirm the configurable history boundary is a valid UTC day since 2018.

    Args:
        tmp_path: The isolated data directory.
        earliest: The invalid proposed history boundary.
    """
    with pytest.raises((TypeError, ValueError), match="earliest_date"):
        Downloader(tmp_path, earliest_date=earliest)
