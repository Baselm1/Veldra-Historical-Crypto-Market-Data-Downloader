"""Test pair resolution and independent availability cleanup."""

from crypto_downloader.binance.datasets import get_dataset


from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from threading import Lock
from time import sleep

import httpx
import pandas as pd
import pytest

import crypto_downloader._core.engine as downloader_module
from crypto_downloader._core.datasets import DatasetSpec
from crypto_downloader.binance.datasets import SPOT_KLINES
from crypto_downloader._core.engine import RetrievalEngine
from crypto_downloader._core.catalog import Catalog, open_catalog
from crypto_downloader._core.models import (
    IngestedResource,
    Market,
    Resource,
    ResourceKey,
    Result,
)
from crypto_downloader._core.request import normalize_pair

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
        *,
        delay: float = 0.0,
    ) -> None:
        """Store market metadata and available archive days.

        Args:
            markets: The complete market snapshot returned to the downloader.
            days: Available daily resources indexed by native symbol.
            delay: Optional source latency used to observe pair concurrency.
        """
        self.market_rows = markets
        self.days = days
        self.delay = delay
        self.market_calls = 0
        self.first_calls: list[tuple[str, date | None, date]] = []
        self.resource_calls: list[tuple[str, date, date]] = []
        self.active_calls = 0
        self.peak_calls = 0
        self._call_lock = Lock()

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return the configured market snapshot.

        Args:
            client: The unused HTTPX client.
            product: The requested source product.

        Returns:
            A copy of the configured markets.
        """
        self.market_calls += 1
        return self.market_rows.copy()

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Return the first configured resource inside a broad range.

        Args:
            client: The unused HTTPX client.
            key: The requested source dataset identity.
            start_day: The earliest acceptable archive day, or ``None`` for all days.
            end_day: The latest acceptable archive day.

        Returns:
            The first matching resource, or ``None`` when none exists.
        """
        self.first_calls.append((key.symbol, start_day, end_day))
        self._pause()
        matching = [
            day
            for day in self.days.get(key.symbol, [])
            if (start_day is None or start_day <= day) and day <= end_day
        ]
        if not matching:
            return None
        day = min(matching)
        return Resource(day, f"memory://{key.symbol}/{day}.zip", "memory://checksum")

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
        self._pause()
        return [
            Resource(day, f"memory://{key.symbol}/{day}.zip", "memory://checksum")
            for day in self.days.get(key.symbol, [])
            if start_day <= day <= end_day
        ]

    def _pause(self) -> None:
        """Apply configured latency while recording concurrent source calls."""
        with self._call_lock:
            self.active_calls += 1
            self.peak_calls = max(self.peak_calls, self.active_calls)
        try:
            if self.delay:
                sleep(self.delay)
        finally:
            with self._call_lock:
                self.active_calls -= 1

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
        return IngestedResource(
            archive_sha256="a" * 64,
            parquet_size=stat.st_size,
            parquet_mtime_ns=stat.st_mtime_ns,
            row_count=1,
            first_timestamp=opened,
            last_timestamp=opened,
        )


class ObservedIngestionSource(RangeSource):
    """Record request-wide ingestion concurrency and configurable failures."""

    def __init__(
        self,
        markets: list[Market],
        days: dict[str, list[date]],
        *,
        failed_symbols: set[str] | None = None,
    ) -> None:
        """Create an observed source for request-wide scheduler tests.

        Args:
            markets: The complete market snapshot returned to the downloader.
            days: Available daily resources indexed by native symbol.
            failed_symbols: Symbols whose ingestion should fail.
        """
        super().__init__(markets, days)
        self.failed_symbols = failed_symbols or set()
        self.active_ingestions = 0
        self.peak_ingestions = 0

    def ingest(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Ingest one resource while recording shared scheduler activity.

        Args:
            client: The unused HTTPX client.
            resource: The daily resource represented by the row.
            dataset: The Spot kline schema used by the downloader.
            destination: The final Parquet path.

        Returns:
            Integrity metadata for the generated test file.
        """
        symbol = resource.url.split("/")[2]
        with self._call_lock:
            self.active_ingestions += 1
            self.peak_ingestions = max(self.peak_ingestions, self.active_ingestions)
        try:
            sleep(0.03)
            if symbol in self.failed_symbols:
                raise RuntimeError(f"cannot ingest {symbol}")
            return super().ingest(client, resource, dataset, destination)
        finally:
            with self._call_lock:
                self.active_ingestions -= 1


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


def service(tmp_path: Path, source: RangeSource) -> RetrievalEngine:
    """Create a downloader using deterministic availability.

    Args:
        tmp_path: The isolated data directory.
        source: The source strategy used by the test.

    Returns:
        A downloader with the 2020 history boundary.
    """
    return RetrievalEngine(
        tmp_path,
        source=source,
        earliest_date=date(2020, 1, 1),
        dataset_resolver=get_dataset,
    )


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


@pytest.mark.parametrize(
    "spelling",
    ["BTcuSDT", "btcusdt", "btc-usdt", "BTC/USDT"],
)
def test_pair_normalization_resolves_case_and_common_separators(
    tmp_path: Path,
    spelling: str,
) -> None:
    """Confirm harmless formatting differences resolve without fuzzy matching.

    Args:
        tmp_path: The isolated downloader directory.
        spelling: One human-formatted spelling of BTCUSDT.
    """
    source = RangeSource([market("BTCUSDT")], {"BTCUSDT": [date(2024, 1, 1)]})

    result = one_result(
        service(tmp_path, source).get_results(spelling, "2024-01-01", "2024-01-01")
    )

    assert result.pair == "BTCUSDT"
    assert result.errors == []


def test_exact_real_market_wins_instead_of_becoming_a_typo_suggestion(
    tmp_path: Path,
) -> None:
    """Confirm a real BTTCUSDT request is never rewritten as BTCUSDT.

    Args:
        tmp_path: The isolated downloader directory.
    """
    source = RangeSource(
        [market("BTCUSDT"), market("BTTCUSDT")],
        {"BTTCUSDT": [date(2024, 1, 1)]},
    )

    result = one_result(
        service(tmp_path, source).get_results("BTTCUSDT", "2024-01-01", "2024-01-01")
    )

    assert result.pair == "BTTCUSDT"
    assert result.errors == []


@pytest.mark.parametrize("spelling", ["BBTCUSDT", "BTCIUSDT", "BCTUSDT"])
def test_unknown_pair_uses_high_confidence_edit_aware_suggestions(
    tmp_path: Path,
    spelling: str,
) -> None:
    """Confirm insertions and transpositions rank BTCUSDT first.

    Args:
        tmp_path: The isolated downloader directory.
        spelling: One misspelled BTCUSDT request.
    """
    source = RangeSource(
        [market("BTCUSDT"), market("BCCUSDT"), market("ETHUSDT")],
        {},
    )

    result = one_result(
        service(tmp_path, source).get_results(spelling, "2024-01-01", "2024-01-01")
    )

    assert result.errors[0].suggestions[0] == "BTCUSDT"
    assert len(result.errors[0].suggestions) <= 3


def test_unrelated_unknown_pair_has_no_low_confidence_suggestions(
    tmp_path: Path,
) -> None:
    """Confirm unrelated text does not produce noisy market guesses.

    Args:
        tmp_path: The isolated downloader directory.
    """
    source = RangeSource([market("BTCUSDT"), market("ETHUSDT")], {})

    result = one_result(
        service(tmp_path, source).get_results("NOTREALPAIR", "2024-01-01", "2024-01-01")
    )

    assert result.errors[0].suggestions == ()


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


def test_start_is_limited_by_configuration_when_source_has_older_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Confirm configured history does not misrepresent older source archives."""
    source = RangeSource(
        [market("BTCUSDT")],
        {"BTCUSDT": [date(2017, 8, 17), date(2020, 1, 1), date(2020, 1, 2)]},
    )

    result = one_result(
        service(tmp_path, source).get_results("BTCUSDT", "2019-01-01", "2020-01-02")
    )

    assert result.available_range == (
        datetime(2017, 8, 17, tzinfo=UTC),
        datetime(2025, 1, 5, tzinfo=UTC),
    )
    assert result.used_range == (
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 3, tzinfo=UTC),
    )
    assert [warning.code for warning in result.warnings] == ["configured_start"]
    assert "BTCUSDT" in result.warnings[0].message
    assert "2017-08-17" in result.warnings[0].message
    assert "2020-01-01" in result.warnings[0].message
    assert source.first_calls == [("BTCUSDT", None, date(2025, 1, 4))]
    assert source.resource_calls == [("BTCUSDT", date(2020, 1, 1), date(2020, 1, 2))]
    output = " ".join(capsys.readouterr().err.split())
    assert "source archive availability 2017-08-17 UTC" in output
    assert "configured history begins 2020-01-01 UTC" in output
    assert "source archive begins 2017-08-17 UTC" in output


def test_pre_cutoff_request_reports_configuration_not_source_unavailability(
    tmp_path: Path,
) -> None:
    """Confirm a wholly pre-cutoff request names the configured history limit.

    Args:
        tmp_path: The isolated data directory.
    """
    source = RangeSource([market("BTCUSDT")], {"BTCUSDT": [date(2017, 8, 17)]})

    result = one_result(
        service(tmp_path, source).get_results("BTCUSDT", "2019-01-01", "2019-01-01")
    )

    assert [warning.code for warning in result.warnings] == [
        "configured_start",
        "no_overlap",
    ]
    assert result.warnings[1].message == (
        "The request does not overlap the configured history range."
    )


def test_all_history_configuration_uses_the_first_real_source_archive(
    tmp_path: Path,
) -> None:
    """Confirm all-history TOML settings permit a pair's real archive start.

    Args:
        tmp_path: The isolated data and settings directory.
    """
    config = tmp_path / "all-history.toml"
    config.write_text(
        """[history]
earliest_date = "all"

[klines]
base_interval = "1m"
""",
        encoding="utf-8",
    )
    source = RangeSource([market("BTCUSDT")], {"BTCUSDT": [date(2017, 8, 17)]})

    result = one_result(
        RetrievalEngine(
            tmp_path / "data",
            source=source,
            config_path=config,
            dataset_resolver=get_dataset,
        ).get_results("BTCUSDT", "2017-08-17", "2017-08-17")
    )

    assert result.available_range == (
        datetime(2017, 8, 17, tzinfo=UTC),
        datetime(2025, 1, 5, tzinfo=UTC),
    )
    assert result.used_range == result.requested_range
    assert result.warnings == []
    assert source.first_calls == [("BTCUSDT", None, date(2025, 1, 4))]


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


def test_historical_requests_reuse_discovery_for_every_market_status(
    tmp_path: Path,
) -> None:
    """Confirm immutable historical ranges do not rescan active or inactive pairs.

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

    assert source.resource_calls == []


def test_active_recent_request_reuses_its_fresh_mutable_tail(tmp_path: Path) -> None:
    """Confirm an active pair reuses a recent successful tail scan."""
    source = RangeSource(
        [market("BTCUSDT")],
        {"BTCUSDT": [date(2025, 1, day) for day in (2, 3, 4)]},
    )
    downloader = service(tmp_path, source)
    downloader.get_results("BTCUSDT", "2025-01-02", "2025-01-04")
    source.resource_calls.clear()

    downloader.get_results("BTCUSDT", "2025-01-02", "2025-01-04")

    assert source.resource_calls == []


def test_active_discovery_is_limited_to_the_cleaned_request(tmp_path: Path) -> None:
    """Confirm an active pair does not catalogue every day since 2020."""
    source = RangeSource(
        [market("BTCUSDT")],
        {
            "BTCUSDT": [
                date(2020, 1, 1),
                date(2024, 6, 1),
                date(2024, 6, 2),
            ]
        },
    )

    result = one_result(
        service(tmp_path, source).get_results("BTCUSDT", "2024-06-01", "2024-06-02")
    )

    assert result.complete
    assert source.first_calls == [("BTCUSDT", None, date(2025, 1, 4))]
    assert source.resource_calls == [("BTCUSDT", date(2024, 6, 1), date(2024, 6, 2))]


def test_fresh_market_snapshot_is_reused_until_refresh_is_requested(
    tmp_path: Path,
) -> None:
    """Confirm archive market folders are not crawled on every online request."""
    source = RangeSource([market("BTCUSDT")], {"BTCUSDT": [date(2024, 1, 1)]})
    downloader = service(tmp_path, source)

    downloader.get_results("BTCUSDT", "2024-01-01", "2024-01-01")
    downloader.get_results("BTCUSDT", "2024-01-01", "2024-01-01")
    downloader.get_results("BTCUSDT", "2024-01-01", "2024-01-01", refresh=True)

    assert source.market_calls == 2


def test_source_boundary_is_reused_until_refresh_is_requested(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Confirm ordinary online requests do not repeat the earliest-file probe.

    Args:
        tmp_path: The isolated downloader directory.
        capsys: The pytest helper used to inspect progress output.
    """
    source = RangeSource([market("BTCUSDT")], {"BTCUSDT": [date(2024, 1, 1)]})
    downloader = service(tmp_path, source)

    downloader.get_results("BTCUSDT", "2024-01-01", "2024-01-01")
    downloader.get_results("BTCUSDT", "2024-01-01", "2024-01-01")
    downloader.get_results(
        "BTCUSDT",
        "2024-01-01",
        "2024-01-01",
        refresh=True,
    )

    assert source.first_calls == [
        ("BTCUSDT", None, date(2025, 1, 4)),
        ("BTCUSDT", None, date(2025, 1, 4)),
    ]
    output = capsys.readouterr().err
    assert output.count("Finding the first BTCUSDT daily file") == 2
    assert output.count("Discovering BTCUSDT daily files") == 1
    assert "reused cached daily-file discovery" in output


def test_stale_market_snapshot_is_refreshed_automatically(tmp_path: Path) -> None:
    """Confirm expired market metadata is replaced before pair resolution."""
    source = RangeSource([market("BTCUSDT")], {"BTCUSDT": [date(2024, 1, 1)]})
    downloader = service(tmp_path, source)
    downloader.get_results("BTCUSDT", "2024-01-01", "2024-01-01")
    with open_catalog(tmp_path / "catalog.duckdb") as catalog:
        catalog.connection.execute(
            "UPDATE markets SET refreshed_at = TIMESTAMP '2000-01-01'"
        )

    downloader.get_results("BTCUSDT", "2024-01-01", "2024-01-01")

    assert source.market_calls == 2


def test_multiple_pair_workflows_run_concurrently_and_preserve_order(
    tmp_path: Path,
) -> None:
    """Confirm independent pairs overlap source work without reordering results."""
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    source = RangeSource(
        [market(symbol) for symbol in symbols],
        {symbol: [date(2020, 1, 1), date(2024, 1, 1)] for symbol in symbols},
        delay=0.05,
    )

    results = service(tmp_path, source).get_results(
        symbols, "2024-01-01", "2024-01-01", progress=False
    )

    assert isinstance(results, list)
    assert [result.pair for result in results] == symbols
    assert all(result.complete for result in results)
    assert source.peak_calls >= 2


def test_dataset_ingestion_limit_is_shared_by_every_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm one dataset limit bounds aggregate ingestion across all pairs.

    Args:
        tmp_path: The isolated downloader directory.
        monkeypatch: The pytest helper used to lower the dataset limit.
    """
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "XRPUSDT"]
    source = ObservedIngestionSource(
        [market(symbol) for symbol in symbols],
        {symbol: [date(2024, 1, 1)] for symbol in symbols},
    )
    specification = replace(SPOT_KLINES, max_concurrency=2)

    results = RetrievalEngine(
        tmp_path,
        source=source,
        earliest_date=date(2020, 1, 1),
        max_workers=8,
        dataset_resolver=lambda *_args, **_kwargs: specification,
    ).get_results(symbols, "2024-01-01", "2024-01-01", progress=False)

    assert isinstance(results, list)
    assert [result.pair for result in results] == symbols
    assert all(not result.errors for result in results)
    assert source.peak_ingestions == 2


def test_shared_ingestion_executor_isolates_failures_and_preserves_order(
    tmp_path: Path,
) -> None:
    """Confirm one failed pair does not reorder or stop sibling workflows.

    Args:
        tmp_path: The isolated downloader directory.
    """
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    source = ObservedIngestionSource(
        [market(symbol) for symbol in symbols],
        {symbol: [date(2024, 1, 1)] for symbol in symbols},
        failed_symbols={"ETHUSDT"},
    )

    results = RetrievalEngine(
        tmp_path,
        source=source,
        earliest_date=date(2020, 1, 1),
        max_workers=3,
        dataset_resolver=get_dataset,
    ).get_results(symbols, "2024-01-01", "2024-01-01", progress=False)

    assert isinstance(results, list)
    assert [result.pair for result in results] == symbols
    assert results[0].complete
    assert [problem.code for problem in results[1].problems] == ["resource_failed"]
    assert results[2].complete


def test_pair_catalog_connections_are_bounded_by_active_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm queued pairs do not each allocate an idle DuckDB connection.

    Args:
        tmp_path: The isolated downloader directory.
        monkeypatch: The pytest helper used to observe catalog creation.
    """
    symbols = [f"PAIR{index}USDT" for index in range(10)]
    source = RangeSource(
        [market(symbol) for symbol in symbols],
        {symbol: [date(2024, 1, 1)] for symbol in symbols},
    )
    initialized: list[bool] = []

    @contextmanager
    def counted_catalog(path: Path, *, initialize: bool = True) -> Iterator[Catalog]:
        """Record whether each catalog connection initializes the schema.

        Args:
            path: The catalog database path.
            initialize: Whether the connection should prepare the schema.

        Yields:
            The open catalog under observation.
        """
        initialized.append(initialize)
        with open_catalog(path, initialize=initialize) as catalog:
            yield catalog

    monkeypatch.setattr(downloader_module, "open_catalog", counted_catalog)

    results = RetrievalEngine(
        tmp_path,
        source=source,
        earliest_date=date(2020, 1, 1),
        max_workers=3,
        dataset_resolver=get_dataset,
    ).get_results(symbols, "2024-01-01", "2024-01-01", progress=False)

    assert isinstance(results, list)
    assert len(results) == len(symbols)
    assert initialized == [True, False, False, False]


def test_equivalent_pair_spellings_share_one_workflow(tmp_path: Path) -> None:
    """Confirm normalized aliases cannot race the same cache destination.

    Args:
        tmp_path: The isolated downloader directory.
    """
    source = RangeSource(
        [market("BTCUSDT")],
        {"BTCUSDT": [date(2024, 1, 1)]},
        delay=0.02,
    )

    results = service(tmp_path, source).get_results(
        ["btc-usdt", "BTCUSDT", "BTC/USDT"],
        "2024-01-01",
        "2024-01-01",
        progress=False,
    )

    assert isinstance(results, list)
    assert [result.pair for result in results] == ["BTCUSDT"] * 3
    assert results[0] is results[1] is results[2]
    assert source.first_calls == [("BTCUSDT", None, date(2025, 1, 4))]


def test_unresolved_pair_spellings_keep_independent_errors(tmp_path: Path) -> None:
    """Confirm unresolved aliases retain the caller's spelling in each error.

    Args:
        tmp_path: The isolated downloader directory.
    """
    source = RangeSource([market("BTCUSDT")], {"BTCUSDT": [date(2024, 1, 1)]})

    results = service(tmp_path, source).get_results(
        ["BTCSUDT", "btc-sudt"],
        "2024-01-01",
        "2024-01-01",
        progress=False,
    )

    assert isinstance(results, list)
    assert [result.pair for result in results] == ["BTCSUDT", "btc-sudt"]
    assert "'BTCSUDT'" in results[0].errors[0].message
    assert "'btc-sudt'" in results[1].errors[0].message


@pytest.mark.parametrize(
    "earliest",
    [
        "not-a-date",
        datetime(2020, 1, 1, 1, tzinfo=UTC),
        TODAY,
    ],
)
def test_invalid_earliest_history_boundary_is_rejected(
    tmp_path: Path, earliest: object
) -> None:
    """Confirm the configurable history boundary is a valid UTC day or all.

    Args:
        tmp_path: The isolated data directory.
        earliest: The invalid proposed history boundary.
    """
    with pytest.raises((TypeError, ValueError), match="earliest_date"):
        RetrievalEngine(
            tmp_path,
            earliest_date=earliest,
            dataset_resolver=get_dataset,
            source=BinanceConnector(),
        )


from crypto_downloader.binance.connector import BinanceConnector
