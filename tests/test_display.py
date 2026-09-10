"""Test optional Rich activity and shared result rendering."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta, timezone
from io import StringIO
import logging
from threading import Barrier

import pandas as pd
import pytest
from rich.console import Console

from veldra.core.reporting import (
    Reporter,
    format_range,
    format_time,
)
from veldra.core.models import Market, Message, Result


def output_console(*, color: bool = False) -> tuple[Console, StringIO]:
    """Create a deterministic in-memory Rich console.

    Args:
        color: Whether ANSI color sequences should be produced.

    Returns:
        The console and its backing text stream.
    """
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=color,
        color_system="standard" if color else None,
        width=120,
    )
    return console, stream


def result(
    pair: str = "BTCUSDT", *, complete: bool = True, empty: bool = False
) -> Result:
    """Build one representative result for rendering tests.

    Args:
        pair: The pair written into the result.
        complete: Whether the result should have full coverage.
        empty: Whether the result should contain no rows.

    Returns:
        A result containing a small OHLCV frame and report.
    """
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 2, tzinfo=UTC)
    frame = (
        pd.DataFrame(columns=["open_time", "open", "close"])
        if empty
        else pd.DataFrame(
            {
                "open_time": [start, start.replace(hour=1)],
                "open": [100.0, 101.0],
                "close": [101.0, 102.0],
            }
        )
    )
    value = Result(
        pair=pair,
        data=frame,
        requested_range=(start, end),
        used_range=(start, end),
        available_range=(datetime(2020, 1, 1, tzinfo=UTC), end),
    )
    if not complete:
        value.warnings.append(Message("start_trimmed", "The start was trimmed."))
        value.problems.append(
            Message("resource_unavailable", "No source file exists.", date(2025, 1, 1))
        )
        value.errors.append(
            Message(
                "unknown_pair",
                "The pair was not found.",
                suggestions=("BTCUSDT", "BTCUSDC"),
            )
        )
    return value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "Not available"),
        (datetime(2025, 1, 1, tzinfo=UTC), "2025-01-01 UTC"),
        (
            datetime(2025, 1, 1, 12, 34, 56, tzinfo=UTC),
            "2025-01-01 12:34:56 UTC",
        ),
        (
            datetime(2025, 1, 1, 2, tzinfo=timezone(timedelta(hours=2))),
            "2025-01-01 UTC",
        ),
        (datetime(2025, 1, 1, 12, 34, 56), "2025-01-01 12:34:56"),
    ],
)
def test_format_time_is_compact(value: datetime | None, expected: str) -> None:
    """Confirm timestamps avoid Python's datetime representation.

    Args:
        value: The timestamp being formatted.
        expected: The expected readable text.
    """
    assert format_time(value) == expected


def test_format_range_explains_its_exclusive_end() -> None:
    """Confirm ranges state their half-open boundary semantics."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 2, tzinfo=UTC)

    assert format_range((start, end)) == (
        "2025-01-01 UTC to 2025-01-02 UTC (end exclusive)"
    )
    assert format_range(None) == "Not available"


def test_reporter_lines_are_colored_and_market_counts_are_sorted() -> None:
    """Confirm interactive summaries retain their markers and Rich styles."""
    console, stream = output_console(color=True)
    reporter = Reporter(console=console)

    reporter.info("checking")
    reporter.success("done")
    reporter.warning("careful")
    reporter.error("failed")
    reporter.market_summary(
        [
            Market("BTCUSDT", "BTCUSDT", status="TRADING", active=True),
            Market("OLDUSDT", "OLDUSDT", status="BREAK"),
            Market("ARCHIVE", "ARCHIVE"),
        ],
        refreshed=True,
    )

    output = stream.getvalue()
    assert all(text in output for text in ("INFO", "OK", "WARN", "ERROR"))
    assert all(text in output for text in ("checking", "done", "careful", "failed"))
    assert "3 pairs" in output
    assert "1 ARCHIVE_ONLY, 1 BREAK, 1 TRADING" in output
    assert "\x1b[" in output


def test_reporter_request_uses_normalized_human_values() -> None:
    """Confirm request output includes the source, workload, and readable range."""
    console, stream = output_console()
    reporter = Reporter(console=console)

    reporter.request(
        "binance",
        "spot",
        "klines",
        2,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 3, tzinfo=UTC),
    )

    output = stream.getvalue()
    assert "Binance spot klines" in output
    assert "2 pairs" in output
    assert "2025-01-01 UTC to 2025-01-03 UTC (end exclusive)" in output


def test_reporter_uses_a_singular_pair_label() -> None:
    """Confirm one-pair requests use natural singular wording."""
    console, stream = output_console()
    reporter = Reporter(console=console)

    reporter.request(
        "binance",
        "spot",
        "klines",
        1,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    )

    assert "1 pair," in stream.getvalue()


def test_status_and_download_progress_are_visible(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Confirm spinners and daily-file outcomes appear when enabled."""
    console, stream = output_console(color=True)
    reporter = Reporter(console=console)

    with caplog.at_level(logging.DEBUG, logger="veldra.core.reporting"):
        with reporter.status("Refreshing markets"):
            pass
    with reporter.downloads("BTCUSDT", 2) as advance:
        advance(date(2025, 1, 1), True)
        advance(date(2025, 1, 2), False)

    output = stream.getvalue()
    assert "Rich status started: Refreshing markets" in caplog.messages
    assert "Rich status finished: Refreshing markets" in caplog.messages
    assert "BTCUSDT 2025-01-02 failed" in output
    assert "2/2" in output


def test_reporter_allows_concurrent_pair_progress() -> None:
    """Confirm parallel pair tasks can share one Rich live display."""
    console, stream = output_console(color=True)
    reporter = Reporter(console=console)
    barrier = Barrier(2)

    def work(symbol: str) -> None:
        """Hold two status tasks open at the same time.

        Args:
            symbol: The pair shown by this concurrent task.
        """
        with reporter.status(f"Discovering {symbol}"):
            barrier.wait()

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(work, ("BTCUSDT", "ETHUSDT")))

    output = stream.getvalue()
    assert "Discovering BTCUSDT" in output
    assert "Discovering ETHUSDT" in output


def test_disabled_reporter_is_completely_silent_but_runs_wrapped_work() -> None:
    """Confirm progress=False hides Rich without skipping requested operations."""
    console, stream = output_console()
    reporter = Reporter(False, console=console)
    executed: list[str] = []

    reporter.info("hidden")
    reporter.success("hidden")
    reporter.warning("hidden")
    reporter.error("hidden")
    reporter.market_summary([Market("BTCUSDT", "BTCUSDT")], refreshed=False)
    with reporter.status("hidden"):
        executed.append("status")
    with reporter.downloads("BTCUSDT", 1) as advance:
        advance(date(2025, 1, 1), True)
        executed.append("downloads")

    assert executed == ["status", "downloads"]
    assert stream.getvalue() == ""


@pytest.mark.parametrize("enabled", [None, 1, "yes"])
def test_reporter_rejects_non_boolean_enabled_values(enabled: object) -> None:
    """Confirm reporter visibility cannot be enabled by truthy accidents.

    Args:
        enabled: The invalid visibility value.
    """
    with pytest.raises(TypeError, match="enabled"):
        Reporter(enabled)  # type: ignore[arg-type]


def test_reporter_does_not_configure_the_callers_root_logger() -> None:
    """Confirm library presentation leaves application logging configuration alone."""
    root = logging.getLogger()
    handlers = tuple(root.handlers)
    level = root.level

    Reporter(False).info("hidden")

    assert tuple(root.handlers) == handlers
    assert root.level == level
