"""Test the behavior provided by downloader result models."""

from datetime import date, datetime, timezone
import json

import pandas as pd

from crypto_downloader import Gap, Message, MissingCandlesError, Result
from crypto_downloader.models import result_report

UTC = timezone.utc
START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 2, tzinfo=UTC)


def make_result() -> Result:
    """Create a small result used by the behavioral tests.

    Returns:
        A result containing one closing price and a valid used range.
    """
    return Result(
        pair="BTCUSDT",
        data=pd.DataFrame({"close": [93_576.0]}),
        requested_range=(START, END),
        used_range=(START, END),
    )


def test_completion_depends_on_used_range_problems_and_errors() -> None:
    """Confirm that warnings are allowed but problems and errors are incomplete."""
    result = make_result()

    assert result.complete is True

    result.warnings.append(Message("trimmed", "The requested start was trimmed."))
    assert result.complete is True

    result.problems.append(Message("missing", "One source candle is missing."))
    assert result.complete is False

    result.problems.clear()
    result.errors.append(Message("failed", "The requested pair failed."))
    assert result.complete is False

    result.errors.clear()
    result.used_range = None
    assert result.complete is False


def test_frame_attaches_a_complete_json_serializable_report() -> None:
    """Confirm that returned frames contain readable structured result metadata."""
    result = make_result()
    result.available_range = (START, END)
    result.data.attrs["owner"] = "caller"
    result.warnings.append(
        Message(
            "start_trimmed",
            "The requested start was trimmed.",
            date(2025, 1, 1),
            ("BTCUSDT",),
        )
    )
    result.problems.append(Message("missing_candles", "One candle is missing."))
    result.errors.append(Message("download_failed", "One archive failed."))
    result.gaps.append(Gap(START, END, 1_440))

    frame = result.frame()
    report = frame.attrs["download"]

    assert frame is result.data
    assert frame.attrs["owner"] == "caller"
    assert report == result_report(result)
    assert report == {
        "pair": "BTCUSDT",
        "source": "binance",
        "product": "spot",
        "dataset": "klines",
        "requested_range": [START.isoformat(), END.isoformat()],
        "used_range": [START.isoformat(), END.isoformat()],
        "available_range": [START.isoformat(), END.isoformat()],
        "complete": False,
        "gap_policy": "forward",
        "gaps": [{"start": START.isoformat(), "end": END.isoformat(), "count": 1_440}],
        "warnings": [
            {
                "code": "start_trimmed",
                "message": "The requested start was trimmed.",
                "date": "2025-01-01",
                "suggestions": ["BTCUSDT"],
            }
        ],
        "problems": [
            {
                "code": "missing_candles",
                "message": "One candle is missing.",
                "date": None,
                "suggestions": [],
            }
        ],
        "errors": [
            {
                "code": "download_failed",
                "message": "One archive failed.",
                "date": None,
                "suggestions": [],
            }
        ],
    }
    json.dumps(report)


def test_report_represents_ranges_and_collections_that_are_not_available() -> None:
    """Confirm that an unused result has null ranges and empty message lists."""
    result = Result("UNKNOWN", pd.DataFrame(), (START, END))

    report = result_report(result)

    assert report["used_range"] is None
    assert report["available_range"] is None
    assert report["warnings"] == []
    assert report["problems"] == []
    assert report["errors"] == []
    assert report["gaps"] == []
    assert report["complete"] is False


def test_missing_candles_error_retains_pair_gaps_and_total() -> None:
    """Confirm that strict gap errors explain every missing candle range."""
    gaps = [
        Gap(START, START.replace(minute=2), 2),
        Gap(START.replace(hour=1), START.replace(hour=1, minute=3), 3),
    ]

    error = MissingCandlesError("BTCUSDT", gaps)

    assert error.pair == "BTCUSDT"
    assert error.gaps == tuple(gaps)
    assert str(error) == "BTCUSDT is missing 5 source candles across 2 gaps"
