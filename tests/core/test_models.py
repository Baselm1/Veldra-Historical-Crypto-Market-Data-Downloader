"""Test the behavior provided by downloader result models."""

from datetime import date, datetime, timezone
import json

import pandas as pd
import pytest

from veldra import Gap, Message, MissingCandlesError, Result
from veldra.core.models import (
    IntegritySpec,
    Market,
    Resource,
    ResourceKey,
    result_report,
)
from veldra.core.subjects import DataSubject

UTC = timezone.utc
START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 2, tzinfo=UTC)


def test_market_activity_is_independent_from_native_status() -> None:
    """Confirm connectors explicitly define activity without status conventions."""
    online = Market("btcusdt", "BTCUSDT", status="online", active=True)
    halted = Market("ETHUSDT", "ETHUSDT", status="TRADING", active=False)

    assert online.active is True
    assert halted.active is False


def test_resource_keys_adapt_symbols_to_explicit_data_subjects() -> None:
    """Confirm legacy keys expose instruments while new keys retain their scope."""
    legacy = ResourceKey("binance", "spot", "klines", "BTCUSDT", "1m")
    family = DataSubject("instrument_family", "BTC-USD")
    scoped = ResourceKey(
        "okx",
        "inverse_futures",
        "trades",
        "BTC-USD",
        None,
        subject=family,
    )

    assert legacy.data_subject == DataSubject("instrument", "BTCUSDT")
    assert scoped.data_subject is family


def test_resource_rejects_an_unknown_checksum_algorithm() -> None:
    """Confirm resources cannot select an unsupported integrity algorithm."""
    with pytest.raises(ValueError, match="checksum algorithm"):
        Resource(
            date(2025, 1, 1),
            "https://data.example/file.zip",
            "https://data.example/file.zip.CHECKSUM",
            checksum_algorithm="crc32",  # type: ignore[arg-type]
        )


def test_resource_adapts_legacy_sidecars_to_an_integrity_policy() -> None:
    """Confirm existing connectors acquire an explicit sidecar policy."""
    resource = Resource(
        date(2025, 1, 1),
        "https://data.example/file.zip",
        "https://data.example/file.zip.CHECKSUM",
        checksum_algorithm="md5",
    )

    assert resource.integrity_spec == IntegritySpec(
        "sidecar",
        algorithm="md5",
        sidecar_url="https://data.example/file.zip.CHECKSUM",
    )


@pytest.mark.parametrize(
    "integrity",
    [
        IntegritySpec("response_header", algorithm="md5"),
        IntegritySpec("archive_only"),
    ],
)
def test_resource_accepts_integrity_without_a_sidecar(
    integrity: IntegritySpec,
) -> None:
    """Confirm response-header and structural policies need no sidecar URL.

    Args:
        integrity: The valid sidecar-free integrity declaration.
    """
    resource = Resource(
        date(2025, 1, 1),
        "https://data.example/file.zip",
        None,
        integrity=integrity,
    )

    assert resource.integrity_spec is integrity


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"mode": "sidecar", "algorithm": "sha256"}, "sidecar_url"),
        ({"mode": "response_header", "algorithm": "sha256"}, "MD5"),
        ({"mode": "archive_only", "expected": "a" * 32}, "expected"),
        ({"mode": "archive_only", "sidecar_url": "https://x"}, "sidecar"),
        (
            {"mode": "response_header", "algorithm": "md5", "expected": "bad"},
            "digest",
        ),
    ],
)
def test_integrity_policy_rejects_contradictory_fields(
    arguments: dict[str, object], message: str
) -> None:
    """Confirm integrity modes cannot silently accept contradictory metadata.

    Args:
        arguments: The malformed integrity constructor values.
        message: The expected validation error fragment.
    """
    with pytest.raises((TypeError, ValueError), match=message):
        IntegritySpec(**arguments)  # type: ignore[arg-type]


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
        "source": "",
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
