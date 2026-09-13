"""Test source-neutral historical data subject validation and resolution."""

import pytest

from veldra.core.subjects import DataSubject, parse_subjects, resolve_subject


def test_subjects_preserve_native_values_and_expose_normalized_values() -> None:
    """Confirm each subject keeps its native identity and comparison spelling."""
    assert DataSubject("instrument", "BTC-USDT").normalized_value == "BTCUSDT"
    assert DataSubject("instrument_family", "BTC-USD").normalized_value == "BTCUSD"
    assert DataSubject("currency", "usdt").normalized_value == "USDT"
    assert DataSubject("all", "ANY").normalized_value == "ANY"


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("unknown", "BTC-USDT"),
        ("instrument", ""),
        ("instrument_family", "---"),
        ("currency", "US-DT"),
        ("all", "BTCUSDT"),
    ],
)
def test_invalid_subjects_are_rejected(kind: object, value: object) -> None:
    """Confirm malformed kinds and values fail before source work begins.

    Args:
        kind: The unsupported subject kind.
        value: The malformed value for that kind.
    """
    with pytest.raises((TypeError, ValueError)):
        DataSubject(kind, value)  # type: ignore[arg-type]


def test_subject_lists_preserve_order_duplicates_and_shape() -> None:
    """Confirm subject parsing retains caller order and single-value shape."""
    values = ["BTC-USDT", "ETH-USDT", "BTC-USDT"]

    parsed, single = parse_subjects(values, "instrument")
    values.append("ADA-USDT")
    one, one_is_single = parse_subjects("USDT", "currency")

    assert tuple(item.value for item in parsed) == (
        "BTC-USDT",
        "ETH-USDT",
        "BTC-USDT",
    )
    assert single is False
    assert one == (DataSubject("currency", "USDT"),)
    assert one_is_single is True


@pytest.mark.parametrize("values", [None, (), {}, set(), [], ["BTC", 1]])
def test_invalid_subject_containers_are_rejected(values: object) -> None:
    """Confirm only a non-empty string or list of strings is accepted.

    Args:
        values: An invalid subject collection.
    """
    with pytest.raises((TypeError, ValueError)):
        parse_subjects(values, "instrument")


def test_resolution_prefers_native_identity_then_unique_normalization() -> None:
    """Confirm exact native IDs win before normalized aliases are considered."""
    candidates = (
        DataSubject("instrument", "BTC-USDT"),
        DataSubject("instrument", "BTCUSDT"),
    )

    assert resolve_subject("BTC-USDT", "instrument", candidates) == candidates[0]
    assert resolve_subject("BTCUSDT", "instrument", candidates) == candidates[1]
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_subject("btc_usdt", "instrument", candidates)


def test_resolution_rejects_wrong_kinds_unknown_values_and_fuzzy_substitutions() -> (
    None
):
    """Confirm subject lookup never silently substitutes another identifier."""
    candidates = (
        DataSubject("currency", "BTC"),
        DataSubject("currency", "USDT"),
    )

    assert resolve_subject("usdt", "currency", candidates) == candidates[1]
    with pytest.raises(ValueError, match="not found"):
        resolve_subject("USDC", "currency", candidates)
    with pytest.raises(ValueError, match="not found"):
        resolve_subject("USDT", "instrument", candidates)
