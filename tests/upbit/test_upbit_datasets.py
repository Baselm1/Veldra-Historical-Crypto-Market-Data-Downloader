"""Test Upbit product, dataset, and physical interval declarations."""

import pytest

from veldra.upbit.datasets import (
    ARCHIVE_KLINE_INTERVALS,
    OUTPUT_INTERVALS,
    PRODUCTS,
    get_dataset,
    supports,
)


def test_upbit_products_and_archive_intervals_are_explicit() -> None:
    """Confirm Upbit exposes only Spot and its observed candle folders."""
    assert PRODUCTS == ("spot",)
    assert ARCHIVE_KLINE_INTERVALS == (
        "1s",
        "1m",
        "3m",
        "5m",
        "10m",
        "15m",
        "30m",
        "60m",
        "240m",
        "day",
        "week",
    )
    assert OUTPUT_INTERVALS == (
        "1m",
        "3m",
        "5m",
        "10m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1mo",
    )


def test_requested_second_interval_selects_second_archives() -> None:
    """Confirm explicit one-second requests use physical one-second files."""
    specification = get_dataset("spot", "klines", requested_interval="1s")

    assert specification.base_interval == "1s"
    assert specification.output_intervals == ("1s",)
    assert specification.gap_semantics == "sparse"


@pytest.mark.parametrize("interval", [None, "1m", "10m", "1h", "1mo"])
def test_coarser_intervals_select_minute_archives(interval: str | None) -> None:
    """Confirm every coarser request reuses physical one-minute files."""
    specification = get_dataset("spot", "klines", requested_interval=interval)

    assert specification.base_interval == "1m"
    assert specification.resolve_interval(interval) == (interval or "1m")
    assert specification.gap_semantics == "sparse"


def test_trades_are_interval_free_and_supported() -> None:
    """Confirm historical trades expose their canonical raw-event schema."""
    specification = get_dataset("spot", "trades")

    assert supports("spot", "trades")
    assert specification.base_interval is None
    assert specification.ordering_columns == ("event_time", "event_number")
    assert specification.integer_columns == ("event_number",)


@pytest.mark.parametrize(
    ("product", "dataset"),
    [("futures", "klines"), ("spot", "order_book_snapshots"), ("spot", "metrics")],
)
def test_unsupported_combinations_are_rejected(product: str, dataset: str) -> None:
    """Confirm absent Upbit products and datasets fail before network access."""
    assert not supports(product, dataset)
    with pytest.raises(ValueError, match="unsupported dataset"):
        get_dataset(product, dataset)


@pytest.mark.parametrize("field", [None, 1, []])
def test_dataset_identifiers_must_be_strings(field: object) -> None:
    """Confirm public dataset identifiers retain strict runtime types."""
    with pytest.raises(TypeError):
        get_dataset(field, "klines")
    with pytest.raises(TypeError):
        get_dataset("spot", field)


def test_nonminute_configured_storage_is_rejected() -> None:
    """Confirm coarse Klines retain the canonical one-minute cache."""
    with pytest.raises(ValueError, match="1m archive interval"):
        get_dataset(
            "spot",
            "klines",
            kline_base_interval="5m",
            requested_interval="1h",
        )
