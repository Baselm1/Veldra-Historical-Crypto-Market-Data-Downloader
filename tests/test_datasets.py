"""Test the declarative Binance Spot kline dataset contract."""

import pytest

from crypto_downloader.datasets import DatasetSpec, get_dataset
from crypto_downloader.request import Request

SOURCE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
STORED_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)
OUTPUT_INTERVALS = (
    "1m",
    "3m",
    "5m",
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


def spot_klines() -> DatasetSpec:
    """Return the Spot kline specification used by these tests.

    Returns:
        The registered Binance Spot kline specification.
    """
    return get_dataset("spot", "klines")


def test_spot_kline_schema_matches_the_daily_archive_and_cache() -> None:
    """Confirm the exact source and canonical cached kline schemas."""
    spec = spot_klines()

    assert spec.product == "spot"
    assert spec.name == "klines"
    assert spec.remote_name == "klines"
    assert spec.source_columns == SOURCE_COLUMNS
    assert spec.stored_columns == STORED_COLUMNS
    assert spec.time_column == "open_time"
    assert spec.base_interval == "1m"
    assert spec.output_intervals == OUTPUT_INTERVALS
    assert spec.max_concurrency == 32
    assert spec.output_columns == (*STORED_COLUMNS, "is_synthetic")
    assert "ignore" not in spec.stored_columns
    assert "ignore" not in spec.output_columns


@pytest.mark.parametrize("interval", OUTPUT_INTERVALS)
def test_supported_output_intervals_are_returned(interval: str) -> None:
    """Confirm every declared Spot kline output interval is accepted.

    Args:
        interval: One supported output interval.
    """
    assert spot_klines().resolve_interval(interval) == interval


@pytest.mark.parametrize(
    "interval",
    [None, 1, True, [], "", "1s", "7m", "1M", "1month"],
)
def test_unsupported_output_intervals_are_rejected(interval: object) -> None:
    """Confirm malformed, finer, and arbitrary output intervals fail.

    Args:
        interval: An interval unavailable for stored one-minute Spot klines.
    """
    with pytest.raises((TypeError, ValueError), match="interval"):
        spot_klines().resolve_interval(interval)


def test_finer_than_stored_intervals_explain_the_constraint() -> None:
    """Confirm second intervals report why they cannot be returned."""
    with pytest.raises(ValueError, match="finer than stored 1m data"):
        spot_klines().resolve_interval("30s")


def test_default_columns_use_the_complete_canonical_output_schema() -> None:
    """Confirm omitted column selections return every available output column."""
    spec = spot_klines()

    assert spec.resolve_columns(None) == {
        column: column for column in spec.output_columns
    }


def test_selected_columns_aliases_and_output_labels_are_resolved() -> None:
    """Confirm aliases select canonical data while retaining requested labels."""
    request = Request.parse(
        "BTCUSDT",
        "2025-01-01",
        "2025-01-01",
        desired_columns={"base_volume": "size", "close": "closing price"},
    )

    columns = spot_klines().resolve_columns(request.columns)

    assert columns == {"volume": "size", "close": "closing price"}


def test_unknown_and_duplicate_resolved_columns_are_rejected() -> None:
    """Confirm unknown names and two names for one stored column fail."""
    spec = spot_klines()

    with pytest.raises(ValueError, match="unknown column"):
        spec.resolve_columns({"ignore": "ignore"})
    with pytest.raises(ValueError, match="duplicate column"):
        spec.resolve_columns({"volume": "volume", "base_volume": "base_volume"})


@pytest.mark.parametrize(
    ("product", "dataset"),
    [
        ("spot", "trades"),
        ("um", "klines"),
        ("cm", "klines"),
        ("unknown", "unknown"),
    ],
)
def test_unimplemented_product_dataset_combinations_are_rejected(
    product: str, dataset: str
) -> None:
    """Confirm only the current Spot kline vertical slice is registered.

    Args:
        product: An unimplemented product identifier.
        dataset: An unimplemented dataset identifier.
    """
    with pytest.raises(ValueError, match=f"{product}/{dataset}"):
        get_dataset(product, dataset)


@pytest.mark.parametrize(
    ("product", "dataset", "field"),
    [(None, "klines", "product"), ("spot", None, "dataset"), (1, "klines", "product")],
)
def test_dataset_lookup_rejects_non_string_identifiers(
    product: object, dataset: object, field: str
) -> None:
    """Confirm direct registry calls reject non-string identifiers.

    Args:
        product: The product value supplied to the registry.
        dataset: The dataset value supplied to the registry.
        field: The invalid field named in the expected error.
    """
    with pytest.raises(TypeError, match=field):
        get_dataset(product, dataset)
