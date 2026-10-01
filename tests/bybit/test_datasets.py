"""Test Bybit product and dataset declarations."""

import pytest

from veldra.bybit.datasets import (
    DATASETS,
    KLINE_INTERVALS,
    PRODUCTS,
    get_dataset,
    supports,
)


def test_every_declared_dataset_has_a_valid_capability() -> None:
    """Every immutable declaration must be reachable through its capability map."""
    assert PRODUCTS == ("spot", "linear", "inverse", "options")
    assert DATASETS
    for (product, dataset), specification in DATASETS.items():
        assert supports(product, dataset)
        assert specification.product == product
        assert specification.name == dataset


@pytest.mark.parametrize("interval", KLINE_INTERVALS)
def test_kline_schemas_store_the_requested_native_interval(interval: str) -> None:
    """Bybit avoids silently fetching one-minute history for coarse requests."""
    specification = get_dataset("spot", "klines", requested_interval=interval)
    assert specification.base_interval == interval
    assert specification.output_intervals == (interval,)
    assert specification.supports_resampling is False


def test_reference_klines_do_not_fabricate_volume_columns() -> None:
    """Reference prices expose only fields actually published by Bybit."""
    specification = get_dataset("linear", "mark_price_klines", requested_interval="1h")
    assert specification.stored_columns == (
        "open_time",
        "open",
        "high",
        "low",
        "close",
    )


def test_trade_units_remain_product_specific() -> None:
    """Linear and inverse contracts cannot silently exchange quantity units."""
    linear = get_dataset("linear", "trades")
    inverse = get_dataset("inverse", "trades")
    assert "quote_quantity" in linear.stored_columns
    assert "contract_quantity" not in linear.stored_columns
    assert "contract_quantity" in inverse.stored_columns
    assert "quote_notional" in inverse.stored_columns


def test_order_book_depth_is_retained_as_data() -> None:
    """One schema spans Bybit's depth changes without hiding their provenance."""
    specification = get_dataset("linear", "order_book_updates")
    assert "source_depth" in specification.stored_columns
    assert specification.max_concurrency == 2
    assert get_dataset("options", "order_book_updates").max_concurrency == 1


@pytest.mark.parametrize(
    ("product", "dataset"),
    [
        ("spot", "funding_rates"),
        ("inverse", "premium_index_klines"),
        ("options", "klines"),
        ("linear", "historical_volatility"),
    ],
)
def test_unsupported_combinations_fail_before_network(
    product: str, dataset: str
) -> None:
    """Source-incompatible requests produce deterministic local errors."""
    with pytest.raises(ValueError, match="unsupported dataset"):
        get_dataset(product, dataset)


@pytest.mark.parametrize("value", [1, None, [], object()])
def test_product_and_dataset_require_strings(value: object) -> None:
    """Unhashable and non-text inputs cannot leak into mapping operations."""
    with pytest.raises(TypeError, match="product"):
        get_dataset(value, "trades")
    with pytest.raises(TypeError, match="dataset"):
        get_dataset("spot", value)


def test_intervals_are_validated_only_for_interval_datasets() -> None:
    """Raw datasets reject irrelevant intervals and Klines reject typos."""
    with pytest.raises(ValueError, match="does not accept"):
        get_dataset("spot", "trades", requested_interval="1m")
    with pytest.raises(ValueError, match="unsupported Bybit interval"):
        get_dataset("spot", "klines", requested_interval="60m")
    with pytest.raises(TypeError, match="interval"):
        get_dataset("spot", "klines", requested_interval=60)
