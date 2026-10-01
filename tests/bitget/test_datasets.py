"""Test Bitget dataset capability declarations."""

from datetime import timedelta

import pytest

from veldra.bitget.datasets import (
    ARCHIVE_DAY_OFFSET,
    DATASETS,
    FUTURES_PRODUCTS,
    OUTPUT_INTERVALS,
    PRODUCTS,
    get_dataset,
    supports,
)


def test_product_and_interval_sets_are_explicit() -> None:
    """Confirm public products and resampled output intervals stay stable."""
    assert PRODUCTS == (
        "spot",
        "usdt_futures",
        "usdc_futures",
        "coin_futures",
    )
    assert FUTURES_PRODUCTS == PRODUCTS[1:]
    assert OUTPUT_INTERVALS[0] == "1m"
    assert OUTPUT_INTERVALS[-1] == "1mo"
    assert ARCHIVE_DAY_OFFSET == timedelta(hours=8)


@pytest.mark.parametrize("product", PRODUCTS)
@pytest.mark.parametrize(
    "dataset",
    ("klines", "trades", "best_book_snapshots", "order_book_snapshots"),
)
def test_archives_exist_for_every_product(product: str, dataset: str) -> None:
    """Confirm every portal dataset is available for Spot and Futures."""
    declaration = get_dataset(product, dataset)
    assert declaration is DATASETS[(product, dataset)]
    assert declaration.archive_day_offset == timedelta(hours=8)
    assert supports(product, dataset)


@pytest.mark.parametrize("product", FUTURES_PRODUCTS)
@pytest.mark.parametrize(
    "dataset",
    (
        "mark_price_klines",
        "index_price_klines",
        "premium_index_klines",
        "funding_rates",
    ),
)
def test_reference_data_is_futures_only(product: str, dataset: str) -> None:
    """Confirm reference prices and funding are declared for Futures only."""
    assert get_dataset(product, dataset).product == product
    assert not supports("spot", dataset)


def test_quantity_units_follow_product_semantics() -> None:
    """Confirm inverse Klines do not mislabel contracts as quote volume."""
    spot = get_dataset("spot", "klines")
    coin = get_dataset("coin_futures", "klines")
    assert "quote_volume" in spot.stored_columns
    assert "contract_volume" not in spot.stored_columns
    assert "contract_volume" in coin.stored_columns
    assert coin.resample_sum_columns == ("base_volume", "contract_volume")


def test_order_book_schemas_distinguish_best_and_deep_snapshots() -> None:
    """Confirm scalar best books and nested level-500 books remain distinct."""
    best = get_dataset("spot", "best_book_snapshots")
    deep = get_dataset("spot", "order_book_snapshots")
    assert best.object_columns == ()
    assert deep.object_columns == ("bids", "asks")
    assert (
        best.ordering_columns
        == deep.ordering_columns
        == (
            "event_time",
            "event_number",
        )
    )


@pytest.mark.parametrize(
    ("product", "dataset", "error"),
    [
        (1, "klines", TypeError),
        ("spot", None, TypeError),
        ("options", "klines", ValueError),
        ("spot", "funding_rates", ValueError),
        ("usdt_futures", "metrics", ValueError),
    ],
)
def test_invalid_dataset_requests_fail_cleanly(
    product: object, dataset: object, error: type[Exception]
) -> None:
    """Confirm malformed and unsupported combinations fail before network I/O."""
    with pytest.raises(error):
        get_dataset(product, dataset)
