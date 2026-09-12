"""Test KuCoin dataset capabilities and routing declarations."""

import pytest

from veldra.kucoin.datasets import (
    ARCHIVE_KLINE_INTERVALS,
    PRODUCTS,
    get_dataset,
    supports,
)


def test_kucoin_products_and_native_archive_intervals_are_explicit() -> None:
    """Confirm public product names and observed source intervals."""
    assert PRODUCTS == ("spot", "linear_futures", "inverse_futures")
    assert ARCHIVE_KLINE_INTERVALS == (
        "1m",
        "5m",
        "15m",
        "1h",
        "8h",
        "12h",
        "1d",
    )


@pytest.mark.parametrize(
    ("product", "dataset"),
    [
        ("spot", "klines"),
        ("spot", "trades"),
        ("linear_futures", "index_price_klines"),
        ("inverse_futures", "mark_price_klines"),
        ("linear_futures", "funding_rates"),
    ],
)
def test_implemented_dataset_combinations_resolve(product: str, dataset: str) -> None:
    """Confirm each initial tabular dataset has a valid declaration."""
    specification = get_dataset(product, dataset)

    assert specification.product == product
    assert specification.name == dataset
    assert supports(product, dataset)
    assert specification.archive_symbol_attribute == "pair"


def test_kucoin_kline_units_are_not_fabricated() -> None:
    """Confirm Spot and Futures retain only quantities present in archives."""
    spot = get_dataset("spot", "klines")
    futures = get_dataset("linear_futures", "klines")
    reference = get_dataset("inverse_futures", "index_price_klines")

    assert spot.resample_sum_columns == ("base_volume", "quote_volume")
    assert futures.resample_sum_columns == ("contract_volume",)
    assert reference.resample_sum_columns == ("sample_count",)


@pytest.mark.parametrize(
    ("product", "dataset"),
    [("spot", "funding_rates"), ("missing", "klines"), ("spot", "metrics")],
)
def test_unsupported_dataset_combinations_are_rejected(
    product: str, dataset: str
) -> None:
    """Confirm invalid source combinations fail before network access."""
    assert not supports(product, dataset)
    with pytest.raises(ValueError, match="unsupported dataset"):
        get_dataset(product, dataset)


@pytest.mark.parametrize("field", [1, None])
def test_dataset_identifiers_must_be_strings(field: object) -> None:
    """Confirm product and dataset identifiers are strictly typed."""
    with pytest.raises(TypeError):
        get_dataset(field, "klines")
    with pytest.raises(TypeError):
        get_dataset("spot", field)


def test_nonminute_storage_is_rejected() -> None:
    """Confirm the implementation stores KuCoin's highest Kline resolution."""
    with pytest.raises(ValueError, match="1m archive interval"):
        get_dataset("spot", "klines", kline_base_interval="5m")
