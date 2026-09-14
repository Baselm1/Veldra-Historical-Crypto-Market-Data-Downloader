"""Test Gate dataset declarations."""

import pytest

from veldra.gate.datasets import PRODUCTS, get_dataset, supports


def test_gate_products_are_explicit() -> None:
    """Confirm Gate exposes Spot and two perpetual Futures products."""
    assert PRODUCTS == ("spot", "um", "cm")


@pytest.mark.parametrize("product", PRODUCTS)
def test_shared_market_datasets_are_available(product: str) -> None:
    """Confirm every product exposes trading and order-book history.

    Args:
        product: The Gate product under test.
    """
    assert supports(product, "klines")
    assert supports(product, "trades")
    assert supports(product, "order_book_updates")
    assert supports(product, "order_book_snapshots")


@pytest.mark.parametrize(
    "dataset", ["mark_prices", "funding_rates", "funding_rate_updates"]
)
def test_reference_datasets_are_futures_only(dataset: str) -> None:
    """Confirm reference and funding history cannot be requested from Spot.

    Args:
        dataset: The Futures-only dataset under test.
    """
    assert not supports("spot", dataset)
    assert supports("um", dataset)
    assert supports("cm", dataset)


def test_kline_resolutions_choose_nonredundant_archives() -> None:
    """Confirm ordinary requests store 1m while Futures can request native 10s."""
    spot = get_dataset("spot", "klines", requested_interval="1h")
    futures = get_dataset("um", "klines", requested_interval="10s")

    assert spot.base_interval == "1m"
    assert spot.remote_name == "candlesticks_1m"
    assert "1mo" in spot.output_intervals
    assert futures.base_interval == "10s"
    assert futures.output_intervals == ("10s",)


def test_gate_quantity_columns_preserve_native_units() -> None:
    """Confirm Spot and Futures declarations do not invent unavailable volume."""
    spot = get_dataset("spot", "klines")
    futures = get_dataset("cm", "klines")
    trades = get_dataset("um", "trades")

    assert "base_volume" in spot.stored_columns
    assert "contract_volume" in futures.stored_columns
    assert "contract_quantity" in trades.stored_columns
    assert "quote_volume" not in spot.stored_columns


def test_monthly_and_hourly_sources_declare_their_logical_cadence() -> None:
    """Confirm monthly datasets differ from daily order-book materializations."""
    assert get_dataset("spot", "klines").archive_cadence == "monthly"
    assert get_dataset("spot", "trades").archive_cadence == "monthly"
    assert get_dataset("um", "funding_rates").archive_cadence == "monthly"
    assert get_dataset("um", "order_book_updates").archive_cadence == "daily"
    assert get_dataset("um", "order_book_snapshots").archive_cadence == "daily"


@pytest.mark.parametrize(
    ("product", "dataset", "message"),
    [
        ("options", "klines", "unsupported dataset"),
        ("spot", "funding_rates", "unsupported dataset"),
        ("spot", "klines_1m", "unsupported dataset"),
    ],
)
def test_unsupported_gate_datasets_fail_early(
    product: str, dataset: str, message: str
) -> None:
    """Confirm unsupported combinations produce useful validation errors.

    Args:
        product: The invalid or incompatible product.
        dataset: The invalid or incompatible dataset.
        message: The expected error text.
    """
    with pytest.raises(ValueError, match=message):
        get_dataset(product, dataset)


def test_spot_rejects_futures_only_ten_second_klines() -> None:
    """Confirm a nonexistent Spot archive interval is rejected."""
    with pytest.raises(ValueError, match="Spot.*10s"):
        get_dataset("spot", "klines", requested_interval="10s")


@pytest.mark.parametrize("field", [None, 1, [], {}])
def test_dataset_identity_requires_text(field: object) -> None:
    """Confirm product and dataset identities require strings.

    Args:
        field: The malformed identity value.
    """
    with pytest.raises(TypeError):
        get_dataset(field, "klines")
    with pytest.raises(TypeError):
        get_dataset("spot", field)
