"""Test Bitget market metadata normalization."""

from datetime import UTC, datetime

import pytest

from veldra.bitget.markets import (
    category,
    markets,
    parse_market,
    parse_volumes,
    quote_volumes,
)


def instrument(**changes: object) -> dict[str, object]:
    """Return one representative Bitget instrument row."""
    row: dict[str, object] = {
        "symbol": "BTCUSDT",
        "baseCoin": "BTC",
        "quoteCoin": "USDT",
        "status": "online",
        "launchTime": "1609459200000",
        "deliveryTime": "0",
        "sizeMultiplier": "0.001",
    }
    row.update(changes)
    return row


@pytest.mark.parametrize(
    ("product", "native"),
    [
        ("spot", "SPOT"),
        ("usdt_futures", "USDT-FUTURES"),
        ("usdc_futures", "USDC-FUTURES"),
        ("coin_futures", "COIN-FUTURES"),
    ],
)
def test_product_categories_are_explicit(product: str, native: str) -> None:
    """Map each public product to exactly one native API category."""
    assert category(product) == native


def test_spot_market_preserves_assets_status_and_onboarding() -> None:
    """Normalize one current Spot instrument without contract metadata."""
    market = parse_market(instrument(), "spot")
    assert market.symbol == "BTCUSDT"
    assert market.normalized_symbol == "BTCUSDT"
    assert market.base_asset == "BTC"
    assert market.quote_asset == "USDT"
    assert market.status == "ONLINE"
    assert market.onboard_time == datetime(2021, 1, 1, tzinfo=UTC)
    assert market.contract_type is None
    assert market.contract_size is None
    assert market.active is True


def test_perpetual_and_delivery_contracts_are_distinguished() -> None:
    """Use delivery time to separate perpetual and dated Futures."""
    perpetual = parse_market(instrument(), "usdt_futures")
    delivery = parse_market(instrument(deliveryTime="1767225600000"), "coin_futures")
    assert perpetual.contract_type == "PERPETUAL"
    assert delivery.contract_type == "DELIVERY"
    assert delivery.delivery_time == datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("status", ["online", "normal", "trading"])
def test_active_statuses_are_case_insensitive(status: str) -> None:
    """Accept the documented active status variants."""
    assert parse_market(instrument(status=status), "spot").active is True


def test_quote_volumes_accept_spot_and_futures_fields() -> None:
    """Read quote turnover from the current and legacy ticker shapes."""
    assert parse_volumes(
        [
            {"symbol": "BTCUSDT", "turnover24h": "12.5"},
            {"symbol": "ETHUSDT", "usdtVolume": 99},
        ]
    ) == {"BTCUSDT": 12.5, "ETHUSDT": 99.0}


@pytest.mark.parametrize(
    "row",
    [
        {},
        [],
        instrument(symbol="unsafe/path"),
        instrument(launchTime="never"),
        instrument(launchTime=[]),
    ],
)
def test_invalid_market_rows_fail_closed(row: object) -> None:
    """Reject unsafe or semantically malformed instrument metadata."""
    with pytest.raises(ValueError):
        parse_market(row, "spot")


def test_invalid_futures_contract_size_fails_closed() -> None:
    """Reject a nonfinite multiplier when contract units depend on it."""
    with pytest.raises(ValueError):
        parse_market(instrument(sizeMultiplier=float("inf")), "coin_futures")


def test_unknown_products_are_rejected_before_network_calls() -> None:
    """Reject products outside Bitget's four explicit categories."""
    with pytest.raises(ValueError, match="unsupported Bitget product"):
        category("options")
    with pytest.raises(TypeError):
        category(None)


class StubClient:
    """Return deterministic duplicate market rows."""

    def __init__(self, *, duplicate: bool = True) -> None:
        """Choose whether instrument rows contain a duplicate."""
        self.duplicate = duplicate

    def get_instruments(self, native: str) -> list[dict[str, object]]:
        """Return two identical instruments."""
        assert native == "SPOT"
        if self.duplicate:
            return [instrument(), instrument()]
        return [instrument(symbol="ETHUSDT", baseCoin="ETH"), instrument()]

    def get_tickers(self, native: str) -> list[dict[str, object]]:
        """Return one representative ticker."""
        assert native == "SPOT"
        return [{"symbol": "BTCUSDT", "turnover24h": "12.5"}]


def test_duplicate_instruments_are_rejected() -> None:
    """Fail rather than silently overwrite conflicting market metadata."""
    with pytest.raises(ValueError, match="duplicate"):
        markets(StubClient(), "spot")


def test_market_and_volume_helpers_delegate_to_native_categories() -> None:
    """Sort parsed markets and fetch quote volumes through one client contract."""
    client = StubClient(duplicate=False)
    assert [market.symbol for market in markets(client, "spot")] == [
        "BTCUSDT",
        "ETHUSDT",
    ]
    assert quote_volumes(client, "spot") == {"BTCUSDT": 12.5}


def test_market_helper_skips_non_ascii_source_symbols() -> None:
    """Localized instruments cannot break an otherwise valid snapshot."""

    class LocalizedClient(StubClient):
        """Include one localized market beside a valid market."""

        def get_instruments(self, native: str) -> list[dict[str, object]]:
            """Return one safe and one localized source symbol."""
            assert native == "SPOT"
            return [instrument(), instrument(symbol="BTC测试USDT")]

    client = LocalizedClient(duplicate=False)
    assert [market.symbol for market in markets(client, "spot")] == ["BTCUSDT"]


def test_volume_helper_skips_non_ascii_source_symbols() -> None:
    """Localized ticker aliases cannot break volume sorting."""
    rows = [
        {"symbol": "BTCUSDT", "turnover24h": "12.5"},
        {"symbol": "\u9f99\u867eUSDT", "turnover24h": "99"},
    ]
    assert parse_volumes(rows) == {"BTCUSDT": 12.5}


@pytest.mark.parametrize(
    "rows",
    [
        ["wrong"],
        [{"symbol": "BTCUSDT", "turnover24h": True}],
        [{"symbol": "BTCUSDT", "turnover24h": "invalid"}],
        [{"symbol": "BTCUSDT", "turnover24h": float("inf")}],
        [{"symbol": "BTCUSDT", "turnover24h": -1}],
    ],
)
def test_invalid_ticker_rows_fail_closed(rows: list[object]) -> None:
    """Reject malformed, nonfinite, or negative quote turnover."""
    with pytest.raises(ValueError):
        parse_volumes(rows)
