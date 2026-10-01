"""Test Bybit market and symbol normalization."""

from datetime import UTC, datetime

import pytest

from veldra.bybit.identities import category, normalized_symbol, symbol
from veldra.bybit.markets import markets, parse_market, parse_volumes, quote_volumes


def instrument(**changes: object) -> dict[str, object]:
    """Return one representative Bybit instrument row."""
    row: dict[str, object] = {
        "symbol": "BTCUSDT",
        "contractType": "LinearPerpetual",
        "status": "Trading",
        "baseCoin": "BTC",
        "quoteCoin": "USDT",
        "launchTime": "1584230400000",
        "deliveryTime": "0",
    }
    row.update(changes)
    return row


@pytest.mark.parametrize(
    ("product", "native"),
    [
        ("spot", "spot"),
        ("linear", "linear"),
        ("inverse", "inverse"),
        ("options", "option"),
    ],
)
def test_product_categories_are_explicit(product: str, native: str) -> None:
    """Map each public product to exactly one native V5 category."""
    assert category(product) == native


def test_pair_and_dated_symbols_keep_distinct_identities() -> None:
    """Use pair aliases for plain markets without collapsing expiries."""
    assert normalized_symbol("BTCUSDT", "BTC", "USDT", "linear") == "BTCUSDT"
    assert (
        normalized_symbol("BTC-25JUN27-100000-C-USDT", "BTC", "USDT", "options")
        == "BTC25JUN27100000CUSDT"
    )


def test_spot_market_preserves_assets_and_onboarding() -> None:
    """Normalize current Spot metadata without contract fields."""
    market = parse_market(instrument(contractType=""), "spot")
    assert market.symbol == "BTCUSDT"
    assert market.normalized_symbol == "BTCUSDT"
    assert market.base_asset == "BTC"
    assert market.quote_asset == "USDT"
    assert market.status == "TRADING"
    assert market.onboard_time == datetime(2020, 3, 15, tzinfo=UTC)
    assert market.contract_type is None
    assert market.active is True


def test_derivative_contract_styles_are_explicit() -> None:
    """Distinguish perpetuals, dated Futures, and Options."""
    perpetual = parse_market(instrument(), "linear")
    future = parse_market(
        instrument(
            symbol="BTCUSD-26DEC25",
            quoteCoin="USD",
            contractType="InverseFutures",
            deliveryTime="1766707200000",
        ),
        "inverse",
    )
    option = parse_market(
        instrument(
            symbol="BTC-25JUN27-100000-C-USDT",
            contractType=None,
            deliveryTime="1813910400000",
        ),
        "options",
    )
    assert perpetual.contract_type == "PERPETUAL"
    assert future.contract_type == "DELIVERY"
    assert future.delivery_time == datetime(2025, 12, 26, tzinfo=UTC)
    assert option.contract_type == "OPTION"


def test_only_trading_status_is_active() -> None:
    """Do not label prelaunch or settled instruments as active."""
    assert parse_market(instrument(status="PreLaunch"), "linear").active is False


@pytest.mark.parametrize(
    "row",
    [
        {},
        [],
        instrument(symbol="unsafe/path"),
        instrument(launchTime="never"),
        instrument(baseCoin=""),
    ],
)
def test_invalid_market_rows_fail_closed(row: object) -> None:
    """Reject unsafe or malformed current instrument metadata."""
    with pytest.raises(ValueError):
        parse_market(row, "linear")


def test_unknown_products_and_symbols_are_rejected() -> None:
    """Reject products and symbol values outside explicit contracts."""
    with pytest.raises(ValueError, match="unsupported Bybit product"):
        category("usdt_futures")
    with pytest.raises(TypeError):
        category(None)
    with pytest.raises(ValueError):
        symbol("BTC/USDT")


def test_quote_volumes_validate_turnover() -> None:
    """Read finite nonnegative current turnover from ticker rows."""
    assert parse_volumes(
        [
            {"symbol": "BTCUSDT", "turnover24h": "12.5"},
            {"symbol": "ETHUSDT", "turnover24h": 99},
        ]
    ) == {"BTCUSDT": 12.5, "ETHUSDT": 99.0}
    for rows in (
        ["wrong"],
        [{"symbol": "BTCUSDT", "turnover24h": True}],
        [{"symbol": "BTCUSDT", "turnover24h": "invalid"}],
        [{"symbol": "BTCUSDT", "turnover24h": float("inf")}],
    ):
        with pytest.raises(ValueError):
            parse_volumes(rows)


class StubClient:
    """Return deterministic Spot and Option metadata."""

    def __init__(self, *, duplicate: bool = False) -> None:
        """Choose whether Spot instruments contain a duplicate."""
        self.duplicate = duplicate
        self.instrument_calls: list[tuple[str, str | None]] = []

    def get_instruments(
        self,
        native_category: str,
        *,
        base_coin: str | None = None,
        status: str | None = None,
        max_pages: int = 20,
    ) -> list[dict[str, object]]:
        """Return representative native instruments."""
        assert status is None
        assert max_pages == 20
        self.instrument_calls.append((native_category, base_coin))
        if native_category == "option":
            assert base_coin in {"BTC", "ETH"}
            return [
                instrument(
                    symbol=f"{base_coin}-25JUN27-100000-C-USDT",
                    baseCoin=base_coin,
                    contractType=None,
                )
            ]
        rows = [instrument(), instrument(symbol="ETHUSDT", baseCoin="ETH")]
        return [instrument(), instrument()] if self.duplicate else rows

    def get_tickers(self, native_category: str) -> list[dict[str, object]]:
        """Return tickers for Option bases or quote turnover."""
        assert native_category != "option"
        return [{"symbol": "BTCUSDT", "turnover24h": "12.5"}]

    def get_option_base_coins(self) -> list[dict[str, object]]:
        """Return two active Option underlyings and one empty family."""
        return [
            {"baseCoin": "ETH", "hasSymbol": 1},
            {"baseCoin": "BTC", "hasSymbol": "1"},
            {"baseCoin": "SOL", "hasSymbol": 0},
        ]


def test_markets_sort_and_reject_duplicate_symbols() -> None:
    """Sort current metadata and reject conflicting duplicates."""
    assert [market.symbol for market in markets(StubClient(), "spot")] == [
        "BTCUSDT",
        "ETHUSDT",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        markets(StubClient(duplicate=True), "spot")


def test_options_discover_each_ticker_underlying() -> None:
    """Avoid V5's default-BTC trap by querying every active Option base."""
    client = StubClient()
    found = markets(client, "options")
    assert [market.symbol for market in found] == [
        "BTC-25JUN27-100000-C-USDT",
        "ETH-25JUN27-100000-C-USDT",
    ]
    assert client.instrument_calls == [("option", "BTC"), ("option", "ETH")]


def test_quote_volume_helper_delegates_to_native_category() -> None:
    """Fetch current quote turnover through the mapped V5 category."""
    assert quote_volumes(StubClient(), "spot") == {"BTCUSDT": 12.5}


def test_localized_symbols_do_not_break_valid_market_snapshots() -> None:
    """Skip localized aliases while retaining safe source symbols."""

    class LocalizedClient(StubClient):
        """Add one unsafe localized instrument."""

        def get_instruments(
            self,
            native_category: str,
            *,
            base_coin: str | None = None,
            status: str | None = None,
            max_pages: int = 20,
        ) -> list[dict[str, object]]:
            """Return one safe and one localized source symbol."""
            del native_category, base_coin, status, max_pages
            return [instrument(), instrument(symbol="BTC测试USDT")]

    assert [market.symbol for market in markets(LocalizedClient(), "spot")] == [
        "BTCUSDT"
    ]
