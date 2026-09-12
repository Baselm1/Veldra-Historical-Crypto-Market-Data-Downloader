"""Test market matching across source-native and archive identifiers."""

from veldra.core.matching import exact_markets, rank_markets, suggest_symbols
from veldra.core.models import Market


def kucoin_market() -> Market:
    """Create a market whose API and archive use different asset names.

    Returns:
        A representative KuCoin linear Futures market.
    """
    return Market(
        symbol="XBTUSDTM",
        normalized_symbol="XBTUSDTM",
        base_asset="XBT",
        quote_asset="USDT",
        pair="BTCUSDTM",
        product="linear_futures",
        active=True,
    )


def test_exact_markets_accepts_native_and_archive_identifiers() -> None:
    """Confirm either KuCoin identifier resolves to one canonical market."""
    market = kucoin_market()

    assert exact_markets("XBTUSDTM", [market]) == [market]
    assert exact_markets("BTC-USDTM", [market]) == [market]


def test_native_identity_wins_over_an_alias_collision() -> None:
    """Confirm an exact source symbol cannot become ambiguous through aliases."""
    native = kucoin_market()
    collision = Market(
        symbol="OTHER",
        normalized_symbol="OTHER",
        pair="XBTUSDTM",
    )

    assert exact_markets("XBTUSDTM", [collision, native]) == [native]


def test_rank_and_suggestions_score_every_market_alias() -> None:
    """Confirm archive-name typos find the market's public native symbol."""
    market = kucoin_market()

    assert rank_markets("BTCUSDMT", [market], 3) == [market]
    assert suggest_symbols("BTCUSDMT", [market]) == ("XBTUSDTM",)
