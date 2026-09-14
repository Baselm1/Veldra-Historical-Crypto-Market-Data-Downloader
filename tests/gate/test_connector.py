"""Test Gate market metadata and deterministic archive discovery."""

from datetime import UTC, date, datetime, timedelta
import json

import httpx
import pytest

from veldra.core.models import ResourceKey
from veldra.gate.connector import GateConnector


def response(request: httpx.Request) -> httpx.Response:
    """Return representative Gate API rows and archive metadata."""
    path = request.url.path
    if path.endswith("/spot/currency_pairs"):
        return httpx.Response(
            200,
            json=[
                {
                    "id": "BTC_USDT",
                    "base": "BTC",
                    "quote": "USDT",
                    "trade_status": "tradable",
                    "sell_start": 1_609_459_200,
                }
            ],
        )
    if path.endswith("/spot/tickers"):
        return httpx.Response(
            200, json=[{"currency_pair": "BTC_USDT", "quote_volume": "12.5"}]
        )
    if path.endswith("/futures/usdt/contracts_all"):
        return httpx.Response(
            200,
            json=[
                {
                    "name": "BTC_USDT",
                    "status": "trading",
                    "in_delisting": False,
                    "quanto_multiplier": "0.0001",
                    "launch_time": 1_609_459_200,
                }
            ],
        )
    if path.endswith("/futures/usdt/tickers"):
        return httpx.Response(
            200, json=[{"contract": "BTC_USDT", "volume_24h_quote": "99"}]
        )
    if request.method == "HEAD" and "202501" in path and "202502" not in path:
        return httpx.Response(
            200,
            headers={"ETag": '"0123456789abcdef0123456789abcdef"'},
        )
    return httpx.Response(404)


def client() -> httpx.Client:
    """Return a mock Gate HTTP client."""
    return httpx.Client(transport=httpx.MockTransport(response))


def test_spot_markets_preserve_native_and_normalized_symbols() -> None:
    """Parse status, assets, activity, and source onboarding time."""
    connector = GateConnector(retries=0)
    with client() as http:
        markets = connector.markets(http, "spot")

    assert len(markets) == 1
    assert markets[0].symbol == "BTC_USDT"
    assert markets[0].normalized_symbol == "BTCUSDT"
    assert markets[0].base_asset == "BTC"
    assert markets[0].active is True
    assert markets[0].onboard_time == datetime(2021, 1, 1, tzinfo=UTC)


def test_unsupported_market_symbols_are_skipped() -> None:
    """Ignore symbols that cannot map to Gate's archive-safe directories."""
    row = {
        "id": "老子_USDT",
        "base": "老子",
        "quote": "USDT",
        "trade_status": "tradable",
    }

    assert GateConnector._market(row, "spot") is None
    assert (
        GateConnector._volumes(
            [{"currency_pair": "老子_USDT", "quote_volume": 1}], "currency_pair"
        )
        == {}
    )


def test_futures_markets_preserve_contract_units() -> None:
    """Parse active USDT perpetuals and their exact contract multiplier."""
    connector = GateConnector(retries=0)
    with client() as http:
        market = connector.markets(http, "um")[0]

    assert market.contract_type == "PERPETUAL"
    assert market.contract_size == 0.0001
    assert market.active is True


@pytest.mark.parametrize(("product", "expected"), [("spot", 12.5), ("um", 99.0)])
def test_quote_volumes_use_each_gate_ticker_shape(
    product: str, expected: float
) -> None:
    """Read Spot and Futures quote turnover without changing units.

    Args:
        product: The Gate product whose ticker is queried.
        expected: The representative parsed quote volume.
    """
    with client() as http:
        assert GateConnector(retries=0).quote_volumes(http, product) == {
            "BTC_USDT": expected
        }


def test_daily_spot_kline_resources_use_compact_dates_and_etags() -> None:
    """Discover existing daily Spot Klines and ignore absent days."""
    key = ResourceKey("gate", "spot", "klines", "BTC_USDT", "1m")
    with client() as http:
        resources = GateConnector(retries=0).resources(
            http, key, date(2025, 1, 1), date(2025, 2, 1)
        )

    assert len(resources) == 31
    assert resources[0].url.endswith("/BTC_USDT-20250101.csv.gz")
    assert resources[0].integrity_spec.expected == "0123456789abcdef0123456789abcdef"
    assert resources[0].coverage[1] - resources[0].coverage[0] == timedelta(days=1)


def test_checksum_refresh_reads_the_current_etag() -> None:
    """Confirm refresh reads Gate instead of trusting cataloged ETag metadata."""
    key = ResourceKey("gate", "spot", "klines", "BTC_USDT", "1m")
    connector = GateConnector(retries=0)
    with client() as http:
        resource = connector.resources(http, key, date(2025, 1, 1), date(2025, 1, 1))[0]
        assert connector.checksum(http, resource) == "0123456789abcdef0123456789abcdef"


def test_monthly_futures_resources_cover_the_complete_source_month() -> None:
    """Represent one Futures archive as a full physical calendar month."""
    key = ResourceKey("gate", "um", "klines", "BTC_USDT", "1m", cadence="monthly")
    with client() as http:
        resources = GateConnector(retries=0).resources(
            http, key, date(2025, 1, 15), date(2025, 2, 2)
        )

    assert len(resources) == 1
    assert resources[0].day == date(2025, 1, 1)
    assert resources[0].last_day == date(2025, 1, 31)
    assert resources[0].url.endswith("/BTC_USDT-202501.csv.gz")


def test_first_resource_uses_the_market_onboarding_hint() -> None:
    """Avoid probing Gate dates that predate the selected market."""
    connector = GateConnector(retries=0)
    key = ResourceKey("gate", "spot", "klines", "BTC_USDT", "1m")
    with client() as http:
        connector.markets(http, "spot")
        first = connector.first_resource(http, key, None, date(2025, 1, 31))

    assert first is not None
    assert first.day == date(2025, 1, 1)


def test_source_bounds_never_probe_before_gate_publication() -> None:
    """Use Gate's documented dataset starts ahead of older market listings."""
    connector = GateConnector(retries=0)
    connector._onboard_dates[("spot", "BTC_USDT")] = date(2017, 1, 1)

    assert connector._source_start(
        ResourceKey("gate", "spot", "klines", "BTC_USDT", "1m")
    ) == date(2023, 1, 1)
    assert connector._source_start(
        ResourceKey("gate", "spot", "order_book_updates", "BTC_USDT", None)
    ) == date(2021, 8, 1)


@pytest.mark.parametrize(
    "key",
    [
        ResourceKey("gate", "options", "klines", "BTC_USDT", "1m"),
        ResourceKey("gate", "spot", "funding_rates", "BTC_USDT", None),
        ResourceKey("gate", "spot", "klines", "../BTC", "1m"),
    ],
)
def test_invalid_resource_identities_never_reach_the_network(key: ResourceKey) -> None:
    """Reject unsupported products, datasets, and unsafe symbols.

    Args:
        key: The invalid resource identity under test.
    """
    with client() as http, pytest.raises(ValueError):
        GateConnector(retries=0).resources(
            http, key, date(2025, 1, 1), date(2025, 1, 2)
        )


def test_malformed_market_payload_is_rejected() -> None:
    """Reject API payloads whose top-level value is not a market list."""

    def invalid(_: httpx.Request) -> httpx.Response:
        """Return one malformed Gate market response."""
        return httpx.Response(200, content=json.dumps({"bad": True}).encode())

    with httpx.Client(transport=httpx.MockTransport(invalid)) as http:
        with pytest.raises(ValueError, match="no snapshot"):
            GateConnector(retries=0).markets(http, "spot")
