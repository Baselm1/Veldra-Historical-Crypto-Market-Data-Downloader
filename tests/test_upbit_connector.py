"""Test Upbit market metadata and archive resource discovery."""

from datetime import date
from urllib.parse import parse_qs

import httpx
import pytest

from veldra.core.models import ResourceKey
from veldra.upbit.connector import UpbitConnector


def directory(key: str) -> dict[str, object]:
    """Return one synthetic portal directory row."""
    return {"key": key, "size": 0, "lastModified": None, "type": "DIRECTORY"}


def file(key: str, size: int = 42) -> dict[str, object]:
    """Return one synthetic portal file row."""
    return {
        "key": key,
        "size": size,
        "lastModified": "2025-01-02T01:00:00Z",
        "type": "FILE",
    }


def market(symbol: str, *, warning: bool = False) -> dict[str, object]:
    """Return one synthetic Upbit current-market row."""
    return {
        "market": symbol,
        "korean_name": "name",
        "english_name": "Name",
        "market_event": {"warning": warning, "caution": {}},
    }


class Source:
    """Serve compact official-shaped Upbit API and portal responses."""

    def __init__(self) -> None:
        """Create an empty request record."""
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return one response selected by host, path, and listing prefix."""
        self.requests.append(request)
        if request.url.host == "api.upbit.com":
            if request.url.path.endswith("/market/all"):
                return httpx.Response(
                    200,
                    json=[market("USDT-BTC"), market("BTC-USDT", warning=True)],
                )
            return httpx.Response(
                200,
                json=[
                    {"market": "USDT-BTC", "acc_trade_price_24h": 125.5},
                    {"market": "BTC-USDT", "acc_trade_price_24h": 2.5},
                ],
            )

        prefix = parse_qs(request.url.query.decode())["prefix"][0]
        rows: list[dict[str, object]] = []
        if prefix in {"candle", "trade"}:
            rows = [
                directory(f"{prefix}/USDT-BTC"),
                directory(f"{prefix}/BTC-USDT"),
                directory(f"{prefix}/OLD-COIN"),
            ]
        return httpx.Response(200, json=rows)


def test_markets_preserve_native_names_and_reverse_quote_first_symbols() -> None:
    """Confirm semantic aliases never confuse BTC-USDT with USDT-BTC."""
    source = Source()
    with httpx.Client(transport=httpx.MockTransport(source)) as client:
        markets = UpbitConnector(retries=0).markets(client, "spot")

    assert [item.symbol for item in markets] == ["BTC-USDT", "OLD-COIN", "USDT-BTC"]
    by_symbol = {item.symbol: item for item in markets}
    assert by_symbol["USDT-BTC"].normalized_symbol == "BTCUSDT"
    assert by_symbol["USDT-BTC"].base_asset == "BTC"
    assert by_symbol["USDT-BTC"].quote_asset == "USDT"
    assert by_symbol["USDT-BTC"].status == "TRADING"
    assert by_symbol["USDT-BTC"].active
    assert by_symbol["BTC-USDT"].normalized_symbol == "USDTBTC"
    assert by_symbol["BTC-USDT"].status == "CAUTION"
    assert not by_symbol["OLD-COIN"].active


def test_quote_volumes_parse_nonnegative_current_turnover() -> None:
    """Confirm volume sorting metadata remains indexed by native symbol."""
    with httpx.Client(transport=httpx.MockTransport(Source())) as client:
        volumes = UpbitConnector(retries=0).quote_volumes(client, "spot")

    assert volumes == {"USDT-BTC": 125.5, "BTC-USDT": 2.5}


def test_resource_discovery_queries_each_year_and_ignores_sidecars() -> None:
    """Confirm a cross-year request yields only exact daily archive objects."""
    key = ResourceKey("upbit", "spot", "klines", "USDT-BTC", "1m", "USDT-BTC")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one relevant file from each requested archive year."""
        prefix = parse_qs(request.url.query.decode())["prefix"][0]
        requested.append(prefix)
        stamp = "20241231" if prefix.endswith("2024") else "20250101"
        archive = f"{prefix}/USDT-BTC_candle-1m_{stamp}.zip"
        return httpx.Response(
            200,
            json=[
                file(archive),
                file(f"{archive}.checksum", 64),
                file(f"{prefix}/unexpected.zip"),
            ],
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resources = UpbitConnector(retries=0).resources(
            client, key, date(2024, 12, 31), date(2025, 1, 1)
        )

    assert requested == [
        "candle/USDT-BTC/daily/1m/2024",
        "candle/USDT-BTC/daily/1m/2025",
    ]
    assert [item.day for item in resources] == [
        date(2024, 12, 31),
        date(2025, 1, 1),
    ]
    assert resources[0].checksum_url is not None
    assert resources[0].checksum_url.endswith(".zip.checksum")
    assert resources[0].checksum_algorithm == "sha256"
    assert resources[0].archive_symbol == "USDT-BTC"
    assert resources[0].timestamp_column == "open_time"


def test_trade_routes_use_no_interval_and_event_timestamps() -> None:
    """Confirm raw trades use the separate daily path and naming scheme."""
    key = ResourceKey("upbit", "spot", "trades", "KRW-BTC", None, "KRW-BTC")
    connector = UpbitConnector()
    route = connector._route(key)
    resource = connector._resource(
        date(2025, 1, 1), f"{route.prefix}/2025/{route.stem}20250101.zip", key
    )

    assert route.prefix == "trade/KRW-BTC/daily"
    assert route.stem == "KRW-BTC_trade_"
    assert resource.timestamp_column == "event_time"


def test_first_resource_walks_available_years_in_ascending_order() -> None:
    """Confirm earliest-file lookup respects a configured lower boundary."""
    key = ResourceKey("upbit", "spot", "trades", "KRW-BTC", None, "KRW-BTC")
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return year folders followed by one archive in the allowed year."""
        prefix = parse_qs(request.url.query.decode())["prefix"][0]
        requests.append(prefix)
        if prefix == "trade/KRW-BTC/daily":
            return httpx.Response(
                200,
                json=[directory(f"{prefix}/2022"), directory(f"{prefix}/2023")],
            )
        if prefix.endswith("2023"):
            archive = f"{prefix}/KRW-BTC_trade_20230102.zip"
            return httpx.Response(200, json=[file(archive)])
        return httpx.Response(200, json=[])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resource = UpbitConnector(retries=0).first_resource(
            client, key, date(2023, 1, 1), date(2024, 1, 1)
        )

    assert resource is not None
    assert resource.day == date(2023, 1, 2)
    assert requests == ["trade/KRW-BTC/daily", "trade/KRW-BTC/daily/2023"]


@pytest.mark.parametrize("product", ["futures", "linear_swap", ""])
def test_connector_rejects_unsupported_products(product: str) -> None:
    """Confirm Upbit product mistakes fail before HTTP requests."""
    with httpx.Client() as client:
        with pytest.raises(ValueError, match="unsupported Upbit product"):
            UpbitConnector().markets(client, product)


def test_resource_requests_reject_unsafe_and_incomplete_identities() -> None:
    """Confirm resource paths cannot escape folders or omit Kline intervals."""
    connector = UpbitConnector()
    with pytest.raises(ValueError, match="unsafe"):
        connector._validate_resource_request(
            ResourceKey("upbit", "spot", "trades", "../BTC", None),
            date(2025, 1, 1),
            date(2025, 1, 2),
        )
    with pytest.raises(ValueError, match="require an interval"):
        connector._validate_resource_request(
            ResourceKey("upbit", "spot", "klines", "KRW-BTC", None),
            date(2025, 1, 1),
            date(2025, 1, 2),
        )


@pytest.mark.parametrize("payload", [{}, ["bad"], [{"key": 3, "type": "FILE"}]])
def test_listing_payload_must_contain_safe_typed_entries(payload: object) -> None:
    """Confirm malformed portal responses never become download paths."""
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(ValueError, match="listing"):
            UpbitConnector(retries=0)._listing(client, "candle")
