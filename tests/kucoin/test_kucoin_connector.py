"""Test KuCoin market metadata and archive resource discovery."""

from datetime import UTC, date, datetime
import json
from urllib.parse import parse_qs

import httpx
import pytest

from veldra.core.models import ResourceKey
from veldra.kucoin.connector import KuCoinConnector


def bucket(*, keys: tuple[str, ...] = (), prefixes: tuple[str, ...] = ()) -> str:
    """Build one non-truncated S3 listing response."""
    contents = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    folders = "".join(
        f"<CommonPrefixes><Prefix>{prefix}</Prefix></CommonPrefixes>"
        for prefix in prefixes
    )
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>false</IsTruncated>{contents}{folders}</ListBucketResult>"
    )


def spot_row() -> dict[str, object]:
    """Return one current Spot API market row."""
    return {
        "symbol": "BTC-USDT",
        "baseCurrency": "BTC",
        "quoteCurrency": "USDT",
        "enableTrading": True,
    }


def futures_row(
    symbol: str = "XBTUSDTM", *, inverse: bool = False
) -> dict[str, object]:
    """Return one current perpetual contract row."""
    return {
        "symbol": symbol,
        "type": "FFWCSX",
        "status": "Open",
        "baseCurrency": "XBT" if symbol.startswith("XBT") else "ETH",
        "quoteCurrency": "USD" if inverse else "USDT",
        "isInverse": inverse,
        "multiplier": -0.001 if inverse else 0.001,
        "firstOpenDate": 1_585_555_200_000,
        "expireDate": None,
        "turnoverOf24h": 123.5,
    }


class Source:
    """Serve compact KuCoin API and archive fixtures."""

    def __init__(self) -> None:
        """Create a request record."""
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return API JSON or S3 XML for one request."""
        self.requests.append(request)
        if request.url.host == "api.kucoin.com":
            if request.url.path.endswith("/symbols"):
                return httpx.Response(200, json={"data": [spot_row()]})
            return httpx.Response(
                200,
                json={"data": {"ticker": [{"symbol": "BTC-USDT", "volValue": "42.5"}]}},
            )
        if request.url.host == "api-futures.kucoin.com":
            rows = [
                futures_row(),
                futures_row("XBTUSDM", inverse=True),
                {**futures_row("XBTMU26"), "type": "FUTURES"},
            ]
            return httpx.Response(200, json={"data": rows})

        query = parse_qs(request.url.query.decode())
        prefix = query["prefix"][0]
        folders: tuple[str, ...]
        if prefix.startswith("data/spot"):
            folders = (f"{prefix}BTCUSDT/", f"{prefix}OLDUSDT/", f"{prefix}null/")
            if "orderbooklv50" in prefix:
                folders = (
                    f"{prefix}BTC-USDT/",
                    f"{prefix}OLD-USDT/",
                    f"{prefix}null/",
                )
        elif query.get("delimiter") == ["/"]:
            folders = (
                f"{prefix}BTCUSDTM/",
                f"{prefix}BTCUSDM/",
                f"{prefix}OLDUSDTM/",
                f"{prefix}XBTMU26/",
            )
        else:
            folders = ()
        return httpx.Response(200, text=bucket(prefixes=folders))


def test_spot_markets_merge_api_and_every_archive_branch() -> None:
    """Confirm Spot native names and compact archive aliases are retained."""
    source = Source()
    with httpx.Client(transport=httpx.MockTransport(source)) as client:
        markets = KuCoinConnector(retries=0).markets(client, "spot")

    assert [market.symbol for market in markets] == ["BTC-USDT", "OLD-USDT"]
    btc = markets[0]
    assert btc.pair == "BTCUSDT"
    assert btc.active
    assert btc.product == "spot"
    assert markets[1].active is False
    listing_requests = [
        request
        for request in source.requests
        if request.url.host == "historical-data.kucoin.com"
    ]
    assert len(listing_requests) == 3


@pytest.mark.parametrize(
    ("product", "symbols", "archive"),
    [
        ("linear_futures", ["OLDUSDTM", "XBTUSDTM"], "BTCUSDTM"),
        ("inverse_futures", ["XBTUSDM"], "BTCUSDM"),
    ],
)
def test_futures_markets_keep_native_xbt_and_exclude_delivery_contracts(
    product: str, symbols: list[str], archive: str
) -> None:
    """Confirm current and archive-only perpetual markets are classified."""
    with httpx.Client(transport=httpx.MockTransport(Source())) as client:
        markets = KuCoinConnector(retries=0).markets(client, product)

    assert [market.symbol for market in markets] == symbols
    current = next(market for market in markets if market.symbol.startswith("XBT"))
    assert current.pair == archive
    assert current.contract_type == "PERPETUAL"
    assert current.contract_size == 0.001
    assert current.onboard_time == datetime(2020, 3, 30, 8, tzinfo=UTC)


def test_quote_volumes_support_spot_and_both_futures_products() -> None:
    """Confirm sorting metadata uses the correct public endpoint fields."""
    with httpx.Client(transport=httpx.MockTransport(Source())) as client:
        connector = KuCoinConnector(retries=0)
        assert connector.quote_volumes(client, "spot") == {"BTC-USDT": 42.5}
        assert connector.quote_volumes(client, "linear_futures") == {"XBTUSDTM": 123.5}
        assert connector.quote_volumes(client, "inverse_futures") == {"XBTUSDM": 123.5}


def test_daily_resource_listing_uses_archive_alias_interval_and_md5() -> None:
    """Confirm a Futures Kline route yields canonical integrity metadata."""
    key = ResourceKey(
        "kucoin",
        "linear_futures",
        "klines",
        "XBTUSDTM",
        "1m",
        archive_symbol="BTCUSDTM",
    )
    object_key = "data/futures/daily/klines/BTCUSDTM/1m/BTCUSDTM-1m-2025-01-01.zip"

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one matching archive plus its sidecar key."""
        return httpx.Response(
            200,
            text=bucket(keys=(object_key, f"{object_key}.CHECKSUM")),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resources = KuCoinConnector(retries=0).resources(
            client, key, date(2025, 1, 1), date(2025, 1, 1)
        )

    assert len(resources) == 1
    resource = resources[0]
    assert resource.day == date(2025, 1, 1)
    assert resource.checksum_algorithm == "md5"
    assert resource.archive_symbol == "BTCUSDTM"
    assert resource.timestamp_column == "open_time"
    assert resource.checksum_url.endswith(".zip.CHECKSUM")


def test_first_resource_ignores_sidecars_and_respects_bounds() -> None:
    """Confirm earliest-resource discovery parses only exact ZIP names."""
    key = ResourceKey("kucoin", "spot", "trades", "BTC-USDT", None, "BTCUSDT")
    root = "data/spot/daily/trades/BTCUSDT/"
    objects = (
        f"{root}BTCUSDT-trades-invalid.zip",
        f"{root}BTCUSDT-trades-2023-01-01.zip",
        f"{root}BTCUSDT-trades-2023-01-01.zip.CHECKSUM",
    )

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=bucket(keys=objects))
        )
    ) as client:
        resource = KuCoinConnector(retries=0).first_resource(
            client, key, None, date(2024, 1, 1)
        )

    assert resource is not None
    assert resource.day == date(2023, 1, 1)
    assert resource.timestamp_column == "event_time"


def test_order_book_resources_include_boundary_padding() -> None:
    """Confirm depth resources declare their observed cross-day tolerance."""
    key = ResourceKey(
        "kucoin", "spot", "order_book_snapshots", "BTC-USDT", None, "BTCUSDT"
    )
    route = KuCoinConnector._route(key)
    resource = KuCoinConnector._resource(
        date(2025, 1, 1), f"{route.prefix}{route.stem}2025-01-01.zip", key
    )

    assert resource.coverage == (
        datetime(2024, 12, 31, 23, 55, tzinfo=UTC),
        datetime(2025, 1, 2, 0, 5, tzinfo=UTC),
    )


@pytest.mark.parametrize("product", ["options", "linear_swap", ""])
def test_connector_rejects_unknown_products(product: str) -> None:
    """Confirm source product mistakes fail before HTTP requests."""
    with httpx.Client() as client:
        with pytest.raises(ValueError, match="unsupported KuCoin product"):
            KuCoinConnector().markets(client, product)


def test_resource_request_rejects_unsafe_or_incomplete_identities() -> None:
    """Confirm route inputs cannot escape paths or omit Kline intervals."""
    connector = KuCoinConnector()
    with pytest.raises(ValueError, match="unsafe"):
        connector._validate_resource_request(
            ResourceKey("kucoin", "spot", "trades", "../BTC", None),
            date(2025, 1, 1),
            date(2025, 1, 2),
        )
    with pytest.raises(ValueError, match="require an interval"):
        connector._validate_resource_request(
            ResourceKey("kucoin", "spot", "klines", "BTCUSDT", None),
            date(2025, 1, 1),
            date(2025, 1, 2),
        )
