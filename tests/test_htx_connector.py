"""Test HTX market metadata and dual-tree archive discovery."""

from datetime import UTC, date, datetime
import httpx
import pytest

from veldra.core.models import Market, ResourceKey
from veldra.htx.connector import (
    ARCHIVE_URL,
    LISTING_URL,
    MARKET_URLS,
    TICKER_URL,
    HTXConnector,
)
from veldra.htx.datasets import (
    KLINE_INTERVALS,
    NEW_INTERVALS,
    OLD_INTERVALS,
    PRODUCTS,
    SUPPORTED_DATASETS,
    supports,
)


def listing(*, keys: tuple[str, ...] = (), prefixes: tuple[str, ...] = ()) -> str:
    """Build one complete S3-compatible XML listing.

    Args:
        keys: Object keys to include.
        prefixes: Common folder prefixes to include.

    Returns:
        A complete XML bucket response.
    """
    objects = "".join(f"<Contents><Key>{value}</Key></Contents>" for value in keys)
    folders = "".join(
        f"<CommonPrefixes><Prefix>{value}</Prefix></CommonPrefixes>"
        for value in prefixes
    )
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>false</IsTruncated>{objects}{folders}</ListBucketResult>"
    )


def test_htx_declares_products_datasets_and_interval_translations() -> None:
    """Confirm public HTX capabilities match the probed archive."""
    source = HTXConnector(timeout=12, retries=2, backoff=0.25)

    assert source.code == "htx"
    assert source.products == PRODUCTS == ("spot", "linear_swap", "coin_swap")
    assert source.max_concurrency == 32
    assert source.monthly_datasets == frozenset()
    assert KLINE_INTERVALS == ("1m", "5m", "15m", "30m", "1h", "4h", "1d")
    assert OLD_INTERVALS["1h"] == "60min"
    assert NEW_INTERVALS["1h"] == "1h"
    assert supports("spot", "klines")
    assert not supports("spot", "funding_rates")
    assert supports("linear_swap", "funding_rates")
    assert not supports("coin_swap", "funding_rates")
    assert set(SUPPORTED_DATASETS) == set(PRODUCTS)


@pytest.mark.parametrize(
    ("product", "payload", "expected"),
    [
        (
            "spot",
            {
                "data": [
                    {
                        "sc": "btcusdt",
                        "bc": "btc",
                        "qc": "usdt",
                        "state": "online",
                        "toa": 1_514_779_200_000,
                    },
                    {
                        "sc": "ethusdt",
                        "bc": "eth",
                        "qc": "usdt",
                        "state": "suspend",
                        "toa": 1_600_000_000_000,
                    },
                ]
            },
            [
                Market(
                    "BTCUSDT",
                    "BTCUSDT",
                    "BTC",
                    "USDT",
                    "online",
                    "BTC-USDT",
                    onboard_time=datetime(2018, 1, 1, 4, tzinfo=UTC),
                    active=True,
                ),
                Market(
                    "ETHUSDT",
                    "ETHUSDT",
                    "ETH",
                    "USDT",
                    "suspend",
                    "ETH-USDT",
                    onboard_time=datetime.fromtimestamp(1_600_000_000, UTC),
                ),
            ],
        ),
        (
            "linear_swap",
            {
                "data": [
                    {
                        "contract_code": "BTC-USDT",
                        "contract_type": "swap",
                        "contract_status": 1,
                        "contract_size": 0.001,
                        "create_date": "20201021",
                        "delivery_time": "1700000000000",
                    },
                    {
                        "contract_code": "ETH-USDT-260925",
                        "contract_type": "this_week",
                        "contract_status": 1,
                        "contract_size": 0.01,
                        "create_date": "20260801",
                    },
                ]
            },
            [
                Market(
                    "BTC-USDT",
                    "BTCUSDT",
                    "BTC",
                    "USDT",
                    "1",
                    "BTC-USDT-PERP",
                    "PERPETUAL",
                    0.001,
                    datetime(2020, 10, 21, tzinfo=UTC),
                    datetime.fromtimestamp(1_700_000_000, UTC),
                    active=True,
                    product="linear_swap",
                )
            ],
        ),
        (
            "coin_swap",
            {
                "data": [
                    {
                        "contract_code": "BTC-USD",
                        "contract_status": 3,
                        "contract_size": 100,
                        "create_date": 20200325,
                        "delivery_time": 1_700_000_000_000,
                    }
                ]
            },
            [
                Market(
                    "BTC-USD",
                    "BTCUSD",
                    "BTC",
                    "USD",
                    "3",
                    "BTC-USD-PERP",
                    "PERPETUAL",
                    100.0,
                    datetime(2020, 3, 25, tzinfo=UTC),
                    datetime.fromtimestamp(1_700_000_000, UTC),
                    product="coin_swap",
                )
            ],
        ),
    ],
)
def test_current_markets_parse_native_product_metadata(
    product: str, payload: object, expected: list[Market]
) -> None:
    """Confirm source statuses, activity, symbols, and contracts are preserved.

    Args:
        product: The HTX product represented by the response.
        payload: The decoded endpoint response.
        expected: The canonical market rows expected from it.
    """
    assert list(HTXConnector._current_markets(payload, product).values()) == expected


def test_markets_merge_both_archives_and_prefer_new_symbol_spelling() -> None:
    """Confirm current, old-only, and new archive markets form one snapshot."""
    old_root = "data/klines/spot/daily/"
    new_root = "historical_data/spot/daily/klines/"

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one current snapshot and two archive listings."""
        if str(request.url).startswith(MARKET_URLS["spot"]):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "sc": "btcusdt",
                            "bc": "btc",
                            "qc": "usdt",
                            "state": "online",
                            "toa": 1_514_779_200_000,
                        }
                    ]
                },
            )
        prefix = request.url.params["prefix"]
        if prefix == old_root:
            return httpx.Response(
                200,
                text=listing(prefixes=(f"{old_root}BTCUSDT/", f"{old_root}OLDUSDT/")),
            )
        assert prefix == new_root
        return httpx.Response(
            200,
            text=listing(prefixes=(f"{new_root}BTC-USDT/", f"{new_root}NEW-USDT/")),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        markets = HTXConnector().markets(client, "spot")

    assert [market.symbol for market in markets] == ["BTCUSDT", "NEWUSDT", "OLDUSDT"]
    assert markets[0].pair == "BTC-USDT"
    assert markets[0].active
    assert markets[1].pair == "NEW-USDT"
    assert not markets[1].active


def test_futures_archive_discovery_filters_other_product_and_dated_contracts() -> None:
    """Confirm one shared new Futures root yields only the selected perpetuals."""
    old_root = "data/klines/linear-swap/daily/"
    new_root = "historical_data/futures/daily/klines/"

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an empty endpoint and mixed Futures archive folders."""
        if str(request.url).startswith(MARKET_URLS["linear_swap"]):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "contract_code": "SOL-USDT",
                            "contract_type": "swap",
                            "contract_status": 1,
                            "contract_size": 0.1,
                            "create_date": 20210101,
                            "delivery_time": "",
                        }
                    ]
                },
            )
        prefix = request.url.params["prefix"]
        if prefix == old_root:
            return httpx.Response(200, text=listing())
        assert prefix == new_root
        return httpx.Response(
            200,
            text=listing(
                prefixes=(
                    f"{new_root}BTC-USDT-PERP/",
                    f"{new_root}BTC-USD-PERP/",
                    f"{new_root}ETH-USDT-260925/",
                )
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        markets = HTXConnector().markets(client, "linear_swap")

    assert [market.symbol for market in markets] == ["BTC-USDT", "SOL-USDT"]
    assert markets[0].pair == "BTC-USDT-PERP"


def test_spot_quote_volumes_use_quote_turnover_and_ignore_non_ascii_symbols() -> None:
    """Confirm HTX Spot volume ranking uses the source's quote turnover."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return representative ticker rows."""
        assert str(request.url).startswith(TICKER_URL)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"symbol": "btcusdt", "vol": 123.5},
                    {"symbol": "中文", "vol": 99},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert HTXConnector().quote_volumes(client, "spot") == {"BTCUSDT": 123.5}


def test_resources_form_sparse_union_and_prefer_new_overlap() -> None:
    """Confirm two archive generations merge by source day without filling holes."""
    key = ResourceKey("htx", "spot", "klines", "BTCUSDT", "1m", "BTC-USDT")
    old_prefix = "data/klines/spot/daily/BTCUSDT/1min/"
    new_prefix = "historical_data/spot/daily/klines/BTC-USDT/1m/"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return old-only, overlapping, and new-only daily archives."""
        requests.append(request)
        prefix = request.url.params["prefix"]
        if prefix == old_prefix:
            keys = tuple(f"{prefix}BTCUSDT-1min-2025-01-0{day}.zip" for day in (1, 2))
        else:
            assert prefix == new_prefix
            keys = tuple(
                f"{prefix}BTC-USDT-klines-1m-2025-01-0{day}.zip" for day in (2, 4)
            )
        return httpx.Response(200, text=listing(keys=keys))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resources = HTXConnector().resources(
            client, key, date(2025, 1, 1), date(2025, 1, 4)
        )

    assert [resource.day.day for resource in resources] == [1, 2, 4]
    assert "/data/klines/" in resources[0].url
    assert "/historical_data/" in resources[1].url
    assert resources[0].checksum_url.endswith("BTCUSDT-1min-2025-01-01.CHECKSUM")
    assert resources[1].checksum_url.endswith(".zip.CHECKSUM")
    assert resources[0].coverage == (
        datetime(2024, 12, 31, 16, tzinfo=UTC),
        datetime(2025, 1, 1, 16, tzinfo=UTC),
    )
    assert all(request.url.host == "www.htx.com" for request in requests)


@pytest.mark.parametrize(
    ("key", "part"),
    [
        (ResourceKey("other", "spot", "klines", "BTCUSDT", "1m"), "another source"),
        (ResourceKey("htx", "options", "klines", "BTCUSDT", "1m"), "product"),
        (ResourceKey("htx", "spot", "metrics", "BTCUSDT", None), "dataset"),
        (ResourceKey("htx", "spot", "klines", "BTCUSDT", "1s"), "interval"),
        (ResourceKey("htx", "spot", "trades", "BTCUSDT", "1m"), "does not use"),
        (ResourceKey("htx", "spot", "trades", "BTC/USDT", None), "unsafe"),
        (
            ResourceKey("htx", "spot", "trades", "BTCUSDT", None, cadence="monthly"),
            "daily cadence",
        ),
    ],
)
def test_resources_reject_unsupported_or_unsafe_requests(
    key: ResourceKey, part: str
) -> None:
    """Confirm invalid source paths fail before an HTTP request.

    Args:
        key: The invalid resource identity.
        part: Text identifying the validation failure.
    """
    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: pytest.fail("unexpected HTTP"))
    ) as client:
        with pytest.raises(ValueError, match=part):
            HTXConnector().resources(client, key, date(2025, 1, 1), date(2025, 1, 2))


def test_first_resource_compares_both_generations() -> None:
    """Confirm broad discovery returns the earliest physical archive."""
    key = ResourceKey("htx", "linear_swap", "trades", "BTC-USDT", None, "BTC-USDT-PERP")

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a different first date from each archive generation."""
        prefix = request.url.params["prefix"]
        if prefix.startswith("data/trades"):
            keys = (f"{prefix}BTC-USDT-trades-2021-05-01.zip",)
        else:
            keys = (f"{prefix}BTC-USDT-PERP-trades-2026-01-01.zip",)
        return httpx.Response(200, text=listing(keys=keys))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resource = HTXConnector().first_resource(client, key, None, date(2026, 9, 1))

    assert resource is not None
    assert resource.day == date(2021, 5, 1)
    assert resource.archive_symbol == "BTC-USDT-PERP"


def test_all_new_dataset_routes_use_the_observed_names() -> None:
    """Confirm special dataset folders and stems match the live archive."""
    expected = {
        "index_price_klines": ("index-klines", "index-klines-1m"),
        "mark_price_klines": ("mark-klines", "mark-price-klines-1m"),
        "funding_rates": ("funding-rates", "fundingRates"),
        "order_book_updates": ("orderbook/lv150", "l2orderbook-150lv"),
    }
    for dataset, (folder, stem) in expected.items():
        interval = "1m" if dataset.endswith("klines") else None
        key = ResourceKey(
            "htx", "linear_swap", dataset, "BTC-USDT", interval, "BTC-USDT-PERP"
        )
        route = HTXConnector._routes(key)[-1]
        assert folder in route.prefix
        assert stem in route.stem
        assert route.suffix == (
            ".tar.gz" if dataset == "order_book_updates" else ".zip"
        )


def test_market_and_ticker_payload_validation_is_strict() -> None:
    """Confirm malformed public metadata is never silently cataloged."""
    with pytest.raises(ValueError, match="no market"):
        HTXConnector._current_markets({}, "spot")
    with pytest.raises(ValueError, match="invalid market"):
        HTXConnector._current_markets({"data": [None]}, "spot")
    with pytest.raises(ValueError, match="unsafe symbol"):
        HTXConnector._current_markets({"data": [{"sc": "BTC/USDT"}]}, "spot")

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a malformed volume snapshot."""
        return httpx.Response(200, json={"data": [{"symbol": "btcusdt", "vol": -1}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="invalid market"):
            HTXConnector().quote_volumes(client, "spot")
        with pytest.raises(ValueError, match="only for spot"):
            HTXConnector().quote_volumes(client, "linear_swap")


def test_swap_markets_ignore_non_ascii_campaign_symbols() -> None:
    """Confirm HTX promotional symbols do not abort Futures discovery."""
    payload = {
        "data": [
            {
                "contract_code": "\u725b\u6765-USDT",
                "contract_type": "swap",
                "contract_status": 1,
                "contract_size": 1,
                "create_date": "20260901",
            },
            {
                "contract_code": "BTC-USDT",
                "contract_type": "swap",
                "contract_status": 1,
                "contract_size": 0.001,
                "create_date": "20201021",
            },
        ]
    }

    markets = HTXConnector._current_markets(payload, "linear_swap")

    assert list(markets) == ["BTC-USDT"]
