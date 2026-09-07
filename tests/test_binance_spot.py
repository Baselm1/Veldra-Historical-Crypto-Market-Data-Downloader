"""Test Binance Spot market and daily kline discovery."""

from copy import deepcopy
from datetime import UTC, date, datetime
import json
from pathlib import Path
from typing import cast

import httpx
import pytest

from crypto_downloader.datasets import DatasetSpec, SPOT_KLINES, get_dataset
from crypto_downloader.models import Market, Resource, ResourceKey
from crypto_downloader.source import Source
from crypto_downloader.sources.binance import (
    ARCHIVE_URL,
    BUCKET_URL,
    EXCHANGE_INFO_URLS,
    BinanceSource,
)

FIXTURES = Path(__file__).parent / "fixtures"
KEY = ResourceKey("binance", "spot", "klines", "BTCUSDT", "1m")
SPOT_KLINES_PREFIX = "data/spot/daily/klines/"
SPOT_EXCHANGE_INFO_URL = EXCHANGE_INFO_URLS["spot"]


def fixture_text(name: str) -> str:
    """Read one UTF-8 Binance response fixture.

    Args:
        name: The fixture filename.

    Returns:
        The fixture contents.
    """
    return (FIXTURES / name).read_text(encoding="utf-8")


def exchange_info() -> dict[str, object]:
    """Return a fresh copy of the Spot exchange-info fixture.

    Returns:
        The parsed exchange-info response.
    """
    return cast(
        dict[str, object],
        json.loads(fixture_text("binance_spot_exchange_info.json")),
    )


def futures_exchange_info(product: str) -> dict[str, object]:
    """Return a fresh copy of one perpetual Futures exchange-info fixture.

    Args:
        product: The Futures product whose fixture is required.

    Returns:
        The parsed USD-M or COIN-M exchange-info response.
    """
    filenames = {
        "um": "binance_um_exchange_info.json",
        "cm": "binance_cm_exchange_info.json",
    }
    return cast(dict[str, object], json.loads(fixture_text(filenames[product])))


def listing(
    *,
    keys: tuple[str, ...] = (),
    prefixes: tuple[str, ...] = (),
    truncated: str = "false",
    marker: str | None = None,
    namespace: bool = True,
) -> str:
    """Build a small S3-style XML listing.

    Args:
        keys: Object keys placed in the listing.
        prefixes: Common folder prefixes placed in the listing.
        truncated: The literal pagination value.
        marker: The optional marker for the next page.
        namespace: Whether to include the normal S3 XML namespace.

    Returns:
        The complete XML listing text.
    """
    xmlns = ' xmlns="http://s3.amazonaws.com/doc/2006-03-01/"' if namespace else ""
    next_marker = f"<NextMarker>{marker}</NextMarker>" if marker is not None else ""
    contents = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    folders = "".join(
        f"<CommonPrefixes><Prefix>{prefix}</Prefix></CommonPrefixes>"
        for prefix in prefixes
    )
    return (
        f"<ListBucketResult{xmlns}><IsTruncated>{truncated}</IsTruncated>"
        f"{next_marker}{contents}{folders}</ListBucketResult>"
    )


def test_binance_declares_supported_market_products() -> None:
    """Confirm Binance advertises Spot and perpetual Futures products."""
    source = BinanceSource(timeout=12.0, retries=2, backoff=0.25)

    assert source.code == "binance"
    assert source.products == ("spot", "um", "cm")
    assert source.active_statuses == frozenset({"TRADING"})
    assert source.timeout == 12.0
    assert source.retries == 2
    assert source.backoff == 0.25
    assert source.max_concurrency == 32


def test_source_contract_remains_limited_to_five_operations() -> None:
    """Confirm sources expose bounded discovery and ingestion without orchestration."""
    operations = {
        name
        for name, value in vars(Source).items()
        if callable(value) and not name.startswith("_")
    }

    assert operations == {
        "checksum",
        "markets",
        "first_resource",
        "resources",
        "ingest",
    }


def test_market_discovery_preserves_native_metadata_and_merges_archive_only() -> None:
    """Confirm Spot metadata and symbol folders form one sorted snapshot."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return exchange metadata and two symbol-folder pages."""
        requests.append(request)
        if str(request.url).startswith(SPOT_EXCHANGE_INFO_URL):
            return httpx.Response(200, json=exchange_info())
        marker = request.url.params.get("marker")
        filename = (
            "binance_spot_symbols_page_2.xml"
            if marker is not None
            else "binance_spot_symbols_page_1.xml"
        )
        return httpx.Response(200, text=fixture_text(filename))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        markets = BinanceSource(timeout=9.0).markets(client, "spot")

    assert markets == [
        Market("BTCUSDT", "BTCUSDT", "BTC", "USDT", "TRADING"),
        Market("OLDUSDT", "OLDUSDT"),
        Market("XRPTUSD", "XRPTUSD", "XRP", "TUSD", "BREAK"),
    ]
    assert len(requests) == 3
    assert requests[0].url.params["showPermissionSets"] == "false"
    assert all(
        set(request.extensions["timeout"].values()) == {9.0} for request in requests
    )
    for request in requests[1:]:
        assert str(request.url).startswith(BUCKET_URL)
        assert request.url.params["prefix"] == SPOT_KLINES_PREFIX
        assert request.url.params["delimiter"] == "/"
    assert requests[2].url.params["marker"] == f"{SPOT_KLINES_PREFIX}BTCUSDT/"


@pytest.mark.parametrize(
    ("product", "archive_symbols", "expected"),
    [
        (
            "um",
            ("ARCHIVEUSDT", "BTCUSDT", "BTCUSDT_260626", "TRADIFIUSDT"),
            [
                Market(
                    symbol="ARCHIVEUSDT",
                    normalized_symbol="ARCHIVEUSDT",
                    pair="ARCHIVEUSDT",
                    contract_type="PERPETUAL",
                ),
                Market(
                    symbol="BTCUSDT",
                    normalized_symbol="BTCUSDT",
                    base_asset="BTC",
                    quote_asset="USDT",
                    status="TRADING",
                    pair="BTCUSDT",
                    contract_type="PERPETUAL",
                    onboard_time=datetime(2019, 9, 8, 17, 55, tzinfo=UTC),
                ),
                Market(
                    symbol="ETHUSDT",
                    normalized_symbol="ETHUSDT",
                    base_asset="ETH",
                    quote_asset="USDT",
                    status="SETTLING",
                    pair="ETHUSDT",
                    contract_type="PERPETUAL",
                    onboard_time=datetime(2019, 11, 27, 7, 45, tzinfo=UTC),
                ),
            ],
        ),
        (
            "cm",
            ("ARCHIVEUSD_PERP", "BTCUSD_PERP", "BTCUSD_260626", "ETHUSD_PERP"),
            [
                Market(
                    symbol="ARCHIVEUSD_PERP",
                    normalized_symbol="ARCHIVEUSDPERP",
                    pair="ARCHIVEUSD",
                    contract_type="PERPETUAL",
                ),
                Market(
                    symbol="BTCUSD_PERP",
                    normalized_symbol="BTCUSDPERP",
                    base_asset="BTC",
                    quote_asset="USD",
                    status="TRADING",
                    pair="BTCUSD",
                    contract_type="PERPETUAL",
                    contract_size=100.0,
                    onboard_time=datetime(2020, 8, 10, 7, tzinfo=UTC),
                ),
                Market(
                    symbol="ETHUSD_PERP",
                    normalized_symbol="ETHUSDPERP",
                    base_asset="ETH",
                    quote_asset="USD",
                    status="TRADING",
                    pair="ETHUSD",
                    contract_type="PERPETUAL",
                    contract_size=10.0,
                    onboard_time=datetime(2020, 8, 10, 7, tzinfo=UTC),
                ),
            ],
        ),
    ],
)
def test_perpetual_market_discovery_reads_product_endpoints_and_filters_contracts(
    product: str,
    archive_symbols: tuple[str, ...],
    expected: list[Market],
) -> None:
    """Confirm Futures discovery keeps only standard perpetual contracts.

    Args:
        product: The Futures product under test.
        archive_symbols: Symbols supplied by the product's Kline folders.
        expected: The perpetual snapshot expected after archive-only merging.
    """
    requests: list[httpx.Request] = []
    archive_root = f"data/futures/{product}/daily/klines/"

    def handler(request: httpx.Request) -> httpx.Response:
        """Return product metadata followed by an archive folder listing."""
        requests.append(request)
        if str(request.url).startswith(EXCHANGE_INFO_URLS[product]):
            return httpx.Response(200, json=futures_exchange_info(product))
        return httpx.Response(
            200,
            text=listing(
                prefixes=tuple(f"{archive_root}{symbol}/" for symbol in archive_symbols)
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        markets = BinanceSource().markets(client, product)

    assert markets == expected
    assert str(requests[0].url) == EXCHANGE_INFO_URLS[product]
    assert not requests[0].url.params
    assert requests[1].url.params["prefix"] == archive_root
    assert requests[1].url.params["delimiter"] == "/"
    assert all(market.delivery_time is None for market in markets)


@pytest.mark.parametrize(
    ("product", "row"),
    [
        (
            "um",
            {
                "symbol": "BTCUSDT",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "contractType": "PERPETUAL",
                "onboardDate": "not-a-timestamp",
            },
        ),
        (
            "cm",
            {
                "symbol": "BTCUSD_PERP",
                "baseAsset": "BTC",
                "quoteAsset": "USD",
                "contractType": "PERPETUAL",
                "contractSize": 100,
                "onboardDate": 1597042800000,
            },
        ),
    ],
)
def test_invalid_perpetual_market_metadata_is_rejected(
    product: str, row: dict[str, object]
) -> None:
    """Confirm malformed retained perpetual markets cannot enter the catalog.

    Args:
        product: The Futures product represented by the malformed row.
        row: The incomplete or invalid exchange-info market object.
    """
    with pytest.raises(ValueError, match="exchangeInfo"):
        BinanceSource._exchange_markets({"symbols": [row]}, product)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"symbols": []},
        {"symbols": [None]},
        {"symbols": [{"symbol": "BTCUSDT"}]},
        {
            "symbols": [
                {
                    "symbol": "../BTCUSDT",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "status": "TRADING",
                }
            ]
        },
    ],
)
def test_invalid_exchange_info_fails_before_archive_listing(payload: object) -> None:
    """Confirm malformed market snapshots are never stored as valid metadata.

    Args:
        payload: The malformed exchange-info response.
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Count and return the malformed exchange response."""
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="exchangeInfo"):
            BinanceSource().markets(client, "spot")

    assert calls == 1


def test_invalid_json_exchange_info_fails_before_archive_listing() -> None:
    """Confirm a non-JSON exchange response cannot become an empty snapshot."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a response that is not JSON."""
        return httpx.Response(200, text="not json")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            BinanceSource().markets(client, "spot")


def test_duplicate_exchange_symbol_is_rejected() -> None:
    """Confirm duplicate native symbols make the snapshot invalid."""
    payload = exchange_info()
    symbols = cast(list[object], payload["symbols"])
    symbols.append(deepcopy(symbols[0]))

    def handler(request: httpx.Request) -> httpx.Response:
        """Return exchange metadata containing a duplicate symbol."""
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="duplicate"):
            BinanceSource().markets(client, "spot")


def test_non_ascii_exchange_symbols_are_ignored_without_rejecting_snapshot() -> None:
    """Confirm unsupported Unicode symbols do not invalidate normal markets."""
    payload = exchange_info()
    symbols = cast(list[object], payload["symbols"])
    symbols.append(
        {
            "symbol": "币安人生USDT",
            "baseAsset": "币安人生",
            "quoteAsset": "USDT",
            "status": "TRADING",
        }
    )

    markets, _ = BinanceSource._exchange_markets(payload, "spot")

    assert [market.symbol for market in markets.values()] == ["BTCUSDT", "XRPTUSD"]


def test_unsupported_market_product_fails_without_http() -> None:
    """Confirm unsupported Binance products cannot request metadata."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Count any unexpected request."""
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="product"):
            BinanceSource().markets(client, "options")

    assert calls == 0


@pytest.mark.parametrize(
    "body",
    [
        "not xml",
        "<Error/>",
        "<ListBucketResult/>",
        listing(truncated="maybe"),
    ],
)
def test_malformed_symbol_listing_is_not_treated_as_empty(body: str) -> None:
    """Confirm malformed archive metadata aborts market discovery.

    Args:
        body: The invalid bucket response body.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """Return valid exchange metadata followed by malformed XML."""
        if str(request.url).startswith(SPOT_EXCHANGE_INFO_URL):
            return httpx.Response(200, json=exchange_info())
        return httpx.Response(200, text=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="listing"):
            BinanceSource().markets(client, "spot")


def test_truncated_listing_without_new_marker_is_rejected() -> None:
    """Confirm pagination cannot silently stop or loop without progress."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an empty truncated page after valid exchange metadata."""
        if str(request.url).startswith(SPOT_EXCHANGE_INFO_URL):
            return httpx.Response(200, json=exchange_info())
        return httpx.Response(200, text=listing(truncated="true"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="marker"):
            BinanceSource().markets(client, "spot")


def test_pagination_cycle_is_rejected() -> None:
    """Confirm a repeated server marker cannot create an infinite loop."""
    bucket_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return two truncated pages with the same next marker."""
        nonlocal bucket_calls
        if str(request.url).startswith(SPOT_EXCHANGE_INFO_URL):
            return httpx.Response(200, json=exchange_info())
        bucket_calls += 1
        return httpx.Response(
            200,
            text=listing(
                prefixes=(f"{SPOT_KLINES_PREFIX}BTCUSDT/",),
                truncated="true",
                marker="same-marker",
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="marker"):
            BinanceSource().markets(client, "spot")

    assert bucket_calls == 2


def test_first_resource_uses_a_small_forward_listing() -> None:
    """Confirm earliest availability needs one bounded bucket request."""
    prefix = f"{SPOT_KLINES_PREFIX}BTCUSDT/1m/"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the first available ZIP and its checksum object."""
        requests.append(request)
        return httpx.Response(
            200,
            text=listing(
                keys=(
                    f"{prefix}BTCUSDT-1m-2022-03-04.zip",
                    f"{prefix}BTCUSDT-1m-2022-03-04.zip.CHECKSUM",
                )
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resource = BinanceSource().first_resource(
            client, KEY, date(2020, 1, 1), date(2025, 1, 1)
        )

    assert resource is not None
    assert resource.day == date(2022, 3, 4)
    assert resource.checksum_url == f"{resource.url}.CHECKSUM"
    assert len(requests) == 1
    assert requests[0].url.params["max-keys"] == "2"
    assert requests[0].url.params["marker"].endswith("BTCUSDT-1m-2020-01-01")


def test_first_resource_can_start_at_the_beginning_of_a_source_folder() -> None:
    """Confirm all-history discovery omits the date marker from its listing.

    The source should choose the first actual archive key rather than assuming an
    arbitrary exchange launch date.
    """
    prefix = f"{SPOT_KLINES_PREFIX}BTCUSDT/1m/"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record an unbounded listing and return Binance's first archive."""
        requests.append(request)
        return httpx.Response(
            200,
            text=listing(keys=(f"{prefix}BTCUSDT-1m-2017-08-17.zip",)),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resource = BinanceSource().first_resource(client, KEY, None, date(2025, 1, 1))

    assert resource is not None
    assert resource.day == date(2017, 8, 17)
    assert requests[0].url.params.get("marker") is None
    assert requests[0].url.params["max-keys"] == "2"


def test_archive_layout_uses_dataset_rules_and_an_optional_archive_symbol() -> None:
    """Confirm interval and raw archive paths stay inside the Binance connector."""
    raw_trades = DatasetSpec(
        product="spot",
        name="trades",
        remote_name="trades",
        source_columns=("event_time",),
        stored_columns=("event_time",),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases={},
    )
    alternate_key = ResourceKey(
        "binance",
        "spot",
        "klines",
        "BTCUSD_PERP",
        "1m",
        archive_symbol="BTCUSD",
    )
    raw_key = ResourceKey("binance", "spot", "trades", "BTCUSDT", None)

    interval_prefix, interval_stem, interval_symbol = BinanceSource._archive_layout(
        alternate_key, SPOT_KLINES
    )
    raw_prefix, raw_stem, raw_symbol = BinanceSource._archive_layout(
        raw_key, raw_trades
    )

    assert interval_prefix == "data/spot/daily/klines/BTCUSD/1m/"
    assert interval_stem == "BTCUSD-1m-"
    assert interval_symbol == "BTCUSD"
    assert raw_prefix == "data/spot/daily/trades/BTCUSDT/"
    assert raw_stem == "BTCUSDT-trades-"
    assert raw_symbol == "BTCUSDT"


def test_resource_discovery_records_dataset_metadata_and_archive_symbol() -> None:
    """Confirm discovered resources retain routing and schema metadata."""
    prefix = f"{SPOT_KLINES_PREFIX}BTCUSDT/1m/"

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one exact daily kline archive."""
        return httpx.Response(
            200,
            text=listing(keys=(f"{prefix}BTCUSDT-1m-2025-01-01.zip",)),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resources = BinanceSource().resources(
            client, KEY, date(2025, 1, 1), date(2025, 1, 1)
        )

    url = f"{ARCHIVE_URL}/{prefix}BTCUSDT-1m-2025-01-01.zip"
    assert resources == [
        Resource(
            day=date(2025, 1, 1),
            url=url,
            checksum_url=f"{url}.CHECKSUM",
            archive_symbol="BTCUSDT",
            timestamp_column="open_time",
            schema_version=1,
        )
    ]


@pytest.mark.parametrize(
    ("dataset", "remote_name", "filename", "timestamp_column"),
    [
        ("trades", "trades", "BTCUSDT-trades-2025-01-01.zip", "event_time"),
        (
            "agg_trades",
            "aggTrades",
            "BTCUSDT-aggTrades-2025-01-01.zip",
            "event_time",
        ),
    ],
)
def test_daily_event_resource_discovery_uses_raw_dataset_layout(
    dataset: str, remote_name: str, filename: str, timestamp_column: str
) -> None:
    """Confirm Spot event archives do not add a kline interval folder.

    Args:
        dataset: The public snake-case event dataset name.
        remote_name: The exact Binance daily archive folder name.
        filename: The expected Binance archive basename.
        timestamp_column: The canonical event timestamp column.
    """
    key = ResourceKey("binance", "spot", dataset, "BTCUSDT", None)
    prefix = f"data/spot/daily/{remote_name}/BTCUSDT/"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the source request and return one matching object key."""
        requests.append(request)
        return httpx.Response(200, text=listing(keys=(f"{prefix}{filename}",)))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resources = BinanceSource().resources(
            client, key, date(2025, 1, 1), date(2025, 1, 1)
        )

    assert len(resources) == 1
    assert resources[0].url == f"{ARCHIVE_URL}/{prefix}{filename}"
    assert resources[0].archive_symbol == "BTCUSDT"
    assert resources[0].timestamp_column == timestamp_column
    assert requests[0].url.params["prefix"] == prefix
    assert requests[0].url.params["marker"] == f"{prefix}{filename[:-4]}"


@pytest.mark.parametrize(
    ("product", "symbol"),
    [("um", "BTCUSDT"), ("cm", "BTCUSD_PERP")],
)
def test_metrics_resource_discovery_uses_the_futures_raw_layout(
    product: str, symbol: str
) -> None:
    """Confirm Futures metrics archives have no interval subdirectory.

    Args:
        product: The USD-M or COIN-M perpetual product.
        symbol: The native perpetual contract archive symbol.
    """
    dataset = get_dataset(product, "metrics")
    key = ResourceKey("binance", product, dataset.name, symbol, None)
    prefix = f"data/futures/{product}/daily/metrics/{symbol}/"
    filename = f"{symbol}-metrics-2024-01-01.zip"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the requested source location and return one archive."""
        requests.append(request)
        return httpx.Response(200, text=listing(keys=(f"{prefix}{filename}",)))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resources = BinanceSource().resources(
            client, key, date(2024, 1, 1), date(2024, 1, 1)
        )

    assert len(resources) == 1
    assert resources[0].url == f"{ARCHIVE_URL}/{prefix}{filename}"
    assert resources[0].timestamp_column == "event_time"
    assert requests[0].url.params["prefix"] == prefix
    assert requests[0].url.params["marker"] == f"{prefix}{filename[:-4]}"


@pytest.mark.parametrize(
    ("product", "symbol"),
    [("um", "BTCUSDT"), ("cm", "BTCUSD_PERP")],
)
def test_book_depth_resource_discovery_uses_the_futures_raw_layout(
    product: str, symbol: str
) -> None:
    """Confirm Futures book-depth archives have no interval subdirectory.

    Args:
        product: The USD-M or COIN-M perpetual product.
        symbol: The native perpetual contract archive symbol.
    """
    dataset = get_dataset(product, "book_depth")
    key = ResourceKey("binance", product, dataset.name, symbol, None)
    prefix = f"data/futures/{product}/daily/bookDepth/{symbol}/"
    filename = f"{symbol}-bookDepth-2024-01-01.zip"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the requested source location and return one archive."""
        requests.append(request)
        return httpx.Response(200, text=listing(keys=(f"{prefix}{filename}",)))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resources = BinanceSource().resources(
            client, key, date(2024, 1, 1), date(2024, 1, 1)
        )

    assert len(resources) == 1
    assert resources[0].url == f"{ARCHIVE_URL}/{prefix}{filename}"
    assert resources[0].timestamp_column == "event_time"
    assert requests[0].url.params["prefix"] == prefix
    assert requests[0].url.params["marker"] == f"{prefix}{filename[:-4]}"


@pytest.mark.parametrize(
    ("product", "symbol"),
    [("um", "BTCUSDT"), ("cm", "BTCUSD_PERP")],
)
def test_mark_price_resource_discovery_uses_the_futures_interval_layout(
    product: str, symbol: str
) -> None:
    """Confirm mark-price Klines reuse the Futures daily archive layout.

    Args:
        product: The Binance perpetual Futures product.
        symbol: The perpetual contract archive symbol.
    """
    dataset = get_dataset(product, "mark_price_klines")
    key = ResourceKey("binance", product, dataset.name, symbol, "1m")
    prefix = f"data/futures/{product}/daily/markPriceKlines/{symbol}/1m/"
    filename = f"{symbol}-1m-2024-01-01.zip"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the requested listing and return one archive object."""
        requests.append(request)
        return httpx.Response(200, text=listing(keys=(f"{prefix}{filename}",)))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resources = BinanceSource().resources(
            client, key, date(2024, 1, 1), date(2024, 1, 1)
        )

    assert len(resources) == 1
    assert resources[0].url == f"{ARCHIVE_URL}/{prefix}{filename}"
    assert resources[0].timestamp_column == "open_time"
    assert resources[0].schema_version == dataset.schema_version
    assert requests[0].url.params["prefix"] == prefix
    assert requests[0].url.params["marker"] == f"{prefix}{filename[:-4]}"


@pytest.mark.parametrize(
    ("product", "symbol", "archive_symbol"),
    [("um", "BTCUSDT", "BTCUSDT"), ("cm", "BTCUSD_PERP", "BTCUSD")],
)
def test_index_price_resource_discovery_uses_the_declared_archive_symbol(
    product: str, symbol: str, archive_symbol: str
) -> None:
    """Confirm index archives route contracts through their source identifier.

    Args:
        product: The Binance perpetual Futures product.
        symbol: The public perpetual contract identifier.
        archive_symbol: The source folder and archive identifier.
    """
    dataset = get_dataset(product, "index_price_klines")
    key = ResourceKey(
        "binance",
        product,
        dataset.name,
        symbol,
        "1m",
        archive_symbol=archive_symbol,
    )
    prefix = f"data/futures/{product}/daily/indexPriceKlines/{archive_symbol}/1m/"
    filename = f"{archive_symbol}-1m-2024-01-01.zip"

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return one source archive listed under its native index symbol."""
        return httpx.Response(200, text=listing(keys=(f"{prefix}{filename}",)))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resources = BinanceSource().resources(
            client, key, date(2024, 1, 1), date(2024, 1, 1)
        )

    assert resources[0].archive_symbol == archive_symbol
    assert resources[0].url == f"{ARCHIVE_URL}/{prefix}{filename}"


def test_first_resource_returns_none_when_no_archive_follows_boundary() -> None:
    """Confirm a valid empty first-page listing means no known availability."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one complete empty listing."""
        return httpx.Response(200, text=listing())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resource = BinanceSource().first_resource(
            client, KEY, date(2020, 1, 1), date(2025, 1, 1)
        )

    assert resource is None


def test_daily_resource_discovery_paginates_filters_and_stops_after_end() -> None:
    """Confirm only exact daily ZIPs inside the requested range are returned."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return two realistic daily archive pages."""
        requests.append(request)
        filename = (
            "binance_spot_klines_page_1.xml"
            if len(requests) == 1
            else "binance_spot_klines_page_2.xml"
        )
        return httpx.Response(200, text=fixture_text(filename))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resources = BinanceSource().resources(
            client, KEY, date(2025, 1, 1), date(2025, 1, 2)
        )

    assert [resource.day for resource in resources] == [
        date(2025, 1, 1),
        date(2025, 1, 2),
    ]
    assert resources[0].url == (
        f"{ARCHIVE_URL}/data/spot/daily/klines/BTCUSDT/1m/" "BTCUSDT-1m-2025-01-01.zip"
    )
    assert resources[0].checksum_url == f"{resources[0].url}.CHECKSUM"
    assert len(requests) == 2
    prefix = f"{SPOT_KLINES_PREFIX}BTCUSDT/1m/"
    assert requests[0].url.params["prefix"] == prefix
    assert requests[0].url.params["marker"] == f"{prefix}BTCUSDT-1m-2025-01-01"
    assert requests[1].url.params["marker"].endswith("BTCUSDT-1m-invalid.zip")


def test_daily_listing_without_files_returns_empty_result() -> None:
    """Confirm a valid empty listing reports no available daily resources."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one complete bucket page without objects."""
        return httpx.Response(200, text=listing(namespace=False))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert (
            BinanceSource().resources(client, KEY, date(2025, 1, 1), date(2025, 1, 2))
            == []
        )


def test_daily_listing_ignores_prior_and_impossible_dates() -> None:
    """Confirm malformed servers cannot leak invalid or earlier archive days."""
    prefix = f"{SPOT_KLINES_PREFIX}BTCUSDT/1m/"
    keys = (
        f"{prefix}BTCUSDT-1m-2025-01-01.zip",
        f"{prefix}BTCUSDT-1m-2025-99-99.zip",
        f"{prefix}BTCUSDT-1m-2025-01-02.zip",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        """Return prior, impossible, and requested archive dates."""
        return httpx.Response(200, text=listing(keys=keys))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resources = BinanceSource().resources(
            client, KEY, date(2025, 1, 2), date(2025, 1, 2)
        )

    assert [resource.day for resource in resources] == [date(2025, 1, 2)]


@pytest.mark.parametrize(
    ("key", "start_day", "end_day", "message"),
    [
        (
            ResourceKey("other", "spot", "klines", "BTCUSDT", "1m"),
            date(2025, 1, 1),
            date(2025, 1, 2),
            "source",
        ),
        (
            ResourceKey("binance", "um", "not_a_dataset", "BTCUSDT", None),
            date(2025, 1, 1),
            date(2025, 1, 2),
            "dataset",
        ),
        (
            ResourceKey("binance", "spot", "trades", "BTCUSDT", "1m"),
            date(2025, 1, 1),
            date(2025, 1, 2),
            "interval",
        ),
        (
            ResourceKey("binance", "spot", "klines", "BTCUSDT", "5m"),
            date(2025, 1, 1),
            date(2025, 1, 2),
            "interval",
        ),
        (
            ResourceKey("binance", "spot", "klines", "../BTC", "1m"),
            date(2025, 1, 1),
            date(2025, 1, 2),
            "symbol",
        ),
        (
            ResourceKey(
                "binance",
                "spot",
                "klines",
                "BTCUSD_PERP",
                "1m",
                archive_symbol="../BTCUSD",
            ),
            date(2025, 1, 1),
            date(2025, 1, 2),
            "archive symbol",
        ),
        (
            KEY,
            date(2025, 1, 2),
            date(2025, 1, 1),
            "range",
        ),
    ],
)
def test_invalid_resource_request_fails_without_http(
    key: ResourceKey, start_day: date, end_day: date, message: str
) -> None:
    """Confirm unsupported or unsafe resource requests never reach the network.

    Args:
        key: The invalid resource identity.
        start_day: The proposed first archive day.
        end_day: The proposed last archive day.
        message: The invalid field named in the expected error.
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Count any unexpected request."""
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match=message):
            BinanceSource().resources(client, key, start_day, end_day)

    assert calls == 0


@pytest.mark.parametrize("body", ["bad", "<Error/>", "<ListBucketResult/>"])
def test_malformed_daily_listing_is_not_treated_as_no_files(body: str) -> None:
    """Confirm invalid resource XML remains distinguishable from no files.

    Args:
        body: The malformed bucket response.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """Return malformed XML for a daily archive request."""
        return httpx.Response(200, text=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="listing"):
            BinanceSource().resources(client, KEY, date(2025, 1, 1), date(2025, 1, 2))


def test_binance_metadata_uses_shared_http_retries() -> None:
    """Confirm Binance discovery delegates temporary failures to the HTTP layer."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Throttle once and then return a complete empty market archive."""
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        if str(request.url).startswith(SPOT_EXCHANGE_INFO_URL):
            return httpx.Response(200, json=exchange_info())
        return httpx.Response(200, text=listing())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        markets = BinanceSource(retries=1, backoff=0).markets(client, "spot")

    assert len(markets) == 2
    assert calls == 3
