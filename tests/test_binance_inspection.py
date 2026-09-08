"""Test Binance market and availability inspection through the public facade."""

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import httpx
import pytest

from crypto_downloader import Availability, Binance, Market
from crypto_downloader.catalog import open_catalog
from crypto_downloader.models import IngestedResource, Resource, ResourceKey
from crypto_downloader.sources.binance import BinanceSource

DAY_1 = date(2024, 1, 1)
DAY_2 = date(2024, 1, 2)
DAY_3 = date(2024, 1, 3)
DAY_5 = date(2024, 1, 5)


@pytest.fixture(autouse=True)
def fixed_inspection_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep active-market policy bounds deterministic in inspection tests.

    Args:
        monkeypatch: The pytest helper used to replace the UTC clock.
    """
    monkeypatch.setattr(
        "crypto_downloader.inspection.utc_today", lambda: DAY_5 + date.resolution
    )


def market(
    symbol: str,
    *,
    status: str | None = "TRADING",
    base: str | None = "BTC",
    quote: str | None = "USDT",
    pair: str | None = None,
) -> Market:
    """Create one source market for inspection tests.

    Args:
        symbol: The native Binance market symbol.
        status: The native Binance status.
        base: The base asset.
        quote: The quote asset.
        pair: The alternate Binance pair identifier.

    Returns:
        One immutable market value.
    """
    return Market(
        symbol,
        "".join(character for character in symbol if character.isalnum()),
        base,
        quote,
        status,
        pair=pair,
        contract_type="PERPETUAL" if pair is not None else None,
    )


def source_for(service: Binance) -> BinanceSource:
    """Return the facade's concrete Binance source.

    Args:
        service: The facade whose internal source is needed for test wiring.

    Returns:
        The concrete Binance source strategy.
    """
    return cast(BinanceSource, service._downloader.source)


def resource(day: date) -> Resource:
    """Create one discovered daily archive.

    Args:
        day: The archive date.

    Returns:
        One deterministic remote resource.
    """
    return Resource(day, f"https://archive/{day}.zip", f"https://archive/{day}.sum")


def seed_availability(
    service: Binance,
    *,
    symbol: str = "BTCUSDT",
    pair_name: str | None = None,
    product: str = "spot",
    dataset: str = "klines",
    interval: str = "1m",
) -> tuple[ResourceKey, Path]:
    """Store a mixed ready, missing, failed, and unavailable coverage example.

    Args:
        service: The facade whose catalog receives the example.
        symbol: The native catalog market.
        pair_name: The optional alternate archive pair.
        product: The Binance product.
        dataset: The Binance dataset.
        interval: The stored resource interval.

    Returns:
        The resource key and valid local Parquet path.
    """
    key = ResourceKey("binance", product, dataset, symbol, interval, pair_name)
    parquet = service.data_dir / "one.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"parquet")
    stat = parquet.stat()
    with open_catalog(service.data_dir / "catalog.duckdb") as catalog:
        catalog.save_markets(
            "binance",
            product,
            [
                market(
                    symbol, pair=pair_name, quote="USD" if product == "cm" else "USDT"
                )
            ],
        )
        catalog.save_discovery(
            key,
            DAY_1,
            DAY_5,
            [resource(DAY_1), resource(DAY_2), resource(DAY_3)],
        )
        catalog.save_source_bounds(key, DAY_1, None)
        catalog.mark_ready(
            key,
            DAY_1,
            parquet,
            IngestedResource(
                archive_sha256="a" * 64,
                parquet_sha256="b" * 64,
                parquet_size=stat.st_size,
                parquet_mtime_ns=stat.st_mtime_ns,
                row_count=2,
                first_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                last_timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                timestamp_column="open_time" if interval != "raw" else "event_time",
                schema_version=1,
            ),
        )
        catalog.mark_failed(key, DAY_3, "broken archive")
    return key, parquet


def test_inspection_models_are_immutable_and_describe_activity() -> None:
    """Confirm public inspection values cannot be changed after construction."""
    value = market("BTCUSDT")
    availability = Availability(
        "binance",
        "spot",
        "klines",
        "BTCUSDT",
        "1h",
        "1m",
        None,
        None,
        None,
        (),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    assert value.active is True
    assert market("BTCUSDT", status="BREAK").active is False
    with pytest.raises(FrozenInstanceError):
        value.status = "BREAK"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        availability.cached_days = 1  # type: ignore[misc]


def test_get_markets_refreshes_once_reuses_ttl_and_decorates_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm market inspection shares refresh TTL and returns public context.

    Args:
        tmp_path: The isolated facade data directory.
        monkeypatch: The pytest helper used to replace source requests.
    """
    service = Binance(tmp_path, progress=False)
    calls: list[str] = []

    def markets(_client: httpx.Client, product: str) -> list[Market]:
        """Return one deterministic source snapshot."""
        calls.append(product)
        return [market("ETHUSDT", base="ETH"), market("BTCUSDT")]

    monkeypatch.setattr(source_for(service), "markets", markets)

    first = service.get_markets()
    second = service.get_markets()
    refreshed = service.get_markets(refresh=True)
    offline = service.get_markets(offline=True)

    assert calls == ["spot", "spot"]
    assert [value.symbol for value in first] == ["BTCUSDT", "ETHUSDT"]
    assert first == second == refreshed == offline
    assert all(value.source == "binance" for value in first)
    assert all(value.product == "spot" for value in first)


def test_get_markets_applies_case_insensitive_exact_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm native status and quote filters preserve deterministic ordering.

    Args:
        tmp_path: The isolated facade data directory.
        monkeypatch: The pytest helper used to replace source requests.
    """
    service = Binance(tmp_path, progress=False)

    def markets(_client: httpx.Client, _product: str) -> list[Market]:
        """Return markets covering each filter outcome."""
        return [
            market("ETHUSDT", base="ETH"),
            market("BTCUSDT"),
            market("BTCFDUSD", quote="FDUSD"),
            market("OLDUSDT", status="BREAK", base="OLD"),
        ]

    monkeypatch.setattr(source_for(service), "markets", markets)

    values = service.get_markets(status="trading", quote_asset="usdt")

    assert [value.symbol for value in values] == ["BTCUSDT", "ETHUSDT"]


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"product": "options"}, ValueError, "product"),
        ({"product": 1}, TypeError, "product"),
        ({"status": ""}, ValueError, "status"),
        ({"status": 1}, TypeError, "status"),
        ({"quote_asset": " "}, ValueError, "quote_asset"),
        ({"quote_asset": []}, TypeError, "quote_asset"),
        ({"refresh": 1}, TypeError, "refresh"),
        ({"offline": None}, TypeError, "offline"),
        ({"refresh": True, "offline": True}, ValueError, "refresh.*offline"),
    ],
)
def test_get_markets_rejects_invalid_options_before_io(
    tmp_path: Path,
    options: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """Confirm malformed market filters fail before disk or network access.

    Args:
        tmp_path: The isolated facade data directory.
        options: The malformed keyword arguments.
        error: The expected exception type.
        message: Text expected in the failure.
    """
    service = Binance(tmp_path, progress=False)

    with pytest.raises(error, match=message):
        service.get_markets(**options)  # type: ignore[arg-type]
    assert not (tmp_path / "catalog.duckdb").exists()


def test_get_markets_offline_requires_a_cached_snapshot(tmp_path: Path) -> None:
    """Confirm offline inspection does not invent an empty market snapshot.

    Args:
        tmp_path: The isolated facade data directory.
    """
    service = Binance(tmp_path, progress=False)

    with pytest.raises(RuntimeError, match="offline.*cached market"):
        service.get_markets(offline=True)
    assert not (tmp_path / "catalog.duckdb").exists()


def test_find_markets_offline_searches_the_product_snapshots_that_exist(
    tmp_path: Path,
) -> None:
    """Confirm all-product offline search tolerates products not cached yet.

    Args:
        tmp_path: The isolated facade data directory.
    """
    service = Binance(tmp_path, progress=False)
    with open_catalog(tmp_path / "catalog.duckdb") as catalog:
        catalog.save_markets("binance", "spot", [market("BTCUSDT")])

    values = service.find_markets("btc", offline=True)

    assert [(value.product, value.symbol) for value in values] == [("spot", "BTCUSDT")]


def test_find_markets_ranks_exact_prefix_and_fuzzy_matches_across_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm search keeps product identity and never selects one collision.

    Args:
        tmp_path: The isolated facade data directory.
        monkeypatch: The pytest helper used to replace source requests.
    """
    service = Binance(tmp_path, progress=False)

    def markets(_client: httpx.Client, product: str) -> list[Market]:
        """Return overlapping symbols across all Binance products."""
        if product == "spot":
            return [
                market("BTCUSDT"),
                market("BTCUSDC", quote="USDC"),
                market("BTCSTUSDT", base="BTCST"),
                market("ZBTUSDT", base="ZBT"),
                market("WCTUSDT", base="WCT"),
            ]
        if product == "um":
            return [market("BTCUSDT", pair="BTCUSDT")]
        return [market("BTCUSD_PERP", pair="BTCUSD", quote="USD")]

    monkeypatch.setattr(source_for(service), "markets", markets)

    exact = service.find_markets("btc-usdt", limit=2)
    prefix = service.find_markets("btcus", product="spot", limit=2)
    fuzzy = service.find_markets("btcsudt", status="trading", limit=2)
    transposed = service.find_markets("bctusdt", product="spot", limit=1)

    assert [(value.product, value.symbol) for value in exact] == [
        ("spot", "BTCUSDT"),
        ("um", "BTCUSDT"),
    ]
    assert [value.symbol for value in prefix] == ["BTCUSDC", "BTCUSDT"]
    assert [(value.product, value.symbol) for value in fuzzy] == [
        ("spot", "BTCUSDT"),
        ("um", "BTCUSDT"),
    ]
    assert [value.symbol for value in transposed] == ["BTCUSDT"]


@pytest.mark.parametrize(
    ("query", "limit", "error", "message"),
    [
        (1, 10, TypeError, "query"),
        ("", 10, ValueError, "query"),
        ("---", 10, ValueError, "query"),
        ("BTC", 0, ValueError, "limit"),
        ("BTC", -1, ValueError, "limit"),
        ("BTC", True, TypeError, "limit"),
        ("BTC", 1.5, TypeError, "limit"),
    ],
)
def test_find_markets_rejects_invalid_query_and_limit_before_io(
    tmp_path: Path,
    query: object,
    limit: object,
    error: type[Exception],
    message: str,
) -> None:
    """Confirm malformed search values fail before disk or network access.

    Args:
        tmp_path: The isolated facade data directory.
        query: The malformed search value.
        limit: The malformed result limit.
        error: The expected exception type.
        message: Text expected in the failure.
    """
    service = Binance(tmp_path, progress=False)

    with pytest.raises(error, match=message):
        service.find_markets(query, limit=limit)  # type: ignore[arg-type]
    assert not (tmp_path / "catalog.duckdb").exists()


def test_get_availability_reports_every_remote_and_local_state(
    tmp_path: Path,
) -> None:
    """Confirm local coverage separates all known daily resource states.

    Args:
        tmp_path: The isolated facade data directory.
    """
    service = Binance(tmp_path, progress=False)
    _, parquet = seed_availability(service)

    value = service.get_availability(
        "btc-usdt", product="spot", dataset="klines", interval="1h"
    )

    assert value == Availability(
        source="binance",
        product="spot",
        dataset="klines",
        symbol="BTCUSDT",
        interval="1h",
        storage_interval="1m",
        remote_range=(DAY_1, DAY_5),
        configured_range=(date(2020, 1, 1), DAY_5),
        cached_range=(DAY_1, DAY_1),
        scanned_ranges=((DAY_1, DAY_5),),
        scanned_days=5,
        available_days=3,
        cached_days=1,
        missing_days=1,
        unavailable_days=2,
        failed_days=1,
        row_count=2,
        local_bytes=parquet.stat().st_size,
    )


def test_get_availability_respects_the_configured_history_boundary(
    tmp_path: Path,
) -> None:
    """Confirm source bounds remain distinct from configured usable history.

    Args:
        tmp_path: The isolated facade data directory.
    """
    service = Binance(tmp_path, earliest_date="2024-01-02", progress=False)
    seed_availability(service)

    value = service.get_availability("BTCUSDT", product="spot", dataset="klines")

    assert value.remote_range == (DAY_1, DAY_5)
    assert value.configured_range == (DAY_2, DAY_5)


def test_get_availability_detects_a_missing_or_changed_ready_file(
    tmp_path: Path,
) -> None:
    """Confirm stale ready metadata is counted as missing local coverage.

    Args:
        tmp_path: The isolated facade data directory.
    """
    service = Binance(tmp_path, progress=False)
    _, parquet = seed_availability(service)
    parquet.write_bytes(b"changed")

    value = service.get_availability("BTCUSDT", product="spot", dataset="klines")

    assert value.cached_days == 0
    assert value.missing_days == 2
    assert value.cached_range is None
    assert value.row_count == 0
    assert value.local_bytes == 0


def test_get_availability_rejects_incompatible_ready_schema(tmp_path: Path) -> None:
    """Confirm availability does not count an obsolete Parquet schema as cached.

    Args:
        tmp_path: The isolated facade data directory.
    """
    service = Binance(tmp_path, progress=False)
    key, parquet = seed_availability(service)
    stat = parquet.stat()
    with open_catalog(service.data_dir / "catalog.duckdb") as catalog:
        catalog.mark_ready(
            key,
            DAY_1,
            parquet,
            IngestedResource(
                archive_sha256="a" * 64,
                parquet_sha256="b" * 64,
                parquet_size=stat.st_size,
                parquet_mtime_ns=stat.st_mtime_ns,
                row_count=2,
                first_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                last_timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                timestamp_column="event_time",
                schema_version=2,
            ),
        )

    value = service.get_availability("BTCUSDT", product="spot", dataset="klines")

    assert value.cached_days == 0
    assert value.missing_days == 2
    assert value.cached_range is None
    assert value.row_count == 0
    assert value.local_bytes == 0


def test_get_availability_is_local_and_requires_cataloged_markets(
    tmp_path: Path,
) -> None:
    """Confirm local availability never creates or refreshes a market catalog.

    Args:
        tmp_path: The isolated facade data directory.
    """
    service = Binance(tmp_path, progress=False)

    with pytest.raises(RuntimeError, match="cached market"):
        service.get_availability("BTCUSDT", product="spot", dataset="klines")
    assert not (tmp_path / "catalog.duckdb").exists()


def test_get_availability_reports_unknown_pairs_with_suggestions(
    tmp_path: Path,
) -> None:
    """Confirm local lookup rejects rather than substitutes a fuzzy market.

    Args:
        tmp_path: The isolated facade data directory.
    """
    service = Binance(tmp_path, progress=False)
    seed_availability(service)

    with pytest.raises(ValueError, match="BTCSUDT.*BTCUSDT"):
        service.get_availability("BTCSUDT", product="spot", dataset="klines")


def test_discover_availability_lists_only_the_bounded_range_without_ingestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm explicit discovery catalogs metadata but no archive data.

    Args:
        tmp_path: The isolated facade data directory.
        monkeypatch: The pytest helper used to replace source requests.
    """
    service = Binance(tmp_path, discovery_tail_days=1, progress=False)
    source = source_for(service)
    scans: list[tuple[ResourceKey, date, date]] = []
    first_calls: list[ResourceKey] = []

    def markets(_client: httpx.Client, _product: str) -> list[Market]:
        """Return the market selected for bounded discovery."""
        return [market("BTCUSDT")]

    def resources(
        _client: httpx.Client, key: ResourceKey, start: date, end: date
    ) -> list[Resource]:
        """Record a bounded listing and return one available day."""
        scans.append((key, start, end))
        return [resource(DAY_2)]

    def first_resource(
        _client: httpx.Client,
        key: ResourceKey,
        _start: date | None,
        _end: date,
    ) -> Resource:
        """Return and record the true first source archive."""
        first_calls.append(key)
        return resource(DAY_1)

    def ingest(*_args: object, **_kwargs: object) -> None:
        """Fail if availability inspection attempts archive ingestion."""
        raise AssertionError("discovery must not ingest archives")

    monkeypatch.setattr(source, "markets", markets)
    monkeypatch.setattr(source, "resources", resources)
    monkeypatch.setattr(source, "first_resource", first_resource)
    monkeypatch.setattr(source, "ingest", ingest)

    first = service.discover_availability(
        "btc-usdt",
        "2024-01-01",
        "2024-01-03",
        product="spot",
        dataset="klines",
        interval="1h",
    )
    second = service.discover_availability(
        "BTCUSDT",
        "2024-01-01",
        "2024-01-03",
        product="spot",
        dataset="klines",
        interval="1h",
    )
    refreshed = service.discover_availability(
        "BTCUSDT",
        "2024-01-01",
        "2024-01-03",
        product="spot",
        dataset="klines",
        interval="1h",
        refresh=True,
    )

    assert scans == [
        (ResourceKey("binance", "spot", "klines", "BTCUSDT", "1m"), DAY_2, DAY_3),
        (ResourceKey("binance", "spot", "klines", "BTCUSDT", "1m"), DAY_1, DAY_3),
    ]
    assert first_calls == [
        ResourceKey("binance", "spot", "klines", "BTCUSDT", "1m"),
        ResourceKey("binance", "spot", "klines", "BTCUSDT", "1m"),
    ]
    assert first == second == refreshed
    assert first.remote_range == (DAY_1, DAY_5)
    assert first.configured_range == (date(2020, 1, 1), DAY_5)
    assert first.scanned_ranges == ((DAY_1, DAY_3),)
    assert first.scanned_days == 3
    assert first.available_days == 2
    assert first.cached_days == 0
    assert first.missing_days == 2
    assert first.unavailable_days == 1


def test_discover_availability_maps_coin_m_index_archive_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm COIN-M index discovery uses its pair folder transparently.

    Args:
        tmp_path: The isolated facade data directory.
        monkeypatch: The pytest helper used to replace source requests.
    """
    service = Binance(tmp_path, progress=False)
    source = source_for(service)
    keys: list[ResourceKey] = []

    def markets(_client: httpx.Client, _product: str) -> list[Market]:
        """Return one COIN-M perpetual contract and index pair."""
        return [market("BTCUSD_PERP", pair="BTCUSD", quote="USD")]

    def resources(
        _client: httpx.Client, key: ResourceKey, _start: date, _end: date
    ) -> list[Resource]:
        """Record the resolved archive key without returning files."""
        keys.append(key)
        return []

    def first_resource(
        _client: httpx.Client,
        _key: ResourceKey,
        _start: date | None,
        _end: date,
    ) -> None:
        """Report that the test source has no matching archive."""
        return None

    monkeypatch.setattr(source, "markets", markets)
    monkeypatch.setattr(source, "resources", resources)
    monkeypatch.setattr(source, "first_resource", first_resource)

    value = service.discover_availability(
        "BTCUSD_PERP",
        DAY_1,
        DAY_1,
        product="cm",
        dataset="index_price_klines",
    )

    assert keys == [
        ResourceKey(
            "binance",
            "cm",
            "index_price_klines",
            "BTCUSD_PERP",
            "1m",
            "BTCUSD",
        )
    ]
    assert value.symbol == "BTCUSD_PERP"
    assert value.remote_range is None
    assert value.unavailable_days == 1


@pytest.mark.parametrize(
    ("method", "arguments", "options", "error", "message"),
    [
        (
            "get_availability",
            ("BTCUSDT",),
            {"product": "bad", "dataset": "klines"},
            ValueError,
            "product",
        ),
        (
            "get_availability",
            ("BTCUSDT",),
            {"product": "spot", "dataset": "metrics"},
            ValueError,
            "dataset",
        ),
        (
            "get_availability",
            ("BTCUSDT",),
            {"product": "spot", "dataset": "trades", "interval": "1m"},
            ValueError,
            "interval",
        ),
        (
            "discover_availability",
            ("BTCUSDT", "bad", DAY_1),
            {"product": "spot", "dataset": "klines"},
            ValueError,
            "dates",
        ),
        (
            "discover_availability",
            ("BTCUSDT", DAY_2, DAY_1),
            {"product": "spot", "dataset": "klines"},
            ValueError,
            "starting_date",
        ),
        (
            "discover_availability",
            ("BTCUSDT", DAY_1, DAY_1),
            {"product": "spot", "dataset": "klines", "refresh": 1},
            TypeError,
            "refresh",
        ),
    ],
)
def test_availability_methods_reject_invalid_requests_before_io(
    tmp_path: Path,
    method: str,
    arguments: tuple[object, ...],
    options: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """Confirm invalid availability requests fail before disk or network access.

    Args:
        tmp_path: The isolated facade data directory.
        method: The availability method under test.
        arguments: The malformed positional request values.
        options: The request keyword arguments.
        error: The expected exception type.
        message: Text expected in the failure.
    """
    service = Binance(tmp_path, progress=False)

    with pytest.raises(error, match=message):
        getattr(service, method)(*arguments, **options)
    assert not (tmp_path / "catalog.duckdb").exists()
