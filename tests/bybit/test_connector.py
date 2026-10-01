"""Test the Bybit connector's archive and metadata routing."""

from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from veldra.bybit.connector import BybitConnector
from veldra.bybit.datasets import get_dataset
from veldra.core.models import (
    ArchiveKey,
    ArchiveObject,
    IntegritySpec,
    Market,
    Resource,
    ResourceKey,
)


def _object(dataset: str = "trades") -> ArchiveObject:
    """Return one representative physical Bybit object."""
    return ArchiveObject(
        ArchiveKey(
            "bybit",
            "spot",
            dataset,
            "portal",
            "instrument",
            "BTCUSDT",
            "daily",
            date(2025, 1, 1),
            date(2025, 1, 1),
            "source.zip",
        ),
        "https://public.bybit.com/source.zip",
        integrity=IntegritySpec("archive_only"),
    )


def _resource() -> Resource:
    """Return one representative archive-only resource."""
    return Resource(
        date(2025, 1, 1),
        "https://public.bybit.com/source.zip",
        None,
        integrity=IntegritySpec("archive_only"),
    )


def test_archive_object_adapter_preserves_book_rollover_coverage() -> None:
    """Expose Bybit's post-midnight book tail to exact-range queries."""
    resource = BybitConnector._resource(_object("order_book_updates"))
    assert resource.coverage_end == datetime(2025, 1, 2, 0, 5, tzinfo=UTC)
    assert resource.archive_symbol == "BTCUSDT"


def test_trade_archive_adapter_uses_the_calendar_day() -> None:
    """Leave ordinary trade coverage at the shared daily default."""
    resource = BybitConnector._resource(_object())
    assert resource.end_day == date(2025, 1, 1)
    assert resource.coverage_end is None


def test_options_use_family_scoped_physical_archives() -> None:
    """Route one Option instrument through its BTC family archive."""
    key = ResourceKey("bybit", "options", "trades", "BTC-27DEC24-100000-C", None)
    subject = BybitConnector._subject(key)
    assert subject.kind == "instrument_family"
    assert subject.value == "BTC"


def test_non_option_archives_keep_the_native_instrument_scope() -> None:
    """Keep Spot and perpetual archive discovery instrument-specific."""
    key = ResourceKey("bybit", "linear", "trades", "BTCUSDT", None)
    subject = BybitConnector._subject(key)
    assert subject.kind == "instrument"
    assert subject.value == "BTCUSDT"


def test_connector_rejects_rest_datasets_and_intervals() -> None:
    """Keep REST histories and interval-less archives on their own routes."""
    connector = BybitConnector()
    with pytest.raises(ValueError, match="REST-backed"):
        connector._validate(
            ResourceKey("bybit", "spot", "klines", "BTCUSDT", "1m"),
            date(2025, 1, 1),
            date(2025, 1, 1),
        )
    with pytest.raises(ValueError, match="no interval"):
        connector._validate(
            ResourceKey("bybit", "spot", "trades", "BTCUSDT", "1m"),
            date(2025, 1, 1),
            date(2025, 1, 1),
        )


@pytest.mark.parametrize(
    ("key", "start_day", "message"),
    [
        (ResourceKey("other", "spot", "trades", "BTCUSDT", None), None, "source"),
        (
            ResourceKey("bybit", "unknown", "trades", "BTCUSDT", None),
            None,
            "product",
        ),
        (
            ResourceKey("bybit", "spot", "trades", "BTCUSDT", None),
            date(2025, 1, 2),
            "begins after",
        ),
    ],
)
def test_connector_rejects_invalid_archive_keys(
    key: ResourceKey, start_day: date | None, message: str
) -> None:
    """Reject foreign, unsupported, and reversed archive requests."""
    with pytest.raises(ValueError, match=message):
        BybitConnector()._validate(key, start_day, date(2025, 1, 1))


def test_checksum_requires_a_source_revision() -> None:
    """Reject archive metadata without a usable ETag."""
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client:
        with pytest.raises(ValueError, match="ETag"):
            BybitConnector().checksum(client, _resource())


def test_checksum_returns_a_normalized_etag() -> None:
    """Use a published ETag as an opaque refresh revision."""
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"ETag": '"revision"'})
        )
    ) as client:
        assert BybitConnector().checksum(client, _resource()) == "revision"


def test_market_onboard_time_narrows_first_resource_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start discovery no earlier than the selected market launch date."""
    market = Market(
        "BTCUSDT",
        "BTCUSDT",
        onboard_time=datetime(2024, 1, 2, tzinfo=UTC),
        source="bybit",
        product="spot",
        active=True,
    )
    monkeypatch.setattr(
        "veldra.bybit.connector.markets", lambda client, product: [market]
    )
    connector = BybitConnector()
    with httpx.Client() as client:
        assert connector.markets(client, "spot") == [market]
    key = ResourceKey("bybit", "spot", "trades", "BTCUSDT", None)
    assert connector._source_start(key) == date(2024, 1, 2)


def test_quote_volumes_route_through_the_public_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose native quote turnover without changing its keys."""
    monkeypatch.setattr(
        "veldra.bybit.connector.quote_volumes",
        lambda client, product: {"BTCUSDT": 12.5},
    )
    with httpx.Client() as client:
        assert BybitConnector().quote_volumes(client, "spot") == {"BTCUSDT": 12.5}


@pytest.mark.parametrize(
    ("dataset", "discovery_name"),
    [
        ("trades", "BybitTradeDiscovery"),
        ("order_book_updates", "BybitOrderBookDiscovery"),
    ],
)
def test_resources_route_to_the_declared_discovery(
    monkeypatch: pytest.MonkeyPatch, dataset: str, discovery_name: str
) -> None:
    """Adapt physical objects from each supported archive manifest."""
    discovered = _object(dataset)
    probe = SimpleNamespace(discover=lambda *args: [discovered])
    monkeypatch.setattr(
        f"veldra.bybit.connector.{discovery_name}", lambda client: probe
    )
    key = ResourceKey("bybit", "spot", dataset, "BTCUSDT", None)
    with httpx.Client() as client:
        resources = BybitConnector().resources(
            client, key, date(2025, 1, 1), date(2025, 1, 1)
        )
    assert [resource.url for resource in resources] == [discovered.url]


def test_first_resource_skips_empty_monthly_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advance bounded scans until the first available daily object."""
    connector = BybitConnector()
    found = BybitConnector._resource(_object())
    calls: list[tuple[date, date]] = []

    def resources(
        _client: httpx.Client, _key: ResourceKey, start: date, end: date
    ) -> list[Resource]:
        calls.append((start, end))
        return [] if len(calls) == 1 else [found]

    monkeypatch.setattr(connector, "resources", resources)
    key = ResourceKey("bybit", "spot", "trades", "BTCUSDT", None)
    with httpx.Client() as client:
        result = connector.first_resource(
            client, key, date(2025, 1, 1), date(2025, 3, 5)
        )
    assert result is found
    assert calls == [
        (date(2025, 1, 1), date(2025, 1, 31)),
        (date(2025, 2, 1), date(2025, 3, 3)),
    ]


def test_first_resource_returns_none_after_the_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return no resource when every bounded manifest probe is empty."""
    connector = BybitConnector()
    monkeypatch.setattr(connector, "resources", lambda *args: [])
    key = ResourceKey("bybit", "options", "trades", "BTC-27DEC24-1-C", None)
    with httpx.Client() as client:
        assert (
            connector.first_resource(client, key, date(2021, 1, 1), date(2022, 1, 2))
            is None
        )


@pytest.mark.parametrize(
    ("dataset_name", "ingestor_name"),
    [("trades", "ingest_trades"), ("order_book_updates", "ingest_order_book")],
)
def test_ingest_routes_supported_archive_schemas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_name: str,
    ingestor_name: str,
) -> None:
    """Route archive formats to their dedicated streaming normalizers."""
    expected = object()
    monkeypatch.setattr(
        f"veldra.bybit.connector.{ingestor_name}", lambda *args, **kwargs: expected
    )
    with httpx.Client() as client:
        result = BybitConnector().ingest(
            client,
            _resource(),
            get_dataset("spot", dataset_name),
            tmp_path / "data.parquet",
        )
    assert result is expected


def test_ingest_rejects_a_rest_dataset(tmp_path: Path) -> None:
    """Fail before network access when ingestion receives a REST schema."""
    with httpx.Client() as client:
        with pytest.raises(ValueError, match="unsupported"):
            BybitConnector().ingest(
                client,
                _resource(),
                get_dataset("spot", "klines", requested_interval="1m"),
                tmp_path / "data.parquet",
            )
