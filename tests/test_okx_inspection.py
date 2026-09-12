"""Test the public OKX market, contract, cache, and coverage tools."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from veldra import OKX
from veldra.core.catalog import catalog_lock, open_catalog
from veldra.core.models import (
    ArchiveKey,
    ArchiveObject,
    Availability,
    IntegritySpec,
    LogicalPartition,
    Market,
    Materialization,
    ResourceKey,
)
from veldra.core.subjects import DataSubject
from veldra.okx.reports import CacheReport


def instrument(
    symbol: str,
    *,
    kind: str = "SPOT",
    family: str = "",
    contract_type: str = "",
    state: str = "live",
    expiry: str = "",
) -> dict[str, object]:
    """Return one valid public instrument response row.

    Args:
        symbol: Native OKX instrument ID.
        kind: Native instrument type.
        family: Optional derivative family.
        contract_type: Optional linear or inverse contract type.
        state: Native current state.
        expiry: Optional epoch-millisecond expiry.

    Returns:
        A source response object accepted by the OKX identity parser.
    """
    option = kind == "OPTION"
    return {
        "instType": kind,
        "instId": symbol,
        "instFamily": family,
        "baseCcy": symbol.split("-")[0],
        "quoteCcy": "USDT" if kind in {"SPOT", "MARGIN"} else "",
        "settleCcy": "USD" if "-USD" in family else "USDT",
        "ctType": contract_type,
        "ctVal": "1" if contract_type else "",
        "ctMult": "1" if contract_type else "",
        "ctValCcy": "BTC" if contract_type else "",
        "state": state,
        "ruleType": "normal",
        "listTime": "1609459200000",
        "expTime": expiry,
        "stk": symbol.split("-")[-2] if option else "",
        "optType": symbol[-1] if option else "",
    }


class InspectionFixture:
    """Serve deterministic current instruments, volumes, and manifests."""

    def __init__(self) -> None:
        """Create counters for cache and refresh assertions."""
        self.instruments = 0
        self.manifests = 0
        self.tickers = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return one response for each public inspection request.

        Args:
            request: The HTTP request emitted by the facade.

        Returns:
            A valid OKX response envelope.
        """
        path = request.url.path
        if path.endswith("/instruments"):
            self.instruments += 1
            native = request.url.params.get("instType")
            rows = {
                "SPOT": [
                    instrument("BTC-USDT"),
                    instrument("ETH-USDT", state="suspend"),
                ],
                "FUTURES": [
                    instrument(
                        "BTC-USDT-250328",
                        kind="FUTURES",
                        family="BTC-USDT",
                        contract_type="linear",
                        expiry="1743120000000",
                    ),
                    instrument(
                        "BTC-USD-250328",
                        kind="FUTURES",
                        family="BTC-USD",
                        contract_type="inverse",
                        expiry="1743120000000",
                    ),
                ],
            }.get(native, [])
            return httpx.Response(
                200,
                json={"code": "0", "msg": "", "data": rows},
                request=request,
            )
        if path.endswith("/tickers"):
            self.tickers += 1
            return httpx.Response(
                200,
                json={
                    "code": "0",
                    "msg": "",
                    "data": [
                        {"instId": "BTC-USDT", "volCcy24h": "100"},
                        {"instId": "ETH-USDT", "volCcy24h": "200"},
                    ],
                },
                request=request,
            )
        if path.endswith("/market-data-history"):
            self.manifests += 1
            subject = (
                request.url.params.get("instIdList")
                or request.url.params.get("instFamilyList")
                or request.url.params.get("ccyList")
                or "ANY"
            )
            filename = f"{subject}-candlesticks-2025-01-01.zip"
            group = {
                "instId": subject,
                "instFamily": subject,
                "ccy": subject,
                "groupDetails": [
                    {
                        "dataTs": "1735689600000",
                        "filename": filename,
                        "sizeMB": "0.01",
                        "url": f"https://files.test/{filename}",
                    }
                ],
            }
            return httpx.Response(
                200,
                json={
                    "code": "0",
                    "msg": "",
                    "data": [{"dateAggrType": "daily", "details": [group]}],
                },
                request=request,
            )
        raise AssertionError(f"unexpected inspection request {request.url}")


def service(tmp_path: Path, fixture: InspectionFixture) -> OKX:
    """Return an isolated facade backed by deterministic source responses.

    Args:
        tmp_path: Isolated local data root.
        fixture: Source response fixture.

    Returns:
        Configured OKX facade.
    """
    return OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )


def test_market_listing_filters_sorts_and_reuses_cached_metadata(
    tmp_path: Path,
) -> None:
    """Confirm product filters and native rolling volume ordering."""
    fixture = InspectionFixture()
    okx = service(tmp_path, fixture)

    values = okx.get_markets(
        product="spot", sort_by="quote_volume", active=True, limit=1
    )

    assert [item.symbol for item in values] == ["BTC-USDT"]
    assert values[0].source == "okx" and values[0].product == "spot"
    assert values[0].quote_volume_24h == 100
    assert okx.get_markets(product="spot", offline=True)[0].symbol == "BTC-USDT"
    assert fixture.instruments == 1
    assert fixture.tickers == 1


def test_market_search_is_product_scoped_and_validates_options(tmp_path: Path) -> None:
    """Confirm fuzzy searches never silently replace caller input."""
    fixture = InspectionFixture()
    okx = service(tmp_path, fixture)

    assert [item.symbol for item in okx.find_markets("BTCSUDT", product="spot")] == [
        "BTC-USDT"
    ]
    assert [item.symbol for item in okx.find_markets("eth", offline=True)] == [
        "ETH-USDT"
    ]
    with pytest.raises(ValueError, match="unsupported OKX product"):
        okx.get_markets(product="fake")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit"):
        okx.find_markets("btc", product="spot", limit=0)
    with pytest.raises(TypeError, match="active"):
        okx.get_markets(product="spot", active=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refresh and offline"):
        okx.get_markets(product="spot", refresh=True, offline=True)
    with pytest.raises(RuntimeError, match="volume sorting"):
        other = service(tmp_path / "other", InspectionFixture())
        other.get_markets(product="spot")
        other.get_markets(product="spot", sort_by="quote_volume", offline=True)


def test_futures_contract_filters_return_only_the_requested_family(
    tmp_path: Path,
) -> None:
    """Confirm linear and inverse contracts remain separate products."""
    fixture = InspectionFixture()
    okx = service(tmp_path, fixture)

    linear = okx.get_contracts(product="linear_futures", family="btc-usdt", active=True)
    inverse = okx.get_contracts(product="inverse_futures", family="BTC-USD")

    assert [item.symbol for item in linear] == ["BTC-USDT-250328"]
    assert [item.symbol for item in inverse] == ["BTC-USD-250328"]
    with pytest.raises(ValueError, match="dated Futures"):
        okx.get_contracts(product="spot", family="BTC-USDT")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contract style"):
        okx.get_contracts(
            product="linear_futures",
            family="BTC-USDT",
            contract_style="dated",  # type: ignore[arg-type]
        )


def test_option_contract_filters_parse_native_contract_terms(
    tmp_path: Path,
) -> None:
    """Confirm Option expiry, strike, and type filters combine safely."""
    fixture = InspectionFixture()
    option_rows = [
        instrument(
            "BTC-USD-251226-90000-C",
            kind="OPTION",
            family="BTC-USD",
            contract_type="inverse",
            expiry="1766707200000",
        ),
        instrument(
            "BTC-USD-251226-90000-P",
            kind="OPTION",
            family="BTC-USD",
            contract_type="inverse",
            expiry="1766707200000",
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an Option family and its current contracts."""
        if request.url.path.endswith("/underlying"):
            return httpx.Response(
                200,
                json={"code": "0", "msg": "", "data": [["BTC-USD"]]},
                request=request,
            )
        return httpx.Response(
            200,
            json={"code": "0", "msg": "", "data": option_rows},
            request=request,
        )

    okx = OKX(
        tmp_path,
        retries=0,
        progress=False,
        transport=httpx.MockTransport(handler),
    )
    values = okx.get_option_contracts(
        family="BTC-USD", expiry="2025-12-26", option_type="put", strike_min=80_000
    )
    assert [item.symbol for item in values] == ["BTC-USD-251226-90000-P"]
    with pytest.raises(ValueError, match="strike_min"):
        okx.get_option_contracts(
            family="BTC-USD", strike_min=100_000, strike_max=90_000
        )
    assert (
        okx.get_option_contracts(
            family="ETH-USD", expiry="2025-12-26", option_type="call"
        )
        == []
    )
    assert (
        okx.get_option_contracts(
            family="BTC-USD", expiry="2025-12-27", strike_max=80_000
        )
        == []
    )
    with pytest.raises(ValueError, match="option_type"):
        okx.get_option_contracts(
            family="BTC-USD", option_type="both"  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="strike_max"):
        okx.get_option_contracts(
            family="BTC-USD", strike_max=True  # type: ignore[arg-type]
        )


def test_generic_cache_method_delegates_one_explicit_bulk_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm loops can cache a dataset without selecting a wrapper."""
    fixture = InspectionFixture()
    okx = service(tmp_path, fixture)
    expected = CacheReport(
        "okx",
        "spot",
        "klines",
        (datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 2, tzinfo=UTC)),
        "bulk",
        "test",
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        True,
        False,
    )
    calls: list[dict[str, object]] = []

    def cache_all(*args: object, **kwargs: object) -> CacheReport:
        """Record one generic bulk delegation."""
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(okx._service, "cache_all", cache_all)
    result = okx.cache_dataset(
        "2025-01-01", "2025-01-01", product="spot", dataset="klines"
    )
    assert result is expected
    assert calls == [
        {
            "product": "spot",
            "dataset": "klines",
            "dry_run": False,
            "refresh": False,
            "offline": False,
        }
    ]


def test_discovery_reports_remote_and_local_coverage_without_downloading(
    tmp_path: Path,
) -> None:
    """Confirm bounded inspection catalogs manifests but not source payloads."""
    fixture = InspectionFixture()
    okx = service(tmp_path, fixture)

    value = okx.discover_availability(
        "BTC-USDT",
        "2025-01-01",
        "2025-01-02",
        product="spot",
        dataset="klines",
        interval="1h",
    )
    cached = okx.get_availability(
        "BTC-USDT", product="spot", dataset="klines", interval="1h"
    )

    assert value == cached
    assert isinstance(value, Availability)
    assert value.remote_range == (date(2025, 1, 1), date(2025, 1, 1))
    assert value.scanned_ranges == ((date(2025, 1, 1), date(2025, 1, 3)),)
    assert value.scanned_days == 3
    assert value.available_days == 1
    assert value.unavailable_days == 2
    assert value.cached_days == value.row_count == value.local_bytes == 0
    assert fixture.manifests == 1


def test_local_availability_requires_metadata_and_unknowns_offer_suggestions(
    tmp_path: Path,
) -> None:
    """Confirm coverage failures are explicit and product-scoped."""
    fixture = InspectionFixture()
    okx = service(tmp_path, fixture)
    with pytest.raises(RuntimeError, match="cached OKX market metadata"):
        okx.get_availability(
            "BTC-USDT", product="spot", dataset="klines", interval="1m"
        )
    okx.get_markets(product="spot")
    with pytest.raises(ValueError, match="Suggestions: BTC-USDT"):
        okx.get_availability("BTCSUDT", product="spot", dataset="klines", interval="1m")


def test_local_availability_counts_ready_partitions_and_physical_bytes(
    tmp_path: Path,
) -> None:
    """Confirm ready shared files become exact logical cache coverage."""
    fixture = InspectionFixture()
    okx = service(tmp_path, fixture)
    okx.get_markets(product="spot")
    path = tmp_path / "ready.parquet"
    path.write_bytes(b"parquet fixture")
    key = ArchiveKey(
        "okx",
        "spot",
        "klines",
        "module_2",
        "all",
        "ANY",
        "daily",
        date(2025, 1, 1),
        date(2025, 1, 1),
        "ANY-candlesticks-2025-01-01.zip",
    )
    archive = ArchiveObject(
        key,
        "https://files.test/ready.zip",
        integrity=IntegritySpec("response_header", algorithm="md5"),
    )
    start = datetime(2024, 12, 31, 16, tzinfo=UTC)
    end = datetime(2025, 1, 1, 16, tzinfo=UTC)
    materialization = Materialization(
        key,
        path,
        1,
        2,
        start,
        end,
        path.stat().st_size,
        local_mtime_ns=path.stat().st_mtime_ns,
    )
    partition = LogicalPartition(
        "okx",
        "spot",
        "klines",
        DataSubject("instrument", "BTC-USDT"),
        "1m",
        start,
        end,
        path,
        "instrument_id",
        "BTC-USDT",
        2,
        source_day=date(2025, 1, 1),
    )
    with catalog_lock(tmp_path / "catalog.duckdb"):
        with open_catalog(tmp_path / "catalog.duckdb") as catalog:
            catalog.save_archives([archive])
            catalog.publish_materialization(materialization, [partition])
            catalog.save_discovery(
                ResourceKey("okx", "spot", "klines", "BTC-USDT", "1m"),
                date(2025, 1, 1),
                date(2025, 1, 1),
                [],
            )

    value = okx.get_availability(
        "BTC-USDT", product="spot", dataset="klines", interval="1m"
    )
    assert value.available_days == value.cached_days == 1
    assert value.missing_days == value.failed_days == 0
    assert value.cached_range == (date(2025, 1, 1), date(2025, 1, 1))
    assert value.row_count == 2
    assert value.local_bytes == len(b"parquet fixture")


def test_currency_and_historical_contract_coverage_resolve_native_subjects(
    tmp_path: Path,
) -> None:
    """Confirm coverage accepts currencies and expired derivative IDs."""
    fixture = InspectionFixture()
    okx = service(tmp_path, fixture)
    borrow = okx.discover_availability(
        "usdt",
        "2025-01-01",
        "2025-01-01",
        product="margin",
        dataset="borrow_rates",
    )
    future = okx.discover_availability(
        "BTC-USDT-241227",
        "2025-01-01",
        "2025-01-01",
        product="linear_futures",
        dataset="klines",
    )
    assert borrow.symbol == "USDT"
    assert future.symbol == "BTC-USDT-241227"
    assert fixture.manifests == 2


@pytest.mark.parametrize(
    ("method", "kwargs", "message"),
    [
        ("discover_availability", {"refresh": "yes"}, "refresh"),
        ("get_markets", {"sort_by": "rank"}, "sort_by"),
        ("find_markets", {"query": "---"}, "query"),
        ("find_markets", {"query": 3}, "query"),
        ("find_markets", {"query": "btc", "limit": True}, "limit"),
        ("get_markets", {"status": 3}, "status"),
        ("get_markets", {"quote_asset": " "}, "quote_asset"),
        ("get_markets", {"offline": "yes"}, "offline"),
    ],
)
def test_inspection_rejects_invalid_scalar_options(
    tmp_path: Path,
    method: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Confirm malformed public inspection values fail before network access.

    Args:
        tmp_path: Isolated local data root.
        method: Public method to call.
        kwargs: Invalid keyword arguments.
        message: Expected failure text.
    """
    okx = OKX(tmp_path, progress=False)
    if method == "discover_availability":
        with pytest.raises((TypeError, ValueError), match=message):
            okx.discover_availability(  # type: ignore[arg-type]
                "BTC-USDT",
                "2025-01-01",
                "2025-01-01",
                product="spot",
                dataset="klines",
                **kwargs,
            )
    elif method == "find_markets":
        with pytest.raises((TypeError, ValueError), match=message):
            okx.find_markets(product="spot", **kwargs)  # type: ignore[arg-type]
    else:
        with pytest.raises((TypeError, ValueError), match=message):
            okx.get_markets(product="spot", **kwargs)  # type: ignore[arg-type]
