"""Test Bybit daily trade archive discovery."""

from datetime import date

import httpx
import pytest

from veldra.bybit.client import BybitClient
from veldra.bybit.manifest import BybitTradeDiscovery, manifest_windows
from veldra.core.subjects import DataSubject


class Limiter:
    """Provide no-op request limits for mock discovery."""

    def acquire(self, key: str, *, cost: int = 1) -> None:
        """Accept one request reservation."""
        del key, cost

    def penalize(self, key: str, retry_after: float) -> None:
        """Accept one request penalty."""
        del key, retry_after


def client(handler: httpx.MockTransport) -> tuple[httpx.Client, BybitClient]:
    """Return externally managed HTTPX and Bybit clients."""
    http = httpx.Client(transport=handler)
    return http, BybitClient(client=http, limiter=Limiter(), retries=0)


def test_manifest_windows_use_eight_inclusive_dates() -> None:
    """Pack long portal requests at the observed source maximum."""
    assert manifest_windows(date(2026, 1, 1), date(2026, 1, 18)) == [
        (date(2026, 1, 1), date(2026, 1, 8)),
        (date(2026, 1, 9), date(2026, 1, 16)),
        (date(2026, 1, 17), date(2026, 1, 18)),
    ]
    with pytest.raises(ValueError, match="must not follow"):
        manifest_windows(date(2026, 1, 2), date(2026, 1, 1))


@pytest.mark.parametrize(
    ("product", "path"),
    [
        ("spot", "/spot/BTCUSDT/BTCUSDT_2025-01-01.csv.gz"),
        ("linear", "/trading/BTCUSDT/BTCUSDT2025-01-01.csv.gz"),
        ("inverse", "/trading/BTCUSDT/BTCUSDT2025-01-01.csv.gz"),
    ],
)
def test_direct_trade_archives_use_deterministic_daily_urls(
    product: str, path: str
) -> None:
    """Probe Spot and derivative public roots without listing pages."""

    def response(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"
        assert request.url.path == path
        return httpx.Response(
            200, headers={"Content-Length": "123", "ETag": '"revision"'}
        )

    http, api = client(httpx.MockTransport(response))
    with http:
        found = BybitTradeDiscovery(api).discover(
            product,
            DataSubject("instrument", "BTCUSDT"),
            date(2025, 1, 1),
            date(2025, 1, 1),
        )
    assert len(found) == 1
    assert found[0].key.period_start == date(2025, 1, 1)
    assert found[0].key.remote_scope_kind == "instrument"
    assert found[0].remote_size == 123
    assert found[0].revision_id == "revision"
    assert found[0].integrity is not None
    assert found[0].integrity.mode == "archive_only"


def test_absent_direct_trade_days_are_omitted() -> None:
    """Treat public-root 404 responses as unavailable source dates."""
    http, api = client(
        httpx.MockTransport(lambda request: httpx.Response(404, request=request))
    )
    with http:
        found = BybitTradeDiscovery(api).discover(
            "spot",
            DataSubject("instrument", "BTCUSDT"),
            date(2021, 1, 1),
            date(2021, 1, 1),
        )
    assert found == []


def test_option_trade_manifest_preserves_family_scope() -> None:
    """Model one shared Option archive rather than copying it per contract."""
    requests: list[httpx.Request] = []

    def response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        row = {
            "bizType": "option",
            "productId": "trade",
            "interval": "daily",
            "symbol": "BTC",
            "date": "2026-09-20",
            "filename": "2026-09-20_BTC_USDT.trades.csv.zip",
            "size": "2082833",
            "url": (
                "https://public.bybit.com/trade/option/BTC/"
                "2026-09-20_BTC_USDT.trades.csv.zip"
            ),
        }
        return httpx.Response(
            200,
            json={"ret_code": 0, "ret_msg": "", "result": {"list": [row]}},
        )

    http, api = client(httpx.MockTransport(response))
    with http:
        found = BybitTradeDiscovery(api).discover(
            "options",
            DataSubject("instrument_family", "BTC"),
            date(2026, 9, 20),
            date(2026, 9, 20),
        )
    assert len(found) == 1
    assert found[0].key.remote_scope_kind == "instrument_family"
    assert found[0].key.remote_scope_value == "BTC"
    assert found[0].remote_size == 2_082_833
    assert requests[0].headers["referer"] == "https://www.bybit.com/data-download"


@pytest.mark.parametrize(
    ("product", "subject"),
    [
        ("options", DataSubject("instrument", "BTCUSDT")),
        ("spot", DataSubject("instrument_family", "BTC")),
    ],
)
def test_archive_subject_kinds_match_physical_scope(
    product: str, subject: DataSubject
) -> None:
    """Reject logical identities that would duplicate shared archives."""
    http, api = client(httpx.MockTransport(lambda request: httpx.Response(404)))
    with http, pytest.raises(ValueError):
        BybitTradeDiscovery(api).discover(
            product, subject, date(2026, 1, 1), date(2026, 1, 1)
        )


@pytest.mark.parametrize(
    "change",
    [
        {"bizType": "spot"},
        {"symbol": "ETH"},
        {"date": "2026-09-19"},
        {"size": "many"},
        {"url": "https://example.com/archive.zip"},
        {"filename": "../archive.zip"},
    ],
)
def test_malformed_manifest_rows_fail_closed(change: dict[str, object]) -> None:
    """Reject portal rows that escape the requested scope or source host."""
    row: dict[str, object] = {
        "bizType": "option",
        "productId": "trade",
        "interval": "daily",
        "symbol": "BTC",
        "date": "2026-09-20",
        "filename": "2026-09-20_BTC_USDT.trades.csv.zip",
        "size": "1",
        "url": (
            "https://public.bybit.com/trade/option/BTC/"
            "2026-09-20_BTC_USDT.trades.csv.zip"
        ),
    }
    row.update(change)
    payload = {"ret_code": 0, "result": {"list": [row]}}
    http, api = client(
        httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )
    with http, pytest.raises(ValueError):
        BybitTradeDiscovery(api).discover(
            "options",
            DataSubject("instrument_family", "BTC"),
            date(2026, 9, 20),
            date(2026, 9, 20),
        )
