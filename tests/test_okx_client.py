"""Test the rate-limited OKX public HTTP client."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Lock
from time import monotonic

import httpx
import pytest

from veldra.okx.client import (
    DEFAULT_POLICIES,
    OKXClient,
    OKXRateLimiter,
    OKXResponseError,
    RatePolicy,
)


def client_for(handler: httpx.MockTransport) -> tuple[OKXClient, httpx.Client]:
    """Build an OKX client over one fixture transport.

    Args:
        handler: The HTTPX fixture transport.

    Returns:
        The OKX wrapper and its underlying client.
    """
    raw = httpx.Client(transport=handler)
    return OKXClient(client=raw, backoff=0, jitter=lambda _limit: 0), raw


def test_rate_policy_rejects_invalid_values() -> None:
    """Confirm unusable limiter capacities and windows are rejected."""
    with pytest.raises(ValueError, match="capacity"):
        RatePolicy(0, 2)
    with pytest.raises(ValueError, match="window"):
        RatePolicy(1, 0)
    with pytest.raises(ValueError, match="safety"):
        RatePolicy(1, 2, -1)


def test_limiter_rejects_unknown_keys_and_costs() -> None:
    """Confirm limiter calls cannot bypass declared policies."""
    limiter = OKXRateLimiter({"test": RatePolicy(1, 0.01, 0)})
    with pytest.raises(KeyError, match="policy"):
        limiter.acquire("missing")
    with pytest.raises(ValueError, match="cost"):
        limiter.acquire("test", cost=0)
    with pytest.raises(ValueError, match="capacity"):
        limiter.acquire("test", cost=2)
    with pytest.raises(ValueError, match="retry_after"):
        limiter.penalize("test", -1)


def test_limiter_serializes_a_concurrent_rolling_window() -> None:
    """Confirm concurrent callers share one rolling-window reservation deque."""
    limiter = OKXRateLimiter({"test": RatePolicy(2, 0.04, 0)})
    lock = Lock()
    completed: list[float] = []

    def acquire() -> None:
        """Reserve one request and record its completion time."""
        limiter.acquire("test")
        with lock:
            completed.append(monotonic())

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _value: acquire(), range(4)))

    completed.sort()
    assert completed[2] - completed[0] >= 0.03


def test_client_reads_instruments_and_tickers() -> None:
    """Confirm standard public envelopes and endpoint parameters are returned."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one typed fixture row for each public endpoint."""
        requests.append(request)
        return httpx.Response(
            200, json={"code": "0", "msg": "", "data": [{"instId": "BTC-USDT"}]}
        )

    api, raw = client_for(httpx.MockTransport(handler))
    with raw:
        assert api.get_instruments("SPOT") == [{"instId": "BTC-USDT"}]
        assert api.get_tickers("SPOT") == [{"instId": "BTC-USDT"}]

    assert [request.url.path for request in requests] == [
        "/api/v5/public/instruments",
        "/api/v5/market/tickers",
    ]
    assert all(request.url.params["instType"] == "SPOT" for request in requests)


@pytest.mark.parametrize("inst_type", ["", "spot", "EVENTS", 1])
def test_client_rejects_invalid_instrument_types(inst_type: object) -> None:
    """Confirm unsupported instrument types fail before network I/O.

    Args:
        inst_type: The invalid public instrument type.
    """
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        """Count an unexpected request."""
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    api, raw = client_for(httpx.MockTransport(handler))
    with raw, pytest.raises((TypeError, ValueError), match="inst_type"):
        api.get_instruments(inst_type)  # type: ignore[arg-type]
    assert calls == 0


def test_manifest_builds_a_bounded_specific_request() -> None:
    """Confirm Spot manifests use instrument IDs and inclusive millisecond dates."""
    request_seen: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture and satisfy one manifest request."""
        nonlocal request_seen
        request_seen = request
        return httpx.Response(
            200,
            json={
                "code": "0",
                "msg": "",
                "data": [{"dateAggrType": "daily", "details": []}],
            },
        )

    api, raw = client_for(httpx.MockTransport(handler))
    with raw:
        rows = api.get_manifest(
            module=2,
            inst_type="SPOT",
            subjects=["BTC-USDT", "ETH-USDT"],
            cadence="daily",
            begin=date(2025, 1, 1),
            end=date(2025, 1, 10),
        )

    assert rows == [{"dateAggrType": "daily", "details": []}]
    assert request_seen is not None
    assert request_seen.url.params["instIdList"] == "BTC-USDT,ETH-USDT"
    assert "instFamilyList" not in request_seen.url.params
    assert request_seen.url.params["begin"] == "1735689600000"
    assert request_seen.url.params["end"] == "1736467200000"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"module": 9}, "module"),
        ({"subjects": []}, "subject"),
        ({"subjects": ["A", "B", "C", "D", "E", "F"]}, "five"),
        ({"cadence": "weekly"}, "cadence"),
        ({"begin": date(2025, 1, 2), "end": date(2025, 1, 1)}, "begin"),
        ({"end": date(2025, 1, 11)}, "ten days"),
        ({"cadence": "monthly", "end": date(2025, 11, 1)}, "ten months"),
        ({"subjects": ["ANY", "BTC-USDT"]}, "ANY"),
        ({"module": 3, "cadence": "daily", "subjects": ["BTC-USDT"]}, "funding"),
    ],
)
def test_manifest_rejects_invalid_requests_before_io(
    changes: dict[str, object], message: str
) -> None:
    """Confirm manifest API limits are locally enforced.

    Args:
        changes: Values overriding an otherwise valid request.
        message: Text expected in the validation error.
    """
    values: dict[str, object] = {
        "module": 2,
        "inst_type": "SPOT",
        "subjects": ["BTC-USDT"],
        "cadence": "daily",
        "begin": date(2025, 1, 1),
        "end": date(2025, 1, 10),
    }
    values.update(changes)
    api, raw = client_for(
        httpx.MockTransport(lambda _request: pytest.fail("unexpected HTTP request"))
    )
    with raw, pytest.raises((TypeError, ValueError), match=message):
        api.get_manifest(**values)  # type: ignore[arg-type]


def test_manifest_uses_families_for_derivatives() -> None:
    """Confirm derivatives send family subjects instead of Spot IDs."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture one successful manifest request."""
        requests.append(request)
        return httpx.Response(200, json={"code": "0", "msg": "", "data": []})

    api, raw = client_for(httpx.MockTransport(handler))
    with raw:
        assert (
            api.get_manifest(
                1, "SWAP", ["BTC-USDT"], "monthly", date(2025, 1, 1), date(2025, 10, 1)
            )
            == []
        )
    assert requests[0].url.params["instFamilyList"] == "BTC-USDT"


def test_client_raises_semantic_errors_from_http_200() -> None:
    """Confirm OKX error envelopes are not mistaken for successful empty data."""
    api, raw = client_for(
        httpx.MockTransport(
            lambda _request: httpx.Response(
                200, json={"code": "51000", "msg": "Parameter error", "data": []}
            )
        )
    )
    with raw, pytest.raises(OKXResponseError) as raised:
        api.get_instruments("SPOT")
    assert raised.value.code == "51000"
    assert not raised.value.retryable


def test_client_accepts_an_empty_success_envelope() -> None:
    """Confirm no-file responses remain ordinary successful empty results."""
    api, raw = client_for(
        httpx.MockTransport(
            lambda _request: httpx.Response(
                200, json={"code": "0", "msg": "", "data": []}
            )
        )
    )
    with raw:
        assert api.get_instruments("SPOT") == []


@pytest.mark.parametrize("failure", [429, 503, "50011", "transport"])
def test_client_retries_transient_failures(failure: object) -> None:
    """Confirm transport, HTTP, and semantic throttling failures are retried.

    Args:
        failure: The transient failure returned on the first attempt.
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Fail once and then return a successful envelope."""
        nonlocal calls
        calls += 1
        if calls > 1:
            return httpx.Response(200, json={"code": "0", "msg": "", "data": []})
        if failure == "transport":
            raise httpx.ReadError("temporary", request=request)
        if failure == "50011":
            return httpx.Response(
                200, json={"code": "50011", "msg": "busy", "data": []}
            )
        assert isinstance(failure, int)
        return httpx.Response(failure, headers={"Retry-After": "0"})

    api, raw = client_for(httpx.MockTransport(handler))
    with raw:
        assert api.get_instruments("SPOT") == []
    assert calls == 2


def test_client_honors_retry_after_and_penalizes_the_bucket() -> None:
    """Confirm throttling uses Retry-After for both sleep and limiter penalty."""
    delays: list[float] = []
    penalties: list[tuple[str, float]] = []
    calls = 0

    class Limiter:
        """Record client limiter operations without waiting."""

        def acquire(self, key: str, *, cost: int = 1) -> None:
            """Accept and record-free one request reservation."""
            assert cost == 1

        def penalize(self, key: str, retry_after: float) -> None:
            """Record one temporary endpoint penalty."""
            penalties.append((key, retry_after))

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return one throttled response before succeeding."""
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200, json={"code": "0", "msg": "", "data": []})

    raw = httpx.Client(transport=httpx.MockTransport(handler))
    api = OKXClient(
        client=raw,
        limiter=Limiter(),
        sleeper=delays.append,
        jitter=lambda _limit: 0,
    )
    with raw:
        api.get_instruments("SPOT")
    assert delays == [0.25]
    assert penalties == [("instruments:SPOT", 0.25)]


def test_client_rejects_malformed_envelopes() -> None:
    """Confirm non-JSON and structurally invalid responses fail clearly."""
    responses = iter(
        [
            httpx.Response(200, text="not json"),
            httpx.Response(200, json={"code": "0", "data": {}}),
        ]
    )
    api, raw = client_for(httpx.MockTransport(lambda _request: next(responses)))
    with raw:
        with pytest.raises(OKXResponseError, match="JSON"):
            api.get_instruments("SPOT")
        with pytest.raises(OKXResponseError, match="data"):
            api.get_instruments("SPOT")


def test_paginate_stops_on_short_page_and_detects_cursor_loops() -> None:
    """Confirm generic page traversal is bounded and loop-safe."""
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return the same full page to trigger cursor-loop detection."""
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "code": "0",
                "msg": "",
                "data": [{"ts": "2"}, {"ts": "1"}],
            },
        )

    api, raw = client_for(httpx.MockTransport(handler))
    with raw, pytest.raises(OKXResponseError, match="cursor"):
        api.paginate(
            "/api/v5/market/history-candles",
            policy_key="history_candles",
            params={"instId": "BTC-USDT"},
            cursor=lambda rows: str(rows[-1]["ts"]),  # type: ignore[index]
            page_size=2,
            max_pages=3,
        )
    assert calls == 2


def test_default_policy_registry_contains_every_planned_endpoint() -> None:
    """Confirm quota declarations cover each historical REST family."""
    assert {
        "manifest",
        "history_candles",
        "history_index_candles",
        "history_mark_candles",
        "premium_history",
    } <= DEFAULT_POLICIES.keys()
