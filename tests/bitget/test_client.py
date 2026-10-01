"""Test Bitget's bounded public HTTP client."""

import json

import httpx
import pytest

from veldra.bitget.client import BitgetClient, BitgetResponseError


class Limiter:
    """Record client quota reservations and penalties."""

    def __init__(self) -> None:
        """Create empty call logs."""
        self.acquired: list[str] = []
        self.penalties: list[tuple[str, float]] = []

    def acquire(self, key: str, *, cost: int = 1) -> None:
        """Record one reservation."""
        assert cost == 1
        self.acquired.append(key)

    def penalize(self, key: str, retry_after: float) -> None:
        """Record one server penalty."""
        self.penalties.append((key, retry_after))


def mock_client(handler: httpx.MockTransport) -> httpx.Client:
    """Return one HTTPX client backed by a mock transport."""
    return httpx.Client(transport=handler)


def test_get_instruments_uses_native_category_and_policy() -> None:
    """Send market metadata through the instruments quota family."""
    limiter = Limiter()

    def response(request: httpx.Request) -> httpx.Response:
        assert request.url.params["category"] == "SPOT"
        return httpx.Response(
            200, json={"code": "00000", "data": [{"symbol": "BTCUSDT"}]}
        )

    with mock_client(httpx.MockTransport(response)) as http:
        client = BitgetClient(client=http, limiter=limiter, retries=0)
        assert client.get_instruments("SPOT") == [{"symbol": "BTCUSDT"}]
    assert limiter.acquired == ["instruments:SPOT"]


def test_portal_accepts_its_distinct_success_code() -> None:
    """Accept the historical portal's successful response envelope."""
    with mock_client(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"code": 200, "data": {"list": []}}
            )
        )
    ) as http:
        value = BitgetClient(client=http, limiter=Limiter(), retries=0).portal(
            "/v1/statistics/public/download/getSymbolList", {"businessLine": 1}
        )
    assert value == {"list": []}


def test_http_429_penalizes_and_retries() -> None:
    """Retry an empty throttled portal response after penalizing its bucket."""
    attempts = 0
    limiter = Limiter()

    def response(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"code": "200", "data": []})

    with mock_client(httpx.MockTransport(response)) as http:
        value = BitgetClient(
            client=http,
            limiter=limiter,
            retries=1,
            backoff=0,
            sleeper=lambda delay: None,
        ).portal("/manifest", {})
    assert value == []
    assert attempts == 2
    assert limiter.penalties == [("portal", 0.0)]


@pytest.mark.parametrize("status", [400, 404])
def test_nonretryable_http_errors_propagate(status: int) -> None:
    """Do not hide deterministic HTTP failures.

    Args:
        status: Representative client-error status.
    """
    with mock_client(
        httpx.MockTransport(lambda request: httpx.Response(status))
    ) as http:
        client = BitgetClient(client=http, limiter=Limiter(), retries=3)
        with pytest.raises(httpx.HTTPStatusError):
            client.get_instruments("SPOT")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"code": "00000"},
        {"code": "00000", "data": "wrong"},
    ],
)
def test_invalid_response_shapes_are_rejected(payload: object) -> None:
    """Reject malformed response envelopes and row collections.

    Args:
        payload: Invalid representative JSON response.
    """
    with mock_client(
        httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as http:
        client = BitgetClient(client=http, limiter=Limiter(), retries=0)
        with pytest.raises(BitgetResponseError):
            client.get_instruments("SPOT")


def test_invalid_json_is_rejected() -> None:
    """Reject a successful response that is not JSON."""
    with mock_client(
        httpx.MockTransport(lambda request: httpx.Response(200, content=b"nope"))
    ) as http:
        with pytest.raises(BitgetResponseError, match="valid JSON"):
            BitgetClient(client=http, limiter=Limiter(), retries=0).get_tickers("SPOT")


def test_retryable_source_error_is_retried() -> None:
    """Retry Bitget's semantic throttle code before returning data."""
    replies = iter(
        [
            {"code": "50011", "msg": "busy", "data": []},
            {"code": "00000", "data": []},
        ]
    )
    limiter = Limiter()
    with mock_client(
        httpx.MockTransport(lambda request: httpx.Response(200, json=next(replies)))
    ) as http:
        assert (
            BitgetClient(
                client=http,
                limiter=limiter,
                retries=1,
                backoff=0,
                sleeper=lambda delay: None,
                jitter=lambda maximum: maximum,
            ).get_tickers("SPOT")
            == []
        )
    assert limiter.penalties == [("tickers:SPOT", 0.0)]


def test_client_settings_reject_nonfinite_or_boolean_values() -> None:
    """Reject settings that would make retries unsafe or unbounded."""
    with pytest.raises(ValueError):
        BitgetClient(timeout=float("nan"))
    with pytest.raises(ValueError):
        BitgetClient(retries=True)
    with pytest.raises(ValueError):
        BitgetClient(backoff=-1)


def test_context_closes_only_owned_clients() -> None:
    """Leave injected HTTP pools under caller ownership."""
    external = mock_client(
        httpx.MockTransport(
            lambda request: httpx.Response(200, json={"code": "00000", "data": []})
        )
    )
    with BitgetClient(client=external, limiter=Limiter()):
        pass
    assert not external.is_closed
    external.close()

    owned = BitgetClient(limiter=Limiter())
    owned.close()
    assert owned.client.is_closed


def test_transport_failures_exhaust_bounded_retries() -> None:
    """Propagate a transport failure after the configured retry budget."""
    attempts = 0

    def fail(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadError("broken", request=request)

    with mock_client(httpx.MockTransport(fail)) as http:
        with pytest.raises(httpx.ReadError):
            BitgetClient(
                client=http,
                limiter=Limiter(),
                retries=1,
                backoff=0,
                sleeper=lambda delay: None,
            ).get_tickers("SPOT")
    assert attempts == 2


def test_source_error_retains_native_details() -> None:
    """Expose deterministic source errors without retrying them."""
    payload = {"code": "40003", "msg": "invalid date range", "data": None}
    with mock_client(
        httpx.MockTransport(
            lambda request: httpx.Response(200, content=json.dumps(payload))
        )
    ) as http:
        with pytest.raises(BitgetResponseError) as caught:
            BitgetClient(client=http, limiter=Limiter(), retries=0).portal(
                "/manifest", {}
            )
    assert caught.value.code == "40003"
    assert caught.value.message == "invalid date range"
