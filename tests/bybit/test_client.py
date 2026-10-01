"""Test Bybit's pooled, bounded public HTTP client."""

from datetime import UTC, datetime, timedelta
import json

import httpx
import pytest

from veldra.bybit.client import BybitClient, BybitResponseError


class Limiter:
    """Record quota reservations and penalties."""

    def __init__(self) -> None:
        """Create empty call logs."""
        self.acquired: list[str] = []
        self.penalties: list[tuple[str, float]] = []

    def acquire(self, key: str, *, cost: int = 1) -> None:
        """Record one request reservation."""
        assert cost == 1
        self.acquired.append(key)

    def penalize(self, key: str, retry_after: float) -> None:
        """Record one source penalty."""
        self.penalties.append((key, retry_after))


def mock_client(handler: httpx.MockTransport) -> httpx.Client:
    """Return an HTTPX client backed by one mock transport."""
    return httpx.Client(transport=handler)


def test_v5_uses_one_global_ip_budget() -> None:
    """Reserve every V5 endpoint against Bybit's shared IP quota."""
    limiter = Limiter()

    def response(request: httpx.Request) -> httpx.Response:
        assert request.url.params["category"] == "spot"
        return httpx.Response(
            200, json={"retCode": 0, "retMsg": "OK", "result": {"list": []}}
        )

    with mock_client(httpx.MockTransport(response)) as http:
        result = BybitClient(client=http, limiter=limiter, retries=0).v5(
            "/v5/market/instruments-info", {"category": "spot"}
        )
    assert result == {"list": []}
    assert limiter.acquired == ["v5"]


def test_manifest_accepts_an_empty_file_list() -> None:
    """Treat a successful empty manifest as source absence, not an error."""
    limiter = Limiter()

    def response(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/download/list-files")
        return httpx.Response(
            200, json={"ret_code": 0, "ret_msg": "success", "result": {"list": []}}
        )

    with mock_client(httpx.MockTransport(response)) as http:
        value = BybitClient(client=http, limiter=limiter, retries=0).manifest(
            {"bizType": "spot", "productId": "trade"}
        )
    assert value == {"list": []}
    assert limiter.acquired == ["manifest"]


def test_http_429_penalizes_and_retries() -> None:
    """Retry a throttle response after pausing Bybit's global V5 budget."""
    attempts = 0
    limiter = Limiter()

    def response(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"retCode": 0, "result": {}})

    with mock_client(httpx.MockTransport(response)) as http:
        result = BybitClient(
            client=http,
            limiter=limiter,
            retries=1,
            backoff=0,
            sleeper=lambda delay: None,
        ).v5("/v5/market/time")
    assert result == {}
    assert attempts == 2
    assert limiter.penalties == [("v5", 0.0)]


def test_semantic_throttle_honors_reset_header() -> None:
    """Use Bybit's reset timestamp for a semantic throttle response."""
    attempts = 0
    limiter = Limiter()
    delays: list[float] = []
    reset = int((datetime.now(UTC) + timedelta(seconds=2)).timestamp() * 1000)

    def response(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                200,
                headers={"X-Bapi-Limit-Reset-Timestamp": str(reset)},
                json={"retCode": 10006, "retMsg": "Too many visits", "result": {}},
            )
        return httpx.Response(200, json={"retCode": 0, "result": []})

    with mock_client(httpx.MockTransport(response)) as http:
        result = BybitClient(
            client=http,
            limiter=limiter,
            retries=1,
            sleeper=delays.append,
        ).v5("/v5/market/recent-trade")
    assert result == []
    assert 0 < delays[0] <= 2
    assert limiter.penalties == [("v5", delays[0])]


def test_http_403_fails_fast_with_ten_minute_cooldown() -> None:
    """Do not repeatedly hit Bybit while its IP-level ban is active."""
    attempts = 0

    def response(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(403)

    with mock_client(httpx.MockTransport(response)) as http:
        with pytest.raises(BybitResponseError) as caught:
            BybitClient(client=http, limiter=Limiter(), retries=3).v5("/v5/market/time")
    assert attempts == 1
    assert caught.value.code == "ip_banned"
    assert caught.value.cooldown_seconds == 600


def test_deterministic_v5_and_manifest_errors_are_not_retried() -> None:
    """Preserve native validation errors without wasting request capacity."""
    scenarios = [
        {"retCode": 10001, "retMsg": "Request parameter error", "result": {}},
        {"ret_code": 10016, "ret_msg": "date range error", "result": {}},
    ]

    def request(payload: object, method: str) -> tuple[int, BybitResponseError]:
        """Run one deterministic error scenario."""
        attempts = 0

        def response(_request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(200, json=payload)

        with mock_client(httpx.MockTransport(response)) as http:
            client = BybitClient(client=http, limiter=Limiter(), retries=3)
            with pytest.raises(BybitResponseError) as caught:
                if method == "v5":
                    client.v5("/bad")
                else:
                    client.manifest({})
        return attempts, caught.value

    for payload, method in zip(scenarios, ["v5", "manifest"], strict=True):
        attempts, error = request(payload, method)
        assert attempts == 1
        assert error.code in {"10001", "10016"}


@pytest.mark.parametrize(
    ("payload", "method"),
    [
        ([], "v5"),
        ({"retCode": "0", "result": {}}, "v5"),
        ({"retCode": 0}, "v5"),
        ({"ret_code": 0}, "manifest"),
    ],
)
def test_invalid_response_envelopes_are_rejected(payload: object, method: str) -> None:
    """Reject malformed portal and V5 envelopes.

    Args:
        payload: Invalid representative response body.
        method: Client method used to parse the body.
    """
    with mock_client(
        httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as http:
        client = BybitClient(client=http, limiter=Limiter(), retries=0)
        with pytest.raises(BybitResponseError):
            if method == "v5":
                client.v5("/bad")
            else:
                client.manifest({})


def test_invalid_json_is_rejected() -> None:
    """Reject a successful response that is not JSON."""
    with mock_client(
        httpx.MockTransport(lambda request: httpx.Response(200, content=b"nope"))
    ) as http:
        with pytest.raises(BybitResponseError, match="valid JSON"):
            BybitClient(client=http, limiter=Limiter(), retries=0).v5("/bad")


def test_http_error_envelope_preserves_source_details() -> None:
    """Decode a useful Bybit error body before wrapping the HTTP status."""
    payload = {"retCode": 10001, "retMsg": "bad category", "result": {}}
    with mock_client(
        httpx.MockTransport(lambda request: httpx.Response(400, json=payload))
    ) as http:
        with pytest.raises(BybitResponseError) as caught:
            BybitClient(client=http, limiter=Limiter(), retries=0).v5("/bad")
    assert caught.value.code == "10001"
    assert caught.value.message == "bad category"


def test_transport_failures_exhaust_bounded_retries() -> None:
    """Propagate transport failure after the configured retry budget."""
    attempts = 0

    def fail(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadError("broken", request=request)

    with mock_client(httpx.MockTransport(fail)) as http:
        with pytest.raises(httpx.ReadError):
            BybitClient(
                client=http,
                limiter=Limiter(),
                retries=1,
                backoff=0,
                sleeper=lambda delay: None,
            ).v5("/bad")
    assert attempts == 2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("timeout", float("nan")),
        ("timeout", True),
        ("retries", True),
        ("retries", -1),
        ("backoff", -1),
    ],
)
def test_client_settings_reject_unsafe_values(name: str, value: object) -> None:
    """Reject settings that would make request bounds unsafe.

    Args:
        name: Invalid client setting.
        value: Invalid representative value.
    """
    with pytest.raises(ValueError):
        if name == "timeout":
            BybitClient(timeout=value)  # type: ignore[arg-type]
        elif name == "retries":
            BybitClient(retries=value)  # type: ignore[arg-type]
        else:
            BybitClient(backoff=value)  # type: ignore[arg-type]


def test_context_closes_only_owned_clients() -> None:
    """Leave injected connection pools under caller ownership."""
    external = mock_client(
        httpx.MockTransport(
            lambda request: httpx.Response(200, json={"retCode": 0, "result": {}})
        )
    )
    with BybitClient(client=external, limiter=Limiter()):
        pass
    assert not external.is_closed
    external.close()

    owned = BybitClient(limiter=Limiter())
    owned.close()
    assert owned.client.is_closed


def test_rows_requires_object_collections() -> None:
    """Reject source collections containing positional or scalar rows."""
    assert BybitClient.rows([{"symbol": "BTCUSDT"}], "instrument") == [
        {"symbol": "BTCUSDT"}
    ]
    with pytest.raises(BybitResponseError, match="list of objects"):
        BybitClient.rows([["BTCUSDT"]], "instrument")


def test_error_can_be_rendered_from_json_encoded_body() -> None:
    """Retain deterministic source details from an encoded response."""
    payload = {"retCode": 10029, "retMsg": "symbol is invalid", "result": {}}
    with mock_client(
        httpx.MockTransport(
            lambda request: httpx.Response(200, content=json.dumps(payload))
        )
    ) as http:
        with pytest.raises(BybitResponseError) as caught:
            BybitClient(client=http, limiter=Limiter(), retries=0).v5("/bad")
    assert caught.value.code == "10029"
