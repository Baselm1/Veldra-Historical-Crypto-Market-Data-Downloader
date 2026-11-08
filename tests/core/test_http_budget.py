"""Exercise shared request permits with independent clients and streams."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event

import httpx
import pytest

from veldra.core.download import get
from veldra.core.http_budget import ExchangeHTTPClient, RequestBudget


class Body(httpx.SyncByteStream):
    """Expose an open response body until the caller reads or closes it."""

    def __iter__(self) -> Iterator[bytes]:
        yield b"fixture"


def transport() -> httpx.MockTransport:
    """Return a mock with both streamed and already-buffered responses."""

    def respond(request: httpx.Request) -> httpx.Response:
        """Send a stream when requested by the fixture URL."""
        if request.url.path == "/stream":
            return httpx.Response(200, stream=Body())
        return httpx.Response(200, content=b"fixture")

    return httpx.MockTransport(respond)


@pytest.mark.parametrize(("limit", "count"), [(1, 1), (100, 32)])
def test_budget_holds_until_stream_closes_and_isolates_exchanges(
    limit: int, count: int
) -> None:
    """Saturate the configured cap while another exchange remains available."""
    budget = RequestBudget("fixture-a", limit)
    sibling_budget = budget if limit == 1 else RequestBudget("fixture-a", limit)
    entered = Event()
    delivered = Event()
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        ExchangeHTTPClient(budget, transport=transport()) as first,
        ExchangeHTTPClient(sibling_budget, transport=transport()) as sibling,
        ExchangeHTTPClient(
            RequestBudget("fixture-b", 1), transport=transport()
        ) as other,
        ExitStack() as streams,
    ):
        held = [
            streams.enter_context(first.stream("GET", "https://example.test/stream"))
            for _ in range(count)
        ]

        def wait_for_permit() -> int:
            """Start the sibling request and signal its eventual completion."""
            entered.set()
            response = sibling.get("https://example.test/")
            delivered.set()
            return response.status_code

        future = executor.submit(wait_for_permit)
        assert entered.wait(2)
        assert not delivered.wait(0.05)
        assert other.get("https://example.test/").status_code == 200
        held[0].close()
        assert future.result(timeout=2) == 200


def test_budget_releases_failed_requests_before_retry_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow another request during retry sleep after a transport exception."""
    budget = RequestBudget("fixture-retry", 1)
    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        """Fail the first transport attempt and succeed on the retry."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("first attempt", request=request)
        return httpx.Response(200, content=b"recovered")

    with (
        ExchangeHTTPClient(budget, transport=httpx.MockTransport(respond)) as client,
        ExchangeHTTPClient(budget, transport=transport()) as sibling,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):

        def during_backoff(delay: float) -> None:
            """Request from another client before the failing call can retry."""
            future = executor.submit(sibling.get, "https://example.test/")
            assert future.result(timeout=2).status_code == 200

        monkeypatch.setattr("veldra.core.download.time.sleep", during_backoff)
        assert get(client, "https://example.test/", retries=1).content == b"recovered"
        assert attempts == 2
