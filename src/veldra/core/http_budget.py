"""Bound in-flight HTTP work across clients serving the same exchange."""

from collections.abc import Iterator
from contextlib import ExitStack
from ssl import SSLContext
from threading import BoundedSemaphore, Lock
from typing import Any

import httpx

MAX_EXCHANGE_REQUESTS = 32
_exchange_permits: dict[str, BoundedSemaphore] = {}
_registry_lock = Lock()


class RequestBudget:
    """Share permits across one runtime and the exchange's other runtimes."""

    def __init__(self, exchange: str, limit: int) -> None:
        """Set a runtime limit within the exchange-wide process limit."""
        self.limit = min(limit, MAX_EXCHANGE_REQUESTS)
        self.runtime = BoundedSemaphore(self.limit)
        with _registry_lock:
            if exchange not in _exchange_permits:
                _exchange_permits[exchange] = BoundedSemaphore(MAX_EXCHANGE_REQUESTS)
            self.exchange = _exchange_permits[exchange]


class _BudgetedStream(httpx.SyncByteStream):
    """Hold the request permits until its response stream finishes."""

    def __init__(self, stream: httpx.SyncByteStream, permits: ExitStack) -> None:
        self._stream = stream
        self._permits = permits
        self._closed = False

    def __iter__(self) -> Iterator[bytes]:
        try:
            yield from self._stream
        finally:
            self.close()

    def close(self) -> None:
        """Close the HTTP stream and return its permits once."""
        if self._closed:
            return
        self._closed = True
        with self._permits:
            self._stream.close()


class ExchangeHTTPClient(httpx.Client):
    """Keep ordinary HTTPX settings while sharing a request budget."""

    def __init__(
        self,
        budget: RequestBudget,
        *,
        transport: httpx.BaseTransport | None = None,
        verify: SSLContext | bool = True,
        follow_redirects: bool = True,
    ) -> None:
        """Configure the pool and retain the request permits for this runtime."""
        super().__init__(
            transport=transport,
            verify=verify,
            follow_redirects=follow_redirects,
            limits=httpx.Limits(
                max_connections=budget.limit, max_keepalive_connections=budget.limit
            ),
        )
        self._budget = budget

    def send(
        self, request: httpx.Request, *, stream: bool = False, **options: Any
    ) -> httpx.Response:
        """Send a request within the shared runtime and exchange budgets.

        Args:
            request: Prepared HTTPX request.
            stream: Whether to let the caller consume or close the response body.
            options: Authentication and redirect options passed through to HTTPX.

        Returns:
            The HTTPX response with its normal streaming behavior.
        """
        with ExitStack() as permits:
            permits.enter_context(self._budget.runtime)
            permits.enter_context(self._budget.exchange)
            response = super().send(request, stream=stream, **options)
            if not response.is_closed:
                assert isinstance(response.stream, httpx.SyncByteStream)
                response.stream = _BudgetedStream(response.stream, permits.pop_all())
            return response
