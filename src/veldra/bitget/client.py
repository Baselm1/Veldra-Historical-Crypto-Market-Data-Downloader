"""Call Bitget public endpoints through shared rate limits and retries."""

from collections.abc import Callable, Mapping
import json
import math
import random
import time
from typing import Protocol, cast

import httpx

from veldra.core.download import retry_delay
from veldra.core.rate_limit import RatePolicy, RollingWindowRateLimiter

REST_URL = "https://api.bitget.com"
PORTAL_URL = "https://www.bitget.com"
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
SUCCESS_CODES = frozenset({"0", "00000", "200"})
DEFAULT_POLICIES: Mapping[str, RatePolicy] = {
    "portal": RatePolicy(2, 1.0, 0.1),
    "instruments": RatePolicy(20, 1.0),
    "tickers": RatePolicy(20, 1.0),
    "history_candles": RatePolicy(20, 1.0),
    "history_funding": RatePolicy(20, 1.0),
}


class Limiter(Protocol):
    """Describe the request limiter operations used by the client."""

    def acquire(self, key: str, *, cost: int = 1) -> None:
        """Reserve capacity for one source request."""
        raise NotImplementedError

    def penalize(self, key: str, retry_after: float) -> None:
        """Pause one quota following source throttling."""
        raise NotImplementedError


class BitgetResponseError(RuntimeError):
    """Report one HTTP, transport, envelope, or source failure."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        """Retain Bitget's error code and retry classification.

        Args:
            code: Native Bitget or local validation code.
            message: Human-readable source failure.
            retryable: Whether another request may succeed.
        """
        super().__init__(f"Bitget {code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable


class BitgetRateLimiter(RollingWindowRateLimiter):
    """Apply Bitget's portal and REST quotas through the shared limiter."""

    def __init__(self, policies: Mapping[str, RatePolicy] | None = None) -> None:
        """Create a limiter with conservative public endpoint policies.

        Args:
            policies: Optional replacement endpoint policies.
        """
        super().__init__(DEFAULT_POLICIES if policies is None else policies)


def _positive_number(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Return one validated finite client setting.

    Args:
        value: Proposed numeric setting.
        name: Setting name used in validation errors.
        allow_zero: Whether zero is accepted.

    Returns:
        The validated number.
    """
    minimum = 0 if allow_zero else 0.0
    if not math.isfinite(value) or value < minimum or (not allow_zero and value == 0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def _retry_count(value: int) -> int:
    """Return a valid retry count.

    Args:
        value: Proposed retries following the first request.

    Returns:
        The validated retry count.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("retries must be a nonnegative integer")
    return value


def _throttled(response: httpx.Response | None, error: Exception) -> bool:
    """Return whether one failed request exhausted source quota.

    Args:
        response: Optional failed HTTP response.
        error: Failure raised by the request attempt.

    Returns:
        Whether the limiter should pause the affected bucket.
    """
    if response is not None and response.status_code == 429:
        return True
    return isinstance(error, BitgetResponseError) and error.retryable


class BitgetClient:
    """Call Bitget's unauthenticated portal and REST endpoints safely."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        limiter: Limiter | None = None,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 0.5,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[float], float] = lambda limit: random.uniform(0, limit),
    ) -> None:
        """Create one pooled Bitget public client.

        Args:
            client: Optional externally managed HTTPX client.
            limiter: Optional shared quota registry.
            timeout: Per-attempt request timeout in seconds.
            retries: Retries following the first attempt.
            backoff: Initial retry delay in seconds.
            sleeper: Delay function used between attempts.
            jitter: Full-jitter function receiving a maximum delay.
        """
        self.client = client or httpx.Client()
        self._owns_client = client is None
        self.limiter = limiter or BitgetRateLimiter()
        self.timeout = _positive_number(timeout, "timeout")
        self.retries = _retry_count(retries)
        self.backoff = _positive_number(backoff, "backoff", allow_zero=True)
        self._sleep = sleeper
        self._jitter = jitter

    def close(self) -> None:
        """Close the internally created connection pool."""
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "BitgetClient":
        """Return this client for a managed context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close owned resources when a context exits.

        Args:
            exc: Optional exception context supplied by Python.
        """
        self.close()

    @staticmethod
    def _envelope(response: httpx.Response) -> object:
        """Validate one Bitget response envelope and return its data.

        Args:
            response: Successful HTTP response to decode.

        Returns:
            The response's source data value.
        """
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            raise BitgetResponseError(
                "invalid_json", "response is not valid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise BitgetResponseError("invalid_envelope", "response must be an object")
        code = str(payload.get("code", ""))
        message = payload.get("msg", payload.get("message", ""))
        if code not in SUCCESS_CODES:
            raise BitgetResponseError(
                code or "invalid_envelope",
                (
                    message
                    if isinstance(message, str) and message
                    else "source rejected request"
                ),
                retryable=code in {"429", "50001", "50011"},
            )
        if "data" not in payload:
            raise BitgetResponseError("invalid_envelope", "response data is missing")
        return payload["data"]

    def _wait(self, response: httpx.Response | None, attempt: int) -> float:
        """Return a declared Retry-After delay or exponential full jitter.

        Args:
            response: Optional failed HTTP response.
            attempt: Zero-based failed attempt.

        Returns:
            Delay in seconds before retrying.
        """
        maximum = retry_delay(response, attempt, self.backoff)
        declared = None if response is None else response.headers.get("Retry-After")
        return maximum if declared is not None else self._jitter(maximum)

    @classmethod
    def _response_value(cls, response: httpx.Response) -> object:
        """Return data while preserving useful source errors on HTTP failures.

        Args:
            response: Completed Bitget response.

        Returns:
            The validated response data.
        """
        try:
            value = cls._envelope(response)
        except BitgetResponseError as error:
            if response.is_error and error.code in {
                "invalid_json",
                "invalid_envelope",
            }:
                response.raise_for_status()
            raise
        response.raise_for_status()
        return value

    def request(
        self,
        method: str,
        url: str,
        *,
        policy_key: str,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, object] | None = None,
    ) -> object:
        """Make one rate-limited and retryable Bitget request.

        Args:
            method: HTTP request method.
            url: Absolute public endpoint URL.
            policy_key: Shared quota key.
            params: Optional query-string values.
            json_body: Optional JSON request body.

        Returns:
            The validated envelope data value.
        """
        for attempt in range(self.retries + 1):
            self.limiter.acquire(policy_key)
            response: httpx.Response | None = None
            try:
                response = self.client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    timeout=self.timeout,
                )
                return self._response_value(response)
            except BitgetResponseError as error:
                caught: Exception = error
                retryable = error.retryable
            except httpx.HTTPStatusError as error:
                caught = error
                retryable = error.response.status_code in RETRYABLE_STATUS
            except httpx.TransportError as error:
                caught = error
                retryable = True
            if not retryable or attempt == self.retries:
                raise caught
            delay = self._wait(response, attempt)
            if _throttled(response, caught):
                self.limiter.penalize(policy_key, delay)
            self._sleep(delay)
        raise RuntimeError("Bitget retry loop ended without a result")

    @staticmethod
    def _rows(value: object, endpoint: str) -> list[dict[str, object]]:
        """Require a list of object rows from one endpoint.

        Args:
            value: Envelope data value.
            endpoint: Endpoint name used in validation errors.

        Returns:
            Validated object rows.
        """
        if not isinstance(value, list) or not all(
            isinstance(row, dict) for row in value
        ):
            raise BitgetResponseError(
                "invalid_data", f"{endpoint} data must be a list of objects"
            )
        return cast(list[dict[str, object]], value)

    def get_instruments(self, category: str) -> list[dict[str, object]]:
        """Return current instruments for one native Bitget category.

        Args:
            category: Native Spot or Futures category.

        Returns:
            Current public instrument rows.
        """
        data = self.request(
            "GET",
            f"{REST_URL}/api/v3/market/instruments",
            policy_key=f"instruments:{category}",
            params={"category": category},
        )
        return self._rows(data, "instrument")

    def get_tickers(self, category: str) -> list[dict[str, object]]:
        """Return current tickers for one native Bitget category.

        Args:
            category: Native Spot or Futures category.

        Returns:
            Current public ticker rows.
        """
        data = self.request(
            "GET",
            f"{REST_URL}/api/v3/market/tickers",
            policy_key=f"tickers:{category}",
            params={"category": category},
        )
        return self._rows(data, "ticker")

    def portal(self, path: str, body: Mapping[str, object]) -> object:
        """Call one rate-limited historical-download portal endpoint.

        Args:
            path: Portal API path below Bitget's public origin.
            body: Validated source-specific request body.

        Returns:
            The validated portal response data.
        """
        return self.request(
            "POST",
            f"{PORTAL_URL}{path}",
            policy_key="portal",
            json_body=body,
        )
