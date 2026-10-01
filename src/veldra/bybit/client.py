"""Call Bybit's public REST and historical-manifest endpoints safely."""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import json
import math
import random
import time
from typing import Protocol, cast

import httpx

from veldra.core.download import retry_delay
from veldra.core.rate_limit import RatePolicy, RollingWindowRateLimiter

REST_URL = "https://api.bybit.com"
PORTAL_URL = "https://www.bybit.com"
MANIFEST_PATH = "/x-api/quote/public/support/download/list-files"
PORTAL_HEADERS: Mapping[str, str] = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.bybit.com/data-download",
}
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
DEFAULT_POLICIES: Mapping[str, RatePolicy] = {
    # Bybit documents one 600-request, five-second IP budget. Reserving one
    # sixth for non-Veldra traffic and clock skew prevents avoidable bans.
    "v5": RatePolicy(500, 5.0, 0.05),
    # The download GUI publishes no quota. Keep this deliberately modest.
    "manifest": RatePolicy(2, 1.0, 0.1),
}


class Limiter(Protocol):
    """Describe the quota operations used by the Bybit client."""

    def acquire(self, key: str, *, cost: int = 1) -> None:
        """Reserve capacity for one source request."""
        raise NotImplementedError

    def penalize(self, key: str, retry_after: float) -> None:
        """Pause one quota after source throttling."""
        raise NotImplementedError


class BybitResponseError(RuntimeError):
    """Report one HTTP, envelope, or Bybit semantic failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        cooldown_seconds: float | None = None,
    ) -> None:
        """Retain Bybit's error details and retry classification.

        Args:
            code: Native Bybit or local validation code.
            message: Human-readable source failure.
            retryable: Whether another bounded attempt may succeed.
            cooldown_seconds: Source-mandated pause before future requests.
        """
        super().__init__(f"Bybit {code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable
        self.cooldown_seconds = cooldown_seconds


class BybitRateLimiter(RollingWindowRateLimiter):
    """Share Bybit's global V5 and conservative portal quotas."""

    def __init__(self, policies: Mapping[str, RatePolicy] | None = None) -> None:
        """Create the limiter with conservative public endpoint policies.

        Args:
            policies: Optional replacement quota policies.
        """
        super().__init__(DEFAULT_POLICIES if policies is None else policies)


def _finite_number(value: object) -> bool:
    """Return whether one value is a finite, non-Boolean number.

    Args:
        value: Candidate numeric value.

    Returns:
        Whether the value is safe for HTTP timing arithmetic.
    """
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _positive_number(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Validate one finite HTTP client setting.

    Args:
        value: Proposed numeric setting.
        name: Setting name included in validation errors.
        allow_zero: Whether zero is accepted.

    Returns:
        The validated number.
    """
    if not _finite_number(value) or value < 0 or (not allow_zero and value == 0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return float(value)


def _retry_count(value: int) -> int:
    """Validate retries following the initial request.

    Args:
        value: Proposed retry count.

    Returns:
        The validated count.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("retries must be a nonnegative integer")
    return value


def _message(payload: Mapping[str, object], key: str) -> str:
    """Return a useful source message from one response envelope.

    Args:
        payload: Decoded response object.
        key: Source-specific message field.

    Returns:
        A nonempty human-readable message.
    """
    value = payload.get(key)
    return value if isinstance(value, str) and value else "source rejected request"


def _retryable(error: Exception) -> bool:
    """Return whether one client failure permits another bounded attempt.

    Args:
        error: Failure raised by one request attempt.

    Returns:
        Whether a retry may succeed.
    """
    if isinstance(error, BybitResponseError):
        return error.retryable
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS
    return isinstance(error, httpx.TransportError)


def _throttled(response: httpx.Response | None, error: Exception) -> bool:
    """Return whether one failure exhausted a Bybit quota.

    Args:
        response: Optional failed response.
        error: Failure raised by the attempt.

    Returns:
        Whether the shared quota should be penalized.
    """
    if response is not None and response.status_code == 429:
        return True
    return isinstance(error, BybitResponseError) and error.code == "10006"


class BybitClient:
    """Call Bybit's unauthenticated endpoints with pooling and bounded retries."""

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
        """Create one pooled Bybit public client.

        Args:
            client: Optional externally managed HTTPX connection pool.
            limiter: Optional shared request limiter.
            timeout: Per-attempt timeout in seconds.
            retries: Retries following the first attempt.
            backoff: Initial retry delay in seconds.
            sleeper: Delay function used between attempts.
            jitter: Full-jitter function receiving the maximum delay.
        """
        self.client = client or httpx.Client()
        self._owns_client = client is None
        self.limiter = limiter or BybitRateLimiter()
        self.timeout = _positive_number(timeout, "timeout")
        self.retries = _retry_count(retries)
        self.backoff = _positive_number(backoff, "backoff", allow_zero=True)
        self._sleep = sleeper
        self._jitter = jitter

    def close(self) -> None:
        """Close the internally created connection pool."""
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "BybitClient":
        """Return this client for a managed context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close owned resources when a context exits.

        Args:
            exc: Optional exception context supplied by Python.
        """
        self.close()

    @staticmethod
    def _object(response: httpx.Response) -> dict[str, object]:
        """Decode one JSON response object.

        Args:
            response: Completed HTTP response.

        Returns:
            The decoded response object.
        """
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            raise BybitResponseError(
                "invalid_json", "response is not valid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise BybitResponseError("invalid_envelope", "response must be an object")
        return cast(dict[str, object], payload)

    @classmethod
    def _v5_envelope(cls, response: httpx.Response) -> object:
        """Validate a V5 envelope and return its result value.

        Args:
            response: Successful V5 HTTP response.

        Returns:
            The envelope's result value.
        """
        payload = cls._object(response)
        code = payload.get("retCode")
        if isinstance(code, bool) or not isinstance(code, int):
            raise BybitResponseError("invalid_envelope", "retCode is missing")
        if code != 0:
            raise BybitResponseError(
                str(code),
                _message(payload, "retMsg"),
                retryable=code == 10006,
            )
        if "result" not in payload:
            raise BybitResponseError("invalid_envelope", "result is missing")
        return payload["result"]

    @classmethod
    def _manifest_envelope(cls, response: httpx.Response) -> object:
        """Validate the download portal's distinct response envelope.

        Args:
            response: Successful portal HTTP response.

        Returns:
            The envelope's result value, including valid empty results.
        """
        payload = cls._object(response)
        code = payload.get("ret_code")
        if isinstance(code, bool) or not isinstance(code, int):
            raise BybitResponseError("invalid_envelope", "ret_code is missing")
        if code != 0:
            raise BybitResponseError(str(code), _message(payload, "ret_msg"))
        if "result" not in payload:
            raise BybitResponseError("invalid_envelope", "result is missing")
        return payload["result"]

    def _wait(self, response: httpx.Response | None, attempt: int) -> float:
        """Return a declared or exponentially backed-off retry delay.

        Args:
            response: Optional failed HTTP response.
            attempt: Zero-based failed attempt.

        Returns:
            Delay in seconds before retrying.
        """
        maximum = retry_delay(response, attempt, self.backoff)
        declared = None if response is None else response.headers.get("Retry-After")
        if declared is not None:
            return maximum
        reset = (
            None
            if response is None
            else response.headers.get("X-Bapi-Limit-Reset-Timestamp")
        )
        if reset is not None:
            try:
                delay = float(reset) / 1000 - datetime.now(UTC).timestamp()
            except ValueError:
                pass
            else:
                if math.isfinite(delay) and delay > 0:
                    return min(delay, 60.0)
        return self._jitter(maximum)

    @staticmethod
    def _cooldown(response: httpx.Response) -> None:
        """Raise the documented ten-minute IP cooldown on HTTP 403.

        Args:
            response: Forbidden Bybit response.
        """
        if response.status_code == 403:
            raise BybitResponseError(
                "ip_banned",
                "IP request limit exceeded; wait at least ten minutes",
                cooldown_seconds=600.0,
            )

    @staticmethod
    def _response_value(
        response: httpx.Response, parser: Callable[[httpx.Response], object]
    ) -> object:
        """Return parsed data while preserving useful HTTP error envelopes.

        Args:
            response: Completed Bybit response.
            parser: Source-envelope parser.

        Returns:
            The validated source result.
        """
        try:
            value = parser(response)
        except BybitResponseError as error:
            if response.is_error and error.code in {
                "invalid_json",
                "invalid_envelope",
            }:
                response.raise_for_status()
            raise
        response.raise_for_status()
        return value

    def _request(
        self,
        url: str,
        *,
        policy_key: str,
        params: Mapping[str, str] | None,
        parser: Callable[[httpx.Response], object],
        headers: Mapping[str, str] | None = None,
    ) -> object:
        """Make one pooled, rate-limited, and retryable request.

        Args:
            url: Absolute public endpoint URL.
            policy_key: Shared request quota key.
            params: Optional query-string values.
            parser: Source-envelope parser.
            headers: Optional source-required request headers.

        Returns:
            The validated envelope result.
        """
        for attempt in range(self.retries + 1):
            self.limiter.acquire(policy_key)
            response: httpx.Response | None = None
            try:
                response = self.client.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
                self._cooldown(response)
                if response.status_code in RETRYABLE_STATUS:
                    response.raise_for_status()
                return self._response_value(response, parser)
            except (
                BybitResponseError,
                httpx.HTTPStatusError,
                httpx.TransportError,
            ) as error:
                caught = error
            if not _retryable(caught) or attempt == self.retries:
                raise caught
            delay = self._wait(response, attempt)
            if _throttled(response, caught):
                self.limiter.penalize(policy_key, delay)
            self._sleep(delay)
        raise RuntimeError("Bybit retry loop ended without a result")

    def v5(self, path: str, params: Mapping[str, str] | None = None) -> object:
        """Call one public V5 endpoint through Bybit's global IP budget.

        Args:
            path: Public V5 API path.
            params: Optional query-string values.

        Returns:
            The validated V5 result value.
        """
        return self._request(
            f"{REST_URL}{path}",
            policy_key="v5",
            params=params,
            parser=self._v5_envelope,
        )

    def manifest(self, params: Mapping[str, str]) -> object:
        """Query the historical-download portal manifest.

        Args:
            params: Validated portal query values.

        Returns:
            The validated manifest result value.
        """
        return self._request(
            f"{PORTAL_URL}{MANIFEST_PATH}",
            policy_key="manifest",
            params=params,
            parser=self._manifest_envelope,
            headers=PORTAL_HEADERS,
        )

    @staticmethod
    def _result_page(
        value: object, endpoint: str
    ) -> tuple[list[dict[str, object]], str]:
        """Validate one paginated V5 result object.

        Args:
            value: Source result returned by V5.
            endpoint: Endpoint name used in validation errors.

        Returns:
            Object rows and the optional next-page cursor.
        """
        if not isinstance(value, dict):
            raise BybitResponseError(
                "invalid_data", f"{endpoint} result must be an object"
            )
        rows = BybitClient.rows(value.get("list"), endpoint)
        cursor = value.get("nextPageCursor", "")
        if not isinstance(cursor, str):
            raise BybitResponseError(
                "invalid_data", f"{endpoint} cursor must be a string"
            )
        return rows, cursor

    def get_instruments(
        self,
        category: str,
        *,
        base_coin: str | None = None,
        status: str | None = None,
        max_pages: int = 20,
    ) -> list[dict[str, object]]:
        """Return current instruments across all V5 cursor pages.

        Args:
            category: Native Bybit instrument category.
            base_coin: Optional Options underlying filter.
            status: Optional native instrument status.
            max_pages: Hard pagination safety bound.

        Returns:
            Current public instrument records.
        """
        if (
            isinstance(max_pages, bool)
            or not isinstance(max_pages, int)
            or max_pages < 1
        ):
            raise ValueError("max_pages must be a positive integer")
        params = {"category": category, "limit": "1000"}
        if base_coin is not None:
            params["baseCoin"] = base_coin
        if status is not None:
            params["status"] = status
        found: list[dict[str, object]] = []
        seen: set[str] = set()
        for _page in range(max_pages):
            value = self.v5("/v5/market/instruments-info", params)
            rows, cursor = self._result_page(value, "instrument")
            found.extend(rows)
            if not cursor:
                return found
            if cursor in seen:
                raise BybitResponseError("cursor_loop", "instrument cursor repeated")
            seen.add(cursor)
            params["cursor"] = cursor
        raise BybitResponseError(
            "page_limit", "instrument pagination exceeded its limit"
        )

    def get_tickers(self, category: str) -> list[dict[str, object]]:
        """Return current tickers for one native Bybit category.

        Args:
            category: Native Bybit instrument category.

        Returns:
            Current public ticker records.
        """
        value = self.v5("/v5/market/tickers", {"category": category})
        rows, cursor = self._result_page(value, "ticker")
        if cursor:
            raise BybitResponseError(
                "invalid_data", "ticker endpoint unexpectedly returned a cursor"
            )
        return rows

    def get_option_base_coins(self) -> list[dict[str, object]]:
        """Return every Option underlying currently published by Bybit.

        Returns:
            Native Option base-coin records.
        """
        value = self.v5("/v5/market/option-base-coins")
        rows, cursor = self._result_page(value, "Option base coin")
        if cursor:
            raise BybitResponseError(
                "invalid_data", "Option base-coin endpoint returned a cursor"
            )
        return rows

    @staticmethod
    def rows(value: object, endpoint: str) -> list[dict[str, object]]:
        """Require a list of object rows from one endpoint result.

        Args:
            value: Source result value.
            endpoint: Endpoint name used in validation errors.

        Returns:
            Validated object rows.
        """
        if not isinstance(value, list) or not all(
            isinstance(row, dict) for row in value
        ):
            raise BybitResponseError(
                "invalid_data", f"{endpoint} data must be a list of objects"
            )
        return cast(list[dict[str, object]], value)
