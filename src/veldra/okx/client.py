"""Call OKX public endpoints through one shared rolling-window limiter."""

from collections import defaultdict, deque
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
import math
import random
from threading import Condition
import time
from typing import Protocol, cast

import httpx

from veldra.core.download import retry_delay

BASE_URL = "https://www.okx.com"
INSTRUMENT_TYPES = frozenset({"SPOT", "MARGIN", "SWAP", "FUTURES", "OPTION"})
MANIFEST_MODULES = frozenset({1, 2, 3, 4, 5, 6, 11})
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RatePolicy:
    """Declare one request-count quota over a rolling time window."""

    capacity: int
    window_seconds: float
    safety_seconds: float = 0.05

    def __post_init__(self) -> None:
        """Reject nonpositive or nonfinite limiter settings."""
        if isinstance(self.capacity, bool) or self.capacity < 1:
            raise ValueError("rate policy capacity must be positive")
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("rate policy window must be finite and positive")
        if not math.isfinite(self.safety_seconds) or self.safety_seconds < 0:
            raise ValueError("rate policy safety must be finite and nonnegative")


DEFAULT_POLICIES: dict[str, RatePolicy] = {
    "manifest": RatePolicy(5, 2),
    "instruments": RatePolicy(20, 2),
    "tickers": RatePolicy(20, 2),
    "history_candles": RatePolicy(20, 2),
    "history_index_candles": RatePolicy(10, 2),
    "history_mark_candles": RatePolicy(20, 2),
    "funding_history": RatePolicy(10, 2),
    "premium_history": RatePolicy(20, 2),
    "settlement_history": RatePolicy(40, 2),
    "delivery_exercise": RatePolicy(40, 2),
}


class OKXRateLimiter:
    """Share keyed rolling-window request limits across worker threads."""

    def __init__(
        self,
        policies: Mapping[str, RatePolicy] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a limiter from endpoint policies.

        Args:
            policies: Endpoint policies keyed by quota family.
            clock: Monotonic clock used to age reservations.
        """
        self.policies = dict(DEFAULT_POLICIES if policies is None else policies)
        self._clock = clock
        self._condition = Condition()
        self._reservations: dict[Hashable, deque[float]] = defaultdict(deque)
        self._penalties: dict[Hashable, float] = {}

    def _policy(self, key: Hashable) -> RatePolicy:
        """Resolve an exact or colon-qualified limiter key.

        Args:
            key: Exact endpoint key or a qualified per-instrument key.

        Returns:
            The declared endpoint policy.
        """
        if key in self.policies:
            return self.policies[key]
        if isinstance(key, str) and ":" in key:
            family = key.split(":", 1)[0]
            if family in self.policies:
                return self.policies[family]
        raise KeyError(f"no rate policy is declared for {key!r}")

    def acquire(self, key: Hashable, *, cost: int = 1) -> None:
        """Wait until one endpoint request fits its rolling quota.

        Args:
            key: Endpoint quota key.
            cost: Request units reserved by this call.
        """
        policy = self._policy(key)
        if isinstance(cost, bool) or not isinstance(cost, int) or cost < 1:
            raise ValueError("rate-limit cost must be a positive integer")
        if cost > policy.capacity:
            raise ValueError("rate-limit cost exceeds policy capacity")
        with self._condition:
            while True:
                now = self._clock()
                reservations = self._reservations[key]
                cutoff = now - policy.window_seconds
                while reservations and reservations[0] <= cutoff:
                    reservations.popleft()
                penalty = self._penalties.get(key, 0.0)
                if penalty <= now and len(reservations) + cost <= policy.capacity:
                    reservations.extend([now] * cost)
                    return
                waits = []
                if penalty > now:
                    waits.append(penalty - now)
                if len(reservations) + cost > policy.capacity:
                    index = len(reservations) + cost - policy.capacity - 1
                    waits.append(reservations[index] + policy.window_seconds - now)
                self._condition.wait(max(0.001, min(waits)) + policy.safety_seconds)

    def penalize(self, key: Hashable, retry_after: float) -> None:
        """Pause one affected quota bucket after server throttling.

        Args:
            key: Endpoint quota key.
            retry_after: Minimum pause in seconds.
        """
        self._policy(key)
        if not math.isfinite(retry_after) or retry_after < 0:
            raise ValueError("retry_after must be finite and nonnegative")
        with self._condition:
            self._penalties[key] = max(
                self._penalties.get(key, 0.0), self._clock() + retry_after
            )
            self._condition.notify_all()


class Limiter(Protocol):
    """Describe the limiter operations used by the client."""

    def acquire(self, key: str, *, cost: int = 1) -> None:
        """Reserve one endpoint request."""
        ...

    def penalize(self, key: str, retry_after: float) -> None:
        """Temporarily pause one endpoint bucket."""
        ...


class OKXResponseError(RuntimeError):
    """Report an HTTP, transport, envelope, or OKX semantic failure."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        """Retain the native code and retry classification.

        Args:
            code: Native OKX or local response code.
            message: Human-readable source failure.
            retryable: Whether another attempt may succeed.
        """
        super().__init__(f"OKX {code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable


def _instrument_type(value: object) -> str:
    """Validate and return one native OKX instrument type.

    Args:
        value: Proposed uppercase instrument type.

    Returns:
        A supported native instrument type.
    """
    if not isinstance(value, str):
        raise TypeError("inst_type must be a string")
    if value not in INSTRUMENT_TYPES:
        raise ValueError(f"unsupported inst_type {value!r}")
    return value


def _epoch_milliseconds(value: date) -> str:
    """Return midnight UTC for one manifest calendar date.

    Args:
        value: Calendar date interpreted by the module's documented timezone.

    Returns:
        Unix epoch milliseconds for the date portion.
    """
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError("manifest dates must be date values")
    return str(
        int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp() * 1000)
    )


def _month_span(begin: date, end: date) -> int:
    """Return the inclusive number of calendar months in a range.

    Args:
        begin: First calendar month.
        end: Last calendar month.

    Returns:
        Inclusive calendar-month count.
    """
    return (end.year - begin.year) * 12 + end.month - begin.month + 1


def _manifest_subjects(subjects: Sequence[str]) -> list[str]:
    """Validate one manifest subject batch.

    Args:
        subjects: Native IDs, families, currencies, or `ANY`.

    Returns:
        A stable copied subject list.
    """
    values = list(subjects)
    if not values or len(values) > 5:
        raise ValueError("manifest requires one to five subjects")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError("manifest subjects must be nonempty strings")
    if "ANY" in values and values != ["ANY"]:
        raise ValueError("ANY must be the only manifest subject")
    return values


def _validate_manifest_range(cadence: str, begin: date, end: date) -> None:
    """Enforce OKX's inclusive ten-day or ten-month range limit.

    Args:
        cadence: `daily` or `monthly` archive grouping.
        begin: First inclusive source date.
        end: Last inclusive source date.
    """
    if cadence not in {"daily", "monthly"}:
        raise ValueError("manifest cadence must be daily or monthly")
    if begin > end:
        raise ValueError("manifest begin must not follow end")
    span = (end - begin).days + 1 if cadence == "daily" else _month_span(begin, end)
    unit = "days" if cadence == "daily" else "months"
    if span > 10:
        raise ValueError(f"manifest range cannot exceed ten {unit}")


def _validate_manifest_combination(
    module: int, cadence: str, subjects: list[str]
) -> None:
    """Reject module and cadence combinations unsupported by OKX.

    Args:
        module: Native historical data module.
        cadence: `daily` or `monthly` grouping.
        subjects: Validated native subject batch.
    """
    if isinstance(module, bool) or module not in MANIFEST_MODULES:
        raise ValueError(f"unsupported manifest module {module!r}")
    if module == 3 and cadence == "daily" and subjects != ["ANY"]:
        raise ValueError("daily funding manifests require ANY")
    if module == 6 and cadence == "monthly":
        raise ValueError("legacy order books do not support monthly manifests")


class OKXClient:
    """Call unauthenticated OKX endpoints with bounded retries and quotas."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        limiter: Limiter | None = None,
        base_url: str = BASE_URL,
        timeout: float = 30,
        retries: int = 3,
        backoff: float = 0.5,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[float], float] = lambda limit: random.uniform(0, limit),
    ) -> None:
        """Create one shared OKX public client.

        Args:
            client: Optional externally managed HTTPX client.
            limiter: Optional shared quota registry.
            base_url: OKX public REST origin.
            timeout: Per-attempt request timeout in seconds.
            retries: Retries following the first attempt.
            backoff: Initial retry backoff in seconds.
            sleeper: Delay function used between attempts.
            jitter: Full-jitter function receiving the maximum delay.
        """
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be finite and positive")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("retries must be a nonnegative integer")
        if backoff < 0 or not math.isfinite(backoff):
            raise ValueError("backoff must be finite and nonnegative")
        self.client = client or httpx.Client(base_url=base_url)
        self._owns_client = client is None
        self.limiter = limiter or OKXRateLimiter()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._sleep = sleeper
        self._jitter = jitter

    def close(self) -> None:
        """Close the internally created HTTP connection pool."""
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "OKXClient":
        """Return this client for a managed context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close owned network resources when a context exits.

        Args:
            exc: Optional exception context supplied by Python.
        """
        self.close()

    @staticmethod
    def _envelope(response: httpx.Response) -> list[object]:
        """Validate one OKX response envelope and return its data list.

        Args:
            response: Successful HTTP response to decode.

        Returns:
            The source data list, which may be empty.
        """
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            raise OKXResponseError(
                "invalid_json", "response is not valid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise OKXResponseError("invalid_envelope", "response must be an object")
        code = payload.get("code")
        message = payload.get("msg", "")
        if not isinstance(code, str):
            raise OKXResponseError("invalid_envelope", "response code is missing")
        if code != "0":
            raise OKXResponseError(
                code,
                message if isinstance(message, str) else "unknown source error",
                retryable=code == "50011",
            )
        data = payload.get("data")
        if not isinstance(data, list):
            raise OKXResponseError("invalid_envelope", "response data must be a list")
        return data

    def _wait(self, response: httpx.Response | None, attempt: int) -> float:
        """Calculate a Retry-After delay or exponential full jitter.

        Args:
            response: Optional failed HTTP response.
            attempt: Zero-based failed attempt.

        Returns:
            Delay in seconds before retrying.
        """
        declared = None if response is None else response.headers.get("Retry-After")
        maximum = retry_delay(response, attempt, self.backoff)
        return maximum if declared is not None else self._jitter(maximum)

    def request(
        self,
        path: str,
        *,
        policy_key: str,
        params: Mapping[str, str] | None = None,
    ) -> list[object]:
        """Make one rate-limited, retryable OKX request.

        Args:
            path: Public REST path.
            policy_key: Shared endpoint quota key.
            params: Optional query values.

        Returns:
            The validated response data list.
        """
        for attempt in range(self.retries + 1):
            self.limiter.acquire(policy_key)
            response: httpx.Response | None = None
            try:
                response = self.client.get(
                    f"{self.base_url}{path}", params=params, timeout=self.timeout
                )
                if response.status_code in RETRYABLE_STATUS:
                    response.raise_for_status()
                response.raise_for_status()
                return self._envelope(response)
            except OKXResponseError as error:
                retryable = error.retryable
                caught: Exception = error
            except httpx.HTTPStatusError as error:
                retryable = error.response.status_code in RETRYABLE_STATUS
                caught = error
            except httpx.TransportError as error:
                retryable = True
                caught = error
            if not retryable or attempt == self.retries:
                raise caught
            delay = self._wait(response, attempt)
            throttled = (response is not None and response.status_code == 429) or (
                isinstance(caught, OKXResponseError) and caught.code == "50011"
            )
            if throttled:
                self.limiter.penalize(policy_key, delay)
            self._sleep(delay)
        raise RuntimeError("OKX retry loop ended without a result")

    @staticmethod
    def _rows(data: list[object], endpoint: str) -> list[dict[str, object]]:
        """Require object rows from a metadata endpoint.

        Args:
            data: Envelope data values.
            endpoint: Endpoint name used in errors.

        Returns:
            The validated object rows.
        """
        if not all(isinstance(item, dict) for item in data):
            raise OKXResponseError("invalid_data", f"{endpoint} rows must be objects")
        return cast(list[dict[str, object]], data)

    def get_instruments(
        self, inst_type: str, *, inst_id: str | None = None
    ) -> list[dict[str, object]]:
        """Return current instruments of one native OKX type.

        Args:
            inst_type: Native `SPOT`, `MARGIN`, `SWAP`, `FUTURES`, or `OPTION`.
            inst_id: Optional exact native instrument ID.

        Returns:
            Current public instrument records.
        """
        native = _instrument_type(inst_type)
        params = {"instType": native}
        if inst_id:
            params["instId"] = inst_id
        data = self.request(
            "/api/v5/public/instruments",
            policy_key=f"instruments:{native}",
            params=params,
        )
        return self._rows(data, "instrument")

    def get_tickers(self, inst_type: str) -> list[dict[str, object]]:
        """Return current tickers of one native OKX type.

        Args:
            inst_type: Native public instrument type.

        Returns:
            Current ticker records.
        """
        native = _instrument_type(inst_type)
        data = self.request(
            "/api/v5/market/tickers",
            policy_key=f"tickers:{native}",
            params={"instType": native},
        )
        return self._rows(data, "ticker")

    @staticmethod
    def _manifest_params(
        module: int,
        inst_type: str,
        subjects: Sequence[str],
        cadence: str,
        begin: date,
        end: date,
    ) -> dict[str, str]:
        """Validate and build one bounded manifest query.

        Args:
            module: Native historical data module.
            inst_type: Native instrument type.
            subjects: Instrument, family, currency, or `ANY` scopes.
            cadence: `daily` or `monthly` archive grouping.
            begin: First inclusive source date.
            end: Last inclusive source date.

        Returns:
            Validated query-string parameters.
        """
        native = _instrument_type(inst_type)
        values = _manifest_subjects(subjects)
        _validate_manifest_range(cadence, begin, end)
        _validate_manifest_combination(module, cadence, values)
        params = {
            "module": str(module),
            "instType": native,
            "dateAggrType": cadence,
            "begin": _epoch_milliseconds(begin),
            "end": _epoch_milliseconds(end),
        }
        params["instIdList" if native == "SPOT" else "instFamilyList"] = ",".join(
            values
        )
        return params

    def get_manifest(
        self,
        module: int,
        inst_type: str,
        subjects: Sequence[str],
        cadence: str,
        begin: date,
        end: date,
    ) -> list[dict[str, object]]:
        """Return bounded historical archive manifest groups.

        Args:
            module: Native historical data module.
            inst_type: Native instrument type.
            subjects: At most five native IDs, families, currencies, or `ANY`.
            cadence: `daily` or `monthly`.
            begin: First inclusive source date.
            end: Last inclusive source date.

        Returns:
            Source manifest response groups, including valid empty groups.
        """
        params = self._manifest_params(module, inst_type, subjects, cadence, begin, end)
        data = self.request(
            "/api/v5/public/market-data-history",
            policy_key="manifest",
            params=params,
        )
        return self._rows(data, "manifest")

    def paginate(
        self,
        path: str,
        *,
        policy_key: str,
        params: Mapping[str, str],
        cursor: Callable[[list[object]], str | None],
        page_size: int,
        max_pages: int,
        cursor_parameter: str = "after",
    ) -> list[object]:
        """Traverse a descending OKX history endpoint with a bounded cursor.

        Args:
            path: Public history endpoint path.
            policy_key: Endpoint quota key.
            params: Initial query parameters.
            cursor: Function deriving the next source cursor from one page.
            page_size: Maximum expected rows per page.
            max_pages: Hard safety bound on requests.
            cursor_parameter: Query parameter receiving the next cursor.

        Returns:
            Concatenated source rows in page order.
        """
        if page_size < 1 or max_pages < 1:
            raise ValueError("page_size and max_pages must be positive")
        query = dict(params)
        query["limit"] = str(page_size)
        found: list[object] = []
        seen: set[str] = set()
        for _page in range(max_pages):
            rows = self.request(path, policy_key=policy_key, params=query)
            found.extend(rows)
            if len(rows) < page_size:
                return found
            next_cursor = cursor(rows)
            if not next_cursor:
                return found
            if next_cursor in seen:
                raise OKXResponseError("cursor_loop", "pagination cursor repeated")
            seen.add(next_cursor)
            query[cursor_parameter] = next_cursor
        return found
