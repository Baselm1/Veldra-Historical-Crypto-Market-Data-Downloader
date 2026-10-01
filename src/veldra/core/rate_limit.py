"""Share rolling-window request limits across source worker threads."""

from collections import defaultdict, deque
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
import math
from threading import Condition
import time


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


def _request_cost(policy: RatePolicy, cost: int) -> int:
    """Return a valid request cost for one policy.

    Args:
        policy: Quota applied to the request.
        cost: Proposed number of reserved units.

    Returns:
        The validated request cost.
    """
    if isinstance(cost, bool) or not isinstance(cost, int) or cost < 1:
        raise ValueError("rate-limit cost must be a positive integer")
    if cost > policy.capacity:
        raise ValueError("rate-limit cost exceeds policy capacity")
    return cost


def _expire(reservations: deque[float], cutoff: float) -> None:
    """Remove reservations outside the rolling window.

    Args:
        reservations: Ordered reservation timestamps.
        cutoff: Oldest timestamp retained by the window.
    """
    while reservations and reservations[0] <= cutoff:
        reservations.popleft()


def _wait_time(
    policy: RatePolicy,
    reservations: deque[float],
    penalty: float,
    now: float,
    cost: int,
) -> float:
    """Return the shortest wait required by a quota or server penalty.

    Args:
        policy: Quota applied to the request.
        reservations: Current rolling-window reservations.
        penalty: Time before which the server bucket remains paused.
        now: Current monotonic time.
        cost: Number of units requested.

    Returns:
        A positive wait in seconds.
    """
    waits = [penalty - now] if penalty > now else []
    if len(reservations) + cost > policy.capacity:
        index = len(reservations) + cost - policy.capacity - 1
        waits.append(reservations[index] + policy.window_seconds - now)
    return max(0.001, min(waits)) + policy.safety_seconds


class RollingWindowRateLimiter:
    """Share keyed rolling-window request limits across worker threads."""

    def __init__(
        self,
        policies: Mapping[str, RatePolicy],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a limiter from endpoint policies.

        Args:
            policies: Endpoint policies keyed by quota family.
            clock: Monotonic clock used to age reservations.
        """
        self.policies = dict(policies)
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
        cost = _request_cost(policy, cost)
        with self._condition:
            while True:
                now = self._clock()
                reservations = self._reservations[key]
                _expire(reservations, now - policy.window_seconds)
                penalty = self._penalties.get(key, 0.0)
                if penalty <= now and len(reservations) + cost <= policy.capacity:
                    reservations.extend([now] * cost)
                    return
                self._condition.wait(
                    _wait_time(policy, reservations, penalty, now, cost)
                )

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
