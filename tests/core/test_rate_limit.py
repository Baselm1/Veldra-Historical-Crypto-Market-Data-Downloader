"""Test the shared rolling-window request limiter."""

import pytest

from veldra.core.rate_limit import RatePolicy, RollingWindowRateLimiter


def test_policy_rejects_invalid_values() -> None:
    """Confirm invalid capacities and time windows fail immediately."""
    with pytest.raises(ValueError, match="capacity"):
        RatePolicy(0, 1)
    with pytest.raises(ValueError, match="window"):
        RatePolicy(1, float("nan"))
    with pytest.raises(ValueError, match="safety"):
        RatePolicy(1, 1, -0.1)


def test_limiter_resolves_qualified_keys_and_validates_costs() -> None:
    """Confirm qualified buckets share policies and reject invalid costs."""
    limiter = RollingWindowRateLimiter({"history": RatePolicy(2, 1)})
    limiter.acquire("history:BTCUSDT")
    with pytest.raises(KeyError, match="policy"):
        limiter.acquire("missing")
    with pytest.raises(ValueError, match="positive"):
        limiter.acquire("history", cost=0)
    with pytest.raises(ValueError, match="capacity"):
        limiter.acquire("history", cost=3)


def test_limiter_rejects_invalid_penalties() -> None:
    """Confirm server penalties must be finite and nonnegative."""
    limiter = RollingWindowRateLimiter({"portal": RatePolicy(1, 1)})
    with pytest.raises(ValueError, match="finite"):
        limiter.penalize("portal", float("inf"))
    with pytest.raises(ValueError, match="nonnegative"):
        limiter.penalize("portal", -1)
