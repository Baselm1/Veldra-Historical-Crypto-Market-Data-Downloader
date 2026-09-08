"""Provide tools for downloading and querying historical cryptocurrency data."""

import logging

from .binance import Binance
from ._core.models import (
    Availability,
    Gap,
    Market,
    Message,
    MissingCandlesError,
    Result,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__: tuple[str, ...] = (
    "Binance",
    "Availability",
    "Gap",
    "Market",
    "Message",
    "MissingCandlesError",
    "Result",
)
