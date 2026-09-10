"""Provide tools for downloading and querying historical cryptocurrency data."""

import logging

from .binance import Binance
from .htx import HTX
from .core.models import (
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
    "HTX",
    "Availability",
    "Gap",
    "Market",
    "Message",
    "MissingCandlesError",
    "Result",
)
