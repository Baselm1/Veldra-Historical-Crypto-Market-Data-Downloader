"""Provide tools for downloading and querying historical cryptocurrency data."""

import logging

from .binance import Binance
from .htx import HTX
from .kucoin import KuCoin
from .okx import OKX
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
    "KuCoin",
    "OKX",
    "Availability",
    "Gap",
    "Market",
    "Message",
    "MissingCandlesError",
    "Result",
)
