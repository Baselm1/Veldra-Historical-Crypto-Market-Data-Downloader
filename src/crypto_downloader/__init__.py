"""Provide tools for downloading and querying historical cryptocurrency data."""

import logging

from .downloader import Downloader, get_data, get_results
from .display import render_result, render_results
from .models import Gap, Message, MissingCandlesError, Result

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__: tuple[str, ...] = (
    "Downloader",
    "Gap",
    "Message",
    "MissingCandlesError",
    "Result",
    "get_data",
    "get_results",
    "render_result",
    "render_results",
)
