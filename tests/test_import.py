"""Test that the package is available to Python."""

from importlib import import_module
import logging

from crypto_downloader.binance.facade import Binance
from crypto_downloader._core.models import (
    Availability,
    Gap,
    Market,
    Message,
    MissingCandlesError,
    Result,
)


def test_package_can_be_imported() -> None:
    """Confirm that Python can import the package and find its public exports."""
    package = import_module("crypto_downloader")

    assert package.__all__ == (
        "Binance",
        "Availability",
        "Gap",
        "Market",
        "Message",
        "MissingCandlesError",
        "Result",
    )
    assert package.Binance is Binance
    assert package.Availability is Availability
    assert package.Gap is Gap
    assert package.Market is Market
    assert package.Message is Message
    assert package.MissingCandlesError is MissingCandlesError
    assert package.Result is Result
    assert any(
        isinstance(handler, logging.NullHandler)
        for handler in logging.getLogger("crypto_downloader").handlers
    )


def test_package_root_hides_internal_services_and_helpers() -> None:
    """Confirm the package root exposes no generic or internal data service."""
    package = import_module("crypto_downloader")

    assert not hasattr(package, "RetrievalEngine")
    assert not hasattr(package, "BinanceConnector")
    assert not hasattr(package, "get_data")
    assert not hasattr(package, "get_results")
    assert not hasattr(package, "aget_data")
    assert not hasattr(package, "render_result")
    assert not hasattr(package, "render_results")
    assert not hasattr(package, "get_klines")
