"""Test that the package is available to Python."""

from importlib import import_module
import logging


def test_package_can_be_imported() -> None:
    """Confirm that Python can import the package and find its public exports."""
    package = import_module("crypto_downloader")

    assert package.__all__ == (
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
    assert any(
        isinstance(handler, logging.NullHandler)
        for handler in logging.getLogger("crypto_downloader").handlers
    )
