"""Test that the package is available to Python."""

from importlib import import_module


def test_package_can_be_imported() -> None:
    """Confirm that Python can import the package and find its public exports."""
    package = import_module("crypto_downloader")

    assert package.__all__ == ()
