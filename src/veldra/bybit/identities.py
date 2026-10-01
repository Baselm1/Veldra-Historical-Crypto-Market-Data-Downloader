"""Map Bybit products and symbols to stable Veldra identities."""

from collections.abc import Mapping
import re

from veldra.bybit.datasets import PRODUCTS
from veldra.core.request import normalize_pair

NATIVE_CATEGORIES: Mapping[str, str] = {
    "spot": "spot",
    "linear": "linear",
    "inverse": "inverse",
    "options": "option",
}
SAFE_SYMBOL = re.compile(r"[A-Z0-9_-]+")


def category(product: object) -> str:
    """Return Bybit's native category for one Veldra product.

    Args:
        product: Proposed public product name.

    Returns:
        Native V5 API category.
    """
    if not isinstance(product, str):
        raise TypeError("product must be a string")
    if product not in PRODUCTS:
        raise ValueError(f"unsupported Bybit product {product!r}")
    return NATIVE_CATEGORIES[product]


def symbol(value: object) -> str:
    """Return one safe uppercase Bybit instrument symbol.

    Args:
        value: Proposed native symbol.

    Returns:
        Validated native instrument symbol.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Bybit market contains an invalid symbol")
    native = value.strip().upper()
    if SAFE_SYMBOL.fullmatch(native) is None:
        raise ValueError("Bybit market contains an unsafe symbol")
    return native


def normalized_symbol(native_symbol: str, base: str, quote: str, product: str) -> str:
    """Return a searchable identity without collapsing dated instruments.

    Args:
        native_symbol: Validated native symbol.
        base: Base asset.
        quote: Quote asset.
        product: Veldra product.

    Returns:
        Pair identity for Spot/perpetuals or full identity for dated products.
    """
    category(product)
    compact = normalize_pair(native_symbol)
    pair = normalize_pair(f"{base}{quote}")
    return pair if compact == pair else compact
