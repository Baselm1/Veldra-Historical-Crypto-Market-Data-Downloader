"""Translate Bitget product metadata into Veldra market records."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import math
import re
from typing import Protocol

from veldra.bitget.datasets import PRODUCTS
from veldra.core.models import Market
from veldra.core.request import normalize_pair

PRODUCT_CATEGORIES: Mapping[str, str] = {
    "spot": "SPOT",
    "usdt_futures": "USDT-FUTURES",
    "usdc_futures": "USDC-FUTURES",
    "coin_futures": "COIN-FUTURES",
}
_SAFE_SYMBOL = re.compile(r"[A-Z0-9_-]+")


class MarketClient(Protocol):
    """Describe the Bitget metadata calls used by market discovery."""

    def get_instruments(self, native_category: str) -> list[dict[str, object]]:
        """Return current instruments for one native category."""
        raise NotImplementedError

    def get_tickers(self, native_category: str) -> list[dict[str, object]]:
        """Return current tickers for one native category."""
        raise NotImplementedError


def category(product: object) -> str:
    """Return the native category for one public product.

    Args:
        product: Proposed Veldra product name.

    Returns:
        Bitget's native API category.
    """
    if not isinstance(product, str):
        raise TypeError("product must be a string")
    if product not in PRODUCTS:
        raise ValueError(f"unsupported Bitget product {product!r}")
    return PRODUCT_CATEGORIES[product]


def _text(value: object, field: str) -> str:
    """Return one nonempty metadata string.

    Args:
        value: Proposed source value.
        field: Field name used in validation errors.

    Returns:
        The stripped source string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Bitget market contains invalid {field}")
    return value.strip()


def _symbol(value: object) -> str:
    """Return one safe uppercase Bitget instrument symbol.

    Args:
        value: Proposed source symbol.

    Returns:
        Validated native instrument symbol.
    """
    symbol = _text(value, "symbol").upper()
    if _SAFE_SYMBOL.fullmatch(symbol) is None:
        raise ValueError("Bitget market contains an unsafe symbol")
    return symbol


def _timestamp(value: object) -> datetime | None:
    """Parse an optional epoch-millisecond metadata timestamp.

    Args:
        value: Source timestamp or an empty value.

    Returns:
        A UTC timestamp or ``None``.
    """
    if value is None or value == "" or value == "0" or value == 0:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Bitget market contains an invalid timestamp")
    try:
        epoch = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Bitget market contains an invalid timestamp") from error
    if not math.isfinite(epoch) or epoch < 0:
        raise ValueError("Bitget market contains an invalid timestamp")
    return datetime.fromtimestamp(epoch / 1_000, UTC)


def _number(value: object) -> float | None:
    """Parse an optional positive contract size.

    Args:
        value: Source multiplier or an empty value.

    Returns:
        A positive finite number or ``None``.
    """
    if value is None or value == "" or value == "0" or value == 0:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Bitget market contains an invalid contract size")
    try:
        number = abs(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError("Bitget market contains an invalid contract size") from error
    if not math.isfinite(number) or number == 0:
        raise ValueError("Bitget market contains an invalid contract size")
    return number


def parse_market(row: object, product: str) -> Market:
    """Parse one Bitget current instrument record.

    Args:
        row: Native instrument object.
        product: Veldra product receiving the market.

    Returns:
        Canonical market metadata.
    """
    category(product)
    if not isinstance(row, dict):
        raise ValueError("Bitget instrument endpoint contains an invalid market")
    symbol = _symbol(row.get("symbol"))
    base = _text(row.get("baseCoin"), "base asset").upper()
    quote = _text(row.get("quoteCoin"), "quote asset").upper()
    status = _text(row.get("status"), "status").upper()
    delivery = _timestamp(row.get("deliveryTime"))
    contract_type = None
    if product != "spot":
        contract_type = "DELIVERY" if delivery is not None else "PERPETUAL"
    return Market(
        symbol=symbol,
        normalized_symbol=normalize_pair(f"{base}{quote}"),
        base_asset=base,
        quote_asset=quote,
        status=status,
        pair=f"{base}/{quote}",
        contract_type=contract_type,
        contract_size=(
            None if product == "spot" else _number(row.get("sizeMultiplier"))
        ),
        onboard_time=_timestamp(row.get("launchTime")),
        delivery_time=delivery,
        source="bitget",
        product=product,
        active=status in {"ONLINE", "NORMAL", "TRADING"},
    )


def parse_volumes(rows: Sequence[object]) -> dict[str, float]:
    """Return finite nonnegative quote turnover indexed by symbol.

    Args:
        rows: Native public ticker records.

    Returns:
        Quote-volume mapping for valid rows.
    """
    volumes: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Bitget ticker endpoint contains an invalid market")
        symbol = _symbol(row.get("symbol"))
        raw = row.get("quoteVolume", row.get("usdtVolume"))
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            raise ValueError("Bitget ticker endpoint contains an invalid volume")
        try:
            volume = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Bitget ticker endpoint contains an invalid volume"
            ) from error
        if not math.isfinite(volume) or volume < 0:
            raise ValueError("Bitget ticker endpoint contains an invalid volume")
        volumes[symbol] = volume
    return volumes


def markets(client: MarketClient, product: str) -> list[Market]:
    """Return sorted current markets for one Bitget product.

    Args:
        client: Shared public Bitget client.
        product: Veldra Spot or settlement product.

    Returns:
        Current markets sorted by native symbol.
    """
    rows = client.get_instruments(category(product))
    found = [
        parse_market(row, product)
        for row in rows
        if isinstance(row.get("symbol"), str)
        and _SAFE_SYMBOL.fullmatch(str(row["symbol"]).upper()) is not None
    ]
    symbols = [market.symbol for market in found]
    if len(symbols) != len(set(symbols)):
        raise ValueError("Bitget instrument endpoint contains a duplicate symbol")
    return sorted(found, key=lambda market: market.symbol)


def quote_volumes(client: MarketClient, product: str) -> dict[str, float]:
    """Return current quote turnover for one Bitget product.

    Args:
        client: Shared public Bitget client.
        product: Veldra Spot or settlement product.

    Returns:
        Quote-volume mapping indexed by native symbol.
    """
    return parse_volumes(client.get_tickers(category(product)))
