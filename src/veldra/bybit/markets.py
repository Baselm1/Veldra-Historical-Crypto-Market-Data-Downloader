"""Translate Bybit V5 metadata into Veldra market records."""

from collections.abc import Sequence
from datetime import UTC, datetime
import math
from typing import Protocol

from veldra.bybit.identities import SAFE_SYMBOL, category, normalized_symbol, symbol
from veldra.core.models import Market


class MarketClient(Protocol):
    """Describe the Bybit metadata calls used by market discovery."""

    def get_instruments(
        self,
        native_category: str,
        *,
        base_coin: str | None = None,
        status: str | None = None,
        max_pages: int = 20,
    ) -> list[dict[str, object]]:
        """Return current instruments for one native category."""
        raise NotImplementedError

    def get_tickers(self, native_category: str) -> list[dict[str, object]]:
        """Return current tickers for one native category."""
        raise NotImplementedError

    def get_option_base_coins(self) -> list[dict[str, object]]:
        """Return every currently published Option underlying."""
        raise NotImplementedError


def _text(value: object, field: str) -> str:
    """Return one nonempty source string.

    Args:
        value: Proposed metadata value.
        field: Field name used in validation errors.

    Returns:
        Stripped source string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Bybit market contains invalid {field}")
    return value.strip()


def _timestamp(value: object) -> datetime | None:
    """Parse one optional epoch-millisecond market timestamp.

    Args:
        value: Source timestamp or zero.

    Returns:
        UTC timestamp or ``None``.
    """
    if value is None or value == "" or value == "0" or value == 0:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Bybit market contains an invalid timestamp")
    try:
        epoch = float(value)
    except ValueError as error:
        raise ValueError("Bybit market contains an invalid timestamp") from error
    if not math.isfinite(epoch) or epoch < 0:
        raise ValueError("Bybit market contains an invalid timestamp")
    return datetime.fromtimestamp(epoch / 1_000, UTC)


def _contract_type(product: str, native: object) -> str | None:
    """Normalize Bybit's contract style.

    Args:
        product: Veldra product.
        native: Native contract type.

    Returns:
        Normalized contract style or ``None`` for Spot.
    """
    if product == "spot":
        return None
    if product == "options":
        return "OPTION"
    value = _text(native, "contract type").upper()
    return "PERPETUAL" if value.endswith("PERPETUAL") else "DELIVERY"


def parse_market(row: object, product: str) -> Market:
    """Parse one Bybit current instrument record.

    Args:
        row: Native instrument object.
        product: Veldra product receiving the market.

    Returns:
        Canonical market metadata.
    """
    category(product)
    if not isinstance(row, dict):
        raise ValueError("Bybit instrument endpoint contains an invalid market")
    native = symbol(row.get("symbol"))
    base = _text(row.get("baseCoin"), "base asset").upper()
    quote = _text(row.get("quoteCoin"), "quote asset").upper()
    status = _text(row.get("status"), "status").upper()
    return Market(
        symbol=native,
        normalized_symbol=normalized_symbol(native, base, quote, product),
        base_asset=base,
        quote_asset=quote,
        status=status,
        pair=f"{base}/{quote}",
        contract_type=_contract_type(product, row.get("contractType")),
        contract_size=None,
        onboard_time=_timestamp(row.get("launchTime")),
        delivery_time=_timestamp(row.get("deliveryTime")),
        source="bybit",
        product=product,
        active=status == "TRADING",
    )


def parse_volumes(rows: Sequence[object]) -> dict[str, float]:
    """Return finite nonnegative turnover indexed by native symbol.

    Args:
        rows: Native ticker records.

    Returns:
        Quote-turnover mapping for valid source rows.
    """
    volumes: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Bybit ticker endpoint contains an invalid market")
        source_symbol = row.get("symbol")
        if (
            isinstance(source_symbol, str)
            and SAFE_SYMBOL.fullmatch(source_symbol.upper()) is None
        ):
            continue
        native = symbol(source_symbol)
        raw = row.get("turnover24h")
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            raise ValueError("Bybit ticker endpoint contains an invalid turnover")
        try:
            volume = float(raw)
        except ValueError as error:
            raise ValueError(
                "Bybit ticker endpoint contains an invalid turnover"
            ) from error
        if not math.isfinite(volume) or volume < 0:
            raise ValueError("Bybit ticker endpoint contains an invalid turnover")
        volumes[native] = volume
    return volumes


def _option_bases(client: MarketClient) -> list[str]:
    """Discover Option underlyings through Bybit's dedicated directory.

    Args:
        client: Shared Bybit metadata client.

    Returns:
        Sorted unique native base assets.
    """
    bases: set[str] = set()
    for row in client.get_option_base_coins():
        if not isinstance(row, dict):
            raise ValueError("Bybit Option base directory contains an invalid row")
        base = symbol(row.get("baseCoin"))
        has_symbol = row.get("hasSymbol")
        if has_symbol in (1, "1"):
            bases.add(base)
    return sorted(bases)


def markets(client: MarketClient, product: str) -> list[Market]:
    """Return sorted current markets for one Bybit product.

    Args:
        client: Shared public Bybit client.
        product: Veldra Spot, derivative, or Option product.

    Returns:
        Current markets sorted by native symbol.
    """
    native_category = category(product)
    if product == "options":
        rows = [
            row
            for base in _option_bases(client)
            for row in client.get_instruments(native_category, base_coin=base)
        ]
    else:
        rows = client.get_instruments(native_category)
    found = [
        parse_market(row, product)
        for row in rows
        if isinstance(row.get("symbol"), str)
        and SAFE_SYMBOL.fullmatch(str(row["symbol"]).upper()) is not None
    ]
    symbols = [market.symbol for market in found]
    if len(symbols) != len(set(symbols)):
        raise ValueError("Bybit instrument endpoint contains a duplicate symbol")
    return sorted(found, key=lambda market: market.symbol)


def quote_volumes(client: MarketClient, product: str) -> dict[str, float]:
    """Return current quote turnover for one Bybit product.

    Args:
        client: Shared public Bybit client.
        product: Veldra product name.

    Returns:
        Turnover mapping indexed by native symbol.
    """
    return parse_volumes(client.get_tickers(category(product)))
