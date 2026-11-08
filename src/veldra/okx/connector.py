"""Discover current OKX markets through the typed public client."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
import logging
import math

import httpx

from veldra.core.matching import rank_markets
from veldra.core.models import Market
from veldra.core.request import normalize_pair
from veldra.okx.client import OKXClient, OKXRateLimiter, OKXResponseError
from veldra.okx.identities import OKXInstrument, OKXProduct, parse_instrument

PRODUCTS: tuple[OKXProduct, ...] = (
    "spot",
    "margin",
    "linear_swap",
    "inverse_swap",
    "linear_futures",
    "inverse_futures",
    "options",
)
_NATIVE_TYPES: dict[OKXProduct, str] = {
    "spot": "SPOT",
    "margin": "MARGIN",
    "linear_swap": "SWAP",
    "inverse_swap": "SWAP",
    "linear_futures": "FUTURES",
    "inverse_futures": "FUTURES",
    "options": "OPTION",
}
LOGGER = logging.getLogger(__name__)


def _historical_spot_market(symbol: str) -> Market:
    """Represent an archive selector entry without inferring active trading."""
    base, _, quote = symbol.rpartition("-")
    return Market(
        symbol=symbol,
        normalized_symbol=normalize_pair(symbol),
        base_asset=base,
        quote_asset=quote,
        status="archive_only",
        pair=symbol,
        source="okx",
        product="spot",
        active=False,
    )


class OKXConnector:
    """Resolve OKX product-scoped current instrument identities."""

    code = "okx"
    products = PRODUCTS
    max_concurrency = 64
    monthly_datasets = frozenset({"klines", "trades", "funding_rates"})

    def __init__(
        self, *, timeout: float = 30, retries: int = 3, backoff: float = 0.5
    ) -> None:
        """Create one connector and source-wide limiter.

        Args:
            timeout: Per-attempt public API timeout.
            retries: Retries following a transient failure.
            backoff: Initial retry delay.
        """
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.limiter = OKXRateLimiter()
        self._identities: dict[tuple[str, str], OKXInstrument] = {}

    def _api(self, client: httpx.Client) -> OKXClient:
        """Wrap a shared HTTPX pool with the connector's limiter.

        Args:
            client: Shared source connection pool.

        Returns:
            A configured OKX public client.
        """
        return OKXClient(
            client=client,
            limiter=self.limiter,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    @staticmethod
    def _product(value: str) -> OKXProduct:
        """Validate one Veldra OKX product.

        Args:
            value: Proposed product name.

        Returns:
            Supported OKX product.
        """
        if value not in PRODUCTS:
            raise ValueError(f"unsupported OKX product {value!r}")
        return value

    def identities(self, client: httpx.Client, product: str) -> list[OKXInstrument]:
        """Return current typed identities for one product.

        Args:
            client: Shared source connection pool.
            product: Veldra OKX product.

        Returns:
            Current instruments in native ID order.
        """
        selected = self._product(product)
        api = self._api(client)
        native = _NATIVE_TYPES[selected]
        rows: list[dict[str, object]] = []
        if selected == "options":
            for family in api.get_underlyings("OPTION"):
                rows.extend(api.get_instruments("OPTION", inst_family=family))
        else:
            rows = api.get_instruments(native)
        parsed = [parse_instrument(row) for row in rows]
        filtered = [item for item in parsed if item.product == selected]
        symbols = [item.instrument_id for item in filtered]
        if len(symbols) != len(set(symbols)):
            raise ValueError("OKX market endpoint contains a duplicate instrument")
        ordered = sorted(filtered, key=lambda item: item.instrument_id)
        self._identities.update(
            {(selected, item.instrument_id): item for item in ordered}
        )
        return ordered

    def markets(
        self,
        client: httpx.Client,
        product: str,
        *,
        historical_fallback: Sequence[Market] = (),
    ) -> list[Market]:
        """Return current markets and historical Spot instruments.

        Args:
            client: Shared source connection pool.
            product: Veldra OKX product.

        Returns:
            Product markets in native ID order. Archive-only IDs have no
            implied file coverage.
        """
        markets = {
            item.instrument_id: item.market for item in self.identities(client, product)
        }
        if product == "spot":
            self._add_historical_spot(client, markets, historical_fallback)
        return sorted(markets.values(), key=lambda market: market.symbol)

    def _add_historical_spot(
        self,
        client: httpx.Client,
        markets: dict[str, Market],
        fallback: Sequence[Market],
    ) -> None:
        """Merge archive IDs while keeping current trading metadata."""
        try:
            symbols = self._api(client).get_historical_spot_instruments()
        except (httpx.HTTPError, OKXResponseError) as error:
            LOGGER.warning("OKX historical Spot instrument selector failed: %s", error)
            symbols = [
                market.symbol for market in fallback if market.status == "archive_only"
            ]
        current = {symbol.upper() for symbol in markets}
        for symbol in symbols:
            if symbol.upper() not in current:
                markets[symbol] = _historical_spot_market(symbol)

    def quote_volumes(self, client: httpx.Client, product: str) -> dict[str, float]:
        """Return finite rolling volume values indexed by native ID.

        Args:
            client: Shared source connection pool.
            product: Veldra OKX product.

        Returns:
            Nonnegative 24-hour volume values.
        """
        selected = self._product(product)
        native = _NATIVE_TYPES[selected]
        values: dict[str, float] = {}
        for row in self._api(client).get_tickers(native):
            symbol = row.get("instId")
            raw = row.get("volCcy24h")
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("OKX ticker contains an invalid instrument ID")
            try:
                volume = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError) as error:
                raise ValueError("OKX ticker contains an invalid volume") from error
            if not math.isfinite(volume) or volume < 0:
                raise ValueError("OKX ticker contains an invalid volume")
            identity = self._identities.get((selected, symbol))
            if identity is None or identity.product == selected:
                values[symbol] = volume
                if identity is not None:
                    self._identities[(selected, symbol)] = replace(
                        identity, quote_volume_24h=volume
                    )
        return values

    def suggest(
        self, client: httpx.Client, query: str, product: str, *, limit: int = 3
    ) -> list[str]:
        """Return product-scoped fuzzy instrument suggestions.

        Args:
            client: Shared source connection pool.
            query: Mistyped instrument text.
            product: Product whose identities are valid.
            limit: Maximum suggestions.

        Returns:
            Similar native IDs in descending relevance.
        """
        markets = self.markets(client, product)
        return [market.symbol for market in rank_markets(query, markets, limit=limit)]
