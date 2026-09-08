"""Discover HTX markets and the sparse union of its daily archives."""

from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
import logging
import math
from pathlib import Path
import re
from typing import Literal
from urllib.parse import quote

import httpx

from crypto_downloader.core.datasets import DatasetSpec
from crypto_downloader.core.download import archive_checksum, get
from crypto_downloader.core.ingest import ingest_archive
from crypto_downloader.core.models import (
    IngestedResource,
    Market,
    Resource,
    ResourceKey,
)
from crypto_downloader.core.portal import pages
from crypto_downloader.core.request import normalize_pair
from crypto_downloader.htx.datasets import (
    ArchiveRoute,
    NEW_INTERVALS,
    OLD_INTERVALS,
    PRODUCTS,
    supports,
)
from crypto_downloader.htx.processing import normalize_chunk, validate_chunk

LISTING_URL = "https://www.htx.com/data/"
ARCHIVE_URL = "https://www.htx.com/data"
MARKET_URLS: Mapping[str, str] = {
    "spot": "https://api.huobi.pro/v2/settings/common/symbols",
    "linear_swap": "https://api.hbdm.vn/linear-swap-api/v1/swap_contract_info",
    "coin_swap": "https://api.hbdm.vn/swap-api/v1/swap_contract_info",
}
TICKER_URL = "https://api.huobi.pro/market/tickers"
ARCHIVE_DAY_OFFSET = timedelta(hours=8)
LOGGER = logging.getLogger(__name__)
_SAFE_SYMBOL = re.compile(r"[A-Za-z0-9-]+")
_DATE = r"(\d{4}-\d{2}-\d{2})"


class HTXConnector:
    """Discover source metadata from HTX's APIs and two archive trees."""

    code = "htx"
    products = PRODUCTS
    archive_day_offset = ARCHIVE_DAY_OFFSET
    max_concurrency = 32
    monthly_datasets: frozenset[str] = frozenset()

    def __init__(
        self, *, timeout: float = 30.0, retries: int = 3, backoff: float = 0.5
    ) -> None:
        """Store HTTP settings used for HTX metadata requests.

        Args:
            timeout: The timeout for each request in seconds.
            retries: The retries after the first request attempt.
            backoff: The initial retry delay in seconds.
        """
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _get(
        self,
        client: httpx.Client,
        url: str,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """Make one retrying HTX metadata request.

        Args:
            client: The shared HTTP client.
            url: The endpoint to request.
            params: Optional query values.

        Returns:
            A successful HTTP response.
        """
        return get(
            client,
            url,
            params=params,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def checksum(self, client: httpx.Client, resource: Resource) -> str:
        """Return HTX's SHA-256 digest for one archive.

        Args:
            client: The shared HTTP client.
            resource: The HTX archive whose sidecar is checked.

        Returns:
            The lowercase SHA-256 digest declared by HTX.
        """
        return archive_checksum(
            client,
            resource,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return current and archive-only HTX perpetual or Spot markets.

        Args:
            client: The shared HTTP client.
            product: The HTX product to inspect.

        Returns:
            The complete known market snapshot ordered by native symbol.
        """
        self._check_product(product)
        payload = self._get(client, MARKET_URLS[product]).json()
        current = self._current_markets(payload, product)
        archive = self._archive_markets(client, product)
        merged = self._merge_markets(current, archive, product)
        LOGGER.info(
            "HTX markets loaded: product=%s exchange=%d archive=%d merged=%d",
            product,
            len(current),
            len(archive),
            len(merged),
        )
        return merged

    def quote_volumes(self, client: httpx.Client, product: str) -> dict[str, float]:
        """Return rolling Spot quote turnover indexed by native symbol.

        Args:
            client: The shared HTTP client.
            product: The HTX product to inspect.

        Returns:
            Nonnegative quote turnover for every valid Spot ticker.
        """
        self._check_product(product)
        if product != "spot":
            raise ValueError("HTX quote-volume ranking is supported only for spot")
        volumes: dict[str, float] = {}
        for value in self._ticker_rows(self._get(client, TICKER_URL).json()):
            parsed = self._quote_volume(value)
            if parsed is None:
                continue
            symbol, volume = parsed
            if symbol in volumes:
                raise ValueError("ticker contains an invalid market")
            volumes[symbol] = volume
        return volumes

    @staticmethod
    def _ticker_rows(payload: object) -> list[object]:
        """Return the rows in one HTX ticker snapshot.

        Args:
            payload: The decoded public ticker response.

        Returns:
            The source ticker rows.
        """
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("ticker contains no market snapshot")
        return rows

    @staticmethod
    def _quote_volume(value: object) -> tuple[str, float] | None:
        """Parse one HTX Spot ticker row.

        Args:
            value: The decoded source ticker object.

        Returns:
            Its symbol and quote turnover, or ``None`` for non-ASCII symbols.
        """
        if not isinstance(value, dict):
            raise ValueError("ticker contains an invalid market")
        symbol = HTXConnector._safe_symbol(value.get("symbol"), compact=True)
        if symbol is None:
            return None
        raw_volume = value.get("vol")
        if isinstance(raw_volume, bool) or not isinstance(raw_volume, (int, float)):
            raise ValueError("ticker contains an invalid market")
        volume = float(raw_volume)
        if not math.isfinite(volume) or volume < 0:
            raise ValueError("ticker contains an invalid market")
        return symbol, volume

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return the preferred HTX resources in an inclusive source-day range.

        Args:
            client: The shared HTTP client.
            key: The requested HTX dataset identity.
            start_day: The first HTX source day to include.
            end_day: The last HTX source day to include.

        Returns:
            Old-only and new-preferred resources ordered by source day.
        """
        self._validate_resource_request(key, start_day, end_day)
        found: dict[date, Resource] = {}
        for route in self._routes(key):
            found.update(self._route_resources(client, key, route, start_day, end_day))
        result = [found[day] for day in sorted(found)]
        LOGGER.info(
            "HTX resources listed: product=%s dataset=%s symbol=%s range=[%s, %s] "
            "resources=%d",
            key.product,
            key.dataset,
            key.symbol,
            start_day,
            end_day,
            len(result),
        )
        return result

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Return HTX's first preferred resource within broad source-day bounds.

        Args:
            client: The shared HTTP client.
            key: The requested HTX dataset identity.
            start_day: The first acceptable source day, or all history.
            end_day: The last acceptable source day.

        Returns:
            The earliest matching resource, preferring the new tree on overlap.
        """
        self._validate_resource_request(key, start_day, end_day)
        candidates = [
            resource
            for route in self._routes(key)
            if (
                resource := self._first_route_resource(
                    client, key, route, start_day, end_day
                )
            )
            is not None
        ]
        if not candidates:
            return None
        first_day = min(resource.day for resource in candidates)
        matching = [resource for resource in candidates if resource.day == first_day]
        return matching[-1]

    def ingest(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Convert one verified HTX CSV archive into Parquet.

        Args:
            client: The shared HTTP client.
            resource: The HTX archive to ingest.
            dataset: The schema used to interpret its CSV.
            destination: The final Parquet path.

        Returns:
            Integrity and range metadata for the cached Parquet file.
        """
        return ingest_archive(
            client,
            resource,
            dataset,
            destination,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
            normalizer=normalize_chunk,
            validator=validate_chunk,
        )

    def _route_resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        route: ArchiveRoute,
        start_day: date,
        end_day: date,
    ) -> dict[date, Resource]:
        """List matching resources from one physical archive route.

        Args:
            client: The shared HTTP client.
            key: The requested dataset identity.
            route: The physical archive route to list.
            start_day: The first source day to include.
            end_day: The last source day to include.

        Returns:
            Resources indexed by source day.
        """
        pattern = re.compile(
            re.escape(route.prefix + route.stem) + _DATE + re.escape(route.suffix)
        )
        marker = f"{route.prefix}{route.stem}{start_day.isoformat()}"
        found: dict[date, Resource] = {}
        max_keys = min(1_000, 2 * ((end_day - start_day).days + 2))
        for keys, _ in self._pages(
            client, route.prefix, marker=marker, max_keys=max_keys
        ):
            past_end = False
            for object_key in keys:
                day = self._resource_day(object_key, pattern)
                if day is None:
                    continue
                if day > end_day:
                    past_end = True
                elif day >= start_day:
                    found[day] = self._resource(day, object_key, key, route)
            if past_end:
                break
        return found

    def _first_route_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        route: ArchiveRoute,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Find the first resource in one physical archive route.

        Args:
            client: The shared HTTP client.
            key: The requested dataset identity.
            route: The physical archive route to search.
            start_day: The optional earliest source day.
            end_day: The latest source day.

        Returns:
            The first resource in the route or ``None``.
        """
        marker = (
            f"{route.prefix}{route.stem}{start_day.isoformat()}"
            if start_day is not None
            else None
        )
        pattern = re.compile(
            re.escape(route.prefix + route.stem) + _DATE + re.escape(route.suffix)
        )
        for keys, _ in self._pages(client, route.prefix, marker=marker, max_keys=4):
            for object_key in keys:
                day = self._resource_day(object_key, pattern)
                if day is None or (start_day is not None and day < start_day):
                    continue
                return (
                    None
                    if day > end_day
                    else self._resource(day, object_key, key, route)
                )
        return None

    @staticmethod
    def _resource_day(object_key: str, pattern: re.Pattern[str]) -> date | None:
        """Extract a valid source day from an exact archive object key.

        Args:
            object_key: The object key returned by HTX.
            pattern: The exact route pattern to match.

        Returns:
            The parsed source day, or ``None`` for unrelated keys.
        """
        match = pattern.fullmatch(object_key)
        if match is None:
            return None
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _resource(
        day: date, object_key: str, key: ResourceKey, route: ArchiveRoute
    ) -> Resource:
        """Build one resource with exact UTC+8 coverage.

        Args:
            day: The HTX source calendar day.
            object_key: The physical archive object key.
            key: The requested canonical identity.
            route: The matching archive route.

        Returns:
            A discovered HTX archive resource.
        """
        url = f"{ARCHIVE_URL}/{quote(object_key, safe='/')}"
        checksum_url = (
            f"{url}.CHECKSUM"
            if route.generation == "new"
            else f"{url.removesuffix(route.suffix)}.CHECKSUM"
        )
        start = datetime.combine(day, time.min, UTC) - ARCHIVE_DAY_OFFSET
        timestamp_columns = {
            "klines": "open_time",
            "trades": "event_time",
            "index_price_klines": "open_time",
            "mark_price_klines": "open_time",
            "funding_rates": "funding_time",
            "order_book_updates": "event_time",
        }
        return Resource(
            day=day,
            url=url,
            checksum_url=checksum_url,
            archive_symbol=key.archive_symbol or key.symbol,
            timestamp_column=timestamp_columns[key.dataset],
            coverage_start=start,
            coverage_end=start + timedelta(days=1),
        )

    @staticmethod
    def _routes(key: ResourceKey) -> tuple[ArchiveRoute, ...]:
        """Return every physical archive route for one canonical key.

        Args:
            key: The requested HTX dataset identity.

        Returns:
            Old routes first and the preferred new route last.
        """
        old_symbol = key.symbol
        new_symbol = key.archive_symbol or key.symbol
        routes: list[ArchiveRoute] = []
        old_product = {
            "spot": "spot",
            "linear_swap": "linear-swap",
            "coin_swap": "swap",
        }[key.product]
        if key.dataset in {"klines", "trades"}:
            if key.dataset == "klines":
                assert key.interval is not None
                interval = OLD_INTERVALS[key.interval]
                prefix = f"data/klines/{old_product}/daily/{old_symbol}/{interval}/"
                stem = f"{old_symbol}-{interval}-"
            else:
                prefix = f"data/trades/{old_product}/daily/{old_symbol}/"
                stem = f"{old_symbol}-trades-"
            routes.append(ArchiveRoute("old", prefix, stem))

        routes.append(HTXConnector._new_route(key, new_symbol))
        return tuple(routes)

    @staticmethod
    def _new_route(key: ResourceKey, symbol: str) -> ArchiveRoute:
        """Build the new archive route for one canonical resource key.

        Args:
            key: The requested HTX dataset identity.
            symbol: The symbol used by the new archive tree.

        Returns:
            The matching new-generation folder and filename declaration.
        """
        product_root = "spot" if key.product == "spot" else "futures"
        folders = {
            "klines": ("klines", "klines"),
            "trades": ("trades", "trades"),
            "index_price_klines": ("index-klines", "index-klines"),
            "mark_price_klines": ("mark-klines", "mark-price-klines"),
            "funding_rates": ("funding-rates", "fundingRates"),
            "order_book_updates": (
                "orderbook/lv400" if key.product == "spot" else "orderbook/lv150",
                "l2orderbook-400lv" if key.product == "spot" else "l2orderbook-150lv",
            ),
        }
        folder, file_name = folders[key.dataset]
        root = f"historical_data/{product_root}/daily/{folder}/{symbol}/"
        suffix: Literal[".zip", ".tar.gz"] = (
            ".tar.gz" if key.dataset == "order_book_updates" else ".zip"
        )
        if key.dataset.endswith("klines"):
            assert key.interval is not None
            interval = NEW_INTERVALS[key.interval]
            root = f"{root}{interval}/"
            stem = f"{symbol}-{file_name}-{interval}-"
        else:
            stem = f"{symbol}-{file_name}-"
        return ArchiveRoute("new", root, stem, suffix)

    @staticmethod
    def _current_markets(payload: object, product: str) -> dict[str, Market]:
        """Parse one complete HTX market API response.

        Args:
            payload: The decoded public endpoint response.
            product: The represented HTX product.

        Returns:
            Valid perpetual or Spot markets indexed by native symbol.
        """
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("HTX market endpoint contains no market snapshot")
        markets: dict[str, Market] = {}
        for value in rows:
            market = (
                HTXConnector._spot_market(value)
                if product == "spot"
                else HTXConnector._swap_market(value, product)
            )
            if market is None:
                continue
            if market.symbol in markets:
                raise ValueError("HTX market endpoint contains a duplicate symbol")
            markets[market.symbol] = market
        return markets

    @staticmethod
    def _spot_market(value: object) -> Market | None:
        """Parse one HTX Spot market row.

        Args:
            value: The decoded Spot market object.

        Returns:
            A canonical market or ``None`` for an unsupported symbol.
        """
        if not isinstance(value, dict):
            raise ValueError("HTX market endpoint contains an invalid market")
        symbol = HTXConnector._safe_symbol(value.get("sc"), compact=True)
        if symbol is None:
            return None
        base = HTXConnector._required_symbol(value, "bc", compact=True)
        quote_asset = HTXConnector._required_symbol(value, "qc", compact=True)
        status = HTXConnector._required_text(value, "state")
        return Market(
            symbol=symbol,
            normalized_symbol=normalize_pair(symbol),
            base_asset=base,
            quote_asset=quote_asset,
            status=status,
            pair=f"{base}-{quote_asset}",
            onboard_time=HTXConnector._source_time(value.get("toa"), milliseconds=True),
            active=status == "online",
        )

    @staticmethod
    def _swap_market(value: object, product: str) -> Market | None:
        """Parse one active or historical HTX perpetual market row.

        Args:
            value: The decoded derivative contract object.
            product: The represented perpetual product.

        Returns:
            A canonical perpetual market or ``None`` for a dated contract.
        """
        if not isinstance(value, dict):
            raise ValueError("HTX market endpoint contains an invalid market")
        contract_type = (
            "swap"
            if product == "coin_swap" and value.get("contract_type") is None
            else HTXConnector._required_text(value, "contract_type")
        )
        if contract_type != "swap":
            return None
        symbol = HTXConnector._safe_symbol(value.get("contract_code"), compact=False)
        if symbol is None:
            return None
        status_value = value.get("contract_status")
        if (
            isinstance(status_value, bool)
            or not isinstance(status_value, int)
            or not 0 <= status_value <= 9
        ):
            raise ValueError("HTX market endpoint contains an invalid market")
        raw_size = value.get("contract_size")
        if isinstance(raw_size, bool) or not isinstance(raw_size, (int, float)):
            raise ValueError("HTX market endpoint contains an invalid market")
        contract_size = float(raw_size)
        if not math.isfinite(contract_size) or contract_size <= 0:
            raise ValueError("HTX market endpoint contains an invalid market")
        base, quote_asset = symbol.split("-", 1)
        return Market(
            symbol=symbol,
            normalized_symbol=normalize_pair(symbol),
            base_asset=base,
            quote_asset=quote_asset,
            status=str(status_value),
            pair=f"{symbol}-PERP",
            contract_type="PERPETUAL",
            contract_size=contract_size,
            onboard_time=HTXConnector._date_time(value.get("create_date")),
            delivery_time=HTXConnector._source_time(
                value.get("delivery_time"), milliseconds=True, optional=True
            ),
            active=status_value == 1,
            product=product,
        )

    def _archive_markets(self, client: httpx.Client, product: str) -> dict[str, str]:
        """Return public symbols and new archive symbols found in Kline folders.

        Args:
            client: The shared HTTP client.
            product: The HTX product to inspect.

        Returns:
            New archive symbols indexed by canonical public symbol.
        """
        found: dict[str, str] = {}
        old_root = self._old_market_root(product)
        for _, prefixes in self._pages(client, old_root, delimiter="/"):
            for prefix in prefixes:
                raw = self._folder_symbol(prefix, old_root)
                if raw is not None:
                    found[self._public_symbol(raw, product)] = self._new_symbol(
                        raw, product
                    )
        new_root = self._new_market_root(product)
        for _, prefixes in self._pages(client, new_root, delimiter="/"):
            for prefix in prefixes:
                raw = self._folder_symbol(prefix, new_root)
                if raw is None or (product != "spot" and not raw.endswith("-PERP")):
                    continue
                public = self._public_symbol(raw, product)
                if self._belongs_to_product(public, product):
                    found[public] = raw
        return found

    @staticmethod
    def _merge_markets(
        current: dict[str, Market], archive: dict[str, str], product: str
    ) -> list[Market]:
        """Merge archive-only symbols into current HTX metadata.

        Args:
            current: Current markets indexed by public symbol.
            archive: New archive symbols indexed by public symbol.
            product: The represented HTX product.

        Returns:
            The merged snapshot ordered by public symbol.
        """
        merged = dict(current)
        for symbol, archive_symbol in archive.items():
            if symbol in merged:
                market = merged[symbol]
                if market.pair != archive_symbol:
                    merged[symbol] = replace(market, pair=archive_symbol)
                continue
            parts = symbol.split("-")
            merged[symbol] = Market(
                symbol=symbol,
                normalized_symbol=normalize_pair(symbol),
                base_asset=parts[0] if len(parts) == 2 else None,
                quote_asset=parts[1] if len(parts) == 2 else None,
                pair=archive_symbol,
                contract_type="PERPETUAL" if product != "spot" else None,
            )
        return [merged[symbol] for symbol in sorted(merged)]

    @staticmethod
    def _old_market_root(product: str) -> str:
        """Return the old Kline root used to enumerate one product."""
        name = {"spot": "spot", "linear_swap": "linear-swap", "coin_swap": "swap"}[
            product
        ]
        return f"data/klines/{name}/daily/"

    @staticmethod
    def _new_market_root(product: str) -> str:
        """Return the new Kline root used to enumerate one product."""
        area = "spot" if product == "spot" else "futures"
        return f"historical_data/{area}/daily/klines/"

    @staticmethod
    def _folder_symbol(prefix: str, root: str) -> str | None:
        """Extract one safe immediate folder name from an archive prefix.

        Args:
            prefix: The folder returned by the archive listing.
            root: The exact parent folder being listed.

        Returns:
            An uppercase archive symbol, or ``None`` for unrelated or legacy
            symbols containing characters unsupported by the public API.
        """
        if not prefix.startswith(root) or not prefix.endswith("/"):
            return None
        symbol = prefix[len(root) : -1].upper()
        if _SAFE_SYMBOL.fullmatch(symbol) is None:
            LOGGER.debug("Ignoring unsupported HTX archive symbol: %s", symbol)
            return None
        return symbol

    @staticmethod
    def _public_symbol(symbol: str, product: str) -> str:
        """Convert an old or new archive symbol into the public symbol."""
        value = symbol.removesuffix("-PERP")
        return value.replace("-", "") if product == "spot" else value

    @staticmethod
    def _new_symbol(symbol: str, product: str) -> str:
        """Convert an old archive symbol into its expected new spelling."""
        if product == "spot":
            return symbol
        return f"{symbol}-PERP"

    @staticmethod
    def _belongs_to_product(symbol: str, product: str) -> bool:
        """Return whether a Futures archive symbol belongs to a product."""
        if product == "spot":
            return True
        quote_asset = symbol.rsplit("-", 1)[-1]
        stable = quote_asset in {"USDT", "USDC"}
        return stable if product == "linear_swap" else not stable

    def _pages(
        self,
        client: httpx.Client,
        prefix: str,
        *,
        delimiter: str | None = None,
        marker: str | None = None,
        max_keys: int | None = None,
    ) -> Iterator[tuple[list[str], list[str]]]:
        """Yield HTX object keys and folders for one archive prefix."""
        yield from pages(
            client,
            prefix,
            listing_url=LISTING_URL,
            delimiter=delimiter,
            marker=marker,
            max_keys=max_keys,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def _check_product(self, product: str) -> None:
        """Reject a product outside the HTX perpetual and Spot scope."""
        if product not in self.products:
            raise ValueError(f"unsupported HTX product: {product}")

    def _validate_resource_request(
        self, key: ResourceKey, start_day: date | None, end_day: date
    ) -> None:
        """Reject an unsupported or unsafe HTX resource request."""
        self._validate_resource_identity(key)
        for symbol in (key.symbol, key.archive_symbol):
            if (
                symbol is not None
                and self._safe_symbol(symbol, compact=False) != symbol.upper()
            ):
                raise ValueError("resource key contains an unsafe symbol")
        if start_day is not None and start_day > end_day:
            raise ValueError("resource date range is reversed")

    def _validate_resource_identity(self, key: ResourceKey) -> None:
        """Reject an unsupported source, product, dataset, cadence, or interval.

        Args:
            key: The requested HTX dataset identity.
        """
        if key.source != self.code:
            raise ValueError("resource key belongs to another source")
        self._check_product(key.product)
        if not supports(key.product, key.dataset):
            raise ValueError(f"unsupported HTX dataset: {key.product}/{key.dataset}")
        if key.cadence != "daily":
            raise ValueError("HTX archives support daily cadence only")
        needs_interval = key.dataset.endswith("klines")
        if needs_interval and key.interval not in OLD_INTERVALS:
            raise ValueError("unsupported HTX archive interval")
        if not needs_interval and key.interval is not None:
            raise ValueError("this HTX dataset does not use an interval")

    @staticmethod
    def _safe_symbol(value: object, *, compact: bool) -> str | None:
        """Return a safe uppercase HTX symbol or ignore non-ASCII values."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("HTX market endpoint contains an invalid market")
        symbol = value.upper()
        if _SAFE_SYMBOL.fullmatch(symbol) is not None and (
            not compact or "-" not in symbol
        ):
            return symbol
        if not symbol.isascii():
            LOGGER.debug("Ignoring unsupported non-ASCII HTX symbol: %s", value)
            return None
        raise ValueError("HTX market endpoint contains an unsafe symbol")

    @staticmethod
    def _required_symbol(
        value: Mapping[object, object], field: str, *, compact: bool
    ) -> str:
        """Return one required safe symbol field."""
        result = HTXConnector._safe_symbol(value.get(field), compact=compact)
        if result is None:
            raise ValueError("HTX market endpoint contains an invalid market")
        return result

    @staticmethod
    def _required_text(value: Mapping[object, object], field: str) -> str:
        """Return one required nonempty source string."""
        result = value.get(field)
        if not isinstance(result, str) or not result.strip():
            raise ValueError("HTX market endpoint contains an invalid market")
        return result

    @staticmethod
    def _source_time(
        value: object, *, milliseconds: bool, optional: bool = False
    ) -> datetime | None:
        """Convert an optional or required numeric source timestamp to UTC."""
        if optional and value in {None, "", 0}:
            return None
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("HTX market endpoint contains an invalid market")
        try:
            divisor = 1_000 if milliseconds else 1
            return datetime.fromtimestamp(float(value) / divisor, UTC)
        except (OSError, OverflowError, ValueError) as error:
            raise ValueError(
                "HTX market endpoint contains an invalid market"
            ) from error

    @staticmethod
    def _date_time(value: object) -> datetime:
        """Convert HTX's YYYYMMDD contract creation date to UTC."""
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("HTX market endpoint contains an invalid market")
        try:
            return datetime.strptime(str(value), "%Y%m%d").replace(tzinfo=UTC)
        except ValueError as error:
            raise ValueError(
                "HTX market endpoint contains an invalid market"
            ) from error
