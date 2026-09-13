"""Discover Upbit Spot markets and daily archive resources."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
import logging
import math
import re
from urllib.parse import quote

import httpx

from veldra.core.download import archive_checksum, get
from veldra.core.models import Market, Resource, ResourceKey
from veldra.upbit.datasets import PRODUCTS, supports

LISTING_URL = "https://crix-data-api.upbit.com/api/v1/market-data/listing"
ARCHIVE_URL = "https://crix-data.upbit.com"
MARKETS_URL = "https://api.upbit.com/v1/market/all"
TICKERS_URL = "https://api.upbit.com/v1/ticker/all"
LOGGER = logging.getLogger(__name__)
_SAFE_SYMBOL = re.compile(r"[A-Z0-9]+-[A-Z0-9]+")
_SAFE_KEY = re.compile(r"[A-Za-z0-9._/-]+")


@dataclass(frozen=True)
class ListingEntry:
    """Describe one validated Upbit portal entry."""

    key: str
    kind: str
    size: int | None


@dataclass(frozen=True)
class ArchiveRoute:
    """Describe one Upbit archive folder and filename prefix."""

    prefix: str
    stem: str


class UpbitConnector:
    """Discover metadata from Upbit's public APIs and archive portal."""

    code = "upbit"
    products = PRODUCTS
    max_concurrency = 32
    monthly_datasets: frozenset[str] = frozenset()

    def __init__(
        self, *, timeout: float = 30.0, retries: int = 3, backoff: float = 0.5
    ) -> None:
        """Store HTTP settings used for Upbit source requests."""
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _get(
        self,
        client: httpx.Client,
        url: str,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """Make one retrying Upbit metadata request."""
        return get(
            client,
            url,
            params=params,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def checksum(self, client: httpx.Client, resource: Resource) -> str:
        """Return Upbit's SHA-256 digest for one archive."""
        return archive_checksum(
            client,
            resource,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return current and archive-only Upbit Spot markets."""
        self._check_product(product)
        current = self._current_markets(client)
        archived = self._archive_markets(client)
        merged = self._merge_markets(current, archived)
        LOGGER.info(
            "Upbit markets loaded: exchange=%d archive=%d merged=%d",
            len(current),
            len(archived),
            len(merged),
        )
        return merged

    def quote_volumes(self, client: httpx.Client, product: str) -> dict[str, float]:
        """Return rolling quote turnover indexed by native Upbit symbol."""
        self._check_product(product)
        payload = self._get(
            client,
            TICKERS_URL,
            {"quote_currencies": "KRW,BTC,USDT"},
        ).json()
        if not isinstance(payload, list):
            raise ValueError("Upbit ticker endpoint contains no market snapshot")
        return self._ticker_volumes(payload)

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return Upbit daily resources across an inclusive date range."""
        self._validate_resource_request(key, start_day, end_day)
        route = self._route(key)
        found: list[Resource] = []
        for year in range(start_day.year, end_day.year + 1):
            found.extend(self._year_resources(client, route, key, year))
        return [item for item in found if start_day <= item.day <= end_day]

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Return the first Upbit archive within broad date bounds."""
        self._validate_resource_request(key, start_day, end_day)
        route = self._route(key)
        years = self._available_years(client, route.prefix)
        for year in years:
            if year > end_day.year or (start_day is not None and year < start_day.year):
                continue
            resources = self._year_resources(client, route, key, year)
            for resource in resources:
                if start_day is not None and resource.day < start_day:
                    continue
                return None if resource.day > end_day else resource
        return None

    def _current_markets(self, client: httpx.Client) -> dict[str, Market]:
        """Parse Upbit's complete current Spot market snapshot."""
        payload = self._get(client, MARKETS_URL, {"is_details": "true"}).json()
        if not isinstance(payload, list):
            raise ValueError("Upbit market endpoint contains no market snapshot")
        markets: dict[str, Market] = {}
        for value in payload:
            item = self._market(value, active=True)
            if item.symbol in markets:
                raise ValueError("Upbit market endpoint contains a duplicate symbol")
            markets[item.symbol] = item
        return markets

    def _archive_markets(self, client: httpx.Client) -> set[str]:
        """Return native symbols present in either historical dataset tree."""
        symbols: set[str] = set()
        for root in ("candle", "trade"):
            for entry in self._listing(client, root):
                if entry.kind != "DIRECTORY":
                    continue
                symbol = self._folder_symbol(entry.key, root)
                if symbol is not None:
                    symbols.add(symbol)
        return symbols

    @staticmethod
    def _merge_markets(
        current: Mapping[str, Market], archived: set[str]
    ) -> list[Market]:
        """Merge current and archive-only markets without deleting old rows."""
        merged = dict(current)
        for symbol in archived - current.keys():
            merged[symbol] = UpbitConnector._market(symbol, active=False)
        return [merged[symbol] for symbol in sorted(merged)]

    @staticmethod
    def _market(value: object, *, active: bool) -> Market:
        """Parse one current row or archive-only native symbol."""
        if isinstance(value, str):
            symbol = UpbitConnector._required_symbol(value)
            warning = False
        elif isinstance(value, dict):
            symbol = UpbitConnector._required_symbol(value.get("market"))
            event = value.get("market_event")
            warning_value = (
                event.get("warning", False) if isinstance(event, dict) else False
            )
            if not isinstance(warning_value, bool):
                raise ValueError("Upbit market endpoint contains an invalid market")
            warning = warning_value
        else:
            raise ValueError("Upbit market endpoint contains an invalid market")
        quote_asset, base_asset = symbol.split("-", maxsplit=1)
        return Market(
            symbol=symbol,
            normalized_symbol=f"{base_asset}{quote_asset}",
            base_asset=base_asset,
            quote_asset=quote_asset,
            status="CAUTION" if warning else "TRADING" if active else None,
            pair=symbol,
            active=active,
            product="spot",
        )

    @staticmethod
    def _ticker_volumes(rows: Sequence[object]) -> dict[str, float]:
        """Parse finite nonnegative quote turnover from ticker rows."""
        volumes: dict[str, float] = {}
        for value in rows:
            if not isinstance(value, dict):
                raise ValueError("Upbit ticker endpoint contains an invalid market")
            symbol = UpbitConnector._required_symbol(value.get("market"))
            raw = value.get("acc_trade_price_24h")
            if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                raise ValueError("Upbit ticker endpoint contains an invalid market")
            try:
                volume = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Upbit ticker endpoint contains an invalid market"
                ) from error
            if not math.isfinite(volume) or volume < 0:
                raise ValueError("Upbit ticker endpoint contains an invalid market")
            volumes[symbol] = volume
        return volumes

    def _listing(self, client: httpx.Client, prefix: str) -> list[ListingEntry]:
        """Return one validated Upbit portal directory listing."""
        payload = self._get(client, LISTING_URL, {"prefix": prefix}).json()
        if not isinstance(payload, list):
            raise ValueError("Upbit listing response must be a list")
        entries: list[ListingEntry] = []
        for value in payload:
            entries.append(self._listing_entry(value))
        return entries

    @staticmethod
    def _listing_entry(value: object) -> ListingEntry:
        """Parse one safe, typed portal listing entry."""
        if not isinstance(value, dict):
            raise ValueError("Upbit listing contains an invalid entry")
        key = value.get("key")
        kind = value.get("type")
        size = value.get("size")
        valid_size = isinstance(size, int) and not isinstance(size, bool) and size >= 0
        if not isinstance(key, str) or not UpbitConnector._safe_key(key):
            raise ValueError("Upbit listing contains an invalid entry")
        if kind not in {"DIRECTORY", "FILE"} or not valid_size:
            raise ValueError("Upbit listing contains an invalid entry")
        return ListingEntry(key, kind, size)

    def _year_resources(
        self,
        client: httpx.Client,
        route: ArchiveRoute,
        key: ResourceKey,
        year: int,
    ) -> list[Resource]:
        """Return exact ZIP resources found in one archive year."""
        prefix = f"{route.prefix}/{year}"
        pattern = re.compile(re.escape(f"{prefix}/{route.stem}") + r"(\d{8})\.zip")
        resources: list[Resource] = []
        for entry in self._listing(client, prefix):
            if entry.kind != "FILE":
                continue
            day = self._resource_day(entry.key, pattern)
            if day is not None:
                resources.append(self._resource(day, entry.key, key))
        return sorted(resources, key=lambda resource: resource.day)

    def _available_years(self, client: httpx.Client, prefix: str) -> list[int]:
        """Return valid immediate year folders beneath one archive route."""
        years: list[int] = []
        for entry in self._listing(client, prefix):
            if entry.kind != "DIRECTORY":
                continue
            suffix = entry.key.removeprefix(f"{prefix}/")
            if len(suffix) == 4 and suffix.isdigit():
                years.append(int(suffix))
        return sorted(set(years))

    @staticmethod
    def _route(key: ResourceKey) -> ArchiveRoute:
        """Build one exact Upbit daily archive route."""
        symbol = key.archive_symbol or key.symbol
        if key.dataset == "klines":
            assert key.interval is not None
            return ArchiveRoute(
                f"candle/{symbol}/daily/{key.interval}",
                f"{symbol}_candle-{key.interval}_",
            )
        return ArchiveRoute(f"trade/{symbol}/daily", f"{symbol}_trade_")

    @staticmethod
    def _resource(day: date, object_key: str, key: ResourceKey) -> Resource:
        """Build one SHA-256-verified Upbit daily resource."""
        url = f"{ARCHIVE_URL}/{quote(object_key, safe='/-_.')}"
        start = datetime.combine(day, time.min, UTC)
        return Resource(
            day=day,
            url=url,
            checksum_url=f"{url}.checksum",
            checksum_algorithm="sha256",
            archive_symbol=key.archive_symbol or key.symbol,
            timestamp_column="open_time" if key.dataset == "klines" else "event_time",
            coverage_start=start,
            coverage_end=start + timedelta(days=1),
        )

    @staticmethod
    def _resource_day(object_key: str, pattern: re.Pattern[str]) -> date | None:
        """Extract one valid compact date from an exact archive key."""
        match = pattern.fullmatch(object_key)
        if match is None:
            return None
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            return None

    @staticmethod
    def _folder_symbol(key: str, root: str) -> str | None:
        """Extract a safe immediate native market folder."""
        prefix = f"{root}/"
        if not key.startswith(prefix) or "/" in key[len(prefix) :]:
            return None
        symbol = key[len(prefix) :].upper()
        return symbol if _SAFE_SYMBOL.fullmatch(symbol) is not None else None

    @staticmethod
    def _safe_key(value: str) -> bool:
        """Return whether a portal key cannot escape its remote path."""
        return (
            bool(value)
            and not value.startswith("/")
            and ".." not in value.split("/")
            and "\\" not in value
            and _SAFE_KEY.fullmatch(value) is not None
        )

    @staticmethod
    def _required_symbol(value: object) -> str:
        """Return one required safe uppercase Upbit native symbol."""
        if not isinstance(value, str):
            raise ValueError("Upbit market endpoint contains an invalid market")
        symbol = value.strip().upper()
        if _SAFE_SYMBOL.fullmatch(symbol) is None:
            raise ValueError("Upbit market endpoint contains an invalid market")
        return symbol

    def _check_product(self, product: str) -> None:
        """Reject products outside Upbit Spot."""
        if product not in self.products:
            raise ValueError(f"unsupported Upbit product: {product}")

    def _validate_resource_request(
        self, key: ResourceKey, start_day: date | None, end_day: date
    ) -> None:
        """Reject unsafe or unsupported Upbit archive requests."""
        self._check_product(key.product)
        if key.source != self.code or not supports(key.product, key.dataset):
            raise ValueError("unsupported Upbit resource identity")
        if start_day is not None and start_day > end_day:
            raise ValueError("resource range must not be reversed")
        symbol = key.archive_symbol or key.symbol
        if _SAFE_SYMBOL.fullmatch(symbol) is None:
            raise ValueError("resource identity contains an unsafe symbol")
        if key.dataset == "klines" and key.interval not in {"1s", "1m"}:
            raise ValueError("Upbit Kline resources require an interval")
