"""Discover KuCoin markets and daily archive resources."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
import logging
import math
from pathlib import Path
import re
from urllib.parse import quote

import httpx

from veldra.core.datasets import DatasetSpec
from veldra.core.download import archive_checksum, get
from veldra.core.ingest import ingest_archive
from veldra.core.models import IngestedResource, Market, Resource, ResourceKey
from veldra.core.portal import pages
from veldra.core.request import normalize_pair
from veldra.kucoin.datasets import ArchiveRoute, PRODUCTS, supports
from veldra.kucoin.processing import normalize_chunk, validate_chunk

LISTING_URL = "https://historical-data.kucoin.com/"
ARCHIVE_URL = "https://historical-data.kucoin.com"
SPOT_MARKETS_URL = "https://api.kucoin.com/api/v2/symbols"
SPOT_TICKERS_URL = "https://api.kucoin.com/api/v1/market/allTickers"
FUTURES_MARKETS_URL = "https://api-futures.kucoin.com/api/v1/contracts/active"
LOGGER = logging.getLogger(__name__)
_SAFE_SYMBOL = re.compile(r"[A-Z0-9-]+")
_DATE = r"(\d{4}-\d{2}-\d{2})"


class KuCoinConnector:
    """Discover source metadata from KuCoin's APIs and archive tree."""

    code = "kucoin"
    products = PRODUCTS
    archive_day_offset = timedelta(0)
    max_concurrency = 64
    monthly_datasets: frozenset[str] = frozenset()

    def __init__(
        self, *, timeout: float = 30.0, retries: int = 3, backoff: float = 0.5
    ) -> None:
        """Store HTTP settings used for KuCoin source requests."""
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _get(
        self,
        client: httpx.Client,
        url: str,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """Make one retrying KuCoin metadata request."""
        return get(
            client,
            url,
            params=params,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def checksum(self, client: httpx.Client, resource: Resource) -> str:
        """Return KuCoin's MD5 digest for one archive."""
        return archive_checksum(
            client,
            resource,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return current and archive-only markets for one KuCoin product."""
        self._check_product(product)
        current = self._current_markets(client, product)
        archived = self._archive_markets(client, product)
        merged = self._merge_markets(current, archived, product)
        LOGGER.info(
            "KuCoin markets loaded: product=%s exchange=%d archive=%d merged=%d",
            product,
            len(current),
            len(archived),
            len(merged),
        )
        return merged

    def quote_volumes(self, client: httpx.Client, product: str) -> dict[str, float]:
        """Return rolling quote turnover indexed by native market symbol."""
        self._check_product(product)
        if product == "spot":
            payload = self._get(client, SPOT_TICKERS_URL).json()
            data = payload.get("data") if isinstance(payload, dict) else None
            rows = data.get("ticker") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                raise ValueError("KuCoin ticker contains no market snapshot")
            return self._ticker_volumes(rows, "volValue")
        payload = self._get(client, FUTURES_MARKETS_URL).json()
        rows = self._rows(payload)
        selected = [
            row
            for row in rows
            if isinstance(row, dict)
            and self._futures_product(row) == product
            and self._is_perpetual(row)
        ]
        return self._ticker_volumes(selected, "turnoverOf24h")

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return KuCoin resources in an inclusive archive-day range."""
        self._validate_resource_request(key, start_day, end_day)
        route = self._route(key)
        pattern = re.compile(re.escape(route.prefix + route.stem) + _DATE + r"\.zip")
        marker = f"{route.prefix}{route.stem}{start_day.isoformat()}"
        found: list[Resource] = []
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
                    found.append(self._resource(day, object_key, key))
            if past_end:
                break
        return sorted(found, key=lambda resource: resource.day)

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Return the first KuCoin resource within broad archive-day bounds."""
        self._validate_resource_request(key, start_day, end_day)
        route = self._route(key)
        marker = (
            f"{route.prefix}{route.stem}{start_day.isoformat()}"
            if start_day is not None
            else None
        )
        pattern = re.compile(re.escape(route.prefix + route.stem) + _DATE + r"\.zip")
        for keys, _ in self._pages(client, route.prefix, marker=marker, max_keys=4):
            for object_key in keys:
                day = self._resource_day(object_key, pattern)
                if day is None or (start_day is not None and day < start_day):
                    continue
                return None if day > end_day else self._resource(day, object_key, key)
        return None

    def ingest(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Convert one verified KuCoin CSV archive into Parquet."""
        if dataset.name == "order_book_snapshots":
            from veldra.kucoin.orderbook import ingest_order_book

            return ingest_order_book(
                client,
                resource,
                dataset,
                destination,
                timeout=self.timeout,
                retries=self.retries,
                backoff=self.backoff,
            )
        return ingest_archive(
            client,
            resource,
            dataset,
            destination,
            normalizer=normalize_chunk,
            validator=validate_chunk,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def _current_markets(self, client: httpx.Client, product: str) -> dict[str, Market]:
        """Parse one complete current market snapshot."""
        url = SPOT_MARKETS_URL if product == "spot" else FUTURES_MARKETS_URL
        rows = self._rows(self._get(client, url).json())
        markets: dict[str, Market] = {}
        for value in rows:
            market = (
                self._spot_market(value)
                if product == "spot"
                else self._futures_market(value, product)
            )
            if market is None:
                continue
            if market.symbol in markets:
                raise ValueError("KuCoin market endpoint contains a duplicate symbol")
            markets[market.symbol] = market
        return markets

    @staticmethod
    def _rows(payload: object) -> list[object]:
        """Return source rows from one successful KuCoin response."""
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("KuCoin market endpoint contains no market snapshot")
        return rows

    @staticmethod
    def _spot_market(value: object) -> Market | None:
        """Parse one KuCoin Spot market row."""
        if not isinstance(value, dict):
            raise ValueError("KuCoin market endpoint contains an invalid market")
        symbol = KuCoinConnector._safe_symbol(value.get("symbol"))
        if symbol is None:
            return None
        base = KuCoinConnector._required_symbol(value, "baseCurrency")
        quote_asset = KuCoinConnector._required_symbol(value, "quoteCurrency")
        enabled = value.get("enableTrading")
        if not isinstance(enabled, bool):
            raise ValueError("KuCoin market endpoint contains an invalid market")
        return Market(
            symbol=symbol,
            normalized_symbol=normalize_pair(symbol),
            base_asset=base,
            quote_asset=quote_asset,
            status="TRADING" if enabled else "DISABLED",
            pair=normalize_pair(symbol),
            active=enabled,
            product="spot",
        )

    @staticmethod
    def _futures_market(value: object, product: str) -> Market | None:
        """Parse one current KuCoin perpetual market row."""
        if not isinstance(value, dict):
            raise ValueError("KuCoin market endpoint contains an invalid market")
        if not KuCoinConnector._is_perpetual(value):
            return None
        if KuCoinConnector._futures_product(value) != product:
            return None
        symbol = KuCoinConnector._required_symbol(value, "symbol")
        base = KuCoinConnector._required_symbol(value, "baseCurrency")
        quote_asset = KuCoinConnector._required_symbol(value, "quoteCurrency")
        status = KuCoinConnector._required_text(value, "status")
        multiplier = KuCoinConnector._positive_number(value.get("multiplier"))
        archive_symbol = KuCoinConnector._archive_contract(symbol)
        return Market(
            symbol=symbol,
            normalized_symbol=normalize_pair(symbol),
            base_asset=base,
            quote_asset=quote_asset,
            status=status,
            pair=archive_symbol,
            contract_type="PERPETUAL",
            contract_size=multiplier,
            onboard_time=KuCoinConnector._source_time(value.get("firstOpenDate")),
            delivery_time=KuCoinConnector._source_time(value.get("expireDate")),
            active=status == "Open",
            product=product,
        )

    def _archive_markets(self, client: httpx.Client, product: str) -> dict[str, str]:
        """Return valid market folders across every KuCoin dataset branch."""
        roots = (
            (
                "data/spot/daily/klines/",
                "data/spot/daily/trades/",
                "data/spot/daily/depth/orderbooklv50/",
            )
            if product == "spot"
            else (
                "data/futures/daily/klines/",
                "data/futures/daily/trades/",
                "data/futures/daily/index/",
                "data/futures/daily/mark/",
                "data/futures/daily/fundingRates/",
                "data/futures/daily/depth/orderbooklv50/",
            )
        )
        found: dict[str, str] = {}
        for root in roots:
            for _, prefixes in self._pages(client, root, delimiter="/"):
                for prefix in prefixes:
                    symbol = self._folder_symbol(prefix, root)
                    if symbol is None:
                        continue
                    compact = normalize_pair(symbol)
                    if product == "spot" or self._archive_product(compact) == product:
                        if "-" in symbol or compact not in found:
                            found[compact] = symbol
        return found

    @staticmethod
    def _merge_markets(
        current: dict[str, Market], archived: Mapping[str, str], product: str
    ) -> list[Market]:
        """Merge archive-only symbols and archive aliases into current markets."""
        merged = dict(current)
        current_by_archive = {
            market.pair: symbol
            for symbol, market in current.items()
            if market.pair is not None
        }
        for archive_symbol, native_archive_symbol in archived.items():
            native = current_by_archive.get(archive_symbol)
            if native is not None:
                market = merged[native]
                if market.pair != archive_symbol:
                    merged[native] = replace(market, pair=archive_symbol)
                continue
            public_symbol = (
                native_archive_symbol if product == "spot" else archive_symbol
            )
            merged[public_symbol] = Market(
                symbol=public_symbol,
                normalized_symbol=archive_symbol,
                pair=archive_symbol,
                contract_type="PERPETUAL" if product != "spot" else None,
                product=product,
            )
        return [merged[symbol] for symbol in sorted(merged)]

    @staticmethod
    def _ticker_volumes(rows: Sequence[object], field: str) -> dict[str, float]:
        """Parse nonnegative ticker turnover from source rows."""
        volumes: dict[str, float] = {}
        for value in rows:
            if not isinstance(value, dict):
                raise ValueError("KuCoin ticker contains an invalid market")
            symbol = KuCoinConnector._safe_symbol(value.get("symbol"))
            if symbol is None:
                continue
            raw = value.get(field)
            if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                raise ValueError("KuCoin ticker contains an invalid market")
            try:
                volume = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError("KuCoin ticker contains an invalid market") from error
            if not math.isfinite(volume) or volume < 0:
                raise ValueError("KuCoin ticker contains an invalid market")
            volumes[symbol] = volume
        return volumes

    @staticmethod
    def _route(key: ResourceKey) -> ArchiveRoute:
        """Build the exact archive route for one canonical resource key."""
        archive = key.archive_symbol or key.symbol
        area = "spot" if key.product == "spot" else "futures"
        folder = {
            "klines": "klines",
            "trades": "trades",
            "index_price_klines": "index",
            "mark_price_klines": "mark",
            "funding_rates": "fundingRates",
            "order_book_snapshots": "depth/orderbooklv50",
        }[key.dataset]
        symbol = (
            key.symbol
            if key.product == "spot" and key.dataset == "order_book_snapshots"
            else archive
        )
        root = f"data/{area}/daily/{folder}/{symbol}/"
        if key.dataset.endswith("klines"):
            assert key.interval is not None
            root = f"{root}{key.interval}/"
            stem = f"{symbol}-{key.interval}-"
        elif key.dataset == "funding_rates":
            stem = f"{symbol}-fundingRates-"
        elif key.dataset == "order_book_snapshots":
            stem = f"{symbol}-orderbooklv50-"
        else:
            stem = f"{symbol}-trades-"
        return ArchiveRoute(root, stem)

    @staticmethod
    def _resource(day: date, object_key: str, key: ResourceKey) -> Resource:
        """Build one MD5-verified KuCoin daily resource."""
        url = f"{ARCHIVE_URL}/{quote(object_key, safe='/')}"
        start = datetime.combine(day, time.min, UTC)
        depth = key.dataset == "order_book_snapshots"
        timestamp = (
            "event_time"
            if key.dataset in {"trades", "order_book_snapshots"}
            else "funding_time" if key.dataset == "funding_rates" else "open_time"
        )
        return Resource(
            day=day,
            url=url,
            checksum_url=f"{url}.CHECKSUM",
            checksum_algorithm="md5",
            archive_symbol=key.archive_symbol or key.symbol,
            timestamp_column=timestamp,
            coverage_start=start - timedelta(minutes=5) if depth else start,
            coverage_end=(
                start + timedelta(days=1, minutes=5)
                if depth
                else start + timedelta(days=1)
            ),
        )

    @staticmethod
    def _resource_day(object_key: str, pattern: re.Pattern[str]) -> date | None:
        """Extract a valid archive day from one exact object key."""
        match = pattern.fullmatch(object_key)
        if match is None:
            return None
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _folder_symbol(prefix: str, root: str) -> str | None:
        """Extract a safe immediate market folder from one listing prefix."""
        if not prefix.startswith(root) or not prefix.endswith("/"):
            return None
        symbol = prefix[len(root) : -1].upper()
        if symbol == "NULL" or _SAFE_SYMBOL.fullmatch(symbol) is None:
            return None
        return symbol

    @staticmethod
    def _archive_contract(symbol: str) -> str:
        """Map KuCoin's leading XBT API asset to its BTC archive alias."""
        return f"BTC{symbol[3:]}" if symbol.startswith("XBT") else symbol

    @staticmethod
    def _archive_product(symbol: str) -> str | None:
        """Classify one archive perpetual while rejecting dated contracts."""
        if not symbol.endswith("M"):
            return None
        if symbol.endswith(("USDTM", "USDCM")):
            return "linear_futures"
        if symbol.endswith("USDM"):
            return "inverse_futures"
        return None

    @staticmethod
    def _is_perpetual(value: Mapping[object, object]) -> bool:
        """Return whether a Futures API row is a perpetual contract."""
        return value.get("type") == "FFWCSX"

    @staticmethod
    def _futures_product(value: Mapping[object, object]) -> str | None:
        """Classify one Futures API row as linear or inverse."""
        inverse = value.get("isInverse")
        if not isinstance(inverse, bool):
            return None
        return "inverse_futures" if inverse else "linear_futures"

    @staticmethod
    def _source_time(value: object) -> datetime | None:
        """Parse an optional millisecond KuCoin timestamp."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("KuCoin market endpoint contains an invalid market")
        return datetime.fromtimestamp(value / 1_000, UTC)

    @staticmethod
    def _positive_number(value: object) -> float:
        """Parse one positive finite contract multiplier."""
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("KuCoin market endpoint contains an invalid market")
        try:
            number = abs(float(value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "KuCoin market endpoint contains an invalid market"
            ) from error
        if not math.isfinite(number) or number <= 0:
            raise ValueError("KuCoin market endpoint contains an invalid market")
        return number

    @staticmethod
    def _safe_symbol(value: object) -> str | None:
        """Return one safe uppercase source symbol or skip unsupported text."""
        if not isinstance(value, str):
            raise ValueError("KuCoin market endpoint contains an invalid market")
        symbol = value.strip().upper()
        return symbol if _SAFE_SYMBOL.fullmatch(symbol) is not None else None

    @staticmethod
    def _required_symbol(value: Mapping[object, object], field: str) -> str:
        """Return one required safe symbol field."""
        symbol = KuCoinConnector._safe_symbol(value.get(field))
        if symbol is None:
            raise ValueError("KuCoin market endpoint contains an invalid market")
        return symbol

    @staticmethod
    def _required_text(value: Mapping[object, object], field: str) -> str:
        """Return one nonempty source text field."""
        text = value.get(field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("KuCoin market endpoint contains an invalid market")
        return text.strip()

    def _pages(
        self,
        client: httpx.Client,
        prefix: str,
        *,
        delimiter: str | None = None,
        marker: str | None = None,
        max_keys: int | None = None,
    ) -> Iterator[tuple[list[str], list[str]]]:
        """Yield KuCoin archive keys and folders for one prefix."""
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
        """Reject products outside KuCoin Spot and perpetual Futures."""
        if product not in self.products:
            raise ValueError(f"unsupported KuCoin product: {product}")

    def _validate_resource_request(
        self, key: ResourceKey, start_day: date | None, end_day: date
    ) -> None:
        """Reject unsafe or unsupported KuCoin archive requests."""
        self._check_product(key.product)
        if key.source != self.code or not supports(key.product, key.dataset):
            raise ValueError("unsupported KuCoin resource identity")
        if start_day is not None and start_day > end_day:
            raise ValueError("resource range must not be reversed")
        symbol = key.archive_symbol or key.symbol
        if _SAFE_SYMBOL.fullmatch(symbol) is None:
            raise ValueError("resource identity contains an unsafe symbol")
        if key.dataset.endswith("klines") and key.interval is None:
            raise ValueError("KuCoin Kline resources require an interval")
