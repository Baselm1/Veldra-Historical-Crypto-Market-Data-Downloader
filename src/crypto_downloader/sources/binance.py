"""Discover Binance Spot markets and daily kline archives."""

from collections.abc import Iterator, Mapping
from datetime import date
import logging
from pathlib import Path
import re
from urllib.parse import quote
import xml.etree.ElementTree as ElementTree

import httpx

from ..http import get
from ..datasets import DatasetSpec
from ..ingest import ingest_archive
from ..models import IngestedResource, Market, Resource, ResourceKey
from ..request import normalize_pair

EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
BUCKET_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE_URL = "https://data.binance.vision"
SPOT_KLINES_PREFIX = "data/spot/daily/klines/"
LOGGER = logging.getLogger(__name__)


class Binance:
    """Discover metadata from Binance and its public archive bucket."""

    code: str = "binance"
    products: tuple[str, ...] = ("spot",)
    active_statuses: frozenset[str] = frozenset({"TRADING"})
    max_concurrency: int = 16

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 0.5,
    ) -> None:
        """Store HTTP settings used for Binance metadata requests.

        Args:
            timeout: The timeout for each request in seconds.
            retries: The number of retries after the first attempt.
            backoff: The initial exponential retry delay in seconds.
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
        """Make one retrying Binance metadata request.

        Args:
            client: The HTTPX client used for the request.
            url: The metadata URL to request.
            params: Optional query string values.

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

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return current and archive-only Binance markets.

        Args:
            client: The HTTPX client used for Binance requests.
            product: The Binance product to inspect.

        Returns:
            The complete known market snapshot ordered by symbol.
        """
        self._check_product(product)
        response = self._get(
            client,
            EXCHANGE_INFO_URL,
            {"showPermissionSets": "false"},
        )
        current = self._exchange_markets(response.json())
        exchange_count = len(current)
        archive_symbols = self._archive_symbols(client)
        for symbol in archive_symbols:
            current.setdefault(symbol, Market(symbol, normalize_pair(symbol)))
        markets = [current[symbol] for symbol in sorted(current)]
        LOGGER.info(
            "Binance markets loaded: product=%s exchange=%d archive=%d merged=%d",
            product,
            exchange_count,
            len(archive_symbols),
            len(markets),
        )
        return markets

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return Binance daily archives in an inclusive date range.

        Args:
            client: The HTTPX client used for Binance requests.
            key: The requested Binance dataset identity.
            start_day: The first archive day to include.
            end_day: The last archive day to include.

        Returns:
            The available daily archives ordered by date.
        """
        self._validate_resource_request(key, start_day, end_day)
        prefix = f"{SPOT_KLINES_PREFIX}{key.symbol}/{key.interval}/"
        stem = f"{key.symbol}-{key.interval}-"
        marker = f"{prefix}{stem}{start_day.isoformat()}"
        pattern = re.compile(re.escape(prefix + stem) + r"(\d{4}-\d{2}-\d{2})\.zip")
        found: dict[date, Resource] = {}

        for keys, _ in self._pages(client, prefix, marker=marker):
            past_end = False
            for object_key in keys:
                day = self._resource_day(object_key, pattern)
                if day is None:
                    continue
                if day > end_day:
                    past_end = True
                elif day >= start_day:
                    url = f"{ARCHIVE_URL}/{quote(object_key, safe='/')}"
                    found[day] = Resource(day, url, f"{url}.CHECKSUM")
            if past_end:
                break
        resources = [found[day] for day in sorted(found)]
        LOGGER.info(
            "Binance resources listed: symbol=%s interval=%s range=[%s, %s] "
            "resources=%d",
            key.symbol,
            key.interval,
            start_day,
            end_day,
            len(resources),
        )
        return resources

    def ingest(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Convert one Binance archive into a verified Parquet file.

        Args:
            client: The HTTPX client used for Binance requests.
            resource: The daily archive to ingest.
            dataset: The schema used to interpret source rows.
            destination: The final Parquet path.

        Returns:
            Integrity metadata for the completed Parquet file.
        """
        return ingest_archive(
            client,
            resource,
            dataset,
            destination,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    @staticmethod
    def _exchange_markets(payload: object) -> dict[str, Market]:
        """Parse and validate a complete Spot exchange-info response.

        Args:
            payload: The decoded exchange-info JSON value.

        Returns:
            Valid markets indexed by their native symbols.
        """
        rows = payload.get("symbols") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("exchangeInfo contains no market snapshot")

        markets: dict[str, Market] = {}
        for row in rows:
            market = Binance._exchange_market(row)
            if market.symbol in markets:
                raise ValueError("exchangeInfo contains a duplicate symbol")
            markets[market.symbol] = market
        return markets

    @staticmethod
    def _exchange_market(value: object) -> Market:
        """Parse one required Spot market from exchange-info.

        Args:
            value: The decoded market object.

        Returns:
            The validated market metadata.
        """
        if not isinstance(value, dict):
            raise ValueError("exchangeInfo contains an invalid market")
        symbol = Binance._required_text(value, "symbol")
        if re.fullmatch(r"[A-Za-z0-9_]+", symbol) is None:
            raise ValueError("exchangeInfo contains an unsafe symbol")
        return Market(
            symbol=symbol,
            normalized_symbol=normalize_pair(symbol),
            base_asset=Binance._required_text(value, "baseAsset"),
            quote_asset=Binance._required_text(value, "quoteAsset"),
            status=Binance._required_text(value, "status"),
        )

    @staticmethod
    def _required_text(value: Mapping[object, object], field: str) -> str:
        """Return one required nonempty exchange-info string.

        Args:
            value: The market object containing the field.
            field: The source field to read.

        Returns:
            The original nonempty string value.
        """
        result = value.get(field)
        if not isinstance(result, str) or not result.strip():
            raise ValueError("exchangeInfo contains an invalid market")
        return result

    def _archive_symbols(self, client: httpx.Client) -> set[str]:
        """Return safe symbols represented by Spot kline folders.

        Args:
            client: The HTTPX client used for bucket listings.

        Returns:
            The unique archive symbol names.
        """
        symbols: set[str] = set()
        for _, prefixes in self._pages(client, SPOT_KLINES_PREFIX, delimiter="/"):
            for prefix in prefixes:
                symbol = self._folder_symbol(prefix)
                if symbol is not None:
                    symbols.add(symbol)
        return symbols

    @staticmethod
    def _folder_symbol(prefix: str) -> str | None:
        """Extract one safe symbol from an exact Spot kline folder.

        Args:
            prefix: The folder prefix returned by the bucket.

        Returns:
            The native symbol, or ``None`` for an unrelated or unsafe folder.
        """
        if not prefix.startswith(SPOT_KLINES_PREFIX) or not prefix.endswith("/"):
            return None
        symbol = prefix[len(SPOT_KLINES_PREFIX) : -1]
        return symbol if re.fullmatch(r"[A-Za-z0-9_]+", symbol) else None

    def _pages(
        self,
        client: httpx.Client,
        prefix: str,
        *,
        delimiter: str | None = None,
        marker: str | None = None,
    ) -> Iterator[tuple[list[str], list[str]]]:
        """Yield every valid page from one bucket listing.

        Args:
            client: The HTTPX client used for bucket listings.
            prefix: The object prefix to list.
            delimiter: The optional folder delimiter.
            marker: The optional key after which listing should begin.

        Yields:
            Object keys and common folder prefixes from each page.
        """
        seen = {marker} if marker is not None else set()
        while True:
            params = {"prefix": prefix}
            if delimiter is not None:
                params["delimiter"] = delimiter
            if marker is not None:
                params["marker"] = marker
            root = self._listing_root(self._get(client, BUCKET_URL, params).content)
            keys, prefixes, truncated, next_marker = self._listing_values(root)
            LOGGER.debug(
                "Binance listing page: prefix=%s marker=%s keys=%d prefixes=%d "
                "truncated=%s next_marker=%s",
                prefix,
                marker,
                len(keys),
                len(prefixes),
                truncated,
                next_marker,
            )
            yield keys, prefixes
            if not truncated:
                return
            marker = next_marker or max([*keys, *prefixes], default=None)
            if marker is None or marker in seen:
                raise ValueError("archive listing is truncated without a new marker")
            seen.add(marker)

    @staticmethod
    def _listing_root(content: bytes) -> ElementTree.Element:
        """Parse and validate the root of one bucket response.

        Args:
            content: The raw XML response bytes.

        Returns:
            The validated XML root element.
        """
        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError as error:
            raise ValueError("archive listing is not valid XML") from error
        if root.tag.rsplit("}", 1)[-1] != "ListBucketResult":
            raise ValueError("archive listing is not a bucket response")
        return root

    @staticmethod
    def _listing_values(
        root: ElementTree.Element,
    ) -> tuple[list[str], list[str], bool, str | None]:
        """Read keys, folders, and pagination state from a bucket page.

        Args:
            root: The validated bucket XML element.

        Returns:
            Keys, folder prefixes, truncation state, and optional next marker.
        """
        truncated = root.findtext("{*}IsTruncated")
        if truncated not in {"true", "false"}:
            raise ValueError("archive listing has no valid pagination status")
        keys = [
            element.text
            for element in root.findall("{*}Contents/{*}Key")
            if element.text is not None
        ]
        prefixes = [
            element.text
            for element in root.findall("{*}CommonPrefixes/{*}Prefix")
            if element.text is not None
        ]
        return keys, prefixes, truncated == "true", root.findtext("{*}NextMarker")

    def _check_product(self, product: str) -> None:
        """Reject a product not implemented by this Binance source.

        Args:
            product: The requested Binance product.
        """
        if product not in self.products:
            raise ValueError(f"unsupported Binance product: {product}")

    def _validate_resource_request(
        self, key: ResourceKey, start_day: date, end_day: date
    ) -> None:
        """Reject unsupported or unsafe daily resource requests.

        Args:
            key: The requested source dataset identity.
            start_day: The first archive day to include.
            end_day: The last archive day to include.
        """
        if key.source != self.code:
            raise ValueError(f"unsupported source: {key.source}")
        self._check_product(key.product)
        if key.dataset != "klines":
            raise ValueError(f"unsupported Binance dataset: {key.dataset}")
        if key.interval != "1m":
            raise ValueError(f"unsupported Binance interval: {key.interval}")
        if re.fullmatch(r"[A-Za-z0-9_]+", key.symbol) is None:
            raise ValueError("invalid Binance symbol")
        if end_day < start_day:
            raise ValueError("Binance date range ends before it starts")

    @staticmethod
    def _resource_day(object_key: str, pattern: re.Pattern[str]) -> date | None:
        """Extract the date from one exact daily archive key.

        Args:
            object_key: The bucket object key to inspect.
            pattern: The exact expected archive key pattern.

        Returns:
            The parsed archive day, or ``None`` for an unrelated key.
        """
        match = pattern.fullmatch(object_key)
        if match is None:
            return None
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            return None
