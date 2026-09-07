"""Discover Binance market metadata and daily archives."""

from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime
import logging
import math
from pathlib import Path
import re
from urllib.parse import quote
import xml.etree.ElementTree as ElementTree

import httpx

from ..http import archive_checksum, get
from ..datasets import DatasetSpec, get_dataset
from ..ingest import ingest_archive
from ..models import IngestedResource, Market, Resource, ResourceKey
from ..request import normalize_pair

EXCHANGE_INFO_URLS: Mapping[str, str] = {
    "spot": "https://api.binance.com/api/v3/exchangeInfo",
    "um": "https://fapi.binance.com/fapi/v1/exchangeInfo",
    "cm": "https://dapi.binance.com/dapi/v1/exchangeInfo",
}
BUCKET_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE_URL = "https://data.binance.vision"
DAILY_ROOTS: Mapping[str, str] = {
    "spot": "data/spot/daily",
    "um": "data/futures/um/daily",
    "cm": "data/futures/cm/daily",
}
LOGGER = logging.getLogger(__name__)


class BinanceSource:
    """Discover metadata from Binance and its public archive bucket."""

    code: str = "binance"
    products: tuple[str, ...] = ("spot", "um", "cm")
    active_statuses: frozenset[str] = frozenset({"TRADING"})
    max_concurrency: int = 32

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
            self._exchange_url(product),
            self._exchange_params(product),
        )
        current, excluded = self._exchange_markets(response.json(), product)
        exchange_count = len(current)
        archive_symbols = self._archive_symbols(client, product)
        self._merge_archive_only(current, excluded, archive_symbols, product)
        markets = [current[symbol] for symbol in sorted(current)]
        LOGGER.info(
            "Binance markets loaded: product=%s exchange=%d archive=%d merged=%d",
            product,
            exchange_count,
            len(archive_symbols),
            len(markets),
        )
        return markets

    def checksum(self, client: httpx.Client, resource: Resource) -> str:
        """Return Binance's current SHA-256 digest for one archive.

        Args:
            client: The HTTPX client used for Binance requests.
            resource: The Binance archive whose sidecar is checked.

        Returns:
            The lowercase SHA-256 digest declared by Binance.
        """
        return archive_checksum(
            client,
            resource,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

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
        dataset = self._validate_resource_request(key, start_day, end_day)
        prefix, stem, archive_symbol = self._archive_layout(key, dataset)
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
                    found[day] = self._resource(day, url, archive_symbol, dataset)
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

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Return the first Binance daily archive on or after a boundary.

        Args:
            client: The HTTPX client used for Binance requests.
            key: The requested Binance dataset identity.
            start_day: The earliest acceptable archive day, or ``None`` for
                the first archive in the source folder.
            end_day: The latest acceptable archive day.

        Returns:
            The first matching daily archive, or ``None`` when none exists.
        """
        dataset = self._validate_resource_request(key, start_day, end_day)
        prefix, stem, archive_symbol = self._archive_layout(key, dataset)
        marker = f"{prefix}{stem}{start_day.isoformat()}" if start_day else None
        pattern = re.compile(re.escape(prefix + stem) + r"(\d{4}-\d{2}-\d{2})\.zip")
        for keys, _ in self._pages(client, prefix, marker=marker, max_keys=2):
            for object_key in keys:
                day = self._resource_day(object_key, pattern)
                if day is None or (start_day is not None and day < start_day):
                    continue
                if day > end_day:
                    return None
                url = f"{ARCHIVE_URL}/{quote(object_key, safe='/')}"
                return self._resource(day, url, archive_symbol, dataset)
        return None

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
    def _exchange_markets(
        payload: object, product: str
    ) -> tuple[dict[str, Market], set[str]]:
        """Parse a complete product-specific exchange-info response.

        Args:
            payload: The decoded exchange-info JSON value.
            product: The Binance product represented by the response.

        Returns:
            Included markets and deliberately excluded native symbols.
        """
        BinanceSource._exchange_url(product)
        markets: dict[str, Market] = {}
        excluded: set[str] = set()
        for value in BinanceSource._exchange_rows(payload):
            parsed = BinanceSource._parsed_exchange_market(value, product)
            if parsed is None:
                continue
            symbol, market = parsed
            if symbol in markets or symbol in excluded:
                raise ValueError("exchangeInfo contains a duplicate symbol")
            if market is None:
                excluded.add(symbol)
            else:
                markets[symbol] = market
        return markets, excluded

    @staticmethod
    def _exchange_rows(payload: object) -> list[object]:
        """Read the nonempty market rows from an exchange-info response.

        Args:
            payload: The decoded exchange-info JSON value.

        Returns:
            The source market objects in their source order.
        """
        rows = payload.get("symbols") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("exchangeInfo contains no market snapshot")
        return rows

    @staticmethod
    def _parsed_exchange_market(
        value: object, product: str
    ) -> tuple[str, Market | None] | None:
        """Parse one source row or classify it outside the perpetual scope.

        Args:
            value: The decoded exchange market object.
            product: The Binance product represented by the market.

        Returns:
            The native symbol and included market, the symbol and ``None`` when
            excluded, or ``None`` for a non-ASCII symbol.
        """
        if not isinstance(value, dict):
            raise ValueError("exchangeInfo contains an invalid market")
        symbol = BinanceSource._exchange_symbol(value)
        if symbol is None:
            return None
        if (
            product != "spot"
            and BinanceSource._required_text(value, "contractType") != "PERPETUAL"
        ):
            return symbol, None
        return symbol, BinanceSource._exchange_market(value, product, symbol)

    @staticmethod
    def _exchange_url(product: str) -> str:
        """Return the exchange-info endpoint for one Binance product.

        Args:
            product: The Binance product identifier.

        Returns:
            The public exchange-info endpoint URL.
        """
        try:
            return EXCHANGE_INFO_URLS[product]
        except KeyError as error:
            raise ValueError(f"unsupported Binance product: {product}") from error

    @staticmethod
    def _exchange_params(product: str) -> Mapping[str, str] | None:
        """Return optional exchange-info parameters for one product.

        Args:
            product: The Binance product identifier.

        Returns:
            The optional query parameters required by the endpoint.
        """
        return {"showPermissionSets": "false"} if product == "spot" else None

    @staticmethod
    def _exchange_symbol(value: object) -> str | None:
        """Read one safe native symbol from exchange-info.

        Args:
            value: The decoded market object.

        Returns:
            The native ASCII symbol, or ``None`` for a non-ASCII symbol.
        """
        if not isinstance(value, dict):
            raise ValueError("exchangeInfo contains an invalid market")
        symbol = BinanceSource._required_text(value, "symbol")
        if re.fullmatch(r"[A-Za-z0-9_]+", symbol) is None:
            if not symbol.isascii():
                LOGGER.debug(
                    "Ignoring unsupported non-ASCII Binance symbol: %s", symbol
                )
                return None
            raise ValueError("exchangeInfo contains an unsafe symbol")
        return symbol

    @staticmethod
    def _exchange_market(value: object, product: str, symbol: str) -> Market:
        """Parse one supported Spot or perpetual Futures market.

        Args:
            value: The decoded exchange market object.
            product: The Binance product represented by the market.
            symbol: The already validated native market symbol.

        Returns:
            The canonical market metadata for the included market.
        """
        if not isinstance(value, dict):
            raise ValueError("exchangeInfo contains an invalid market")
        if product == "spot":
            return Market(
                symbol=symbol,
                normalized_symbol=normalize_pair(symbol),
                base_asset=BinanceSource._required_text(value, "baseAsset"),
                quote_asset=BinanceSource._required_text(value, "quoteAsset"),
                status=BinanceSource._required_text(value, "status"),
            )

        status_field = "contractStatus" if product == "cm" else "status"
        return Market(
            symbol=symbol,
            normalized_symbol=normalize_pair(symbol),
            base_asset=BinanceSource._required_text(value, "baseAsset"),
            quote_asset=BinanceSource._required_text(value, "quoteAsset"),
            status=BinanceSource._required_text(value, status_field),
            pair=BinanceSource._required_text(value, "pair"),
            contract_type="PERPETUAL",
            contract_size=BinanceSource._contract_size(value, product),
            onboard_time=BinanceSource._source_time(value.get("onboardDate")),
            delivery_time=BinanceSource._delivery_time(value.get("deliveryDate")),
        )

    @staticmethod
    def _contract_size(value: Mapping[object, object], product: str) -> float | None:
        """Return the explicit COIN-M contract size when required.

        Args:
            value: The decoded perpetual Futures market object.
            product: The Binance Futures product represented by the object.

        Returns:
            The positive COIN-M contract size, or ``None`` for USD-M.
        """
        if product == "um":
            return None
        raw_size = value.get("contractSize")
        if isinstance(raw_size, bool) or not isinstance(raw_size, (int, float)):
            raise ValueError("exchangeInfo contains an invalid market")
        size = float(raw_size)
        if not math.isfinite(size) or size <= 0:
            raise ValueError("exchangeInfo contains an invalid market")
        return size

    @staticmethod
    def _source_time(value: object) -> datetime:
        """Convert one required Binance millisecond timestamp to UTC.

        Args:
            value: The millisecond timestamp supplied by exchange-info.

        Returns:
            The matching UTC timestamp.
        """
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("exchangeInfo contains an invalid market")
        try:
            return datetime.fromtimestamp(value / 1000, UTC)
        except (OverflowError, OSError, ValueError) as error:
            raise ValueError("exchangeInfo contains an invalid market") from error

    @staticmethod
    def _delivery_time(value: object) -> datetime | None:
        """Discard Binance's far-future perpetual delivery placeholder.

        Args:
            value: The millisecond delivery timestamp supplied by exchange-info.

        Returns:
            A real delivery timestamp, or ``None`` for the perpetual placeholder.
        """
        timestamp = BinanceSource._source_time(value)
        return None if timestamp.year >= 2100 else timestamp

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

    def _archive_symbols(self, client: httpx.Client, product: str) -> set[str]:
        """Return safe symbols represented by a product's Kline folders.

        Args:
            client: The HTTPX client used for bucket listings.
            product: The Binance product whose primary archive folders to list.

        Returns:
            The unique archive symbol names.
        """
        prefix = self._market_archive_root(product)
        symbols: set[str] = set()
        for _, prefixes in self._pages(client, prefix, delimiter="/"):
            for prefix in prefixes:
                symbol = self._folder_symbol(prefix, self._market_archive_root(product))
                if symbol is not None:
                    symbols.add(symbol)
        return symbols

    @staticmethod
    def _market_archive_root(product: str) -> str:
        """Return the Kline folder root used to enumerate market symbols.

        Args:
            product: The Binance product identifier.

        Returns:
            The slash-terminated Kline archive folder root.
        """
        try:
            return f"{DAILY_ROOTS[product]}/klines/"
        except KeyError as error:
            raise ValueError(f"unsupported Binance product: {product}") from error

    @staticmethod
    def _merge_archive_only(
        current: dict[str, Market],
        excluded: set[str],
        archive_symbols: set[str],
        product: str,
    ) -> None:
        """Merge valid archive-only perpetual or Spot markets into a snapshot.

        Args:
            current: Current included markets indexed by native symbol.
            excluded: Current endpoint symbols deliberately outside the scope.
            archive_symbols: Valid symbols found in the archive Kline folders.
            product: The Binance product represented by the snapshot.
        """
        for symbol in archive_symbols:
            if (
                symbol in current
                or symbol in excluded
                or not BinanceSource._is_archive_perpetual(symbol, product)
            ):
                continue
            current[symbol] = Market(
                symbol=symbol,
                normalized_symbol=normalize_pair(symbol),
                pair=(
                    symbol.removesuffix("_PERP")
                    if product == "cm"
                    else symbol if product == "um" else None
                ),
                contract_type="PERPETUAL" if product != "spot" else None,
            )

    @staticmethod
    def _is_archive_perpetual(symbol: str, product: str) -> bool:
        """Return whether an archive symbol belongs to the current scope.

        Args:
            symbol: The safe archive folder symbol to inspect.
            product: The Binance product represented by the archive folder.

        Returns:
            True for Spot symbols or a standard perpetual Futures symbol.
        """
        if product == "spot":
            return True
        if product == "um":
            return re.search(r"_\d{6}$", symbol) is None
        return symbol.endswith("_PERP")

    @staticmethod
    def _folder_symbol(prefix: str, dataset_root: str) -> str | None:
        """Extract one safe symbol from an exact Spot kline folder.

        Args:
            prefix: The folder prefix returned by the bucket.
            dataset_root: The exact product and dataset folder root.

        Returns:
            The native symbol, or ``None`` for an unrelated or unsafe folder.
        """
        if not prefix.startswith(dataset_root) or not prefix.endswith("/"):
            return None
        symbol = prefix[len(dataset_root) : -1]
        return symbol if re.fullmatch(r"[A-Za-z0-9_]+", symbol) else None

    @staticmethod
    def _resource(
        day: date, url: str, archive_symbol: str, dataset: DatasetSpec
    ) -> Resource:
        """Build one discovered resource with source routing and schema metadata.

        Args:
            day: The UTC day represented by the archive.
            url: The public ZIP archive URL.
            archive_symbol: The symbol used in the archive path and filename.
            dataset: The declaration used to parse the archive.

        Returns:
            A discovered resource ready for catalog persistence.
        """
        return Resource(
            day=day,
            url=url,
            checksum_url=f"{url}.CHECKSUM",
            archive_symbol=archive_symbol,
            timestamp_column=dataset.time_column,
            schema_version=dataset.schema_version,
        )

    @staticmethod
    def _dataset_root(product: str, dataset: DatasetSpec) -> str:
        """Return the Binance folder root for one supported dataset.

        Args:
            product: The Binance product identifier.
            dataset: The dataset declaration with its Binance folder name.

        Returns:
            The slash-terminated bucket folder root.

        Raises:
            ValueError: If the product has no declared daily archive root.
        """
        try:
            daily_root = DAILY_ROOTS[product]
        except KeyError as error:
            raise ValueError(f"unsupported Binance product: {product}") from error
        return f"{daily_root}/{dataset.remote_name}/"

    @staticmethod
    def _archive_layout(key: ResourceKey, dataset: DatasetSpec) -> tuple[str, str, str]:
        """Build the archive folder and filename prefix for one dataset key.

        Args:
            key: The requested local dataset identity and optional source symbol.
            dataset: The matching declarative dataset schema.

        Returns:
            The archive folder prefix, file stem, and effective archive symbol.
        """
        archive_symbol = key.archive_symbol or key.symbol
        root = BinanceSource._dataset_root(key.product, dataset)
        if dataset.needs_interval:
            assert key.interval is not None
            prefix = f"{root}{archive_symbol}/{key.interval}/"
            stem = f"{archive_symbol}-{key.interval}-"
        else:
            prefix = f"{root}{archive_symbol}/"
            stem = f"{archive_symbol}-{dataset.remote_name}-"
        return prefix, stem, archive_symbol

    def _pages(
        self,
        client: httpx.Client,
        prefix: str,
        *,
        delimiter: str | None = None,
        marker: str | None = None,
        max_keys: int | None = None,
    ) -> Iterator[tuple[list[str], list[str]]]:
        """Yield every valid page from one bucket listing.

        Args:
            client: The HTTPX client used for bucket listings.
            prefix: The object prefix to list.
            delimiter: The optional folder delimiter.
            marker: The optional key after which listing should begin.
            max_keys: The optional maximum objects returned per page.

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
            if max_keys is not None:
                params["max-keys"] = str(max_keys)
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
        self, key: ResourceKey, start_day: date | None, end_day: date
    ) -> DatasetSpec:
        """Reject unsupported or unsafe daily resource requests.

        Args:
            key: The requested source dataset identity.
            start_day: The first archive day to include, or ``None`` when
                finding the first source archive.
            end_day: The last archive day to include.
        """
        if key.source != self.code:
            raise ValueError(f"unsupported source: {key.source}")
        self._check_product(key.product)
        try:
            dataset = get_dataset(key.product, key.dataset)
        except ValueError as error:
            raise ValueError(f"unsupported Binance dataset: {key.dataset}") from error
        if key.interval != dataset.base_interval:
            raise ValueError(f"unsupported Binance interval: {key.interval}")
        if re.fullmatch(r"[A-Za-z0-9_]+", key.symbol) is None:
            raise ValueError("invalid Binance symbol")
        if (
            key.archive_symbol is not None
            and re.fullmatch(r"[A-Za-z0-9_]+", key.archive_symbol) is None
        ):
            raise ValueError("invalid Binance archive symbol")
        if start_day is not None and end_day < start_day:
            raise ValueError("Binance date range ends before it starts")
        return dataset

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
