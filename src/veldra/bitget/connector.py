"""Connect Bitget public markets, manifests, and XLSX archives to core."""

from datetime import date, timedelta
from pathlib import Path
import re

import httpx

from veldra.bitget.client import BitgetClient, BitgetRateLimiter
from veldra.bitget.datasets import ARCHIVE_DATASETS, PRODUCTS, get_dataset, supports
from veldra.bitget.manifest import BitgetManifestDiscovery
from veldra.bitget.markets import markets, quote_volumes
from veldra.bitget.processing import normalize_chunk, validate_chunk
from veldra.bitget.xlsx import ingest_xlsx_archive
from veldra.core.datasets import DatasetSpec
from veldra.core.models import IngestedResource, Market, Resource, ResourceKey

_ETAG = re.compile(r"[0-9a-fA-F]{32}")


class BitgetConnector:
    """Discover and ingest Bitget daily public archives."""

    code = "bitget"
    products: tuple[str, ...] = tuple(PRODUCTS)
    max_concurrency = 32
    monthly_datasets: frozenset[str] = frozenset()

    def __init__(
        self, *, timeout: float = 30.0, retries: int = 3, backoff: float = 0.5
    ) -> None:
        """Store HTTP settings and one source-wide quota registry."""
        self.timeout, self.retries, self.backoff = timeout, retries, backoff
        self.limiter = BitgetRateLimiter()

    def _client(self, client: httpx.Client) -> BitgetClient:
        """Adapt an engine-managed HTTP pool to the Bitget client."""
        return BitgetClient(
            client=client,
            limiter=self.limiter,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def checksum(self, client: httpx.Client, resource: Resource) -> str:
        """Return the archive's current plain MD5 ETag."""
        response = client.head(resource.url, timeout=self.timeout)
        response.raise_for_status()
        value = str(response.headers.get("ETag", "")).strip().strip('"')
        if _ETAG.fullmatch(value) is None:
            raise ValueError("Bitget archive does not publish a plain MD5 ETag")
        return value.lower()

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return current Bitget markets for one product."""
        self._check_product(product)
        return markets(self._client(client), product)

    def quote_volumes(self, client: httpx.Client, product: str) -> dict[str, float]:
        """Return rolling quote turnover indexed by native symbol."""
        self._check_product(product)
        return quote_volumes(self._client(client), product)

    def resources(
        self, client: httpx.Client, key: ResourceKey, start_day: date, end_day: date
    ) -> list[Resource]:
        """Return manifest archives in one inclusive UTC+8 date range."""
        self._validate(key, start_day, end_day)
        return BitgetManifestDiscovery(self._client(client)).discover(
            key.product,
            key.dataset,
            [key.archive_symbol or key.symbol],
            start_day,
            end_day,
        )

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Return the earliest archive found in bounded seven-day probes."""
        self._validate(key, start_day, end_day)
        cursor = max(start_day or date(2023, 1, 1), date(2023, 1, 1))
        while cursor <= end_day:
            last = min(end_day, cursor + timedelta(days=6))
            found = self.resources(client, key, cursor, last)
            if found:
                return found[0]
            cursor = last + timedelta(days=1)
        return None

    def ingest(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Convert one verified Bitget workbook archive into Parquet."""
        return ingest_xlsx_archive(
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

    def _check_product(self, product: str) -> None:
        """Reject products outside Bitget's declared set."""
        if product not in self.products:
            raise ValueError(f"unsupported Bitget product: {product}")

    def _validate(
        self, key: ResourceKey, start_day: date | None, end_day: date
    ) -> None:
        """Reject unsupported identities and reversed resource ranges."""
        self._check_product(key.product)
        if (
            key.source != self.code
            or key.dataset not in ARCHIVE_DATASETS
            or not supports(key.product, key.dataset)
        ):
            raise ValueError("unsupported Bitget resource identity")
        if start_day is not None and start_day > end_day:
            raise ValueError("resource range must not be reversed")
        get_dataset(key.product, key.dataset).resolve_interval(key.interval)
