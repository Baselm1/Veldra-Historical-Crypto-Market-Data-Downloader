"""Connect Bitget public markets, manifests, and XLSX archives to core."""

from datetime import date, timedelta
from dataclasses import replace
import base64
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit, urlunsplit

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from veldra.bitget.client import BitgetClient, BitgetRateLimiter
from veldra.bitget.datasets import ARCHIVE_DATASETS, PRODUCTS, get_dataset, supports
from veldra.bitget.manifest import BitgetManifestDiscovery
from veldra.bitget.markets import markets, quote_volumes
from veldra.bitget.processing import normalize_chunk, validate_chunk
from veldra.bitget.xlsx import ingest_xlsx_archive
from veldra.core.datasets import DatasetSpec
from veldra.core.ingest import ingest_archive
from veldra.core.models import (
    IngestedResource,
    Market,
    Resource,
    ResourceKey,
)

_ETAG = re.compile(r"[0-9a-fA-F]{32}")
_PARTS_PREFIX = "veldra-parts="


def _packed_resource(parts: tuple[Resource, ...]) -> Resource:
    """Return one catalog-safe resource retaining every physical shard.

    Args:
        parts: Same-day physical archives returned by the portal.

    Returns:
        A logical resource whose URL fragment contains all physical URLs.
    """
    primary = parts[0]
    if len(parts) == 1:
        return primary
    payload = json.dumps([part.url for part in parts], separators=(",", ":"))
    encoded = base64.urlsafe_b64encode(payload.encode()).decode()
    return replace(
        primary,
        url=f"{primary.url}#{_PARTS_PREFIX}{encoded}",
    )


def _physical_resources(resource: Resource) -> tuple[Resource, ...]:
    """Restore physical CDN resources from one cataloged logical resource.

    Args:
        resource: Discovered or catalog-restored daily resource.

    Returns:
        One or more physical archives carrying MD5 ETag policies.
    """
    parts = urlsplit(resource.url)
    if not parts.fragment.startswith(_PARTS_PREFIX):
        return (resource,)
    try:
        encoded = parts.fragment.removeprefix(_PARTS_PREFIX)
        value = json.loads(base64.urlsafe_b64decode(encoded).decode())
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Bitget shard manifest is invalid") from error
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("Bitget shard manifest is invalid")
    resources: list[Resource] = []
    for url in value:
        parsed = urlsplit(url) if isinstance(url, str) else None
        if (
            parsed is None
            or parsed.scheme != "https"
            or parsed.hostname != "img.bitgetimg.com"
        ):
            raise ValueError("Bitget shard manifest contains an unsafe URL")
        resources.append(
            replace(
                resource,
                url=urlunsplit(parsed._replace(fragment="")),
            )
        )
    return tuple(resources)


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
        self._onboard: dict[tuple[str, str], date] = {}

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
        """Return a stable source revision from every physical archive ETag."""
        checksums: list[str] = []
        for part in _physical_resources(resource):
            response = client.head(part.url, timeout=self.timeout)
            response.raise_for_status()
            value = str(response.headers.get("ETag", "")).strip().strip('"')
            if _ETAG.fullmatch(value) is None:
                raise ValueError("Bitget archive does not publish a plain MD5 ETag")
            checksums.append(value.lower())
        if len(checksums) == 1:
            return checksums[0]
        return hashlib.sha256("".join(checksums).encode()).hexdigest()

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return current Bitget markets for one product."""
        self._check_product(product)
        found = markets(self._client(client), product)
        for market in found:
            if market.onboard_time is not None:
                self._onboard[(product, market.symbol)] = market.onboard_time.date()
        return found

    def quote_volumes(self, client: httpx.Client, product: str) -> dict[str, float]:
        """Return rolling quote turnover indexed by native symbol."""
        self._check_product(product)
        return quote_volumes(self._client(client), product)

    def resources(
        self, client: httpx.Client, key: ResourceKey, start_day: date, end_day: date
    ) -> list[Resource]:
        """Return manifest archives in one inclusive UTC+8 date range."""
        self._validate(key, start_day, end_day)
        archive_symbol = key.archive_symbol or key.symbol
        if key.product != "spot":
            archive_symbol = archive_symbol.replace("/", "")
        found = BitgetManifestDiscovery(self._client(client)).discover(
            key.product,
            key.dataset,
            [archive_symbol],
            start_day,
            end_day,
        )
        grouped: dict[date, list[Resource]] = {}
        for resource in found:
            grouped.setdefault(resource.day, []).append(resource)
        result: list[Resource] = []
        for day in sorted(grouped):
            parts = tuple(grouped[day])
            result.append(_packed_resource(parts))
        return result

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Return the earliest archive found in bounded seven-day probes."""
        self._validate(key, start_day, end_day)
        hinted = self._onboard.get((key.product, key.symbol), date(2023, 1, 1))
        cursor = max(start_day or hinted, hinted, date(2023, 1, 1))
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
        parts = _physical_resources(resource)
        if len(parts) == 1:
            return self._ingest_one(client, resource, dataset, destination)
        return self._ingest_parts(client, parts, dataset, destination)

    def _ingest_one(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Ingest one physical CSV or workbook archive."""
        if dataset.name == "trades":
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
                allow_member_name_prefix=True,
            )
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

    def _ingest_parts(
        self,
        client: httpx.Client,
        parts: tuple[Resource, ...],
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Merge verified same-day shards into one deterministic Parquet file."""
        with TemporaryDirectory(prefix="veldra-bitget-parts-") as directory:
            root = Path(directory)
            ingested = [
                self._ingest_one(client, part, dataset, root / f"{index}.parquet")
                for index, part in enumerate(parts)
            ]
            table = pa.concat_tables(
                [
                    pq.read_table(root / f"{index}.parquet")
                    for index in range(len(parts))
                ]
            ).sort_by([(column, "ascending") for column in dataset.ordering_columns])
            validate_chunk(table, dataset, parts[0].day, None, parts[0].end_day)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination, compression="zstd", use_dictionary=False)
        bounds = pc.min_max(table[dataset.time_column]).as_py()
        first, last = bounds["min"], bounds["max"]
        assert first is not None and last is not None
        stat = destination.stat()
        digest = hashlib.sha256(
            "".join(item.archive_checksum for item in ingested).encode()
        ).hexdigest()
        return IngestedResource(
            digest,
            stat.st_size,
            stat.st_mtime_ns,
            table.num_rows,
            first,
            last,
            dataset.time_column,
            dataset.schema_version,
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
