"""Connect Bybit markets and daily public archives to Veldra core."""

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import httpx

from veldra.bybit.client import BybitClient, BybitRateLimiter
from veldra.bybit.datasets import PRODUCTS, get_dataset
from veldra.bybit.manifest import BybitOrderBookDiscovery, BybitTradeDiscovery
from veldra.bybit.markets import markets, quote_volumes
from veldra.bybit.orderbook import ingest_order_book
from veldra.bybit.processing import ingest_trades
from veldra.core.datasets import DatasetSpec
from veldra.core.download import head
from veldra.core.models import (
    ArchiveObject,
    IngestedResource,
    Market,
    Resource,
    ResourceKey,
)
from veldra.core.subjects import DataSubject

_ARCHIVE_DATASETS = frozenset({"trades", "order_book_updates"})
_SOURCE_START = {
    "spot": date(2021, 7, 1),
    "linear": date(2019, 1, 1),
    "inverse": date(2019, 1, 1),
    "options": date(2022, 1, 1),
}


class BybitConnector:
    """Discover current markets and normalize Bybit daily archives."""

    code = "bybit"
    products: tuple[str, ...] = tuple(PRODUCTS)
    max_concurrency = 32
    monthly_datasets: frozenset[str] = frozenset()

    def __init__(
        self, *, timeout: float = 30.0, retries: int = 3, backoff: float = 0.5
    ) -> None:
        """Store source settings and one exchange-wide rate limiter."""
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.limiter = BybitRateLimiter()
        self._onboard: dict[tuple[str, str], date] = {}

    def _client(self, client: httpx.Client) -> BybitClient:
        """Adapt one engine-managed connection pool to Bybit's client."""
        return BybitClient(
            client=client,
            limiter=self.limiter,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def checksum(self, client: httpx.Client, resource: Resource) -> str:
        """Return one opaque source revision when Bybit publishes an ETag."""
        response = head(
            client,
            resource.url,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )
        revision = str(response.headers.get("ETag", "")).strip().strip('"')
        if not revision:
            raise ValueError("Bybit archive does not publish an ETag")
        return revision

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return current instruments for one Bybit product."""
        self._check_product(product)
        found = markets(self._client(client), product)
        for market in found:
            if market.onboard_time is not None:
                self._onboard[(product, market.symbol)] = market.onboard_time.date()
        return found

    def quote_volumes(self, client: httpx.Client, product: str) -> dict[str, float]:
        """Return current quote turnover indexed by native market symbol."""
        self._check_product(product)
        return quote_volumes(self._client(client), product)

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return daily trade or order-book archives in an inclusive range."""
        self._validate(key, start_day, end_day)
        subject = self._subject(key)
        api = self._client(client)
        if key.dataset == "trades":
            found = BybitTradeDiscovery(api).discover(
                key.product, subject, start_day, end_day
            )
        else:
            found = BybitOrderBookDiscovery(api).discover(
                key.product, subject, start_day, end_day
            )
        return [self._resource(item) for item in found]

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Return the earliest archive found in bounded monthly probes."""
        self._validate(key, start_day, end_day)
        earliest = self._source_start(key)
        cursor = max(start_day or earliest, earliest)
        while cursor <= end_day:
            last = min(end_day, cursor + timedelta(days=30))
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
        """Stream one Bybit archive into a validated Parquet resource."""
        if dataset.name == "order_book_updates":
            return ingest_order_book(
                client,
                resource,
                dataset,
                destination,
                timeout=self.timeout,
                retries=self.retries,
                backoff=self.backoff,
            )
        if dataset.name == "trades":
            return ingest_trades(
                client,
                resource,
                dataset,
                destination,
                timeout=self.timeout,
                retries=self.retries,
                backoff=self.backoff,
            )
        raise ValueError(f"unsupported Bybit archive dataset: {dataset.name}")

    @staticmethod
    def _resource(value: ArchiveObject) -> Resource:
        """Adapt one stable physical object to the shared archive engine."""
        coverage_end = None
        if value.key.dataset == "order_book_updates":
            coverage_end = datetime.combine(
                value.key.period_end + timedelta(days=1), time.min, UTC
            ) + timedelta(minutes=5)
        return Resource(
            value.key.period_start,
            value.url,
            None,
            archive_symbol=value.key.remote_scope_value,
            end_day=value.key.period_end,
            coverage_end=coverage_end,
            integrity=value.integrity,
        )

    @staticmethod
    def _subject(key: ResourceKey) -> DataSubject:
        """Return the physical archive scope for one logical market."""
        if key.product == "options":
            base = key.symbol.split("-", 1)[0]
            return DataSubject("instrument_family", base)
        return DataSubject("instrument", key.symbol)

    def _source_start(self, key: ResourceKey) -> date:
        """Return the narrowest known source scan boundary."""
        onboard = self._onboard.get((key.product, key.symbol))
        return max(_SOURCE_START[key.product], onboard or date.min)

    @staticmethod
    def _check_product(product: str) -> None:
        """Reject products outside Bybit's declared public categories."""
        if product not in PRODUCTS:
            raise ValueError(f"unsupported Bybit product {product!r}")

    def _validate(
        self, key: ResourceKey, start_day: date | None, end_day: date
    ) -> None:
        """Reject unsupported or contradictory archive requests."""
        self._check_product(key.product)
        if key.source != self.code:
            raise ValueError("resource source must be bybit")
        if key.dataset not in _ARCHIVE_DATASETS:
            raise ValueError(f"Bybit dataset {key.dataset!r} is REST-backed")
        get_dataset(key.product, key.dataset)
        if key.interval is not None:
            raise ValueError("Bybit trade and order-book archives have no interval")
        if start_day is not None and start_day > end_day:
            raise ValueError("archive range begins after it ends")
