"""Discover Gate markets and deterministic historical archive resources."""

from calendar import monthrange
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
import math
from pathlib import Path
import re

import httpx

from veldra.core.datasets import DatasetSpec
from veldra.core.download import head, get
from veldra.core.ingest import ingest_gzip_archive
from veldra.core.models import (
    IngestedResource,
    IntegritySpec,
    Market,
    Resource,
    ResourceKey,
)
from veldra.core.request import normalize_pair
from veldra.gate.datasets import PRODUCTS, supports

API_URL = "https://api.gateio.ws/api/v4"
ARCHIVE_URL = "https://download.gatedata.org"
PRODUCT_PATHS: Mapping[str, str] = {
    "spot": "spot",
    "um": "futures_usdt",
    "cm": "futures_btc",
}
_SAFE_SYMBOL = re.compile(r"[A-Z0-9_]+")
_PLAIN_MD5 = re.compile(r"[0-9a-fA-F]{32}")


class GateConnector:
    """Discover Gate metadata and download its public historical files."""

    code = "gate"
    products = PRODUCTS
    max_concurrency = 32
    monthly_datasets: frozenset[str] = frozenset()

    def __init__(
        self, *, timeout: float = 30.0, retries: int = 3, backoff: float = 0.5
    ) -> None:
        """Store HTTP settings and source listing hints.

        Args:
            timeout: The timeout for each HTTP request in seconds.
            retries: The retries allowed after the first request attempt.
            backoff: The initial exponential retry delay in seconds.
        """
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._onboard_dates: dict[tuple[str, str], date] = {}

    def _get(self, client: httpx.Client, url: str) -> httpx.Response:
        """Make one retrying Gate API request."""
        return get(
            client,
            url,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def _head(self, client: httpx.Client, url: str) -> httpx.Response:
        """Make one retrying Gate archive metadata request."""
        return head(
            client,
            url,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def checksum(self, client: httpx.Client, resource: Resource) -> str:
        """Return the current plain MD5 ETag for one Gate archive.

        Args:
            client: The shared HTTP client.
            resource: The archive whose current revision is requested.

        Returns:
            The lowercase MD5 value published by Gate.
        """
        digest = self._etag(self._head(client, resource.url))
        if digest is None:
            raise ValueError("Gate archive does not publish a plain MD5 ETag")
        return digest

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return current and delisted Gate markets for one product.

        Args:
            client: The shared HTTP client.
            product: Spot, USDT-margined, or BTC-margined Futures.

        Returns:
            Valid markets ordered by native symbol.
        """
        self._check_product(product)
        endpoint = (
            f"{API_URL}/spot/currency_pairs"
            if product == "spot"
            else f"{API_URL}/futures/{self._settle(product)}/contracts_all"
        )
        rows = self._rows(self._get(client, endpoint).json(), "market")
        parsed = [self._market(row, product) for row in rows]
        markets = [market for market in parsed if market is not None]
        if len({market.symbol for market in markets}) != len(markets):
            raise ValueError("Gate market endpoint contains a duplicate symbol")
        for market in markets:
            if market.onboard_time is not None:
                self._onboard_dates[(product, market.symbol)] = (
                    market.onboard_time.date()
                )
        return sorted(markets, key=lambda market: market.symbol)

    def quote_volumes(self, client: httpx.Client, product: str) -> dict[str, float]:
        """Return 24-hour quote turnover indexed by native symbol.

        Args:
            client: The shared HTTP client.
            product: Spot, USDT-margined, or BTC-margined Futures.

        Returns:
            Finite nonnegative quote volumes.
        """
        self._check_product(product)
        endpoint = (
            f"{API_URL}/spot/tickers"
            if product == "spot"
            else f"{API_URL}/futures/{self._settle(product)}/tickers"
        )
        rows = self._rows(self._get(client, endpoint).json(), "ticker")
        symbol_field = "currency_pair" if product == "spot" else "contract"
        return self._volumes(rows, symbol_field)

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return Gate resources found in an inclusive source range.

        Args:
            client: The shared HTTP client.
            key: The exact product, dataset, symbol, interval, and cadence.
            start_day: The first requested source day.
            end_day: The last requested source day.

        Returns:
            Existing resources ordered by their first covered day.
        """
        self._validate_resource_request(key, start_day, end_day)
        periods = (
            self._month_starts(start_day, end_day)
            if key.cadence == "monthly"
            else self._days(start_day, end_day)
        )
        candidates = [self._candidate(key, period) for period in periods]
        if not candidates:
            return []
        workers = min(self.max_concurrency, len(candidates))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            found = list(pool.map(lambda item: self._probe(client, item), candidates))
        return [resource for resource in found if resource is not None]

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Return the first Gate resource within broad source bounds.

        Args:
            client: The shared HTTP client.
            key: The exact dataset and market identity.
            start_day: The earliest allowed source day, or ``None`` for all history.
            end_day: The final allowed source day.

        Returns:
            The earliest existing resource, or ``None``.
        """
        self._validate_resource_request(key, start_day, end_day)
        cursor = max(start_day or self._source_start(key), self._source_start(key))
        while cursor <= end_day:
            last = min(
                end_day,
                date(
                    cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1]
                ),
            )
            resources = self.resources(client, key, cursor, last)
            if resources:
                return resources[0]
            cursor = last + timedelta(days=1)
        return None

    def ingest(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Convert one Gate archive or logical order-book day into Parquet.

        Args:
            client: The shared HTTP client.
            resource: The discovered Gate resource.
            dataset: The canonical schema declaration.
            destination: The final Parquet path.

        Returns:
            Integrity and local materialization metadata.
        """
        if dataset.name in {"order_book_updates", "order_book_snapshots"}:
            from veldra.gate.orderbook import ingest_order_book_day

            return ingest_order_book_day(
                client,
                resource,
                dataset,
                destination,
                timeout=self.timeout,
                retries=self.retries,
                backoff=self.backoff,
            )
        from veldra.gate.processing import normalize_chunk, validate_chunk

        return ingest_gzip_archive(
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

    def _probe(self, client: httpx.Client, resource: Resource) -> Resource | None:
        """Attach available metadata or skip one absent candidate archive."""
        try:
            response = self._head(client, resource.url)
        except httpx.HTTPStatusError as error:
            if error.response.status_code in {404, 410}:
                return None
            raise
        digest = (
            None
            if resource.url.endswith(("00.csv.gz", "00.gz"))
            else self._etag(response)
        )
        integrity = (
            IntegritySpec("response_header", "md5", expected=digest)
            if digest is not None
            else IntegritySpec("archive_only")
        )
        return replace(resource, integrity=integrity)

    def _candidate(self, key: ResourceKey, period: date) -> Resource:
        """Build one deterministic Gate archive candidate."""
        symbol = key.archive_symbol or key.symbol
        folder = period.strftime("%Y%m")
        compact = period.strftime("%Y%m" if key.cadence == "monthly" else "%Y%m%d")
        if key.dataset in {"order_book_updates", "order_book_snapshots"}:
            compact = f"{period:%Y%m%d}00"
        suffix = ".gz" if key.dataset == "order_book_snapshots" else ".csv.gz"
        url = (
            f"{ARCHIVE_URL}/{PRODUCT_PATHS[key.product]}/{self._remote_name(key)}/"
            f"{folder}/{symbol}-{compact}{suffix}"
        )
        start = datetime.combine(period, time.min, UTC)
        monthly = key.cadence == "monthly"
        end_day = (
            date(period.year, period.month, monthrange(period.year, period.month)[1])
            if monthly
            else period
        )
        end = datetime.combine(end_day + timedelta(days=1), time.min, UTC)
        return Resource(
            day=period,
            url=url,
            checksum_url=None,
            archive_symbol=symbol,
            timestamp_column=self._time_column(key.dataset),
            end_day=end_day,
            cadence=key.cadence,
            coverage_start=start,
            coverage_end=end,
            integrity=IntegritySpec("archive_only"),
        )

    @staticmethod
    def _remote_name(key: ResourceKey) -> str:
        """Return the native archive directory for one canonical dataset."""
        if key.dataset == "klines":
            if key.interval is None:
                raise ValueError("Gate Kline resources require an interval")
            return f"candlesticks_{key.interval}"
        return {
            "trades": "deals" if key.product == "spot" else "trades",
            "order_book_updates": "orderbooks",
            "order_book_snapshots": "orderbooks_slice",
            "mark_prices": "mark_prices",
            "funding_rates": "funding_applies",
            "funding_rate_updates": "funding_updates",
        }[key.dataset]

    @staticmethod
    def _time_column(dataset: str) -> str:
        """Return the canonical timestamp field for one Gate dataset."""
        return "open_time" if dataset == "klines" else "event_time"

    def _source_start(self, key: ResourceKey) -> date:
        """Return the best inexpensive lower-bound hint for one market."""
        hint = self._onboard_dates.get((key.product, key.symbol))
        if hint is not None:
            return hint
        if key.dataset in {"order_book_updates", "order_book_snapshots"}:
            return date(2021, 8, 1)
        return date(2018, 1, 1) if key.product == "spot" else date(2021, 1, 1)

    @staticmethod
    def _rows(payload: object, kind: str) -> list[object]:
        """Return a validated list from one Gate public endpoint."""
        if not isinstance(payload, list):
            raise ValueError(f"Gate {kind} endpoint contains no snapshot")
        return payload

    @staticmethod
    def _market(value: object, product: str) -> Market | None:
        """Parse one Gate Spot or perpetual Futures market."""
        if not isinstance(value, dict):
            raise ValueError("Gate market endpoint contains an invalid market")
        if product == "spot":
            symbol = GateConnector._required_symbol(value.get("id"))
            base = GateConnector._required_asset(value.get("base"))
            quote = GateConnector._required_asset(value.get("quote"))
            status = GateConnector._required_text(value.get("trade_status"))
            onboard = GateConnector._source_time(value.get("sell_start"))
            active = status == "tradable"
            contract_size = None
        else:
            symbol = GateConnector._required_symbol(value.get("name"))
            base, quote = symbol.split("_", maxsplit=1)
            status = GateConnector._required_text(value.get("status"))
            onboard = GateConnector._source_time(
                value.get("launch_time") or value.get("create_time")
            )
            active = status == "trading" and value.get("in_delisting") is not True
            contract_size = GateConnector._positive_number(
                value.get("quanto_multiplier")
            )
        return Market(
            symbol=symbol,
            normalized_symbol=normalize_pair(symbol),
            base_asset=base,
            quote_asset=quote,
            status=status.upper(),
            pair=symbol,
            contract_type="PERPETUAL" if product != "spot" else None,
            contract_size=contract_size,
            onboard_time=onboard,
            active=active,
            product=product,
        )

    @staticmethod
    def _volumes(rows: Sequence[object], symbol_field: str) -> dict[str, float]:
        """Parse nonnegative Gate quote volumes."""
        volumes: dict[str, float] = {}
        for value in rows:
            if not isinstance(value, dict):
                raise ValueError("Gate ticker endpoint contains an invalid market")
            symbol = GateConnector._required_symbol(value.get(symbol_field))
            raw = value.get("quote_volume", value.get("volume_24h_quote"))
            if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                raise ValueError("Gate ticker endpoint contains an invalid market")
            try:
                volume = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Gate ticker endpoint contains an invalid market"
                ) from error
            if not math.isfinite(volume) or volume < 0:
                raise ValueError("Gate ticker endpoint contains an invalid market")
            volumes[symbol] = volume
        return volumes

    @staticmethod
    def _etag(response: httpx.Response) -> str | None:
        """Return a plain non-multipart MD5 ETag when Gate publishes one."""
        value = response.headers.get("ETag")
        if value is None:
            return None
        candidate = value.strip().removeprefix("W/").strip().strip('"')
        return candidate.lower() if _PLAIN_MD5.fullmatch(candidate) else None

    @staticmethod
    def _required_symbol(value: object) -> str:
        """Return one safe native Gate market symbol."""
        if not isinstance(value, str):
            raise ValueError("Gate endpoint contains an invalid market symbol")
        symbol = value.strip().upper()
        if _SAFE_SYMBOL.fullmatch(symbol) is None or "_" not in symbol:
            raise ValueError("Gate endpoint contains an invalid market symbol")
        return symbol

    @staticmethod
    def _required_asset(value: object) -> str:
        """Return one safe required asset code."""
        if not isinstance(value, str) or not value.strip().isalnum():
            raise ValueError("Gate market endpoint contains an invalid asset")
        return value.strip().upper()

    @staticmethod
    def _required_text(value: object) -> str:
        """Return one nonempty required text value."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Gate endpoint contains an invalid text value")
        return value.strip()

    @staticmethod
    def _positive_number(value: object) -> float | None:
        """Return a positive finite number or ``None`` for an unknown multiplier."""
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("Gate market endpoint contains an invalid multiplier")
        try:
            number = abs(float(value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Gate market endpoint contains an invalid multiplier"
            ) from error
        if not math.isfinite(number):
            raise ValueError("Gate market endpoint contains an invalid multiplier")
        return number or None

    @staticmethod
    def _source_time(value: object) -> datetime | None:
        """Parse one optional nonnegative epoch-second timestamp."""
        if value in {None, 0, "0"}:
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("Gate market endpoint contains an invalid timestamp")
        try:
            seconds = int(value)
        except ValueError as error:
            raise ValueError(
                "Gate market endpoint contains an invalid timestamp"
            ) from error
        if seconds < 0:
            raise ValueError("Gate market endpoint contains an invalid timestamp")
        return datetime.fromtimestamp(seconds, UTC)

    @staticmethod
    def _month_starts(start_day: date, end_day: date) -> list[date]:
        """Return calendar-month starts overlapping an inclusive day range."""
        periods: list[date] = []
        cursor = start_day.replace(day=1)
        while cursor <= end_day:
            periods.append(cursor)
            cursor = date(
                cursor.year + (cursor.month == 12),
                1 if cursor.month == 12 else cursor.month + 1,
                1,
            )
        return periods

    @staticmethod
    def _days(start_day: date, end_day: date) -> list[date]:
        """Return every day in one inclusive range."""
        return [
            start_day + timedelta(days=offset)
            for offset in range((end_day - start_day).days + 1)
        ]

    @staticmethod
    def _settle(product: str) -> str:
        """Return Gate's native Futures settlement path."""
        return "usdt" if product == "um" else "btc"

    def _check_product(self, product: str) -> None:
        """Reject products outside Gate Spot and perpetual Futures."""
        if product not in self.products:
            raise ValueError(f"unsupported Gate product: {product}")

    def _validate_resource_request(
        self, key: ResourceKey, start_day: date | None, end_day: date
    ) -> None:
        """Reject unsafe or unsupported Gate archive requests."""
        self._check_product(key.product)
        if key.source != self.code or not supports(key.product, key.dataset):
            raise ValueError("unsupported Gate resource identity")
        if start_day is not None and start_day > end_day:
            raise ValueError("resource range must not be reversed")
        symbol = key.archive_symbol or key.symbol
        if _SAFE_SYMBOL.fullmatch(symbol) is None:
            raise ValueError("resource identity contains an unsafe symbol")
        if key.dataset == "klines" and key.interval is None:
            raise ValueError("Gate Kline resources require an interval")
