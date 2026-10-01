"""Discover Bitget daily archives from the bounded public portal API."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
import re
from typing import Protocol, cast
from urllib.parse import urlsplit

from veldra.bitget.datasets import ARCHIVE_DATASETS, PRODUCTS
from veldra.core.models import IntegritySpec, Resource

SYMBOL_ENDPOINT = "/v1/statistics/public/download/getSymbolList"
MANIFEST_ENDPOINT = "/v1/statistics/public/download/getPublicDataV2"
SOURCE_TIMEZONE = timezone(timedelta(hours=8))
_SAFE_NAME = re.compile(r"[A-Za-z0-9_./() -]+")
_BUSINESS_LINES: Mapping[str, int] = {
    "spot": 1,
    "usdt_futures": 2,
    "usdc_futures": 2,
    "coin_futures": 2,
}
_BUSINESS_TYPES: Mapping[str, int] = {
    "klines": 1,
    "trades": 2,
    "best_book_snapshots": 3,
    "order_book_snapshots": 3,
}
_DEPTH_TYPES: Mapping[str, int] = {
    "best_book_snapshots": 1,
    "order_book_snapshots": 2,
}


class PortalClient(Protocol):
    """Describe the rate-limited portal operation used by discovery."""

    def portal(self, path: str, body: Mapping[str, object]) -> object:
        """Return one validated portal response data value."""
        raise NotImplementedError


@dataclass(frozen=True)
class ManifestRequest:
    """Describe one legal portal manifest request."""

    symbols: tuple[str, ...]
    begin: date
    end: date


def _route(product: object, dataset: object) -> tuple[int, int, int | None]:
    """Return portal route identifiers for one archive dataset.

    Args:
        product: Veldra Bitget product.
        dataset: Canonical archive dataset.

    Returns:
        Business line, business type, and optional depth type.
    """
    if not isinstance(product, str):
        raise TypeError("product must be a string")
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")
    if product not in PRODUCTS or dataset not in ARCHIVE_DATASETS:
        raise ValueError(f"unsupported Bitget archive dataset '{product}/{dataset}'")
    return (
        _BUSINESS_LINES[product],
        _BUSINESS_TYPES[dataset],
        _DEPTH_TYPES.get(dataset),
    )


def _symbols(values: Sequence[str]) -> tuple[str, ...]:
    """Return a unique stable list of safe portal display symbols.

    Args:
        values: Native display symbols selected for discovery.

    Returns:
        Validated symbols in caller order.
    """
    if not values:
        raise ValueError("manifest discovery requires at least one symbol")
    found: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("manifest symbols must be nonempty strings")
        symbol = value.strip()
        if _SAFE_NAME.fullmatch(symbol) is None:
            raise ValueError("manifest contains an unsafe symbol")
        if symbol not in found:
            found.append(symbol)
    return tuple(found)


def _windows(begin: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield inclusive seven-day portal windows.

    Args:
        begin: First inclusive source-calendar date.
        end: Last inclusive source-calendar date.

    Yields:
        Consecutive legal request ranges.
    """
    if isinstance(begin, datetime) or isinstance(end, datetime):
        raise TypeError("manifest bounds must be date values")
    if not isinstance(begin, date) or not isinstance(end, date):
        raise TypeError("manifest bounds must be date values")
    if begin > end:
        raise ValueError("manifest begin must not follow end")
    current = begin
    while current <= end:
        last = min(end, current + timedelta(days=6))
        yield current, last
        current = last + timedelta(days=1)


def chunk_requests(
    symbols: Sequence[str], begin: date, end: date
) -> list[ManifestRequest]:
    """Pack symbols and dates at the portal's accepted maxima.

    Args:
        symbols: Native display symbols.
        begin: First inclusive source-calendar date.
        end: Last inclusive source-calendar date.

    Returns:
        Five-symbol and seven-day request chunks.
    """
    safe = _symbols(symbols)
    requests: list[ManifestRequest] = []
    for index in range(0, len(safe), 5):
        batch = safe[index : index + 5]
        requests.extend(
            ManifestRequest(batch, first, last) for first, last in _windows(begin, end)
        )
    return requests


def _source_day(row: Mapping[str, object]) -> date:
    """Return one manifest row's UTC+8 archive date.

    Args:
        row: Native portal file record.

    Returns:
        Source-calendar date represented by the archive.
    """
    text = row.get("dateTimeStr")
    if isinstance(text, str):
        try:
            return date.fromisoformat(text)
        except ValueError as error:
            raise ValueError("Bitget manifest dateTimeStr is invalid") from error
    value = row.get("dateTime")
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Bitget manifest date is missing")
    try:
        return (
            datetime.fromtimestamp(float(value) / 1_000, UTC)
            .astimezone(SOURCE_TIMEZONE)
            .date()
        )
    except (ValueError, OverflowError) as error:
        raise ValueError("Bitget manifest dateTime is invalid") from error


def _required_text(row: Mapping[str, object], field: str) -> str:
    """Return one safe nonempty manifest string.

    Args:
        row: Native portal file record.
        field: Field to parse.

    Returns:
        Validated source string.
    """
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Bitget manifest {field} is invalid")
    text = value.strip()
    if field != "fileUrl" and _SAFE_NAME.fullmatch(text) is None:
        raise ValueError(f"Bitget manifest {field} is unsafe")
    return text


def _resource(value: object, dataset: str) -> Resource:
    """Parse one portal file row into a canonical daily resource.

    Args:
        value: Native manifest file row.
        dataset: Canonical archive dataset.

    Returns:
        Daily resource with UTC+8 source coverage.
    """
    if not isinstance(value, dict):
        raise ValueError("Bitget manifest rows must be objects")
    row = cast(dict[str, object], value)
    day = _source_day(row)
    url = _required_text(row, "fileUrl")
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != "img.bitgetimg.com":
        raise ValueError("Bitget manifest URL is not a trusted CDN URL")
    _required_text(row, "fileName")
    symbol = _required_text(row, "displayName")
    start = datetime.combine(day, time.min, SOURCE_TIMEZONE).astimezone(UTC)
    return Resource(
        day=day,
        url=url,
        checksum_url=None,
        archive_symbol=symbol,
        timestamp_column="open_time" if dataset == "klines" else "event_time",
        coverage_start=start,
        coverage_end=start + timedelta(days=1),
        integrity=IntegritySpec("response_header", algorithm="md5"),
    )


def _rows(value: object, endpoint: str) -> list[dict[str, object]]:
    """Require object rows from one portal endpoint.

    Args:
        value: Portal response data.
        endpoint: Endpoint name used in validation errors.

    Returns:
        Validated object rows.
    """
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Bitget {endpoint} data must be a list of objects")
    return cast(list[dict[str, object]], value)


class BitgetManifestDiscovery:
    """Search symbols and discover stable Bitget archive URLs."""

    def __init__(self, client: PortalClient) -> None:
        """Retain one source-wide rate-limited portal client.

        Args:
            client: Shared Bitget public client.
        """
        self.client = client

    def search_symbols(
        self, query: object, product: object, dataset: object
    ) -> list[str]:
        """Return deduplicated portal display symbols matching a query.

        Args:
            query: Source search text.
            product: Veldra Bitget product.
            dataset: Canonical archive dataset.

        Returns:
            Portal symbols in source order.
        """
        line, kind, _depth = _route(product, dataset)
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        rows = _rows(
            self.client.portal(
                SYMBOL_ENDPOINT,
                {
                    "displaySymbol": query,
                    "businessLine": line,
                    "businessType": kind,
                    "languageType": 0,
                },
            ),
            "symbol search",
        )
        found: list[str] = []
        for row in rows:
            symbol = _required_text(row, "displaySymbol")
            if symbol not in found:
                found.append(symbol)
        return found

    def discover(
        self,
        product: str,
        dataset: str,
        symbols: Sequence[str],
        begin: date,
        end: date,
    ) -> list[Resource]:
        """Return deduplicated daily resources across bounded requests.

        Args:
            product: Veldra Bitget product.
            dataset: Canonical archive dataset.
            symbols: Native portal display symbols.
            begin: First inclusive UTC+8 archive date.
            end: Last inclusive UTC+8 archive date.

        Returns:
            Stable daily resources sorted by date, symbol, and URL.
        """
        line, kind, depth = _route(product, dataset)
        found: dict[str, Resource] = {}
        for request in chunk_requests(symbols, begin, end):
            body: dict[str, object] = {
                "displaySymbol": list(request.symbols),
                "businessLine": line,
                "businessType": kind,
                "dateType": 1,
                "beginTimeStr": request.begin.isoformat(),
                "endTimeStr": request.end.isoformat(),
            }
            if depth is not None:
                body["deptType"] = depth
            rows = _rows(self.client.portal(MANIFEST_ENDPOINT, body), "manifest")
            for row in rows:
                resource = _resource(row, dataset)
                found.setdefault(resource.url, resource)
        return sorted(
            found.values(),
            key=lambda resource: (
                resource.day,
                resource.archive_symbol or "",
                resource.url,
            ),
        )
