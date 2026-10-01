"""Discover stable Bybit daily trade archives from public sources."""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import cast
from urllib.parse import urlsplit

import httpx

from veldra.bybit.client import BybitClient
from veldra.bybit.datasets import PRODUCTS
from veldra.bybit.identities import symbol
from veldra.core.download import head
from veldra.core.models import ArchiveKey, ArchiveObject, IntegritySpec
from veldra.core.subjects import DataSubject

ARCHIVE_URL = "https://public.bybit.com"
MAX_MANIFEST_DAYS = 8


def manifest_windows(begin: date, end: date) -> list[tuple[date, date]]:
    """Split an inclusive range into Bybit's eight-date portal windows.

    Args:
        begin: First inclusive archive date.
        end: Last inclusive archive date.

    Returns:
        Contiguous legal manifest windows.
    """
    if begin > end:
        raise ValueError("manifest begin must not follow end")
    windows: list[tuple[date, date]] = []
    cursor = begin
    while cursor <= end:
        last = min(end, cursor + timedelta(days=MAX_MANIFEST_DAYS - 1))
        windows.append((cursor, last))
        cursor = last + timedelta(days=1)
    return windows


def _size(value: object, field: str = "size") -> int | None:
    """Parse optional nonnegative archive bytes.

    Args:
        value: Source size value or an empty value.
        field: Field name used in validation errors.

    Returns:
        Byte count or ``None``.
    """
    if value in {None, ""}:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Bybit archive {field} is invalid")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"Bybit archive {field} is invalid") from error
    if parsed < 0:
        raise ValueError(f"Bybit archive {field} is invalid")
    return parsed


def _https_url(value: object) -> str:
    """Return one safe Bybit archive URL.

    Args:
        value: Proposed source URL.

    Returns:
        Validated HTTPS URL.
    """
    if not isinstance(value, str):
        raise ValueError("Bybit archive URL is invalid")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc.endswith("bybit.com"):
        raise ValueError("Bybit archive URL is invalid")
    return value


def _source_date(value: object) -> date:
    """Parse one ISO portal archive date.

    Args:
        value: Portal date value.

    Returns:
        Parsed calendar date.
    """
    if not isinstance(value, str):
        raise ValueError("Bybit archive date is invalid")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Bybit archive date is invalid") from error


def _filename(url: str, declared: object) -> str:
    """Validate a manifest filename against its URL.

    Args:
        url: Validated archive URL.
        declared: Portal filename value.

    Returns:
        Safe filename.
    """
    if not isinstance(declared, str) or not declared or "/" in declared:
        raise ValueError("Bybit archive filename is invalid")
    if urlsplit(url).path.rsplit("/", 1)[-1] != declared:
        raise ValueError("Bybit archive filename does not match its URL")
    return declared


def _archive(
    *,
    product: str,
    subject: DataSubject,
    day: date,
    filename: str,
    url: str,
    provider: str,
    remote_size: int | None,
    revision_id: str | None = None,
) -> ArchiveObject:
    """Build one immutable physical trade-archive record."""
    key = ArchiveKey(
        "bybit",
        product,
        "trades",
        provider,
        subject.kind,
        subject.value,
        "daily",
        day,
        day,
        filename,
    )
    return ArchiveObject(
        key,
        url,
        remote_size=remote_size,
        integrity=IntegritySpec("archive_only"),
        revision_id=revision_id,
    )


def _manifest_params(subject: DataSubject, begin: date, end: date) -> dict[str, str]:
    """Build one legal Options trade-manifest query."""
    if subject.kind != "instrument_family":
        raise ValueError("Bybit Option trade archives require an instrument family")
    return {
        "bizType": "option",
        "productId": "trade",
        "symbols": symbol(subject.value),
        "interval": "daily",
        "periods": "",
        "startDay": begin.isoformat(),
        "endDay": end.isoformat(),
    }


def _manifest_rows(value: object) -> list[dict[str, object]]:
    """Return validated file objects from one portal result."""
    if not isinstance(value, dict) or not isinstance(value.get("list"), list):
        raise ValueError("Bybit manifest result must contain a file list")
    rows = value["list"]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Bybit manifest file rows must be objects")
    return cast(list[dict[str, object]], rows)


def _parse_manifest_row(
    row: Mapping[str, object], subject: DataSubject, begin: date, end: date
) -> ArchiveObject:
    """Parse one Options trade archive returned by the portal."""
    if row.get("bizType") != "option" or row.get("productId") != "trade":
        raise ValueError("Bybit manifest returned the wrong dataset")
    if row.get("interval") != "daily" or row.get("symbol") != subject.value:
        raise ValueError("Bybit manifest returned the wrong archive scope")
    day = _source_date(row.get("date"))
    if day < begin or day > end:
        raise ValueError("Bybit manifest returned a date outside the request")
    url = _https_url(row.get("url"))
    return _archive(
        product="options",
        subject=subject,
        day=day,
        filename=_filename(url, row.get("filename")),
        url=url,
        provider="portal",
        remote_size=_size(row.get("size")),
    )


class BybitTradeDiscovery:
    """Discover daily public trade archives without scraping HTML listings."""

    def __init__(
        self,
        client: BybitClient,
        *,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 0.5,
        max_workers: int = 32,
    ) -> None:
        """Retain shared clients and bounded archive-probe settings."""
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")
        self.client = client
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.max_workers = max_workers

    def discover(
        self,
        product: str,
        subject: DataSubject,
        begin: date,
        end: date,
    ) -> list[ArchiveObject]:
        """Return existing daily trade archives for one logical subject."""
        if product not in PRODUCTS:
            raise ValueError(f"unsupported Bybit product {product!r}")
        if begin > end:
            raise ValueError("archive discovery begins after it ends")
        if product == "options":
            return self._options(subject, begin, end)
        if subject.kind != "instrument":
            raise ValueError("Bybit trade archives require an instrument subject")
        native = symbol(subject.value)
        days = [
            begin + timedelta(days=offset) for offset in range((end - begin).days + 1)
        ]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(days))) as pool:
            found = list(
                pool.map(lambda day: self._direct(product, native, subject, day), days)
            )
        return [item for item in found if item is not None]

    def _options(
        self, subject: DataSubject, begin: date, end: date
    ) -> list[ArchiveObject]:
        """Return Options family archives through bounded portal windows."""
        found: list[ArchiveObject] = []
        for first, last in manifest_windows(begin, end):
            result = self.client.manifest(_manifest_params(subject, first, last))
            found.extend(
                _parse_manifest_row(row, subject, first, last)
                for row in _manifest_rows(result)
            )
        unique = {item.key.archive_id: item for item in found}
        if len(unique) != len(found):
            raise ValueError("Bybit manifest returned a duplicate archive")
        return sorted(unique.values(), key=lambda item: item.key.period_start)

    def _direct(
        self, product: str, native: str, subject: DataSubject, day: date
    ) -> ArchiveObject | None:
        """Probe one deterministic Spot or derivative trade URL."""
        filename = (
            f"{native}_{day.isoformat()}.csv.gz"
            if product == "spot"
            else f"{native}{day.isoformat()}.csv.gz"
        )
        folder = "spot" if product == "spot" else "trading"
        url = f"{ARCHIVE_URL}/{folder}/{native}/{filename}"
        try:
            response = head(
                self.client.client,
                url,
                timeout=self.timeout,
                retries=self.retries,
                backoff=self.backoff,
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code in {404, 410}:
                return None
            raise
        etag = response.headers.get("ETag")
        revision = None if etag is None else etag.strip().strip('"') or None
        return _archive(
            product=product,
            subject=subject,
            day=day,
            filename=filename,
            url=url,
            provider="public_archive",
            remote_size=_size(response.headers.get("Content-Length"), "Content-Length"),
            revision_id=revision,
        )
