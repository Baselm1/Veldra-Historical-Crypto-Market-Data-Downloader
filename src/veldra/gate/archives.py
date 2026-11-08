"""List Gate Spot candlestick objects from its public archive bucket."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
import re
from urllib.parse import quote
from xml.etree import ElementTree

import httpx

from veldra.core.download import get

ARCHIVE_URL = "https://download.gatedata.org"
BUCKET_URL = "https://gateio-public-data.s3.ap-northeast-1.amazonaws.com/"
_OBJECT_NAME = re.compile(r"([^/]+)-(\d{6}|\d{8})\.csv\.gz")
_PLAIN_MD5 = re.compile(r"[0-9a-fA-F]{32}")


@dataclass(frozen=True)
class SpotKlineArchive:
    """Identify one observed Spot archive independently of the trading roster."""

    symbol: str
    day: date
    cadence: str
    url: str
    checksum: str | None


def _page(content: bytes, prefix: str) -> tuple[ElementTree.Element, str | None]:
    """Validate the listing scope and its continuation state."""
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise ValueError("Gate archive listing is not valid XML") from error
    if root.tag.rsplit("}", 1)[-1] != "ListBucketResult":
        raise ValueError("Gate archive listing has an invalid root")
    if root.findtext("{*}Prefix") != prefix:
        raise ValueError("Gate archive listing returned a different prefix")
    truncated = root.findtext("{*}IsTruncated")
    if truncated == "false":
        return root, None
    token = root.findtext("{*}NextContinuationToken")
    if truncated != "true" or not token:
        raise ValueError("Gate archive listing has an invalid continuation state")
    return root, token


def _object(
    item: ElementTree.Element, prefix: str, month: date
) -> SpotKlineArchive | None:
    """Decode one candle object while rejecting results outside the prefix."""
    key = item.findtext("{*}Key")
    if not key or not key.startswith(prefix) or "/" in key.removeprefix(prefix):
        raise ValueError("Gate archive listing returned an object outside its prefix")
    match = _OBJECT_NAME.fullmatch(key.rsplit("/", 1)[-1])
    if match is None:
        return None
    symbol, compact = match.groups()
    monthly = len(compact) == 6
    day = datetime.strptime(compact, "%Y%m" if monthly else "%Y%m%d").date()
    if day.replace(day=1) != month:
        raise ValueError("Gate archive date does not match its folder")
    etag = (item.findtext("{*}ETag") or "").strip('"')
    return SpotKlineArchive(
        symbol,
        day,
        "monthly" if monthly else "daily",
        f"{ARCHIVE_URL}/{quote(key, safe='/')}",
        etag.lower() if _PLAIN_MD5.fullmatch(etag) else None,
    )


def list_spot_klines(
    client: httpx.Client,
    month: date,
    interval: str,
    *,
    symbol: str | None = None,
    timeout: float,
    retries: int,
    backoff: float,
) -> Iterator[SpotKlineArchive]:
    """Yield the daily and monthly files in a native interval and source month."""
    if month.day != 1 or interval not in {"10s", "1m", "5m"}:
        raise ValueError(
            "Gate archive listing requires a month start and native interval"
        )
    if symbol is not None and re.fullmatch(r"[A-Z0-9_]+", symbol) is None:
        raise ValueError("Gate archive listing contains an unsafe symbol")
    prefix = f"spot/candlesticks_{interval}/{month:%Y%m}/"
    if symbol is not None:
        prefix += f"{symbol}-"
    params = {"list-type": "2", "prefix": prefix}
    seen: set[str] = set()
    while True:
        response = get(
            client,
            BUCKET_URL,
            params=params,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
        root, token = _page(response.content, prefix)
        if token is not None and token in seen:
            raise ValueError("Gate archive listing repeated a continuation token")
        for item in root.findall("{*}Contents"):
            archive = _object(item, prefix, month)
            if archive is not None:
                yield archive
        if token is None:
            return
        seen.add(token)
        params["continuation-token"] = token
