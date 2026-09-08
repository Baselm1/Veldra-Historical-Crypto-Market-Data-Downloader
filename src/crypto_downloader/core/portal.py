"""Read paginated XML archive listings shared by archive portals."""

from collections.abc import Iterator
import logging
import xml.etree.ElementTree as ElementTree
import httpx
from .download import get

LOGGER = logging.getLogger(__name__)


def pages(
    client: httpx.Client,
    prefix: str,
    *,
    listing_url: str,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.5,
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
        root = _listing_root(
            get(
                client,
                listing_url,
                params=params,
                timeout=timeout,
                retries=retries,
                backoff=backoff,
            ).content
        )
        keys, prefixes, truncated, next_marker = _listing_values(root)
        LOGGER.debug(
            "Archive listing page: prefix=%s marker=%s keys=%d prefixes=%d "
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
