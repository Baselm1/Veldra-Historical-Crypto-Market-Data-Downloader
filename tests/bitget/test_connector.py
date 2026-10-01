"""Test Bitget connector validation and source wiring."""

from datetime import date

import httpx
import pytest

from veldra.bitget.connector import BitgetConnector
from veldra.core.models import IntegritySpec, Resource, ResourceKey


def key(dataset: str = "klines") -> ResourceKey:
    """Return one representative archive key."""
    return ResourceKey(
        "bitget",
        "spot",
        dataset,
        "BTCUSDT",
        "1m" if dataset == "klines" else None,
        "BTC/USDT",
    )


def test_connector_rejects_bad_identity_and_range() -> None:
    """Unsupported products, datasets, and reversed dates fail locally."""
    connector = BitgetConnector()
    with pytest.raises(ValueError, match="identity"):
        connector.resources(
            httpx.Client(),
            ResourceKey("other", "spot", "klines", "BTCUSDT", "1m"),
            date(2025, 1, 1),
            date(2025, 1, 1),
        )
    with pytest.raises(ValueError, match="reversed"):
        connector.resources(httpx.Client(), key(), date(2025, 1, 2), date(2025, 1, 1))


def test_checksum_reads_plain_etag() -> None:
    """Revision checks reuse the CDN's published MD5 ETag."""
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"ETag": '"0123456789abcdef0123456789abcdef"'},
                request=request,
            )
        )
    )
    resource = Resource(
        date(2025, 1, 1),
        "https://img.bitgetimg.com/a.zip",
        None,
        integrity=IntegritySpec("response_header", "md5"),
    )
    assert (
        BitgetConnector().checksum(client, resource)
        == "0123456789abcdef0123456789abcdef"
    )
    client.close()
