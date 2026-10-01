"""Test Bitget connector validation and source wiring."""

from datetime import UTC, date, datetime
import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch
import zipfile

import httpx
import pyarrow.parquet as pq
import pytest

from veldra.bitget.connector import (
    BitgetConnector,
    _packed_resource,
    _physical_resources,
)
from veldra.bitget.datasets import get_dataset
from veldra.core.models import IngestedResource, IntegritySpec, Resource, ResourceKey


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


def test_shard_manifest_survives_catalog_round_trip() -> None:
    """Logical trade resources retain every physical same-day archive."""
    first = Resource(
        date(2025, 1, 1),
        "https://img.bitgetimg.com/a.zip",
        None,
        integrity=IntegritySpec("response_header", "md5"),
    )
    second = Resource(
        date(2025, 1, 1),
        "https://img.bitgetimg.com/b.zip",
        None,
        integrity=IntegritySpec("response_header", "md5"),
    )
    packed = _packed_resource((first, second))
    restored = _physical_resources(packed)
    assert packed.integrity_spec.algorithm == "md5"
    assert [item.url for item in restored] == [first.url, second.url]
    assert all(item.integrity_spec.algorithm == "md5" for item in restored)


def test_shard_checksum_combines_every_etag() -> None:
    """Refresh compares one deterministic revision across every shard."""
    values = {
        "/a.zip": "0123456789abcdef0123456789abcdef",
        "/b.zip": "fedcba9876543210fedcba9876543210",
    }
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"ETag": values[request.url.path]},
                request=request,
            )
        )
    )
    resources = tuple(
        Resource(
            date(2025, 1, 1),
            f"https://img.bitgetimg.com/{name}.zip",
            None,
            integrity=IntegritySpec("response_header", "md5"),
        )
        for name in ("a", "b")
    )
    expected = hashlib.sha256("".join(values.values()).encode()).hexdigest()
    assert BitgetConnector().checksum(client, _packed_resource(resources)) == expected
    client.close()


@pytest.mark.parametrize(
    "url",
    [
        "https://img.bitgetimg.com/a.zip#veldra-parts=bad",
        "https://img.bitgetimg.com/a.zip#veldra-parts=W10=",
    ],
)
def test_rejects_invalid_shard_manifests(url: str) -> None:
    """Malformed persisted shard metadata fails before network access."""
    resource = Resource(
        date(2025, 1, 1),
        url,
        None,
        integrity=IntegritySpec("response_header", "md5"),
    )
    with pytest.raises(ValueError, match="shard manifest"):
        _physical_resources(resource)


@pytest.mark.parametrize(
    ("dataset_name", "operation"),
    [("trades", "ingest_archive"), ("klines", "ingest_xlsx_archive")],
)
def test_ingestion_selects_physical_member_format(
    dataset_name: str, operation: str, tmp_path: Path
) -> None:
    """Trade CSV ZIPs and workbook ZIPs use their matching readers."""
    connector = BitgetConnector()
    resource = Resource(
        date(2025, 1, 1),
        "https://img.bitgetimg.com/a.zip",
        None,
        integrity=IntegritySpec("response_header", "md5"),
    )
    ingested = IngestedResource(
        "0" * 32,
        1,
        1,
        1,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
        "event_time" if dataset_name == "trades" else "open_time",
        1,
    )
    target = f"veldra.bitget.connector.{operation}"
    with patch(target, return_value=ingested) as reader:
        result = connector._ingest_one(
            Mock(),
            resource,
            get_dataset("spot", dataset_name),
            tmp_path / "data.parquet",
        )
    assert result is ingested
    reader.assert_called_once()


def test_ingests_real_trade_archive_member_naming(tmp_path: Path) -> None:
    """Bitget's market-prefixed trade CSV is accepted and normalized."""
    source = BytesIO()
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "BTCUSDT_SPBL_20250101_001.csv",
            "trade_id,timestamp,price,side,volume(quote),size(base)\n"
            "10,1735660800000,100,buy,200,2\n",
        )
    payload = source.getvalue()
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=payload,
                headers={"ETag": digest},
                request=request,
            )
        )
    )
    resource = Resource(
        date(2025, 1, 1),
        "https://img.bitgetimg.com/20250101_001.zip",
        None,
        integrity=IntegritySpec("response_header", "md5"),
    )
    destination = tmp_path / "trades.parquet"
    result = BitgetConnector()._ingest_one(
        client, resource, get_dataset("spot", "trades"), destination
    )
    assert result.row_count == 1
    assert pq.read_table(destination)["event_number"].to_pylist() == [10]
    client.close()
