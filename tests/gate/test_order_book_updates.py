"""Test Gate Spot and Futures depth-update ingestion."""

from datetime import UTC, date, datetime
import gzip
import hashlib
from pathlib import Path

import httpx
import pandas as pd
import pytest

from veldra.core.ingest import ArchiveError
from veldra.core.models import IntegritySpec, Resource
from veldra.gate.orderbook import ingest_order_book_day
from veldra.gate.datasets import get_dataset
from veldra.gate.processing import normalize_chunk, validate_chunk
from tests.gate.test_spot import raw


@pytest.mark.parametrize(
    ("product", "row", "side", "quantity"),
    [
        ("spot", ["1735689600", "2", "set", "100", "2", "10", "0"], "bid", 2.0),
        ("um", ["1735689600", "make", "100", "-3", "10", "1"], "ask", 3.0),
        ("cm", ["1735689600", "take", "100", "4", "10", "2"], "bid", 4.0),
    ],
)
def test_depth_updates_preserve_level_actions_and_native_quantities(
    product: str,
    row: list[str],
    side: str,
    quantity: float,
) -> None:
    """Normalize update direction, action, identifiers, and quantity units.

    Args:
        product: The Gate product under test.
        row: One native update row.
        side: The expected canonical book side.
        quantity: The expected positive source quantity.
    """
    dataset = get_dataset(product, "order_book_updates")
    normalized = normalize_chunk(raw(dataset.source_columns, [row]), dataset)

    assert normalized["side"].to_pylist() == [side]
    assert normalized["action"].to_pylist() == [row[2] if product == "spot" else row[1]]
    field = "base_quantity" if product == "spot" else "contract_quantity"
    assert normalized[field].to_pylist() == [quantity]
    assert normalized["update_id"].to_pylist() == [10]
    validate_chunk(normalized, dataset, date(2025, 1, 1))


def test_hourly_updates_are_combined_into_one_ordered_daily_parquet(
    tmp_path: Path,
) -> None:
    """Download available hours concurrently and append them chronologically."""
    payloads = {
        "00": gzip.compress(b"1735689600,2,set,100,2,10,0\n"),
        "01": gzip.compress(b"1735693200,1,make,101,3,11,1\n"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve two hourly files and mark all remaining hours absent."""
        hour = request.url.path[-9:-7]
        payload = payloads.get(hour)
        if payload is None:
            return httpx.Response(404)
        digest = hashlib.md5(payload).hexdigest()
        if request.method == "HEAD":
            return httpx.Response(200, headers={"ETag": f'"{digest}"'})
        return httpx.Response(200, content=payload, headers={"ETag": f'"{digest}"'})

    resource = Resource(
        date(2025, 1, 1),
        "https://example/spot/orderbooks/202501/BTC_USDT-2025010100.csv.gz",
        None,
        integrity=IntegritySpec("archive_only"),
    )
    destination = tmp_path / "updates.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        metadata = ingest_order_book_day(
            client,
            resource,
            get_dataset("spot", "order_book_updates"),
            destination,
            timeout=5,
            retries=0,
            backoff=0,
        )

    frame = pd.read_parquet(destination)
    assert metadata.row_count == 2
    assert frame.update_id.tolist() == [10, 11]
    assert frame.event_time.tolist() == [
        pd.Timestamp(datetime(2025, 1, 1, 0, 0, tzinfo=UTC)),
        pd.Timestamp(datetime(2025, 1, 1, 1, 0, tzinfo=UTC)),
    ]


def test_multipart_etags_fall_back_to_gzip_integrity(tmp_path: Path) -> None:
    """Accept valid Gzip bytes when large Gate objects lack a plain MD5 ETag."""
    payload = gzip.compress(b"1735689600,set,100,-2,10,0\n")

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve only hour zero with a multipart S3 ETag."""
        if not request.url.path.endswith("00.csv.gz"):
            return httpx.Response(404)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"ETag": '"abcdef-2"'})
        return httpx.Response(200, content=payload, headers={"ETag": '"abcdef-2"'})

    resource = Resource(
        date(2025, 1, 1),
        "https://example/futures_usdt/orderbooks/202501/BTC_USDT-2025010100.csv.gz",
        None,
        integrity=IntegritySpec("archive_only"),
    )
    destination = tmp_path / "updates.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = ingest_order_book_day(
            client,
            resource,
            get_dataset("um", "order_book_updates"),
            destination,
            timeout=5,
            retries=0,
            backoff=0,
        )
    assert result.archive_checksum


def test_absent_order_book_day_fails_without_partial_output(tmp_path: Path) -> None:
    """Reject a logical day when none of its 24 hourly objects exists."""

    def missing(_: httpx.Request) -> httpx.Response:
        """Report every hourly candidate as absent."""
        return httpx.Response(404)

    resource = Resource(
        date(2025, 1, 1),
        "https://example/spot/orderbooks/202501/BTC_USDT-2025010100.csv.gz",
        None,
        integrity=IntegritySpec("archive_only"),
    )
    destination = tmp_path / "missing.parquet"
    with httpx.Client(transport=httpx.MockTransport(missing)) as client:
        with pytest.raises(ArchiveError, match="no hourly files"):
            ingest_order_book_day(
                client,
                resource,
                get_dataset("spot", "order_book_updates"),
                destination,
                timeout=5,
                retries=0,
                backoff=0,
            )
    assert not destination.exists()
