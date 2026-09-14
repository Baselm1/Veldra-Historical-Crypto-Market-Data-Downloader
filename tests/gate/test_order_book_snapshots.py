"""Test Gate complete order-book snapshot ingestion."""

from datetime import UTC, date, datetime
import gzip
import hashlib
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from veldra.core.ingest import ArchiveError
from veldra.core.models import IntegritySpec, Resource
from veldra.gate.datasets import get_dataset
from veldra.gate.orderbook import ingest_order_book_day


def logical_resource(product: str = "spot") -> Resource:
    """Return one logical Gate snapshot day for a product."""
    return Resource(
        date(2025, 1, 1),
        f"https://example/{product}/orderbooks_slice/202501/BTC_USDT-2025010100.gz",
        None,
        integrity=IntegritySpec("archive_only"),
    )


def source(payload: bytes, *, etag: str | None = None) -> httpx.MockTransport:
    """Serve hour zero and report the other Gate hours as absent."""
    digest = etag or hashlib.md5(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve one hourly compressed JSON Lines object."""
        if not request.url.path.endswith("00.gz"):
            return httpx.Response(404)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"ETag": f'"{digest}"'})
        return httpx.Response(200, content=payload, headers={"ETag": f'"{digest}"'})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    ("product", "row", "expected_time", "expected_quantity"),
    [
        (
            "spot",
            {
                "id": 10,
                "current": 1_735_689_600_891,
                "update": 1_735_689_600_833,
                "bids": [["100", "2"]],
                "asks": [["101", "3"]],
            },
            datetime(2025, 1, 1, 0, 0, 0, 891000, tzinfo=UTC),
            2.0,
        ),
        (
            "um",
            {
                "id": 11,
                "current": 1_735_689_602.168,
                "update": 1_735_689_602.167,
                "bids": [{"p": "100", "s": 4}],
                "asks": [{"p": "101", "s": 5}],
            },
            datetime(2025, 1, 1, 0, 0, 2, 168000, tzinfo=UTC),
            4.0,
        ),
    ],
)
def test_snapshot_formats_become_complete_nested_book_states(
    tmp_path: Path,
    product: str,
    row: dict[str, object],
    expected_time: datetime,
    expected_quantity: float,
) -> None:
    """Normalize both source layouts with exact times and nested levels.

    Args:
        tmp_path: The isolated cache directory.
        product: The Spot or Futures product.
        row: One native JSON snapshot.
        expected_time: The exact normalized observation time.
        expected_quantity: The first normalized bid quantity.
    """
    payload = gzip.compress((json.dumps(row) + "\n").encode())
    destination = tmp_path / "snapshots.parquet"
    with httpx.Client(transport=source(payload)) as client:
        metadata = ingest_order_book_day(
            client,
            logical_resource(product),
            get_dataset(product, "order_book_snapshots"),
            destination,
            timeout=5,
            retries=0,
            backoff=0,
        )

    frame = pd.read_parquet(destination)
    assert metadata.row_count == 1
    assert frame.event_time.iloc[0] == expected_time
    assert frame.bids.iloc[0][0]["price"] == 100.0
    assert frame.bids.iloc[0][0]["quantity"] == expected_quantity


def test_snapshot_rows_are_sorted_by_observation_and_update_id(tmp_path: Path) -> None:
    """Produce deterministic query order even when source JSON Lines are reversed."""
    rows = [
        {
            "id": identifier,
            "current": timestamp,
            "update": timestamp,
            "bids": [["100", "1"]],
            "asks": [["101", "1"]],
        }
        for identifier, timestamp in [(2, 1_735_689_601_000), (1, 1_735_689_600_000)]
    ]
    payload = gzip.compress("\n".join(map(json.dumps, rows)).encode())
    destination = tmp_path / "snapshots.parquet"
    with httpx.Client(transport=source(payload)) as client:
        ingest_order_book_day(
            client,
            logical_resource(),
            get_dataset("spot", "order_book_snapshots"),
            destination,
            timeout=5,
            retries=0,
            backoff=0,
        )
    assert pd.read_parquet(destination).update_id.tolist() == [1, 2]


@pytest.mark.parametrize(
    "row",
    [
        {
            "id": -1,
            "current": 1_735_689_600_000,
            "update": 1_735_689_600_000,
            "bids": [],
            "asks": [],
        },
        {
            "id": 1,
            "current": "bad",
            "update": 1_735_689_600_000,
            "bids": [],
            "asks": [],
        },
        {
            "id": 1,
            "current": 1_735_689_600_000,
            "update": 1_735_689_600_000,
            "bids": [["bad", "1"]],
            "asks": [],
        },
    ],
)
def test_malformed_snapshot_rows_are_rejected_without_output(
    tmp_path: Path, row: dict[str, object]
) -> None:
    """Reject malformed identifiers, timestamps, and levels atomically.

    Args:
        tmp_path: The isolated cache directory.
        row: The malformed snapshot object.
    """
    payload = gzip.compress((json.dumps(row) + "\n").encode())
    destination = tmp_path / "snapshots.parquet"
    with httpx.Client(transport=source(payload)) as client:
        with pytest.raises(ArchiveError):
            ingest_order_book_day(
                client,
                logical_resource(),
                get_dataset("spot", "order_book_snapshots"),
                destination,
                timeout=5,
                retries=0,
                backoff=0,
            )
    assert not destination.exists()
