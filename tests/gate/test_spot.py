"""Test Gate Spot Kline and trade normalization and ingestion."""

from datetime import UTC, date, datetime
import gzip
import hashlib
from pathlib import Path

import httpx
import pandas as pd
import pyarrow as pa
import pytest

from veldra.core.models import DataValidationError, IntegritySpec, Resource
from veldra.gate.connector import GateConnector
from veldra.gate.datasets import get_dataset
from veldra.gate.processing import normalize_chunk, validate_chunk


def raw(columns: tuple[str, ...], rows: list[list[str]]) -> pa.Table:
    """Build the string-typed table emitted by Arrow CSV ingestion."""
    return pa.table(
        {
            column: pa.array([row[index] for row in rows])
            for index, column in enumerate(columns)
        }
    )


def test_spot_klines_map_gate_column_order_and_seconds() -> None:
    """Normalize Gate's close-first source layout into canonical OHLCV order."""
    dataset = get_dataset("spot", "klines")
    table = raw(
        dataset.source_columns,
        [["1735689600", "0.5", "101", "102", "99", "100"]],
    )

    normalized = normalize_chunk(table, dataset)

    assert normalized.column_names == list(dataset.stored_columns)
    assert normalized["open_time"][0].as_py() == datetime(2025, 1, 1, tzinfo=UTC)
    assert normalized["open"][0].as_py() == 100.0
    assert normalized["base_volume"][0].as_py() == 0.5
    assert validate_chunk(normalized, dataset, date(2025, 1, 1)) == datetime(
        2025, 1, 1, tzinfo=UTC
    )


def test_spot_trades_preserve_microseconds_and_derive_quote_quantity() -> None:
    """Normalize exact fractional seconds, identifiers, quantities, and sides."""
    dataset = get_dataset("spot", "trades")
    table = raw(
        dataset.source_columns,
        [
            ["1735689940.250206", "12778724858", "10", "2", "2"],
            ["1735689940.250207", "12778724859", "11", "3", "1"],
        ],
    )

    normalized = normalize_chunk(table, dataset)

    assert normalized["event_time"].to_pylist() == [
        datetime(2025, 1, 1, 0, 5, 40, 250206, tzinfo=UTC),
        datetime(2025, 1, 1, 0, 5, 40, 250207, tzinfo=UTC),
    ]
    assert normalized["quote_quantity"].to_pylist() == [20.0, 33.0]
    assert normalized["side"].to_pylist() == ["buy", "sell"]
    validate_chunk(
        normalized,
        dataset,
        date(2025, 1, 1),
        end_day=date(2025, 1, 31),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timestamp", "1735689940250", "timestamp unit"),
        ("deal_id", "1.5", "integer"),
        ("price", "nan", "finite"),
        ("amount", "0", "positive"),
        ("side", "3", "side"),
    ],
)
def test_invalid_spot_trade_values_are_rejected(
    field: str, value: str, message: str
) -> None:
    """Reject malformed source values before they reach Parquet.

    Args:
        field: The raw Gate field changed for the test.
        value: Its invalid source value.
        message: The expected validation message.
    """
    dataset = get_dataset("spot", "trades")
    values = {
        "timestamp": "1735689940.250206",
        "deal_id": "1",
        "price": "10",
        "amount": "2",
        "side": "2",
    }
    values[field] = value
    table = raw(
        dataset.source_columns,
        [[values[column] for column in dataset.source_columns]],
    )

    with pytest.raises(DataValidationError, match=message):
        normalized = normalize_chunk(table, dataset)
        validate_chunk(
            normalized,
            dataset,
            date(2025, 1, 1),
            end_day=date(2025, 1, 31),
        )


def test_gzip_ingestion_sorts_spot_trades_and_verifies_etag(tmp_path: Path) -> None:
    """Convert an out-of-order verified monthly archive into sorted Parquet."""
    dataset = get_dataset("spot", "trades")
    csv = b"1735689940.250207,2,11,3,1\n" b"1735689940.250206,1,10,2,2\n"
    payload = gzip.compress(csv)
    digest = hashlib.md5(payload).hexdigest()
    resource = Resource(
        date(2025, 1, 1),
        "https://example/BTC_USDT-202501.csv.gz",
        None,
        end_day=date(2025, 1, 31),
        cadence="monthly",
        integrity=IntegritySpec("response_header", "md5", expected=digest),
    )

    def handler(_: httpx.Request) -> httpx.Response:
        """Return one source Gzip with matching response metadata."""
        return httpx.Response(200, content=payload, headers={"ETag": f'"{digest}"'})

    destination = tmp_path / "trades.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        metadata = GateConnector(retries=0).ingest(
            http,
            resource,
            dataset,
            destination,
        )

    frame = pd.read_parquet(destination)
    assert metadata.archive_checksum == digest
    assert frame.event_number.tolist() == [1, 2]
    assert frame.event_time.is_monotonic_increasing
