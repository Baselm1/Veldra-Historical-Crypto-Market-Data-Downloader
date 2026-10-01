"""Test Bybit trade normalization, validation, and archive ingestion."""

from datetime import UTC, date, datetime
import gzip
import io
from pathlib import Path
import zipfile

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from veldra.bybit.datasets import get_dataset
from veldra.bybit.processing import ingest_trades, normalize_chunk, validate_chunk
from veldra.core.models import DataValidationError, IntegritySpec, Resource


def raw(columns: tuple[str, ...], rows: list[list[object]]) -> pa.Table:
    """Return a string-valued source table for representative CSV rows."""
    return pa.table(
        {
            column: pa.array([str(row[index]) for row in rows], type=pa.string())
            for index, column in enumerate(columns)
        }
    )


def test_legacy_spot_trade_derives_quote_quantity_and_false_rpi() -> None:
    """Normalize archives written before Bybit added the RPI flag."""
    dataset = get_dataset("spot", "trades")
    table = raw(
        dataset.source_schemas[0].columns,
        [["1", "1668038400525", "15910.61", "0.003383", "sell"]],
    )
    result = normalize_chunk(table, dataset)
    assert result["event_time"][0].as_py() == datetime(
        2022, 11, 10, 0, 0, 0, 525000, tzinfo=UTC
    )
    assert result["trade_id"].to_pylist() == ["1"]
    assert result["side"].to_pylist() == ["sell"]
    assert result["quote_quantity"][0].as_py() == pytest.approx(15910.61 * 0.003383)
    assert result["is_rpi"].to_pylist() == [False]


def test_current_spot_trade_parses_rpi_case_insensitively() -> None:
    """Preserve the current Spot retail-price-improvement flag."""
    dataset = get_dataset("spot", "trades")
    result = normalize_chunk(
        raw(
            dataset.source_columns,
            [["1", "1735689600000", "100", "2", "Buy", "TRUE"]],
        ),
        dataset,
    )
    assert result["is_rpi"].to_pylist() == [True]
    assert result["side"].to_pylist() == ["buy"]


def test_linear_trade_preserves_fractional_second_precision() -> None:
    """Map home and foreign notionals to base and quote quantities."""
    dataset = get_dataset("linear", "trades")
    result = normalize_chunk(
        raw(
            dataset.source_schemas[0].columns,
            [
                [
                    "1585180700.0647",
                    "BTCUSDT",
                    "Buy",
                    "0.042",
                    "6698.5",
                    "PlusTick",
                    "trade-a",
                    "28133700000",
                    "0.042",
                    "281.337",
                ]
            ],
        ),
        dataset,
    )
    assert result["event_time"][0].as_py() == datetime(
        2020, 3, 25, 23, 58, 20, 64700, tzinfo=UTC
    )
    assert result["base_quantity"].to_pylist() == [0.042]
    assert result["quote_quantity"].to_pylist() == [281.337]


def test_inverse_trade_keeps_contract_base_and_quote_units() -> None:
    """Represent inverse contract count, base amount, and quote notional."""
    dataset = get_dataset("inverse", "trades")
    result = normalize_chunk(
        raw(
            dataset.source_schemas[0].columns,
            [
                [
                    "1569974396.557895",
                    "BTCUSD",
                    "Buy",
                    "11668",
                    "8319.5",
                    "PlusTick",
                    "trade-a",
                    "140248813",
                    "11668",
                    "1.40248813",
                ]
            ],
        ),
        dataset,
    )
    assert result["contract_quantity"].to_pylist() == [11668.0]
    assert result["base_quantity"].to_pylist() == [1.40248813]
    assert result["quote_notional"].to_pylist() == [11668.0]


def test_option_trade_preserves_instrument_and_volatility_fields() -> None:
    """Normalize one row from a shared Option-family ZIP."""
    dataset = get_dataset("options", "trades")
    row: list[object] = [
        "b58d7ff2-6710-53d8-9d02-10beb13533e0",
        "75905525121",
        "1789862410029",
        "BTC-20SEP26-81000-C-USDT",
        "Sell",
        "325",
        "0.01",
        "0.1683",
        "81254.12186745",
        "322.98232785",
        "0.1649",
    ]
    result = normalize_chunk(raw(dataset.source_columns, [row]), dataset)
    assert result["instrument"].to_pylist() == ["BTC-20SEP26-81000-C-USDT"]
    assert result["trade_sequence"].to_pylist() == [75905525121]
    assert result["contract_quantity"].to_pylist() == [0.01]
    assert result["side"].to_pylist() == ["sell"]


def test_validation_accepts_equal_times_after_stable_secondary_sort() -> None:
    """Permit simultaneous trades while requiring unique stable IDs."""
    dataset = get_dataset("spot", "trades")
    table = normalize_chunk(
        raw(
            dataset.source_schemas[0].columns,
            [
                ["1", "1735689600000", "100", "1", "buy"],
                ["2", "1735689600000", "101", "1", "sell"],
            ],
        ),
        dataset,
    )
    assert validate_chunk(table, dataset, date(2025, 1, 1)) == datetime(
        2025, 1, 1, tzinfo=UTC
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("timestamp", "seconds-not-milliseconds"),
        ("price", "nan"),
        ("volume", "0"),
        ("side", "hold"),
        ("rpi", "maybe"),
    ],
)
def test_malformed_spot_values_fail_closed(column: str, value: str) -> None:
    """Reject invalid timestamps, values, directions, and flags."""
    dataset = get_dataset("spot", "trades")
    source = dict(
        zip(
            dataset.source_columns,
            ["1", "1735689600000", "100", "1", "buy", "false"],
            strict=True,
        )
    )
    source[column] = value
    table = raw(
        dataset.source_columns, [[source[name] for name in dataset.source_columns]]
    )
    if column in {"timestamp", "price", "side", "rpi"}:
        with pytest.raises(DataValidationError):
            normalize_chunk(table, dataset)
    else:
        normalized = normalize_chunk(table, dataset)
        with pytest.raises(DataValidationError):
            validate_chunk(normalized, dataset, date(2025, 1, 1))


def test_validation_rejects_duplicates_order_and_wrong_days() -> None:
    """Reject duplicate IDs, descending times, and cross-day contamination."""
    dataset = get_dataset("spot", "trades")

    def rows(first_id: str, second_id: str, first: str, second: str) -> pa.Table:
        """Return two normalized Spot rows."""
        return normalize_chunk(
            raw(
                dataset.source_schemas[0].columns,
                [
                    [first_id, first, "100", "1", "buy"],
                    [second_id, second, "100", "1", "sell"],
                ],
            ),
            dataset,
        )

    with pytest.raises(DataValidationError, match="unique"):
        validate_chunk(
            rows("1", "1", "1735689600000", "1735689600001"),
            dataset,
            date(2025, 1, 1),
        )
    with pytest.raises(DataValidationError, match="nondecreasing"):
        validate_chunk(
            rows("1", "2", "1735689600001", "1735689600000"),
            dataset,
            date(2025, 1, 1),
        )
    with pytest.raises(DataValidationError, match="outside"):
        validate_chunk(
            rows("1", "2", "1735776000000", "1735776000001"),
            dataset,
            date(2025, 1, 1),
        )


def test_csv_zip_member_is_ingested_without_a_duplicated_suffix(
    tmp_path: Path,
) -> None:
    """Accept Bybit's unusual filename.csv.zip Option archives."""
    dataset = get_dataset("options", "trades")
    filename = "2026-09-20_BTC_USDT.trades.csv.zip"
    row = (
        "trade-a,75905525121,1789862410029,BTC-20SEP26-81000-C-USDT,"
        "Sell,325,0.01,0.1683,81254.12186745,322.98232785,0.1649\n"
    )
    csv = ",".join(dataset.source_columns) + "\n" + row
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename.removesuffix(".zip"), csv)
    resource = Resource(
        date(2026, 9, 20),
        f"https://public.bybit.com/trade/option/BTC/{filename}",
        None,
        integrity=IntegritySpec("archive_only"),
    )
    destination = tmp_path / "option.parquet"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload.getvalue())
        )
    ) as client:
        metadata = ingest_trades(client, resource, dataset, destination)
    assert metadata.row_count == 1
    assert pq.read_table(destination)["trade_id"].to_pylist() == ["trade-a"]


def test_gzip_trade_ingestion_sorts_source_rows_stably(tmp_path: Path) -> None:
    """Sort out-of-order derivative rows by timestamp and trade ID."""
    dataset = get_dataset("linear", "trades")
    rows = [
        "1585180700.0647,BTCUSDT,Buy,0.042,6698.5,PlusTick,b,1,0.042,281.337",
        "1585180700.02,BTCUSDT,Buy,0.072,6698,PlusTick,a,1,0.072,482.256",
    ]
    csv = ",".join(dataset.source_schemas[0].columns) + "\n" + "\n".join(rows)
    resource = Resource(
        date(2020, 3, 25),
        "https://public.bybit.com/trading/BTCUSDT/BTCUSDT2020-03-25.csv.gz",
        None,
        integrity=IntegritySpec("archive_only"),
    )
    destination = tmp_path / "linear.parquet"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=gzip.compress(csv.encode()))
        )
    ) as client:
        metadata = ingest_trades(client, resource, dataset, destination)
    assert metadata.row_count == 2
    table = pq.read_table(destination)
    assert table["trade_id"].to_pylist() == ["a", "b"]
