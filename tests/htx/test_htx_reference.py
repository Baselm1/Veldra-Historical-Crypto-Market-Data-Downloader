"""Test HTX perpetual reference-price and funding-rate datasets."""

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from veldra.core.models import DataValidationError
from veldra.htx.datasets import (
    COIN_INDEX_PRICE_KLINES,
    COIN_MARK_PRICE_KLINES,
    LINEAR_FUNDING_RATES,
    LINEAR_INDEX_PRICE_KLINES,
    LINEAR_MARK_PRICE_KLINES,
    NEW_FUNDING_COLUMNS,
    NEW_REFERENCE_KLINE_COLUMNS,
    get_dataset,
)
from veldra.htx.facade import HTX
from veldra.htx.processing import normalize_chunk, validate_chunk


def raw_table(columns: tuple[str, ...], rows: list[tuple[object, ...]]) -> Any:
    """Build a string-valued Arrow table resembling CSV ingestion.

    Args:
        columns: The source field names.
        rows: The source records.

    Returns:
        An Arrow table whose values retain their source spelling.
    """
    return pa.table(
        {
            column: pa.array([str(row[index]) for row in rows])
            for index, column in enumerate(columns)
        }
    )


@pytest.mark.parametrize(
    "specification",
    [
        LINEAR_INDEX_PRICE_KLINES,
        COIN_INDEX_PRICE_KLINES,
        LINEAR_MARK_PRICE_KLINES,
        COIN_MARK_PRICE_KLINES,
    ],
)
def test_reference_kline_declarations_are_resampleable(specification: Any) -> None:
    """Confirm every perpetual reference Kline has one canonical schema.

    Args:
        specification: The product and price-kind dataset declaration.
    """
    assert get_dataset(specification.product, specification.name) is specification
    assert specification.source_columns == NEW_REFERENCE_KLINE_COLUMNS
    assert specification.stored_columns == (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "sample_count",
    )
    assert specification.resample_sum_columns == ("sample_count",)
    assert specification.integer_columns == ("sample_count",)
    assert specification.archive_symbol_attribute == "pair"


@pytest.mark.parametrize(
    "specification",
    [
        LINEAR_INDEX_PRICE_KLINES,
        COIN_INDEX_PRICE_KLINES,
        LINEAR_MARK_PRICE_KLINES,
        COIN_MARK_PRICE_KLINES,
    ],
)
def test_reference_klines_normalize_observed_new_archives(
    specification: Any,
) -> None:
    """Confirm reference candles retain OHLC and derive sample counts.

    Args:
        specification: The product and price-kind dataset declaration.
    """
    table = raw_table(
        NEW_REFERENCE_KLINE_COLUMNS,
        [
            ("BTC-USDT-PERP", 78688.42, 78753.68, 78687.61, 78753.68, 1788795720),
            ("BTC-USDT-PERP", 78716.09, 78751.86, 78705.88, 78738.78, 1788795960),
        ],
    )

    result = normalize_chunk(table, specification)
    last = validate_chunk(result, specification, date(2026, 9, 7))

    assert result.column_names == list(specification.stored_columns)
    assert result["sample_count"].to_pylist() == [1, 1]
    assert last == datetime(2026, 9, 7, 15, 46, tzinfo=UTC)


def test_funding_rates_preserve_negative_rates_and_millisecond_times() -> None:
    """Confirm signed funding observations normalize without candle behavior."""
    table = raw_table(
        NEW_FUNDING_COLUMNS,
        [
            ("BTC-USDT-PERP", -0.000045439204046426, 1788710401532),
            ("BTC-USDT-PERP", 0.000065850356674215, 1788739201019),
        ],
    )

    result = normalize_chunk(table, LINEAR_FUNDING_RATES)
    last = validate_chunk(result, LINEAR_FUNDING_RATES, date(2026, 9, 7))

    assert get_dataset("linear_swap", "funding_rates") is LINEAR_FUNDING_RATES
    assert LINEAR_FUNDING_RATES.source_columns == NEW_FUNDING_COLUMNS
    assert result["funding_rate"].to_pylist() == pytest.approx(
        [-0.000045439204046426, 0.000065850356674215]
    )
    assert last == datetime(2026, 9, 7, 0, 0, 1, 19000, tzinfo=UTC)
    with pytest.raises(ValueError, match="unsupported dataset"):
        get_dataset("coin_swap", "funding_rates")


def test_reference_validation_rejects_invalid_prices_and_funding() -> None:
    """Confirm malformed reference observations cannot reach Parquet."""
    prices = raw_table(
        NEW_REFERENCE_KLINE_COLUMNS,
        [("BTC-USDT-PERP", 2, 1, 1, 1, 1788795720)],
    )
    funding = raw_table(
        NEW_FUNDING_COLUMNS,
        [("BTC-USDT-PERP", "nan", 1788710401532)],
    )

    with pytest.raises(DataValidationError, match="high"):
        validate_chunk(
            normalize_chunk(prices, LINEAR_INDEX_PRICE_KLINES),
            LINEAR_INDEX_PRICE_KLINES,
            date(2026, 9, 7),
        )
    with pytest.raises(DataValidationError, match="finite"):
        normalize_chunk(funding, LINEAR_FUNDING_RATES)


@pytest.mark.parametrize(
    ("method_name", "dataset"),
    [
        ("get_index_price_klines", "index_price_klines"),
        ("get_mark_price_klines", "mark_price_klines"),
    ],
)
def test_facade_delegates_reference_klines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    dataset: str,
) -> None:
    """Confirm reference-price facade methods retain declarative options.

    Args:
        tmp_path: The isolated data directory.
        monkeypatch: Pytest's attribute replacement helper.
        method_name: The public facade method to call.
        dataset: The expected internal dataset name.
    """
    service = HTX(tmp_path, progress=False)
    expected = pd.DataFrame({"open": [1.0]})
    received: dict[str, object] = {}

    def get_data(*args: object, **kwargs: object) -> pd.DataFrame:
        """Record one delegated reference request."""
        received.update(kwargs)
        return expected

    monkeypatch.setattr(service._downloader, "get_data", get_data)
    method = getattr(service, method_name)
    actual = method(
        "BTCUSDT",
        "2026-09-07",
        "2026-09-07",
        product="linear_swap",
        interval="1h",
        columns=["open_time", "close"],
        gap_policy="keep",
    )

    assert actual is expected
    assert received == {
        "product": "linear_swap",
        "dataset": dataset,
        "interval": "1h",
        "desired_columns": ["open_time", "close"],
        "refresh": False,
        "offline": False,
        "gap_policy": "keep",
        "progress": False,
    }


def test_facade_delegates_linear_funding_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm the funding facade fixes the only supported product.

    Args:
        tmp_path: The isolated data directory.
        monkeypatch: Pytest's attribute replacement helper.
    """
    service = HTX(tmp_path, progress=False)
    expected = [pd.DataFrame({"funding_rate": [0.1]})]
    received: dict[str, object] = {}

    def get_data(*args: object, **kwargs: object) -> list[pd.DataFrame]:
        """Record one delegated funding request."""
        received.update(kwargs)
        return expected

    monkeypatch.setattr(service._downloader, "get_data", get_data)
    actual = service.get_funding_rates(
        ["BTCUSDT"], "2026-09-07", "2026-09-07", columns=["funding_rate"]
    )

    assert actual is expected
    assert received == {
        "product": "linear_swap",
        "dataset": "funding_rates",
        "desired_columns": ["funding_rate"],
        "refresh": False,
        "offline": False,
        "progress": False,
    }
