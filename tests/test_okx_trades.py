"""Test OKX Spot trade normalization and public facade behavior."""

from base64 import b64encode
from datetime import date
from hashlib import md5
from io import BytesIO
from pathlib import Path
import zipfile

import httpx
import pandas as pd
import pytest

from veldra.core.models import (
    ArchiveKey,
    ArchiveObject,
    DataValidationError,
    IntegritySpec,
)
from veldra.okx.datasets import SPOT_TRADES, TRADE_SOURCE_COLUMNS, get_dataset
from veldra.okx.processing import OKXArchiveProvider, normalize_trades


def trade_frame(**changes: object) -> pd.DataFrame:
    """Build valid Spot trade rows with optional replacements.

    Args:
        changes: Source columns replacing valid values.

    Returns:
        Mutable source rows.
    """
    values: dict[str, object] = {
        "instrument_name": ["BTC-USDT", "BTC-USDT"],
        "trade_id": [10, 11],
        "side": ["buy", "sell"],
        "price": [100.0, 101.0],
        "size": [2.0, 3.0],
        "created_time": [1735689600000, 1735689600000],
    }
    values.update(changes)
    return pd.DataFrame(values, columns=TRADE_SOURCE_COLUMNS)


def trade_archive(frame: pd.DataFrame) -> tuple[ArchiveObject, bytes]:
    """Build one manifest object and its ZIP bytes.

    Args:
        frame: Trade CSV rows.

    Returns:
        Physical metadata and downloadable bytes.
    """
    name = "BTC-USDT-trades-2025-01-01.zip"
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name.removesuffix(".zip") + ".csv", frame.to_csv(index=False))
    key = ArchiveKey(
        "okx",
        "spot",
        "trades",
        "module_1",
        "instrument",
        "BTC-USDT",
        "daily",
        date(2025, 1, 1),
        date(2025, 1, 1),
        name,
    )
    item = ArchiveObject(
        key,
        f"https://files.test/{name}",
        integrity=IntegritySpec("response_header", algorithm="md5"),
    )
    return item, output.getvalue()


def test_spot_trades_preserve_side_ids_and_explicit_quantities() -> None:
    """Confirm equal timestamps and derived quote quantities remain deterministic."""
    dataset = get_dataset("spot", "trades")
    frame = normalize_trades(trade_frame(), dataset)
    assert list(frame.columns) == ["instrument_id", *dataset.stored_columns]
    assert frame["trade_id"].tolist() == [10, 11]
    assert frame["side"].tolist() == ["buy", "sell"]
    assert frame["quote_quantity"].tolist() == [200.0, 303.0]
    assert str(frame["event_time"].dtype) == "datetime64[ms, UTC]"


def test_trade_exact_duplicates_are_removed_but_conflicts_fail() -> None:
    """Confirm trade IDs identify events without rejecting equal timestamps."""
    raw = trade_frame()
    exact = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    assert len(normalize_trades(exact, SPOT_TRADES)) == 2
    conflict = raw.copy()
    conflict.loc[1, "trade_id"] = 10
    with pytest.raises(DataValidationError, match="conflicting"):
        normalize_trades(conflict, SPOT_TRADES)


@pytest.mark.parametrize(
    "changes",
    [
        {"side": ["bid", "sell"]},
        {"price": [0, 1]},
        {"size": [-1, 1]},
        {"trade_id": [1.5, 2]},
        {"trade_id": [-1, 2]},
        {"created_time": ["bad", 1735689600000]},
        {"instrument_name": ["", "BTC-USDT"]},
    ],
)
def test_invalid_spot_trade_rows_fail_visibly(changes: dict[str, object]) -> None:
    """Confirm invalid event values never become queryable.

    Args:
        changes: Invalid fixture column replacement.
    """
    with pytest.raises(DataValidationError):
        normalize_trades(trade_frame(**changes), SPOT_TRADES)


def test_trade_provider_writes_queryable_raw_events(tmp_path: Path) -> None:
    """Confirm verified module 1 archives become raw logical partitions."""
    item, content = trade_archive(trade_frame())

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the integrity-bearing ZIP fixture."""
        return httpx.Response(
            200,
            content=content,
            headers={"Content-MD5": b64encode(md5(content).digest()).decode()},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        value = OKXArchiveProvider(client, retries=0).materialize(
            item, tmp_path / "trades.parquet"
        )
    assert value.materialization.row_count == 2
    assert value.partitions[0].interval is None
    assert value.partitions[0].predicate_value == "BTC-USDT"
