"""Test contract-size context supplied to COIN-M archive ingestion."""

from datetime import UTC, date, datetime

import pandas as pd

from veldra.binance.datasets import (
    CM_INDEX_PRICE_KLINES,
    CM_TRADES,
    UM_INDEX_PRICE_KLINES,
    UM_TRADES,
)
from veldra.core.models import Market, Resource, Result
from veldra.core.pair import _archive_symbol, _with_contract_size


def result() -> Result:
    """Create a minimal result used to collect context errors.

    Returns:
        A new empty result with a valid requested range.
    """
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return Result(
        "BTCUSD_PERP",
        data=pd.DataFrame(),
        requested_range=(start, start.replace(day=2)),
    )


def resource() -> Resource:
    """Create one discovered COIN-M archive resource.

    Returns:
        A resource without request-time contract context.
    """
    return Resource(
        date(2024, 1, 1), "https://example/archive.zip", "https://example/checksum"
    )


def test_cm_trade_resources_receive_the_cataloged_contract_size() -> None:
    """Confirm the pair workflow makes contract metadata available to ingestion."""
    report = result()
    market = Market("BTCUSD_PERP", "BTCUSDPERP", contract_size=100.0)

    resources = _with_contract_size([resource()], market, CM_TRADES, report)

    assert resources is not None
    assert resources[0].contract_size == 100.0
    assert report.errors == []


def test_cm_trade_request_fails_clearly_when_contract_size_is_unavailable() -> None:
    """Confirm COIN-M quote notional is not guessed for archive-only metadata."""
    report = result()
    market = Market("BTCUSD_PERP", "BTCUSDPERP")

    resources = _with_contract_size([resource()], market, CM_TRADES, report)

    assert resources is None
    assert [message.code for message in report.errors] == ["contract_size_unavailable"]


def test_um_trade_resources_do_not_need_contract_context() -> None:
    """Confirm USD-M trades retain a resource unchanged without contract size."""
    item = resource()
    report = result()

    resources = _with_contract_size(
        [item], Market("BTCUSDT", "BTCUSDT"), UM_TRADES, report
    )

    assert resources == [item]
    assert report.errors == []


def test_index_price_archive_symbols_follow_the_declared_market_attribute() -> None:
    """Confirm CM index files use a pair while UM files use the native symbol."""
    um_report = result()
    cm_report = result()

    um_symbol = _archive_symbol(
        Market("BTCUSDT", "BTCUSDT", pair="BTCUSDT"), UM_INDEX_PRICE_KLINES, um_report
    )
    cm_symbol = _archive_symbol(
        Market("BTCUSD_PERP", "BTCUSDPERP", pair="BTCUSD"),
        CM_INDEX_PRICE_KLINES,
        cm_report,
    )

    assert um_symbol is None
    assert cm_symbol == "BTCUSD"
    assert um_report.errors == []
    assert cm_report.errors == []


def test_cm_index_price_request_reports_a_missing_pair_identifier() -> None:
    """Confirm CM index routing does not guess an archive identifier."""
    report = result()

    archive_symbol = _archive_symbol(
        Market("BTCUSD_PERP", "BTCUSDPERP"), CM_INDEX_PRICE_KLINES, report
    )

    assert archive_symbol is None
    assert [message.code for message in report.errors] == ["archive_symbol_unavailable"]
