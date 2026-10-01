"""Test bounded Bitget portal archive discovery."""

from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import cast

import pytest

from veldra.bitget.manifest import (
    MANIFEST_ENDPOINT,
    SYMBOL_ENDPOINT,
    BitgetManifestDiscovery,
    ManifestRequest,
    chunk_requests,
)


def file_row(day: str = "2025-01-01", *, symbol: str = "BTC/USDT") -> dict[str, object]:
    """Return one representative portal file row."""
    compact = day.replace("-", "")
    return {
        "dateTime": 1_735_660_800_000,
        "dateTimeStr": day,
        "displayName": symbol,
        "fileName": f"{symbol}-{compact}.zip",
        "fileUrl": f"https://img.bitgetimg.com/online/kline/BTCUSDT/SP/{compact}.zip",
    }


class Portal:
    """Return deterministic portal rows and record request bodies."""

    def __init__(self, rows: object) -> None:
        """Store one response value."""
        self.rows = rows
        self.calls: list[tuple[str, dict[str, object]]] = []

    def portal(self, path: str, body: Mapping[str, object]) -> object:
        """Record and return one portal response."""
        self.calls.append((path, dict(body)))
        return self.rows


def test_request_chunks_use_five_symbols_and_seven_days() -> None:
    """Pack long ranges without exceeding either observed portal maximum."""
    requests = chunk_requests(
        [f"COIN{index}/USDT" for index in range(6)],
        date(2025, 1, 1),
        date(2025, 1, 9),
    )
    assert requests == [
        ManifestRequest(
            tuple(f"COIN{index}/USDT" for index in range(5)),
            date(2025, 1, 1),
            date(2025, 1, 7),
        ),
        ManifestRequest(
            tuple(f"COIN{index}/USDT" for index in range(5)),
            date(2025, 1, 8),
            date(2025, 1, 9),
        ),
        ManifestRequest(("COIN5/USDT",), date(2025, 1, 1), date(2025, 1, 7)),
        ManifestRequest(("COIN5/USDT",), date(2025, 1, 8), date(2025, 1, 9)),
    ]


def test_duplicate_symbols_are_removed_before_chunking() -> None:
    """Avoid repeating a portal request for the same display symbol."""
    assert chunk_requests(["BTC/USDT", "BTC/USDT"], date(2025, 1, 1), date(2025, 1, 1))[
        0
    ].symbols == ("BTC/USDT",)


@pytest.mark.parametrize(
    "symbols,begin,end,error",
    [
        ([], date(2025, 1, 1), date(2025, 1, 1), ValueError),
        (["unsafe\\path"], date(2025, 1, 1), date(2025, 1, 1), ValueError),
        (["BTC/USDT"], date(2025, 1, 2), date(2025, 1, 1), ValueError),
        (
            ["BTC/USDT"],
            cast(date, datetime(2025, 1, 1)),
            date(2025, 1, 1),
            TypeError,
        ),
    ],
)
def test_invalid_manifest_requests_fail_locally(
    symbols: list[str], begin: date, end: date, error: type[Exception]
) -> None:
    """Reject malformed discovery work before consuming portal quota."""
    with pytest.raises(error):
        chunk_requests(symbols, begin, end)


def test_symbol_search_uses_archive_dataset_scope() -> None:
    """Search Spot Kline symbols through the corresponding portal route."""
    portal = Portal(
        [
            {"displaySymbol": "BTC/USDT"},
            {"displaySymbol": "BTC/USDT"},
            {"displaySymbol": "BTC/EUR"},
        ]
    )
    discovery = BitgetManifestDiscovery(portal)
    assert discovery.search_symbols("btc", "spot", "klines") == [
        "BTC/USDT",
        "BTC/EUR",
    ]
    assert portal.calls == [
        (
            SYMBOL_ENDPOINT,
            {
                "displaySymbol": "btc",
                "businessLine": 1,
                "businessType": 1,
                "languageType": 0,
            },
        )
    ]


def test_daily_resources_preserve_utc_plus_eight_coverage() -> None:
    """Map a source day to its exact exclusive UTC coverage range."""
    portal = Portal([file_row()])
    resources = BitgetManifestDiscovery(portal).discover(
        "spot",
        "klines",
        ["BTC/USDT"],
        date(2025, 1, 1),
        date(2025, 1, 1),
    )
    assert len(resources) == 1
    resource = resources[0]
    assert resource.day == date(2025, 1, 1)
    assert resource.archive_symbol == "BTC/USDT"
    assert resource.timestamp_column == "open_time"
    assert resource.integrity_spec.mode == "response_header"
    assert resource.coverage == (
        datetime(2024, 12, 31, 16, tzinfo=UTC),
        datetime(2025, 1, 1, 16, tzinfo=UTC),
    )
    assert portal.calls[0][0] == MANIFEST_ENDPOINT


@pytest.mark.parametrize(
    ("dataset", "depth"),
    [("best_book_snapshots", 1), ("order_book_snapshots", 2)],
)
def test_depth_datasets_send_their_native_depth_type(dataset: str, depth: int) -> None:
    """Distinguish level-one and level-500 snapshot manifests."""
    portal = Portal([])
    BitgetManifestDiscovery(portal).discover(
        "coin_futures",
        dataset,
        ["BTCCM"],
        date(2025, 1, 1),
        date(2025, 1, 1),
    )
    assert portal.calls[0][1]["deptType"] == depth
    assert portal.calls[0][1]["businessLine"] == 2


def test_duplicate_manifest_urls_are_collapsed() -> None:
    """Ignore duplicate shard entries observed in older portal responses."""
    resources = BitgetManifestDiscovery(Portal([file_row(), file_row()])).discover(
        "spot",
        "trades",
        ["BTC/USDT"],
        date(2025, 1, 1),
        date(2025, 1, 1),
    )
    assert len(resources) == 1
    assert resources[0].timestamp_column == "event_time"


def test_epoch_fallback_preserves_source_calendar() -> None:
    """Use the source epoch when a manifest omits its display date."""
    row = file_row()
    row.pop("dateTimeStr")
    resource = BitgetManifestDiscovery(Portal([row])).discover(
        "spot", "klines", ["BTC/USDT"], date(2025, 1, 1), date(2025, 1, 1)
    )[0]
    assert resource.day == date(2025, 1, 1)
    assert resource.coverage[1] - resource.coverage[0] == timedelta(days=1)


@pytest.mark.parametrize(
    "row",
    [
        "wrong",
        {**file_row(), "dateTimeStr": "not-a-date"},
        {**file_row(), "fileUrl": "http://img.bitgetimg.com/file.zip"},
        {**file_row(), "fileUrl": "https://attacker.example/file.zip"},
        {**file_row(), "displayName": "unsafe\\name"},
    ],
)
def test_malformed_manifest_rows_fail_closed(row: object) -> None:
    """Reject invalid dates, identities, and untrusted archive URLs."""
    with pytest.raises(ValueError):
        BitgetManifestDiscovery(Portal([row])).discover(
            "spot",
            "klines",
            ["BTC/USDT"],
            date(2025, 1, 1),
            date(2025, 1, 1),
        )


def test_invalid_routes_queries_and_payloads_are_rejected() -> None:
    """Reject unsupported datasets, empty searches, and malformed response data."""
    discovery = BitgetManifestDiscovery(Portal("wrong"))
    with pytest.raises(ValueError, match="unsupported"):
        discovery.discover(
            "spot", "options", ["BTC/USDT"], date(2025, 1, 1), date(2025, 1, 1)
        )
    with pytest.raises(ValueError, match="empty"):
        discovery.search_symbols(" ", "spot", "klines")
    with pytest.raises(ValueError, match="list of objects"):
        discovery.search_symbols("BTC", "spot", "klines")
