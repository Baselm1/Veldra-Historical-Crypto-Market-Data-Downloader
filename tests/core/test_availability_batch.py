"""Verify parallel availability and durable successful checkpoints."""

from collections import Counter
from datetime import date
from pathlib import Path
from threading import Barrier, Lock
from unittest.mock import patch

import httpx
import pytest

from veldra import Binance
from veldra.core.models import IntegritySpec, Market, Resource, ResourceKey


def test_batch_overlaps_requests_and_reuses_committed_empty_and_present_ranges(
    tmp_path: Path,
) -> None:
    """Keep successes when a sibling request fails and retry only its window."""
    pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    calls: Counter[str] = Counter()
    lock = Lock()
    both = Barrier(2, timeout=10)
    active = peak = 0
    fail = True

    def serve(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        pair = request.url.path.removeprefix("/")
        with lock:
            calls[pair] += 1
            active += 1
            peak = max(peak, active)
        try:
            if fail and pair != "BNBUSDT":
                both.wait()
            return httpx.Response(503 if fail and pair == "BNBUSDT" else 200)
        finally:
            with lock:
                active -= 1

    service = Binance(
        tmp_path,
        earliest_date="all",
        progress=False,
        max_workers=2,
    )
    service._downloader.transport = httpx.MockTransport(serve)
    source = service._downloader.source
    markets = [Market(pair, pair, active=True) for pair in pairs]
    old = Resource(
        day=date(2020, 1, 1),
        url="https://example.com/old",
        checksum_url=None,
        integrity=IntegritySpec("archive_only"),
    )
    day = date(2022, 1, 1)

    def resources(
        client: httpx.Client,
        key: ResourceKey,
        first: date,
        last: date,
    ) -> list[Resource]:
        assert first == last == day
        client.get(f"https://example.com/{key.symbol}").raise_for_status()
        if key.symbol == "ETHUSDT":
            return []
        return [
            Resource(
                day=day,
                url=f"https://example.com/{key.symbol}",
                checksum_url=None,
                integrity=IntegritySpec("archive_only"),
            )
        ]

    with (
        patch.object(source, "markets", return_value=markets),
        patch.object(source, "first_resource", return_value=old),
        patch.object(source, "resources", side_effect=resources),
    ):
        with pytest.raises(ExceptionGroup) as raised:
            service.discover_availability_many(
                pairs, day, day, product="spot", dataset="klines"
            )
        assert len(raised.value.exceptions) == 1
        assert "BNBUSDT" in raised.value.exceptions[0].__notes__[0]
        assert peak == 2
        fail = False
        result = service.discover_availability_many(
            pairs, day, day, product="spot", dataset="klines"
        )
        single = [
            service.discover_availability(
                pair, day, day, product="spot", dataset="klines"
            )
            for pair in pairs
        ]
    assert result == single
    assert [report.symbol for report in result] == pairs
    assert calls == {"BTCUSDT": 1, "ETHUSDT": 1, "BNBUSDT": 2}
    assert result[1].unavailable_days == 1
