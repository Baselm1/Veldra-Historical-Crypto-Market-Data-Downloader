"""Test exact UTC coverage for source-labeled archive days."""

from datetime import UTC, date, datetime, timedelta

import duckdb
import httpx

from crypto_downloader.core.catalog import Catalog
from crypto_downloader.core.discovery import discover_resources, requested_days
from crypto_downloader.core.models import Resource, ResourceKey
from crypto_downloader.core.planner import select_archives
from crypto_downloader.binance.datasets import SPOT_KLINES

KEY = ResourceKey("shifted", "spot", "klines", "BTCUSDT", "1m")


def shifted_resource(day: date) -> Resource:
    """Create one UTC+8 source-day resource.

    Args:
        day: The source-local archive date.

    Returns:
        A resource spanning midnight-to-midnight at UTC+8.
    """
    start = datetime.combine(day, datetime.min.time(), UTC) - timedelta(hours=8)
    return Resource(
        day,
        f"https://example/{day}.zip",
        f"https://example/{day}.zip.CHECKSUM",
        coverage_start=start,
        coverage_end=start + timedelta(days=1),
    )


class ShiftedSource:
    """Expose deterministic resources labeled in UTC+8."""

    code = "shifted"
    products = ("spot",)
    archive_day_offset = timedelta(hours=8)

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return every requested source-local archive day.

        Args:
            client: The unused HTTP client.
            key: The unused dataset identity.
            start_day: The first UTC+8 archive label.
            end_day: The last UTC+8 archive label.

        Returns:
            Consecutive shifted resources.
        """
        return [
            shifted_resource(date.fromordinal(ordinal))
            for ordinal in range(start_day.toordinal(), end_day.toordinal() + 1)
        ]


def test_resource_defaults_to_utc_calendar_coverage() -> None:
    """Confirm existing resources retain UTC day coverage by default."""
    resource = Resource(date(2025, 1, 1), "archive", "checksum")

    assert resource.coverage == (
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    )


def test_requested_days_can_use_a_source_day_offset() -> None:
    """Confirm one UTC day touches two UTC+8 archive labels."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 2, tzinfo=UTC)

    assert requested_days(start, end, timedelta(hours=8)) == (
        date(2025, 1, 1),
        date(2025, 1, 2),
    )


def test_catalog_round_trips_and_queries_exact_coverage() -> None:
    """Confirm timestamp overlap selects both shifted files for one UTC day."""
    connection = duckdb.connect()
    catalog = Catalog(connection)
    first = shifted_resource(date(2025, 1, 1))
    second = shifted_resource(date(2025, 1, 2))
    catalog.save_discovery(KEY, first.day, second.day, [first, second])

    found = catalog.resources_between(
        KEY,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    )

    assert found == [first, second]
    assert catalog.resource_coverage_bounds(KEY) == (
        datetime(2024, 12, 31, 16, tzinfo=UTC),
        datetime(2025, 1, 2, 16, tzinfo=UTC),
    )
    connection.close()


def test_discovery_scans_shifted_labels_but_returns_exact_overlap() -> None:
    """Confirm discovery translates the request and filters exact UTC coverage."""
    connection = duckdb.connect()
    catalog = Catalog(connection)
    source = ShiftedSource()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 2, tzinfo=UTC)

    with httpx.Client() as client:
        found = discover_resources(source, catalog, client, KEY, start, end)

    assert [resource.day for resource in found] == [date(2025, 1, 1), date(2025, 1, 2)]
    connection.close()


def test_archive_selection_uses_timestamps_instead_of_source_labels() -> None:
    """Confirm archives with overlapping UTC coverage are never selected twice."""
    first = shifted_resource(date(2025, 1, 1))
    duplicate = Resource(
        date(2024, 12, 31),
        "duplicate",
        "checksum",
        coverage_start=first.coverage[0],
        coverage_end=first.coverage[1],
    )

    selected = select_archives([first, duplicate], SPOT_KLINES)

    assert len(selected) == 1
