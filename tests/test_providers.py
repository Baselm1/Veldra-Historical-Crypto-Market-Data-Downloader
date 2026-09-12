"""Test source-neutral historical provider contracts and archive adaptation."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from veldra.binance.datasets import SPOT_KLINES
from veldra.core.models import IngestedResource, Resource, ResourceKey
from veldra.core.providers import (
    ArchiveProvider,
    HistoricalProvider,
    MaterializedArchive,
    PaginatedProvider,
    ProviderRequest,
    ProviderSlice,
    preferred_slices,
)
from veldra.core.subjects import DataSubject

START = datetime(2025, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 2, tzinfo=UTC)


@dataclass
class FakeConnector:
    """Provide one deterministic legacy archive for adapter tests."""

    code: str = "fake"
    products: tuple[str, ...] = ("spot",)
    requested_key: ResourceKey | None = None
    ingested: Resource | None = None

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return one archive and retain the translated request.

        Args:
            client: The unused HTTP client.
            key: The translated legacy resource key.
            start_day: The first requested source day.
            end_day: The final requested source day.

        Returns:
            One discovered daily resource.
        """
        self.requested_key = key
        assert (start_day, end_day) == (date(2025, 1, 1), date(2025, 1, 1))
        return [
            Resource(
                start_day,
                "https://example/BTCUSDT-2025-01-01.zip",
                "https://example/BTCUSDT-2025-01-01.zip.CHECKSUM",
                timestamp_column="open_time",
            )
        ]

    def ingest(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: object,
        destination: Path,
    ) -> IngestedResource:
        """Return representative normalized file metadata.

        Args:
            client: The unused HTTP client.
            resource: The archive selected by discovery.
            dataset: The unused dataset declaration.
            destination: The requested local Parquet path.

        Returns:
            Metadata for a complete one-minute day.
        """
        self.ingested = resource
        return IngestedResource(
            "a" * 64,
            1_234,
            567,
            1_440,
            START,
            END - timedelta(minutes=1),
            timestamp_column="open_time",
        )


def request(subject: DataSubject | None = None) -> ProviderRequest:
    """Create one daily Spot Kline provider request.

    Args:
        subject: The native subject or a default BTCUSDT instrument.

    Returns:
        A valid provider request.
    """
    return ProviderRequest(
        "fake",
        "spot",
        "klines",
        subject or DataSubject("instrument", "BTCUSDT"),
        "1m",
        START,
        END,
        SPOT_KLINES,
    )


def test_archive_provider_translates_discovery_and_materialization(
    tmp_path: Path,
) -> None:
    """Confirm existing connectors produce new physical and logical records."""
    connector = FakeConnector()
    with httpx.Client() as client:
        provider = ArchiveProvider(connector, client)
        objects = provider.discover(request())
        destination = tmp_path / "one.parquet"
        completed = provider.materialize(objects[0], destination)

    assert connector.requested_key == ResourceKey(
        "fake",
        "spot",
        "klines",
        "BTCUSDT",
        "1m",
        subject=DataSubject("instrument", "BTCUSDT"),
    )
    assert objects[0].key.remote_name == "BTCUSDT-2025-01-01.zip"
    assert objects[0].key.subject == DataSubject("instrument", "BTCUSDT")
    assert connector.ingested is not None
    assert completed.materialization.local_path == destination
    assert completed.materialization.row_count == 1_440
    assert completed.partitions[0].subject == DataSubject("instrument", "BTCUSDT")
    assert completed.partitions[0].predicate_column is None


def test_archive_provider_rejects_unsupported_scope_and_unknown_objects() -> None:
    """Confirm one-symbol legacy connectors cannot pretend to support bulk files."""
    with httpx.Client() as client:
        provider = ArchiveProvider(FakeConnector(), client)
        with pytest.raises(ValueError, match="instrument"):
            provider.discover(request(DataSubject("all", "ANY")))
        other = ArchiveProvider(FakeConnector(), client).discover(request())[0]
        with pytest.raises(KeyError, match="discovered"):
            provider.materialize(other, Path("missing.parquet"))


@pytest.mark.parametrize(
    ("start", "end"),
    [(END, START), (START.replace(tzinfo=None), END)],
)
def test_provider_requests_require_an_increasing_aware_range(
    start: datetime, end: datetime
) -> None:
    """Confirm provider boundaries are unambiguous before discovery.

    Args:
        start: The proposed inclusive start.
        end: The proposed exclusive end.
    """
    with pytest.raises(ValueError, match="range|timezone"):
        ProviderRequest(
            "fake",
            "spot",
            "klines",
            DataSubject("instrument", "BTCUSDT"),
            "1m",
            start,
            end,
            SPOT_KLINES,
        )


def test_provider_protocols_accept_matching_implementations() -> None:
    """Confirm provider contracts remain runtime-checkable for integrations."""
    with httpx.Client() as client:
        provider = ArchiveProvider(FakeConnector(), client)
        assert isinstance(provider, HistoricalProvider)
        assert not isinstance(provider, PaginatedProvider)


def test_archive_slices_override_rest_only_where_they_overlap() -> None:
    """Confirm immutable archives win while a REST provider may fill the tail."""
    candidates = [
        ProviderSlice("rest", START, START + timedelta(days=3), priority=20),
        ProviderSlice("archive", START, START + timedelta(days=2), priority=10),
    ]

    assert preferred_slices(candidates) == [
        ProviderSlice("archive", START, START + timedelta(days=2), priority=10),
        ProviderSlice(
            "rest", START + timedelta(days=2), START + timedelta(days=3), priority=20
        ),
    ]


def test_provider_slice_selection_merges_neighbors_and_rejects_bad_ranges() -> None:
    """Confirm adjacent winning slices merge and invalid coverage is rejected."""
    middle = START + timedelta(hours=12)
    assert preferred_slices(
        [
            ProviderSlice("archive", START, middle, priority=10),
            ProviderSlice("archive", middle, END, priority=10),
        ]
    ) == [ProviderSlice("archive", START, END, priority=10)]
    with pytest.raises(ValueError, match="end"):
        ProviderSlice("archive", END, START, priority=10)


def test_materialized_archive_requires_logical_partitions(tmp_path: Path) -> None:
    """Confirm completed provider work cannot publish an inaccessible file."""
    connector = FakeConnector()
    with httpx.Client() as client:
        provider = ArchiveProvider(connector, client)
        item = provider.discover(request())[0]
        completed = provider.materialize(item, tmp_path / "one.parquet")

    with pytest.raises(ValueError, match="partition"):
        MaterializedArchive(completed.materialization, ())
