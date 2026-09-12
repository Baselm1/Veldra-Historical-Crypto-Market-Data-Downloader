"""Define historical providers independently from their transport."""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, cast, runtime_checkable
from urllib.parse import urlsplit

import httpx

from veldra.core.datasets import DatasetSpec
from veldra.core.discovery import requested_days
from veldra.core.models import (
    ArchiveKey,
    ArchiveObject,
    ArchiveStatus,
    IngestedResource,
    LogicalPartition,
    Materialization,
    Resource,
    ResourceKey,
)
from veldra.core.subjects import DataSubject


@dataclass(frozen=True)
class ProviderRequest:
    """Describe an exact logical range requested from one history provider."""

    source: str
    product: str
    dataset: str
    subject: DataSubject
    interval: str | None
    start: datetime
    end: datetime
    dataset_spec: DatasetSpec
    cadence: str = "daily"

    def __post_init__(self) -> None:
        """Reject mismatched datasets and ambiguous timestamp ranges."""
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("provider range timestamps must include a timezone")
        if self.start >= self.end:
            raise ValueError("provider range must end after it starts")
        if (self.product, self.dataset) != (
            self.dataset_spec.product,
            self.dataset_spec.name,
        ):
            raise ValueError("provider dataset declaration does not match request")
        if not self.source or not self.cadence:
            raise ValueError("provider source and cadence must not be empty")


@dataclass(frozen=True)
class MaterializedArchive:
    """Return one verified local file and every logical range it exposes."""

    materialization: Materialization
    partitions: tuple[LogicalPartition, ...]

    def __post_init__(self) -> None:
        """Require at least one partition mapped to the materialized path."""
        if not self.partitions:
            raise ValueError("materialized archive requires a logical partition")
        if any(
            partition.materialization_path != self.materialization.local_path
            for partition in self.partitions
        ):
            raise ValueError("logical partition path does not match materialization")


@runtime_checkable
class HistoricalProvider(Protocol):
    """Discover and materialize physical historical data archives."""

    def discover(self, request: ProviderRequest) -> list[ArchiveObject]:
        """Return physical archives that may cover a logical request."""
        ...

    def materialize(
        self, resource: ArchiveObject, destination: Path
    ) -> MaterializedArchive:
        """Retrieve and normalize one physical archive."""
        ...


@runtime_checkable
class PaginatedProvider(Protocol):
    """Materialize bounded API pages without pretending they are archives."""

    def materialize_pages(
        self, request: ProviderRequest, destination: Path
    ) -> MaterializedArchive:
        """Fetch bounded pages and normalize them into one local file."""
        ...


class ArchiveConnector(Protocol):
    """Describe the existing connector operations used by the adapter."""

    code: str

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return physical resources for an inclusive source-day range."""
        ...

    def ingest(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Normalize one existing source resource into Parquet."""
        ...


class ArchiveProvider:
    """Adapt a one-symbol archive connector to the provider contract."""

    def __init__(self, connector: ArchiveConnector, client: httpx.Client) -> None:
        """Retain one connector and its shared HTTP client.

        Args:
            connector: The existing source archive implementation.
            client: The source-wide shared HTTP connection pool.
        """
        self.connector = connector
        self.client = client
        self._resources: dict[str, tuple[Resource, ProviderRequest]] = {}

    def discover(self, request: ProviderRequest) -> list[ArchiveObject]:
        """Translate a provider request into existing connector resources.

        Args:
            request: The logical range requested from the connector.

        Returns:
            Stable physical archive objects in source order.
        """
        if request.source != self.connector.code:
            raise ValueError("provider source does not match connector")
        if request.subject.kind != "instrument":
            raise ValueError("legacy archive providers require an instrument subject")
        start_day, end_day = requested_days(
            request.start,
            request.end,
            request.dataset_spec.archive_day_offset,
        )
        key = ResourceKey(
            request.source,
            request.product,
            request.dataset,
            request.subject.value,
            request.interval,
            cadence=request.cadence,
            subject=request.subject,
        )
        resources = self.connector.resources(self.client, key, start_day, end_day)
        objects = [self._object(request, resource) for resource in resources]
        self._resources.update(
            {
                item.key.archive_id: (resource, request)
                for item, resource in zip(objects, resources)
            }
        )
        return objects

    @staticmethod
    def _object(request: ProviderRequest, resource: Resource) -> ArchiveObject:
        """Convert one existing connector resource into a physical object.

        Args:
            request: The logical request that discovered the resource.
            resource: The existing source resource metadata.

        Returns:
            A stable provider archive object.
        """
        name = Path(urlsplit(resource.url).path).name or resource.url
        key = ArchiveKey(
            request.source,
            request.product,
            request.dataset,
            "archive",
            request.subject.kind,
            request.subject.value,
            resource.cadence,
            resource.day,
            resource.last_day,
            name,
        )
        status = (
            resource.status
            if resource.status in {"discovered", "ready", "failed", "missing"}
            else "discovered"
        )
        return ArchiveObject(
            key,
            resource.url,
            integrity=resource.integrity_spec,
            status=cast(ArchiveStatus, status),
            revision_id=resource.archive_checksum,
            last_attempt_at=resource.last_attempt_at,
            error=resource.error,
        )

    def materialize(
        self, resource: ArchiveObject, destination: Path
    ) -> MaterializedArchive:
        """Normalize one adapted resource and expose its logical instrument.

        Args:
            resource: The physical archive returned by this provider.
            destination: The final local Parquet path.

        Returns:
            The local file and its one logical instrument partition.
        """
        known = self._resources.get(resource.key.archive_id)
        if known is None:
            raise KeyError("archive was not discovered by this provider")
        legacy, request = known
        metadata = self.connector.ingest(
            self.client,
            legacy,
            request.dataset_spec,
            destination,
        )
        materialization = Materialization(
            resource.key,
            destination,
            metadata.schema_version,
            metadata.row_count,
            metadata.first_timestamp,
            metadata.last_timestamp,
            metadata.parquet_size,
            local_mtime_ns=metadata.parquet_mtime_ns,
            archive_revision=metadata.archive_checksum,
        )
        coverage_start, coverage_end = legacy.coverage
        partition = LogicalPartition(
            request.source,
            request.product,
            request.dataset,
            request.subject,
            request.interval,
            coverage_start,
            coverage_end,
            destination,
            None,
            None,
            metadata.row_count,
            source_day=legacy.day,
        )
        return MaterializedArchive(materialization, (partition,))


@dataclass(frozen=True)
class ProviderSlice:
    """Describe one provider's candidate coverage and precedence."""

    provider: str
    start: datetime
    end: datetime
    priority: int

    def __post_init__(self) -> None:
        """Reject empty providers, naive timestamps, and invalid coverage."""
        if not self.provider:
            raise ValueError("provider must not be empty")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("provider slice timestamps must include a timezone")
        if self.start >= self.end:
            raise ValueError("provider slice end must follow its start")
        if self.priority < 0:
            raise ValueError("provider priority cannot be negative")


def preferred_slices(candidates: list[ProviderSlice]) -> list[ProviderSlice]:
    """Choose lower-priority-number providers over overlapping alternatives.

    Args:
        candidates: Candidate archive and paginated coverage slices.

    Returns:
        Non-overlapping winning slices with identical neighbors merged.
    """
    boundaries = sorted(
        {point for item in candidates for point in (item.start, item.end)}
    )
    selected: list[ProviderSlice] = []
    for start, end in zip(boundaries, boundaries[1:]):
        covering = [
            item for item in candidates if item.start <= start and item.end >= end
        ]
        if not covering:
            continue
        winner = min(covering, key=lambda item: (item.priority, item.provider))
        segment = ProviderSlice(winner.provider, start, end, winner.priority)
        if (
            selected
            and selected[-1].provider == segment.provider
            and selected[-1].priority == segment.priority
            and selected[-1].end == segment.start
        ):
            previous = selected[-1]
            selected[-1] = ProviderSlice(
                previous.provider, previous.start, segment.end, previous.priority
            )
        else:
            selected.append(segment)
    return selected
