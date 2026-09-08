"""Define the small contract implemented by historical data sources."""

from datetime import date
from pathlib import Path
from typing import Protocol

import httpx

from crypto_downloader.core.datasets import DatasetSpec
from crypto_downloader.core.models import (
    IngestedResource,
    Market,
    Resource,
    ResourceKey,
)


class Connector(Protocol):
    """Describe the operations supplied by a historical data source."""

    code: str
    products: tuple[str, ...]
    active_statuses: frozenset[str]

    def checksum(self, client: httpx.Client, resource: Resource) -> str:
        """Return the current SHA-256 digest for one source archive.

        Args:
            client: The HTTPX client used for source requests.
            resource: The archive resource whose sidecar is checked.

        Returns:
            The lowercase SHA-256 digest declared by the source.
        """
        pass

    def markets(self, client: httpx.Client, product: str) -> list[Market]:
        """Return current and archive-only markets for one product.

        Args:
            client: The HTTPX client used for source requests.
            product: The source product to inspect.

        Returns:
            The complete known market snapshot.
        """
        pass

    def resources(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date,
        end_day: date,
    ) -> list[Resource]:
        """Return daily resources found in an inclusive date range.

        Args:
            client: The HTTPX client used for source requests.
            key: The requested source dataset identity.
            start_day: The first source day to include.
            end_day: The last source day to include.

        Returns:
            The discovered daily resources ordered by date.
        """
        pass

    def first_resource(
        self,
        client: httpx.Client,
        key: ResourceKey,
        start_day: date | None,
        end_day: date,
    ) -> Resource | None:
        """Return the first daily resource inside a broad date range.

        Args:
            client: The HTTPX client used for source requests.
            key: The requested source dataset identity.
            start_day: The earliest acceptable source day, or ``None`` for all
                source history.
            end_day: The latest acceptable source day.

        Returns:
            The first resource, or ``None`` when the source has no matching file.
        """
        pass

    def ingest(
        self,
        client: httpx.Client,
        resource: Resource,
        dataset: DatasetSpec,
        destination: Path,
    ) -> IngestedResource:
        """Convert one source archive into a verified Parquet file.

        Args:
            client: The HTTPX client used for source requests.
            resource: The daily archive to ingest.
            dataset: The schema used to interpret the archive.
            destination: The final Parquet path.

        Returns:
            Integrity metadata for the completed Parquet file.
        """
        pass
