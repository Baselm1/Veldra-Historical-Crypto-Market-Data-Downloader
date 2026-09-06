"""Define the small contract implemented by historical data sources."""

from datetime import date
from pathlib import Path
from typing import Protocol

import httpx

from .datasets import DatasetSpec
from .models import IngestedResource, Market, Resource, ResourceKey


class Source(Protocol):
    """Describe the operations supplied by a historical data source."""

    code: str
    products: tuple[str, ...]
    active_statuses: frozenset[str]

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
