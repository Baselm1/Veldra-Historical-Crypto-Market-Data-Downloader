"""Ingest Gate hourly order-book archives."""

from pathlib import Path

import httpx

from veldra.core.datasets import DatasetSpec
from veldra.core.models import IngestedResource, Resource


def ingest_order_book_day(
    client: httpx.Client,
    resource: Resource,
    dataset: DatasetSpec,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    backoff: float,
) -> IngestedResource:
    """Reserve hourly Gate order-book ingestion for its dataset commits.

    Args:
        client: The shared HTTP client.
        resource: The logical source day.
        dataset: The update or snapshot schema.
        destination: The final daily Parquet path.
        timeout: The timeout for each request in seconds.
        retries: The retries after the first attempt.
        backoff: The initial retry delay in seconds.

    Returns:
        Metadata for a completed daily Parquet file.
    """
    raise NotImplementedError("Gate order-book ingestion is not implemented yet")
