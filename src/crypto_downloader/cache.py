"""Materialize discovered daily resources in the local Parquet cache."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx

from .catalog import Catalog
from .datasets import DatasetSpec
from .models import Message, Resource, ResourceKey
from .source import Source


@dataclass
class CacheCoverage:
    """Hold usable Parquet paths and problems found while caching."""

    paths: list[Path] = field(default_factory=list)
    problems: list[Message] = field(default_factory=list)


def parquet_path(data_dir: Path, key: ResourceKey, day: date) -> Path:
    """Return the deterministic Parquet path for one daily resource.

    Args:
        data_dir: The root downloader data directory.
        key: The resource dataset identity.
        day: The daily resource date.

    Returns:
        The final local Parquet path.
    """
    return (
        data_dir
        / "parquet"
        / key.source
        / key.product
        / key.dataset
        / key.symbol
        / key.interval
        / f"{day.isoformat()}.parquet"
    )


def valid_cached_path(resource: Resource) -> Path | None:
    """Return a cache path when its inexpensive metadata checks succeed.

    Args:
        resource: The cataloged resource and cache metadata.

    Returns:
        The valid local path, or ``None`` when recaching is required.
    """
    path = resource.parquet_path
    if (
        resource.status != "ready"
        or path is None
        or resource.parquet_size is None
        or resource.parquet_mtime_ns is None
    ):
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    expected = resource.parquet_size, resource.parquet_mtime_ns
    return path if (stat.st_size, stat.st_mtime_ns) == expected else None


def _cache_resource(
    source: Source,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    dataset: DatasetSpec,
    resource: Resource,
    destination: Path,
) -> Message | None:
    """Ingest one resource and record its outcome.

    Args:
        source: The source strategy used for ingestion.
        catalog: The metadata catalog receiving the outcome.
        client: The HTTPX client used for archive downloads.
        key: The requested source dataset identity.
        dataset: The schema used to interpret source rows.
        resource: The discovered resource to ingest.
        destination: The final daily Parquet path.

    Returns:
        A structured problem when ingestion fails, otherwise ``None``.
    """
    try:
        metadata = source.ingest(client, resource, dataset, destination)
        catalog.mark_ready(key, resource.day, destination, metadata)
        return None
    except Exception as error:
        catalog.mark_failed(key, resource.day, str(error))
        return Message("resource_failed", str(error), resource.day)


def cache_resources(
    source: Source,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    dataset: DatasetSpec,
    resources: list[Resource],
    data_dir: Path,
) -> CacheCoverage:
    """Reuse valid files and ingest every missing known resource.

    Args:
        source: The source strategy used for ingestion.
        catalog: The metadata catalog receiving cache outcomes.
        client: The HTTPX client used for archive downloads.
        key: The requested source dataset identity.
        dataset: The schema used to interpret source rows.
        resources: The discovered daily resources to materialize.
        data_dir: The root downloader data directory.

    Returns:
        Usable paths and isolated resource problems.
    """
    coverage = CacheCoverage()
    for resource in resources:
        cached = valid_cached_path(resource)
        if cached is not None:
            coverage.paths.append(cached)
            continue
        destination = parquet_path(data_dir, key, resource.day)
        problem = _cache_resource(
            source, catalog, client, key, dataset, resource, destination
        )
        if problem is None:
            coverage.paths.append(destination)
        else:
            coverage.problems.append(problem)
    return coverage
