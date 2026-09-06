"""Materialize discovered daily resources in the local Parquet cache."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
import logging
from pathlib import Path

import httpx

from .catalog import Catalog
from .datasets import DatasetSpec
from .display import Reporter
from .models import IngestedResource, Message, Resource, ResourceKey
from .processing import file_sha256
from .source import Source

LOGGER = logging.getLogger(__name__)


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
        LOGGER.debug(
            "Cached resource unavailable: day=%s status=%s path=%s size=%s mtime=%s",
            resource.day,
            resource.status,
            path,
            resource.parquet_size,
            resource.parquet_mtime_ns,
        )
        return None
    try:
        stat = path.stat()
    except OSError as error:
        LOGGER.debug("Cached resource cannot be read: path=%s error=%s", path, error)
        return None
    expected = resource.parquet_size, resource.parquet_mtime_ns
    if (stat.st_size, stat.st_mtime_ns) != expected:
        LOGGER.debug(
            "Cached resource metadata changed: path=%s expected=%s actual=%s",
            path,
            expected,
            (stat.st_size, stat.st_mtime_ns),
        )
        return None
    try:
        valid_hash = (
            resource.parquet_sha256 is None
            or file_sha256(path) == resource.parquet_sha256
        )
    except OSError as error:
        LOGGER.debug("Cached resource hashing failed: path=%s error=%s", path, error)
        return None
    LOGGER.debug("Cached resource hash checked: path=%s valid=%s", path, valid_hash)
    return path if valid_hash else None


def _ingest_resource(
    source: Source,
    client: httpx.Client,
    dataset: DatasetSpec,
    resource: Resource,
    destination: Path,
) -> tuple[IngestedResource | None, Exception | None]:
    """Ingest one resource without mutating the shared catalog.

    Args:
        source: The source strategy used for ingestion.
        client: The HTTPX client used for archive downloads.
        dataset: The schema used to interpret source rows.
        resource: The discovered resource to ingest.
        destination: The final daily Parquet path.

    Returns:
        Either the completed metadata or the isolated exception.
    """
    try:
        metadata = source.ingest(client, resource, dataset, destination)
        LOGGER.debug(
            "Daily resource ingested: source=%s day=%s rows=%d destination=%s",
            source.code,
            resource.day,
            metadata.row_count,
            destination,
        )
        return metadata, None
    except Exception as error:
        LOGGER.exception(
            "Daily resource ingestion failed: source=%s day=%s url=%s",
            source.code,
            resource.day,
            resource.url,
        )
        return None, error


def _worker_count(source: Source, dataset: DatasetSpec, max_workers: object) -> int:
    """Return the lowest valid caller, dataset, and source concurrency limit.

    Args:
        source: The source whose download limit applies.
        dataset: The dataset whose processing limit applies.
        max_workers: The caller's proposed concurrency limit.

    Returns:
        The validated effective worker count.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError("max_workers must be an integer")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    source_limit = getattr(source, "max_concurrency", max_workers)
    if (
        isinstance(source_limit, bool)
        or not isinstance(source_limit, int)
        or source_limit < 1
    ):
        raise ValueError("source max_concurrency must be a positive integer")
    return min(max_workers, dataset.max_concurrency, source_limit)


def _cache_plan(
    resources: list[Resource],
    data_dir: Path,
    key: ResourceKey,
    offline: bool,
) -> tuple[CacheCoverage, list[tuple[Resource, Path]]]:
    """Separate valid cache entries from resources requiring ingestion.

    Args:
        resources: The cataloged resources requested by the caller.
        data_dir: The root downloader data directory.
        key: The requested source dataset identity.
        offline: Whether missing cache entries can be downloaded.

    Returns:
        Initial cache coverage and the resources still requiring work.
    """
    coverage = CacheCoverage()
    pending: list[tuple[Resource, Path]] = []
    for resource in resources:
        cached = valid_cached_path(resource)
        if cached is not None:
            coverage.paths.append(cached)
        elif offline:
            coverage.problems.append(
                Message(
                    "offline_missing",
                    "A valid local Parquet file is not available in offline mode.",
                    resource.day,
                )
            )
        else:
            pending.append((resource, parquet_path(data_dir, key, resource.day)))
    return coverage, pending


def _record_outcomes(
    catalog: Catalog,
    key: ResourceKey,
    pending: list[tuple[Resource, Path]],
    outcomes: list[tuple[IngestedResource | None, Exception | None]],
    coverage: CacheCoverage,
) -> None:
    """Persist ingestion outcomes and extend cache coverage in day order.

    Args:
        catalog: The metadata catalog receiving resource outcomes.
        key: The requested source dataset identity.
        pending: The resources and destinations that were attempted.
        outcomes: The matching ingestion metadata or exceptions.
        coverage: The cache coverage to update in place.
    """
    for (resource, destination), (metadata, error) in zip(pending, outcomes):
        if metadata is not None:
            catalog.mark_ready(key, resource.day, destination, metadata)
            coverage.paths.append(destination)
            LOGGER.info(
                "Daily resource cached: key=%s day=%s rows=%d path=%s",
                key,
                resource.day,
                metadata.row_count,
                destination,
            )
            continue
        message = str(error) if error is not None else "unknown ingestion failure"
        catalog.mark_failed(key, resource.day, message)
        coverage.problems.append(Message("resource_failed", message, resource.day))
        LOGGER.warning(
            "Daily resource failed: key=%s day=%s error=%s",
            key,
            resource.day,
            message,
        )


def cache_resources(
    source: Source,
    catalog: Catalog,
    client: httpx.Client,
    key: ResourceKey,
    dataset: DatasetSpec,
    resources: list[Resource],
    data_dir: Path,
    *,
    offline: bool = False,
    max_workers: int = 16,
    reporter: Reporter | None = None,
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
        offline: Whether archive downloads must be skipped.
        max_workers: The caller's maximum concurrent daily ingestions.
        reporter: The optional Rich activity reporter.

    Returns:
        Usable paths and isolated resource problems.
    """
    worker_count = _worker_count(source, dataset, max_workers)
    coverage, pending = _cache_plan(resources, data_dir, key, offline)
    display = reporter if reporter is not None else Reporter(False)
    noun = "file" if len(resources) == 1 else "files"
    missing_label = "not cached" if offline else "to download"
    missing_count = len(coverage.problems) if offline else len(pending)
    display.info(
        f"{key.symbol}: {len(resources):,} daily {noun} | "
        f"{len(coverage.paths):,} cached, {missing_count:,} {missing_label}"
    )
    LOGGER.debug(
        "Cache plan complete: key=%s resources=%d cached=%d pending=%d "
        "problems=%d offline=%s workers=%d",
        key,
        len(resources),
        len(coverage.paths),
        len(pending),
        len(coverage.problems),
        offline,
        worker_count,
    )
    outcomes: list[tuple[IngestedResource | None, Exception | None]] = []
    if pending:
        with display.downloads(key.symbol, len(pending)) as advance:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                completed = executor.map(
                    lambda item: _ingest_resource(
                        source, client, dataset, item[0], item[1]
                    ),
                    pending,
                )
                for (resource, _destination), outcome in zip(pending, completed):
                    outcomes.append(outcome)
                    advance(resource.day, outcome[0] is not None)
    _record_outcomes(catalog, key, pending, outcomes, coverage)
    if pending:
        succeeded = sum(metadata is not None for metadata, _error in outcomes)
        display.info(
            f"{key.symbol}: cached {succeeded:,} new daily file(s); "
            f"{len(pending) - succeeded:,} failed"
        )
    coverage.paths.sort()
    return coverage
