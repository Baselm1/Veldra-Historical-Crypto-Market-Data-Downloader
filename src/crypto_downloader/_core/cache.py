"""Materialize discovered daily resources in the local Parquet cache."""

from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import date
import logging
from pathlib import Path

import httpx
import pyarrow.parquet as parquet

from crypto_downloader._core.catalog import Catalog
from crypto_downloader._core.datasets import DatasetSpec
from crypto_downloader._core.reporting import Reporter
from crypto_downloader._core.models import (
    IngestedResource,
    Message,
    Resource,
    ResourceKey,
)
from crypto_downloader._core.source import Source

LOGGER = logging.getLogger(__name__)


@dataclass
class CacheCoverage:
    """Hold usable Parquet paths and problems found while caching."""

    paths: list[Path] = field(default_factory=list)
    warnings: list[Message] = field(default_factory=list)
    problems: list[Message] = field(default_factory=list)


def invalid_parquet_paths(paths: list[Path]) -> list[Path]:
    """Return unreadable Parquet files before a DuckDB query.

    Args:
        paths: The cached partitions about to be queried.

    Returns:
        Paths whose Parquet metadata cannot be read.
    """
    invalid: list[Path] = []
    for path in paths:
        try:
            with path.open("rb") as source:
                parquet.ParquetFile(source)
        except Exception as error:
            LOGGER.warning(
                "Cached Parquet is unreadable: path=%s error=%s", path, error
            )
            invalid.append(path)
    return invalid


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
        / (key.interval or "raw")
        / f"{day.isoformat()}.parquet"
    )


def valid_cached_path(resource: Resource, dataset: DatasetSpec) -> Path | None:
    """Return a cache path when its file and schema metadata are valid.

    Args:
        resource: The cataloged resource and cache metadata.
        dataset: The current schema expected by the caller.

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
    if (
        resource.schema_version != dataset.schema_version
        or resource.timestamp_column != dataset.time_column
    ):
        LOGGER.info(
            "Cached resource schema changed: day=%s "
            "stored_version=%s expected_version=%s "
            "stored_timestamp=%s expected_timestamp=%s",
            resource.day,
            resource.schema_version,
            dataset.schema_version,
            resource.timestamp_column,
            dataset.time_column,
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
    return path


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
    dataset: DatasetSpec,
    offline: bool,
) -> tuple[CacheCoverage, list[tuple[Resource, Path]], list[tuple[Resource, Path]]]:
    """Separate valid cache entries from resources requiring ingestion.

    Args:
        resources: The cataloged resources requested by the caller.
        data_dir: The root downloader data directory.
        key: The requested source dataset identity.
        dataset: The current schema expected by the caller.
        offline: Whether missing cache entries can be downloaded.

    Returns:
        Initial cache coverage, resources requiring ingestion, and valid local
        resource paths.
    """
    coverage = CacheCoverage()
    pending: list[tuple[Resource, Path]] = []
    cached: list[tuple[Resource, Path]] = []
    for resource in resources:
        path = valid_cached_path(resource, dataset)
        if path is not None:
            cached.append((resource, path))
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
    return coverage, pending, cached


def _revalidate_cached_resource(
    source: Source,
    client: httpx.Client,
    resource: Resource,
) -> tuple[bool, Message | None]:
    """Check whether one locally cached archive remains current at its source.

    Args:
        source: The source strategy that owns the archive.
        client: The HTTPX client used for the sidecar request.
        resource: The cataloged archive and local cache metadata.

    Returns:
        Whether the local partition remains usable and an optional warning.
    """
    if resource.archive_sha256 is None:
        LOGGER.debug(
            "Cached archive lacks a source checksum and will be rebuilt: day=%s",
            resource.day,
        )
        return False, None
    try:
        current = source.checksum(client, resource)
    except Exception as error:
        LOGGER.warning(
            "Archive checksum revalidation failed; reusing cache: day=%s error=%s",
            resource.day,
            error,
        )
        return True, Message(
            "archive_revalidation_failed",
            "Could not revalidate the source archive; reused the verified local "
            "Parquet file.",
            resource.day,
        )
    unchanged = current == resource.archive_sha256
    LOGGER.debug(
        "Archive checksum revalidated: day=%s unchanged=%s expected=%s actual=%s",
        resource.day,
        unchanged,
        resource.archive_sha256,
        current,
    )
    return unchanged, None


def _revalidate_cached_resources(
    source: Source,
    client: httpx.Client,
    cached: list[tuple[Resource, Path]],
    worker_count: int,
    executor: Executor | None,
) -> tuple[list[Path], list[tuple[Resource, Path]], list[Message]]:
    """Split cached partitions into reusable and source-corrected groups.

    Args:
        source: The source strategy that owns the archives.
        client: The HTTPX client used for checksum sidecars.
        cached: Valid local resource paths to inspect.
        worker_count: The bounded concurrent checksum request count.
        executor: An optional request-wide cache executor.

    Returns:
        Reusable local paths, partitions requiring ingestion, and warnings.
    """
    if not cached:
        return [], [], []
    with ExitStack() as stack:
        active_executor = executor or stack.enter_context(
            ThreadPoolExecutor(max_workers=worker_count)
        )
        checked = list(
            active_executor.map(
                lambda item: _revalidate_cached_resource(source, client, item[0]),
                cached,
            )
        )
    paths: list[Path] = []
    pending: list[tuple[Resource, Path]] = []
    warnings: list[Message] = []
    for (resource, path), (unchanged, warning) in zip(cached, checked):
        if unchanged:
            paths.append(path)
        else:
            pending.append((resource, path))
        if warning is not None:
            warnings.append(warning)
    return paths, pending, warnings


def _cached_coverage(
    source: Source,
    client: httpx.Client,
    dataset: DatasetSpec,
    resources: list[Resource],
    data_dir: Path,
    key: ResourceKey,
    *,
    offline: bool,
    refresh: bool,
    worker_count: int,
    executor: Executor | None,
) -> tuple[CacheCoverage, list[tuple[Resource, Path]]]:
    """Build cache coverage and optionally revalidate local source archives.

    Args:
        source: The source strategy that owns the archives.
        client: The HTTPX client used for checksum sidecars.
        dataset: The schema expected in every cached partition.
        resources: The cataloged daily resources requested by the caller.
        data_dir: The root downloader data directory.
        key: The requested source dataset identity.
        offline: Whether source access is forbidden.
        refresh: Whether ready resources must check remote checksums.
        worker_count: The bounded concurrent checksum request count.
        executor: An optional request-wide cache executor.

    Returns:
        Cache coverage and resources still requiring ingestion.
    """
    coverage, pending, cached = _cache_plan(
        resources,
        data_dir,
        key,
        dataset,
        offline,
    )
    if not refresh or offline:
        coverage.paths.extend(path for _resource, path in cached)
        return coverage, pending

    paths, changed, warnings = _revalidate_cached_resources(
        source,
        client,
        cached,
        worker_count,
        executor,
    )
    coverage.paths.extend(paths)
    coverage.warnings.extend(warnings)
    pending.extend(changed)
    if cached:
        LOGGER.info(
            "Archive cache revalidated: key=%s cached=%d unchanged=%d changed=%d "
            "unavailable=%d",
            key,
            len(cached),
            len(paths),
            len(changed),
            len(warnings),
        )
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
    ready: list[tuple[date, Path, IngestedResource]] = []
    failed: list[tuple[date, str]] = []
    for (resource, destination), (metadata, error) in zip(pending, outcomes):
        if metadata is not None:
            ready.append((resource.day, destination, metadata))
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
        failed.append((resource.day, message))
        coverage.problems.append(Message("resource_failed", message, resource.day))
        LOGGER.warning(
            "Daily resource failed: key=%s day=%s error=%s",
            key,
            resource.day,
            message,
        )
    catalog.mark_outcomes(key, ready, failed)


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
    refresh: bool = False,
    max_workers: int = 16,
    reporter: Reporter | None = None,
    executor: Executor | None = None,
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
        refresh: Whether ready resources must check their remote checksums.
        max_workers: The caller's maximum concurrent daily ingestions.
        reporter: The optional Rich activity reporter.
        executor: An optional request-wide executor shared by every pair.

    Returns:
        Usable paths and isolated resource problems.
    """
    worker_count = _worker_count(source, dataset, max_workers)
    coverage, pending = _cached_coverage(
        source,
        client,
        dataset,
        resources,
        data_dir,
        key,
        offline=offline,
        refresh=refresh,
        worker_count=worker_count,
        executor=executor,
    )
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
        "warnings=%d problems=%d offline=%s refresh=%s workers=%d",
        key,
        len(resources),
        len(coverage.paths),
        len(pending),
        len(coverage.warnings),
        len(coverage.problems),
        offline,
        refresh,
        worker_count,
    )
    outcomes: list[tuple[IngestedResource | None, Exception | None]] = []
    if pending:
        with display.downloads(key.symbol, len(pending)) as advance:
            with ExitStack() as stack:
                active_executor = executor or stack.enter_context(
                    ThreadPoolExecutor(max_workers=worker_count)
                )
                completed = active_executor.map(
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
