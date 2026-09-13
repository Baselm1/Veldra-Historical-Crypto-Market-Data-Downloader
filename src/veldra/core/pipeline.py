"""Bound concurrent archive downloads and normalization by byte pressure."""

from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from queue import Queue
from threading import Condition, Event, Lock, Thread
from time import perf_counter
from typing import cast


def _positive_integer(value: object, name: str) -> int:
    """Validate one positive integer pipeline setting.

    Args:
        value: The proposed setting.
        name: The setting name used in errors.

    Returns:
        The validated integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


class ByteReservation(AbstractContextManager["ByteReservation"]):
    """Hold byte capacity until one downloaded object finishes processing."""

    def __init__(self, budget: "InFlightByteBudget", reserved: int) -> None:
        """Retain the owning budget and reservation size.

        Args:
            budget: The byte budget that granted capacity.
            reserved: The bytes reserved for this object.
        """
        self._budget = budget
        self.bytes_reserved = reserved
        self._released = False

    def release(self) -> None:
        """Return this reservation to its owning budget once."""
        if self._released:
            return
        self._released = True
        self._budget._release(self.bytes_reserved)

    def __exit__(self, *exc: object) -> None:
        """Release byte capacity when a context exits.

        Args:
            exc: Optional exception details supplied by the context protocol.
        """
        self.release()


class InFlightByteBudget:
    """Limit downloaded-but-not-processed bytes across worker threads."""

    def __init__(
        self, capacity: int, *, unknown_reservation: int | None = None
    ) -> None:
        """Create a byte budget and unknown-file estimate.

        Args:
            capacity: Maximum ordinary bytes allowed in flight.
            unknown_reservation: Bytes reserved when a source omits its size.
        """
        self.capacity = _positive_integer(capacity, "capacity")
        default_unknown = min(self.capacity, 64 * 1024 * 1024)
        self.unknown_reservation = _positive_integer(
            default_unknown if unknown_reservation is None else unknown_reservation,
            "unknown_reservation",
        )
        if self.unknown_reservation > self.capacity:
            raise ValueError("unknown_reservation cannot exceed capacity")
        self._condition = Condition()
        self._in_flight = 0

    @property
    def in_flight_bytes(self) -> int:
        """Return the bytes currently reserved by active pipeline objects."""
        with self._condition:
            return self._in_flight

    @staticmethod
    def _size(advertised_bytes: int | None, unknown: int) -> int:
        """Validate and resolve one advertised archive size.

        Args:
            advertised_bytes: The known remote size or ``None``.
            unknown: The configured estimate for unknown files.

        Returns:
            The bytes this file must reserve.
        """
        if advertised_bytes is None:
            return unknown
        if isinstance(advertised_bytes, bool) or not isinstance(advertised_bytes, int):
            raise TypeError("advertised_bytes must be an integer or None")
        if advertised_bytes < 0:
            raise ValueError("advertised_bytes cannot be negative")
        return advertised_bytes

    def reserve(self, advertised_bytes: int | None) -> ByteReservation:
        """Wait for and reserve capacity for one physical object.

        Args:
            advertised_bytes: The known object size or ``None``.

        Returns:
            A reservation that must be released after processing.
        """
        requested = self._size(advertised_bytes, self.unknown_reservation)
        with self._condition:
            while self._in_flight and self._in_flight + requested > self.capacity:
                self._condition.wait()
            self._in_flight += requested
        return ByteReservation(self, requested)

    def acquire(self, advertised_bytes: int | None) -> ByteReservation:
        """Return a context-managed byte reservation.

        Args:
            advertised_bytes: The known object size or ``None``.

        Returns:
            A reservation usable as a context manager.
        """
        return self.reserve(advertised_bytes)

    def _release(self, reserved: int) -> None:
        """Release bytes and wake workers waiting for capacity.

        Args:
            reserved: The reservation size being returned.
        """
        with self._condition:
            self._in_flight -= reserved
            if self._in_flight < 0:
                raise RuntimeError("byte budget was released too many times")
            self._condition.notify_all()


@dataclass(frozen=True)
class PipelineItem[InputT]:
    """Pair one pipeline input with its advertised physical byte size."""

    value: InputT
    advertised_bytes: int | None


@dataclass(frozen=True)
class Downloaded[DownloadedT]:
    """Return downloaded state and time spent verifying its integrity."""

    value: DownloadedT
    verification_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Reject a negative integrity-verification duration."""
        if self.verification_seconds < 0:
            raise ValueError("verification_seconds cannot be negative")


@dataclass(frozen=True)
class WorkTimings:
    """Record time spent waiting, downloading, verifying, and normalizing."""

    queue_seconds: float = 0.0
    network_seconds: float = 0.0
    verification_seconds: float = 0.0
    normalization_seconds: float = 0.0


@dataclass(frozen=True)
class PipelineOutcome[InputT, OutputT]:
    """Describe one ordered pipeline result or isolated failure."""

    item: PipelineItem[InputT]
    value: OutputT | None
    error: BaseException | None
    timings: WorkTimings


@dataclass(frozen=True)
class PipelineMetrics:
    """Summarize one bounded download and processing batch."""

    completed: int
    failed: int
    queue_seconds: float
    network_seconds: float
    verification_seconds: float
    normalization_seconds: float
    catalog_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class _DownloadedWork[InputT, DownloadedT]:
    """Carry one downloaded value and held byte reservation to processing."""

    index: int
    item: PipelineItem[InputT]
    downloaded: Downloaded[DownloadedT]
    reservation: ByteReservation
    timings: WorkTimings


def _metrics[InputT, OutputT](
    outcomes: Sequence[PipelineOutcome[InputT, OutputT]],
    *,
    catalog_seconds: float,
    elapsed_seconds: float,
) -> PipelineMetrics:
    """Aggregate per-item timings into one batch report.

    Args:
        outcomes: Ordered completed and failed work items.
        catalog_seconds: Time spent publishing the completed batch.
        elapsed_seconds: Total wall time for the pipeline.

    Returns:
        Aggregate stage timings and outcome counts.
    """
    return PipelineMetrics(
        completed=sum(outcome.error is None for outcome in outcomes),
        failed=sum(outcome.error is not None for outcome in outcomes),
        queue_seconds=sum(outcome.timings.queue_seconds for outcome in outcomes),
        network_seconds=sum(outcome.timings.network_seconds for outcome in outcomes),
        verification_seconds=sum(
            outcome.timings.verification_seconds for outcome in outcomes
        ),
        normalization_seconds=sum(
            outcome.timings.normalization_seconds for outcome in outcomes
        ),
        catalog_seconds=catalog_seconds,
        elapsed_seconds=elapsed_seconds,
    )


def _join_pipeline_workers[HandoffT](
    downloaders: Sequence[Thread],
    processors: Sequence[Thread],
    handoff: Queue[HandoffT | object],
    sentinel: object,
    stop: Event,
) -> None:
    """Join both pipeline stages and preserve an interrupt until cleanup ends.

    Args:
        downloaders: Active download-stage threads.
        processors: Active processing-stage threads.
        handoff: Queue connecting the two stages.
        sentinel: Unique value that stops a processing worker.
        stop: Shared cancellation signal.
    """
    interrupted: BaseException | None = None
    sentinels_sent = 0
    try:
        for worker in downloaders:
            worker.join()
        for _ in processors:
            handoff.put(sentinel)
            sentinels_sent += 1
        for worker in processors:
            worker.join()
    except BaseException as error:
        stop.set()
        interrupted = error
    finally:
        for worker in downloaders:
            if worker.is_alive():
                worker.join()
        while sentinels_sent < len(processors):
            handoff.put(sentinel)
            sentinels_sent += 1
        for worker in processors:
            if worker.is_alive():
                worker.join()
    if interrupted is not None:
        raise interrupted


def run_bounded_pipeline[InputT, DownloadedT, OutputT](
    items: Sequence[PipelineItem[InputT]],
    download: Callable[[InputT], Downloaded[DownloadedT]],
    process: Callable[[DownloadedT], OutputT],
    *,
    download_workers: int,
    processing_workers: int,
    byte_budget: InFlightByteBudget,
    publish: Callable[[Sequence[PipelineOutcome[InputT, OutputT]]], None] | None = None,
    cleanup: Callable[[DownloadedT], None] | None = None,
    cancellation: Event | None = None,
) -> tuple[list[PipelineOutcome[InputT, OutputT]], PipelineMetrics]:
    """Download and normalize objects through a bounded concurrent handoff.

    Args:
        items: Physical objects and their advertised sizes.
        download: Function that streams and verifies one object.
        process: Function that normalizes one downloaded object.
        download_workers: Maximum simultaneous network workers.
        processing_workers: Maximum simultaneous normalization workers.
        byte_budget: Shared downloaded-object byte budget.
        publish: Optional one-transaction catalog publisher.
        cleanup: Optional cleanup for downloaded values that fail processing.
        cancellation: Optional event that prevents new work.

    Returns:
        Ordered isolated outcomes and aggregate timing metrics.
    """
    download_count = _positive_integer(download_workers, "download_workers")
    process_count = _positive_integer(processing_workers, "processing_workers")
    if len({id(item) for item in items}) != len(items):
        raise ValueError("pipeline contains a duplicate item object")
    started = perf_counter()
    stop = cancellation if cancellation is not None else Event()
    task_queue: Queue[tuple[int, PipelineItem[InputT], float] | object] = Queue()
    handoff: Queue[_DownloadedWork[InputT, DownloadedT] | object] = Queue(
        maxsize=max(1, process_count * 2)
    )
    sentinel = object()
    outcomes: dict[int, PipelineOutcome[InputT, OutputT]] = {}
    outcome_lock = Lock()
    enqueued = perf_counter()
    for index, item in enumerate(items):
        task_queue.put((index, item, enqueued))
    for _ in range(download_count):
        task_queue.put(sentinel)

    def record(index: int, outcome: PipelineOutcome[InputT, OutputT]) -> None:
        """Store one isolated outcome by its caller order.

        Args:
            index: The original input position.
            outcome: The completed or failed result.
        """
        with outcome_lock:
            outcomes[index] = outcome

    def download_worker() -> None:
        """Reserve bytes, download objects, and feed the processing queue."""
        while True:
            queued = task_queue.get()
            if queued is sentinel:
                return
            index, item, queued_at = cast(
                tuple[int, PipelineItem[InputT], float], queued
            )
            timing = WorkTimings(queue_seconds=perf_counter() - queued_at)
            if stop.is_set():
                record(index, PipelineOutcome(item, None, CancelledError(), timing))
                continue
            reservation: ByteReservation | None = None
            try:
                reservation = byte_budget.reserve(item.advertised_bytes)
                network_started = perf_counter()
                downloaded = download(item.value)
                total = perf_counter() - network_started
                timing = replace(
                    timing,
                    network_seconds=max(0.0, total - downloaded.verification_seconds),
                    verification_seconds=downloaded.verification_seconds,
                )
                handoff.put(
                    _DownloadedWork(index, item, downloaded, reservation, timing)
                )
            except BaseException as error:
                if reservation is not None:
                    reservation.release()
                record(index, PipelineOutcome(item, None, error, timing))

    def process_worker() -> None:
        """Normalize downloaded objects and release their held byte capacity."""
        while True:
            queued = handoff.get()
            if queued is sentinel:
                return
            work = cast(_DownloadedWork[InputT, DownloadedT], queued)
            try:
                if stop.is_set():
                    raise CancelledError()
                normalize_started = perf_counter()
                value = process(work.downloaded.value)
                timing = replace(
                    work.timings,
                    normalization_seconds=perf_counter() - normalize_started,
                )
                record(work.index, PipelineOutcome(work.item, value, None, timing))
            except BaseException as error:
                if cleanup is not None:
                    cleanup(work.downloaded.value)
                record(
                    work.index,
                    PipelineOutcome(work.item, None, error, work.timings),
                )
            finally:
                work.reservation.release()

    processors = [Thread(target=process_worker) for _ in range(process_count)]
    downloaders = [Thread(target=download_worker) for _ in range(download_count)]
    for worker in (*processors, *downloaders):
        worker.start()
    _join_pipeline_workers(downloaders, processors, handoff, sentinel, stop)

    ordered = [outcomes[index] for index in range(len(items))]
    catalog_seconds = 0.0
    if publish is not None:
        catalog_started = perf_counter()
        publish(ordered)
        catalog_seconds = perf_counter() - catalog_started
    metrics = _metrics(
        ordered,
        catalog_seconds=catalog_seconds,
        elapsed_seconds=perf_counter() - started,
    )
    return ordered, metrics
