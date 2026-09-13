"""Test bounded physical download and normalization pipelines."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock, Thread
import time

import pytest

from veldra.core.pipeline import (
    Downloaded,
    InFlightByteBudget,
    PipelineItem,
    run_bounded_pipeline,
)


def test_byte_budget_blocks_until_capacity_is_released() -> None:
    """Confirm simultaneous reservations cannot exceed configured bytes."""
    budget = InFlightByteBudget(100, unknown_reservation=25)
    first = budget.reserve(80)
    acquired = Event()

    def reserve_second() -> None:
        """Wait for and release a second reservation."""
        with budget.acquire(30):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(reserve_second)
        time.sleep(0.02)
        assert acquired.is_set() is False
        first.release()
        future.result(timeout=1)

    assert acquired.is_set() is True
    assert budget.in_flight_bytes == 0


def test_unknown_and_oversized_files_have_safe_reservations() -> None:
    """Confirm unknown files use a default and one oversized file runs alone."""
    budget = InFlightByteBudget(100, unknown_reservation=40)
    with budget.acquire(None) as unknown:
        assert unknown.bytes_reserved == 40
    with budget.acquire(250) as oversized:
        assert oversized.bytes_reserved == 250
        assert budget.in_flight_bytes == 250
    assert budget.in_flight_bytes == 0


@pytest.mark.parametrize(
    ("capacity", "unknown"),
    [(0, 1), (1, 0), (True, 1), (10, 11)],
)
def test_invalid_budget_settings_are_rejected(
    capacity: object, unknown: object
) -> None:
    """Confirm byte budgets require useful positive integer settings.

    Args:
        capacity: The proposed total byte capacity.
        unknown: The proposed unknown-file reservation.
    """
    with pytest.raises((TypeError, ValueError)):
        InFlightByteBudget(capacity, unknown_reservation=unknown)  # type: ignore[arg-type]


def test_pipeline_preserves_order_bounds_bytes_and_records_timings() -> None:
    """Confirm concurrent stages remain ordered and publish successful outcomes."""
    lock = Lock()
    active = 0
    peak = 0
    published: list[int] = []

    def download(value: int) -> Downloaded[int]:
        """Return one downloaded value after tracking concurrent bytes."""
        nonlocal active, peak
        with lock:
            active += 60
            peak = max(peak, active)
        time.sleep(0.005)
        with lock:
            active -= 60
        return Downloaded(value * 2, verification_seconds=0.001)

    def process(value: int) -> int:
        """Convert one downloaded value."""
        time.sleep(0.002)
        return value + 1

    def publish(outcomes: object) -> None:
        """Retain successful result values from the batch."""
        published.extend(
            outcome.value for outcome in outcomes if outcome.value is not None  # type: ignore[union-attr]
        )

    outcomes, metrics = run_bounded_pipeline(
        [PipelineItem(value, 60) for value in range(5)],
        download,
        process,
        download_workers=4,
        processing_workers=2,
        byte_budget=InFlightByteBudget(120),
        publish=publish,
    )

    assert [outcome.value for outcome in outcomes] == [1, 3, 5, 7, 9]
    assert all(outcome.error is None for outcome in outcomes)
    assert peak <= 120
    assert published == [1, 3, 5, 7, 9]
    assert metrics.completed == 5
    assert metrics.failed == 0
    assert metrics.network_seconds >= 0
    assert metrics.verification_seconds == pytest.approx(0.005)
    assert metrics.normalization_seconds > 0
    assert metrics.catalog_seconds >= 0


def test_pipeline_isolates_download_and_processing_failures_and_cleans_up() -> None:
    """Confirm worker failures do not leak artifacts or stop neighboring work."""
    cleaned: list[int] = []

    def download(value: int) -> Downloaded[int]:
        """Fail one download and return all other values."""
        if value == 2:
            raise RuntimeError("download failed")
        return Downloaded(value)

    def process(value: int) -> int:
        """Fail one normalization and return all other values."""
        if value == 3:
            raise ValueError("processing failed")
        return value

    outcomes, metrics = run_bounded_pipeline(
        [PipelineItem(value, 10) for value in range(5)],
        download,
        process,
        download_workers=3,
        processing_workers=2,
        byte_budget=InFlightByteBudget(30),
        cleanup=cleaned.append,
    )

    assert [type(outcome.error) for outcome in outcomes] == [
        type(None),
        type(None),
        RuntimeError,
        ValueError,
        type(None),
    ]
    assert cleaned == [3]
    assert metrics.completed == 3
    assert metrics.failed == 2


def test_pipeline_honors_preexisting_cancellation() -> None:
    """Confirm cancellation prevents new downloads from being scheduled."""
    cancelled = Event()
    cancelled.set()
    called: list[int] = []

    outcomes, metrics = run_bounded_pipeline(
        [PipelineItem(1, None), PipelineItem(2, None)],
        lambda value: (called.append(value), Downloaded(value))[1],
        lambda value: value,
        download_workers=1,
        processing_workers=1,
        byte_budget=InFlightByteBudget(10),
        cancellation=cancelled,
    )

    assert called == []
    assert all(isinstance(outcome.error, CancelledError) for outcome in outcomes)
    assert metrics.failed == 2


def test_keyboard_interrupt_cancels_and_joins_pipeline_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm a caller interrupt stops queued work and leaves no live workers."""
    cancellation = Event()
    joined: list[Thread] = []
    original_join = Thread.join
    interrupted = False

    def interrupt_once(self: Thread, timeout: float | None = None) -> None:
        """Interrupt the first join and delegate every cleanup join afterward."""
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        original_join(self, timeout)
        joined.append(self)

    monkeypatch.setattr(Thread, "join", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        run_bounded_pipeline(
            [PipelineItem(value, 1) for value in range(20)],
            lambda value: (time.sleep(0.002), Downloaded(value))[1],
            lambda value: value,
            download_workers=2,
            processing_workers=1,
            byte_budget=InFlightByteBudget(2),
            cancellation=cancellation,
        )

    assert cancellation.is_set()
    assert joined
    assert all(not worker.is_alive() for worker in joined)


def test_pipeline_validates_workers_and_duplicate_item_identity() -> None:
    """Confirm malformed concurrency and duplicate item objects fail clearly."""
    item = PipelineItem(Path("one"), 1)
    arguments = (lambda value: Downloaded(value), lambda value: value)
    with pytest.raises(ValueError, match="workers"):
        run_bounded_pipeline(
            [item],
            *arguments,
            download_workers=0,
            processing_workers=1,
            byte_budget=InFlightByteBudget(1),
        )
    with pytest.raises(ValueError, match="duplicate"):
        run_bounded_pipeline(
            [item, item],
            *arguments,
            download_workers=1,
            processing_workers=1,
            byte_budget=InFlightByteBudget(1),
        )
