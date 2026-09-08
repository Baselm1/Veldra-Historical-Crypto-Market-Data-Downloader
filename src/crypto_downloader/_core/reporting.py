"""Show optional Rich progress and render downloader results."""

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
import logging
from threading import RLock

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.progress import TaskID
from rich.text import Text

from crypto_downloader._core.models import Market, TimeRange

ProgressCallback = Callable[[date, bool], None]
LOGGER = logging.getLogger(__name__)


def format_time(value: datetime | None) -> str:
    """Format one UTC timestamp for a person.

    Args:
        value: The timestamp to format when available.

    Returns:
        A compact timestamp without Python's object representation.
    """
    if value is None:
        return "Not available"
    aware = value.utcoffset() is not None
    rendered = value.astimezone(UTC) if aware else value
    suffix = " UTC" if aware else ""
    pattern = (
        "%Y-%m-%d" if rendered.time() == datetime.min.time() else "%Y-%m-%d %H:%M:%S"
    )
    return rendered.strftime(pattern) + suffix


def format_range(value: TimeRange | None) -> str:
    """Format one half-open timestamp range.

    Args:
        value: The optional inclusive-start and exclusive-end range.

    Returns:
        A compact human-readable range.
    """
    if value is None:
        return "Not available"
    return f"{format_time(value[0])} to {format_time(value[1])} (end exclusive)"


class Reporter:
    """Show optional interactive activity without changing result data."""

    def __init__(self, enabled: bool = True, *, console: Console | None = None) -> None:
        """Create an enabled or silent reporter.

        Args:
            enabled: Whether Rich output should be shown.
            console: The optional destination used for Rich output.
        """
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a Boolean")
        self.enabled = enabled
        self.console = console if console is not None else Console(stderr=True)
        self._progress_lock = RLock()
        self._progress: Progress | None = None
        self._progress_users = 0
        LOGGER.debug("Rich reporter created: enabled=%s", enabled)

    def _start_task(
        self, description: str, total: int | None
    ) -> tuple[Progress, TaskID]:
        """Start one task on the reporter's shared live display.

        Args:
            description: The task text shown to the caller.
            total: The optional number of work items.

        Returns:
            The shared progress display and new task identifier.
        """
        with self._progress_lock:
            if self._progress is None:
                self._progress = Progress(
                    SpinnerColumn(),
                    TextColumn("{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=self.console,
                )
                self._progress.start()
            self._progress_users += 1
            task = self._progress.add_task(description, total=total)
            return self._progress, task

    def _finish_task(self, progress: Progress, task: TaskID) -> None:
        """Finish one shared progress task and stop an unused display.

        Args:
            progress: The shared progress display containing the task.
            task: The completed task identifier.
        """
        with self._progress_lock:
            progress.update(task, completed=progress.tasks[task].total)
            self._progress_users -= 1
            if self._progress_users == 0:
                progress.stop()
                self._progress = None

    def _line(self, marker: str, style: str, message: str) -> None:
        """Show one styled line when reporting is enabled.

        Args:
            marker: The short message category.
            style: The Rich style applied to the category.
            message: The text shown after the category.
        """
        if self.enabled:
            self.console.print(Text.assemble((f"{marker} ", style), message))

    def info(self, message: str) -> None:
        """Show one informational message.

        Args:
            message: The message to show.
        """
        self._line("INFO", "cyan", message)

    def success(self, message: str) -> None:
        """Show one successful outcome.

        Args:
            message: The message to show.
        """
        self._line("OK", "green", message)

    def warning(self, message: str) -> None:
        """Show one recoverable problem.

        Args:
            message: The message to show.
        """
        self._line("WARN", "yellow", message)

    def error(self, message: str) -> None:
        """Show one failed outcome.

        Args:
            message: The message to show.
        """
        self._line("ERROR", "bold red", message)

    def request(
        self,
        source: str,
        product: str,
        dataset: str,
        pair_count: int,
        start: datetime,
        end: datetime,
    ) -> None:
        """Show the normalized request summary.

        Args:
            source: The requested source code.
            product: The requested source product.
            dataset: The requested dataset.
            pair_count: The number of requested pairs.
            start: The inclusive request start.
            end: The exclusive request end.
        """
        noun = "pair" if pair_count == 1 else "pairs"
        self.info(
            f"{source.title()} {product} {dataset}: {pair_count:,} {noun}, "
            f"{format_range((start, end))}"
        )

    def market_summary(self, markets: Sequence[Market], *, refreshed: bool) -> None:
        """Show market counts grouped by native status.

        Args:
            markets: The markets loaded for this request.
            refreshed: Whether they came from the source in this run.
        """
        counts = Counter(market.status or "ARCHIVE_ONLY" for market in markets)
        statuses = ", ".join(
            f"{count:,} {status}" for status, count in sorted(counts.items())
        )
        origin = "refreshed" if refreshed else "loaded from cache"
        details = f" ({statuses})" if statuses else ""
        noun = "pair" if len(markets) == 1 else "pairs"
        self.success(f"Markets {origin}: {len(markets):,} {noun}{details}")

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        """Show a spinner while one operation runs.

        Args:
            message: The active operation description.

        Yields:
            Control to the operation wrapped by the spinner.
        """
        if not self.enabled:
            LOGGER.debug("Rich status suppressed: %s", message)
            yield
            return
        LOGGER.debug("Rich status started: %s", message)
        progress, task = self._start_task(message, None)
        try:
            yield
        finally:
            self._finish_task(progress, task)
        LOGGER.debug("Rich status finished: %s", message)

    @contextmanager
    def downloads(self, symbol: str, total: int) -> Iterator[ProgressCallback]:
        """Show progress and yield a callback for completed daily files.

        Args:
            symbol: The source market being downloaded.
            total: The number of daily files being attempted.

        Yields:
            A callback accepting the completed date and success state.
        """
        if not self.enabled:
            LOGGER.debug(
                "Rich download progress suppressed: symbol=%s total=%d", symbol, total
            )
            yield lambda _day, _succeeded: None
            return
        LOGGER.debug(
            "Rich download progress started: symbol=%s total=%d", symbol, total
        )
        progress, task = self._start_task(f"{symbol}: downloading archives", total)

        def advance(day: date, succeeded: bool) -> None:
            """Advance the display after one daily-file attempt.

            Args:
                day: The daily resource date.
                succeeded: Whether the file entered the cache.
            """
            state = "cached" if succeeded else "failed"
            with self._progress_lock:
                progress.update(
                    task,
                    description=f"{symbol} {day.isoformat()} {state}",
                    advance=1,
                )

        try:
            yield advance
        finally:
            with self._progress_lock:
                current = progress.tasks[task]
                if current.completed < total:
                    progress.update(task, completed=total)
            self._finish_task(progress, task)
        LOGGER.debug("Rich download progress finished: symbol=%s", symbol)
