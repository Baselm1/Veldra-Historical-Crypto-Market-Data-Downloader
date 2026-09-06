"""Show optional Rich progress and render downloader results."""

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
import logging

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from .models import Market, Message, Result, TimeRange

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
        LOGGER.debug("Rich reporter created: enabled=%s", enabled)

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
        with self.console.status(Text(message), spinner="dots"):
            yield
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
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        with progress:
            task = progress.add_task(f"{symbol}: downloading daily files", total=total)

            def advance(day: date, succeeded: bool) -> None:
                """Advance the display after one daily-file attempt.

                Args:
                    day: The daily resource date.
                    succeeded: Whether the file entered the cache.
                """
                state = "cached" if succeeded else "failed"
                progress.update(
                    task,
                    description=f"{symbol} {day.isoformat()} {state}",
                    advance=1,
                )

            yield advance
        LOGGER.debug("Rich download progress finished: symbol=%s", symbol)


def _row_limit(rows: object) -> int:
    """Validate a result preview row limit.

    Args:
        rows: The proposed number of rows to show.

    Returns:
        The validated non-negative row limit.
    """
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise TypeError("rows must be an integer")
    if rows < 0:
        raise ValueError("rows must not be negative")
    return rows


def _message_text(kind: str, message: Message) -> Text:
    """Build one styled structured result message.

    Args:
        kind: The warning, problem, or error category.
        message: The structured message being rendered.

    Returns:
        Rich text containing every available message detail.
    """
    styles = {"WARNING": "yellow", "PROBLEM": "yellow", "ERROR": "bold red"}
    parts = [f"{kind} {message.code}: {message.message}"]
    if message.date is not None:
        parts.append(f"Date: {message.date.isoformat()}.")
    if message.suggestions:
        parts.append(f"Suggestions: {', '.join(message.suggestions)}.")
    return Text(" ".join(parts), style=styles[kind])


def _data_table(result: Result, rows: int) -> Table:
    """Build a Rich table containing the requested DataFrame preview.

    Args:
        result: The result whose rows should be previewed.
        rows: The maximum number of rows to show.

    Returns:
        A Rich table containing the selected columns and rows.
    """
    table = Table(show_header=True, header_style="bold cyan")
    for column in result.data.columns:
        table.add_column(str(column))
    for values in result.data.head(rows).itertuples(index=False, name=None):
        table.add_row(*(str(value) for value in values))
    return table


def render_result(
    result: Result, *, console: Console | None = None, rows: int = 10
) -> None:
    """Render one result summary and a preview of its data.

    Args:
        result: The result to render.
        console: The optional destination used for Rich output.
        rows: The maximum number of data rows to preview.
    """
    limit = _row_limit(rows)
    target = console if console is not None else Console()
    state = "complete" if result.complete else "incomplete"
    style = "green" if result.complete else "yellow"
    target.print(
        Text.assemble(
            (result.pair, "bold cyan"),
            f": {len(result.data):,} rows, ",
            (state, style),
        )
    )
    metadata = Table.grid(padding=(0, 2))
    metadata.add_column(style="bold")
    metadata.add_column()
    metadata.add_row("Requested", format_range(result.requested_range))
    metadata.add_row("Available", format_range(result.available_range))
    metadata.add_row("Used", format_range(result.used_range))
    target.print(metadata)
    for kind, messages in (
        ("WARNING", result.warnings),
        ("PROBLEM", result.problems),
        ("ERROR", result.errors),
    ):
        for message in messages:
            target.print(_message_text(kind, message))
    if result.data.empty:
        target.print(Text("No rows returned.", style="dim"))
    elif limit == 0:
        target.print(Text("Data preview disabled.", style="dim"))
    else:
        target.print(_data_table(result, limit))
        if len(result.data) > limit:
            target.print(
                Text(f"Showing {limit:,} of {len(result.data):,} rows.", style="dim")
            )


def render_results(
    results: Result | Sequence[Result],
    *,
    console: Console | None = None,
    rows: int = 10,
) -> None:
    """Render one result or an ordered collection of results.

    Args:
        results: One result or the results to render in order.
        console: The optional destination used for Rich output.
        rows: The maximum number of rows to preview per result.
    """
    limit = _row_limit(rows)
    target = console if console is not None else Console()
    values = [results] if isinstance(results, Result) else list(results)
    for index, result in enumerate(values):
        if index:
            target.print()
        render_result(result, console=target, rows=limit)
