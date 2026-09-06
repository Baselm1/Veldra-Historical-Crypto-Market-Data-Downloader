"""Adapt command-line arguments to the public downloader service."""

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import logging

import httpx
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

from .display import render_results
from .downloader import get_results
from .models import Result

LOGGER = logging.getLogger(__name__)


def _nonnegative_integer(value: str) -> int:
    """Parse a non-negative integer command-line value.

    Args:
        value: The command-line text to parse.

    Returns:
        The parsed non-negative integer.
    """
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


@contextmanager
def _cli_logging(debug: bool) -> Iterator[None]:
    """Temporarily show package debug logs when requested by the CLI.

    Args:
        debug: Whether DEBUG-level package records should be shown.

    Yields:
        Control while the CLI owns its package logging handler.
    """
    package_logger = logging.getLogger("crypto_downloader")
    previous_level = package_logger.level
    previous_propagate = package_logger.propagate
    level = logging.DEBUG if debug else logging.CRITICAL
    handler = RichHandler(
        console=Console(stderr=True), rich_tracebacks=True, markup=False
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False
    try:
        yield
    finally:
        package_logger.removeHandler(handler)
        handler.close()
        package_logger.setLevel(previous_level)
        package_logger.propagate = previous_propagate


def build_parser() -> argparse.ArgumentParser:
    """Create the crypto downloader command-line parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Download, cache, and query Binance historical data.",
        epilog=(
            "Example: python -m crypto_downloader BTCUSDT --start 2025-01-01 "
            "--end 2025-01-03 --interval 1h"
        ),
    )
    parser.add_argument("pairs", nargs="+", help="one or more source pair symbols")
    parser.add_argument(
        "--start", required=True, help="inclusive start date or timestamp"
    )
    parser.add_argument(
        "--end", required=True, help="inclusive date or exclusive exact timestamp"
    )
    parser.add_argument(
        "--product", default="spot", help="source product (default: spot)"
    )
    parser.add_argument("--dataset", default="klines", help="dataset (default: klines)")
    parser.add_argument("--interval", help="output interval (default: stored interval)")
    parser.add_argument("--columns", nargs="+", help="columns to select")
    parser.add_argument("--config", default="config.toml", help="settings TOML file")
    parser.add_argument("--data-dir", default="data", help="cache directory")
    parser.add_argument(
        "--earliest-date", help="override configured earliest date or use 'all'"
    )
    parser.add_argument("--max-workers", type=int, default=32, help="download workers")
    parser.add_argument(
        "--discovery-tail-days",
        type=int,
        default=7,
        help="recent active-market days to rescan",
    )
    parser.add_argument(
        "--market-refresh-hours",
        type=float,
        default=24.0,
        help="hours to reuse cached market metadata",
    )
    parser.add_argument(
        "--gap-policy",
        choices=("forward", "backward", "nan", "keep", "raise"),
        default="forward",
        help="internal missing-candle behavior",
    )
    parser.add_argument(
        "--rows", type=_nonnegative_integer, default=10, help="preview rows per pair"
    )
    parser.add_argument("--offline", action="store_true", help="use cached data only")
    parser.add_argument("--refresh", action="store_true", help="repeat full discovery")
    parser.add_argument("--quiet", action="store_true", help="hide Rich activity")
    parser.add_argument("--debug", action="store_true", help="show package debug logs")
    return parser


def _run(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> int:
    """Call the public service and render its results.

    Args:
        parser: The parser used to report invalid library arguments.
        arguments: The values parsed from the command line.

    Returns:
        Zero for complete results or one for incomplete results.
    """
    try:
        results = get_results(
            arguments.pairs,
            arguments.start,
            arguments.end,
            data_dir=arguments.data_dir,
            product=arguments.product,
            dataset=arguments.dataset,
            interval=arguments.interval,
            desired_columns=arguments.columns,
            config_path=arguments.config,
            earliest_date=arguments.earliest_date,
            max_workers=arguments.max_workers,
            discovery_tail_days=arguments.discovery_tail_days,
            market_refresh_hours=arguments.market_refresh_hours,
            refresh=arguments.refresh,
            offline=arguments.offline,
            gap_policy=arguments.gap_policy,
            progress=not arguments.quiet,
        )
        values = [results] if isinstance(results, Result) else results
        render_results(values, rows=arguments.rows)
    except (TypeError, ValueError) as error:
        LOGGER.debug("CLI request rejected", exc_info=True)
        parser.error(str(error))
    except (httpx.HTTPError, OSError, RuntimeError) as error:
        LOGGER.debug("CLI operation failed", exc_info=True)
        Console(stderr=True).print(Text.assemble(("ERROR ", "bold red"), str(error)))
        return 1
    exit_code = int(any(result.errors or result.problems for result in values))
    LOGGER.debug(
        "CLI request complete: results=%d exit_code=%d", len(values), exit_code
    )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command-line downloader request.

    Args:
        argv: Optional arguments excluding the executable name.

    Returns:
        Zero for complete results or one for incomplete results.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    with _cli_logging(arguments.debug):
        LOGGER.debug("CLI arguments parsed: %s", vars(arguments))
        return _run(parser, arguments)
