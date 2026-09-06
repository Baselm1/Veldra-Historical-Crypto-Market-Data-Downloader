"""Test the thin command-line adapter without live network requests."""

from datetime import UTC, date, datetime
import logging
from pathlib import Path
import runpy
import sys

import httpx
import pandas as pd
import pytest

import crypto_downloader.cli as cli
from crypto_downloader.models import Message, Result


def result(
    *,
    warnings: list[Message] | None = None,
    problems: list[Message] | None = None,
    errors: list[Message] | None = None,
) -> Result:
    """Build one representative CLI result.

    Args:
        warnings: Optional request adjustments.
        problems: Optional incomplete coverage details.
        errors: Optional pair-specific failures.

    Returns:
        A result suitable for shared rendering.
    """
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 2, tzinfo=UTC)
    return Result(
        pair="BTCUSDT",
        data=pd.DataFrame({"open_time": [start], "close": [100.0]}),
        requested_range=(start, end),
        used_range=(start, end),
        available_range=(start, end),
        warnings=warnings or [],
        problems=problems or [],
        errors=errors or [],
    )


def test_cli_help_describes_the_primary_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Confirm module users can discover the command-line interface.

    Args:
        capsys: Pytest's captured output streams.
    """
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])

    output = capsys.readouterr().out
    assert error.value.code == 0
    for option in (
        "--start",
        "--end",
        "--product",
        "--dataset",
        "--interval",
        "--columns",
        "--config",
        "--gap-policy",
        "--market-refresh-hours",
        "--offline",
        "--refresh",
        "--quiet",
        "--debug",
    ):
        assert option in output


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["BTCUSDT"],
        ["BTCUSDT", "--start", "2025-01-01"],
        ["--start", "2025-01-01", "--end", "2025-01-02"],
    ],
)
def test_cli_requires_pairs_and_both_date_arguments(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Confirm missing command-line fields receive an argparse error.

    Args:
        arguments: The incomplete command-line values.
        capsys: Pytest's captured output streams.
    """
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)

    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_forwards_options_and_renders_complete_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Confirm CLI arguments call the same public service and renderer.

    Args:
        tmp_path: The isolated data directory.
        monkeypatch: Pytest's replacement helper.
        capsys: Pytest's captured output streams.
    """
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_get_results(*args: object, **kwargs: object) -> list[Result]:
        """Record CLI forwarding and return one complete result."""
        calls.append((args, kwargs))
        return [result()]

    monkeypatch.setattr(cli, "get_results", fake_get_results)

    exit_code = cli.main(
        [
            "BTCUSDT",
            "ETHUSDT",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-03",
            "--product",
            "spot",
            "--dataset",
            "klines",
            "--interval",
            "1h",
            "--columns",
            "open_time",
            "close",
            "--gap-policy",
            "keep",
            "--data-dir",
            str(tmp_path),
            "--earliest-date",
            "2019-01-01",
            "--max-workers",
            "8",
            "--discovery-tail-days",
            "4",
            "--market-refresh-hours",
            "12",
            "--refresh",
            "--quiet",
            "--rows",
            "1",
        ]
    )

    assert exit_code == 0
    assert calls[0][0] == (["BTCUSDT", "ETHUSDT"], "2025-01-01", "2025-01-03")
    assert calls[0][1] == {
        "data_dir": str(tmp_path),
        "product": "spot",
        "dataset": "klines",
        "interval": "1h",
        "desired_columns": ["open_time", "close"],
        "config_path": "config.toml",
        "earliest_date": "2019-01-01",
        "max_workers": 8,
        "discovery_tail_days": 4,
        "market_refresh_hours": 12.0,
        "refresh": True,
        "offline": False,
        "gap_policy": "keep",
        "progress": False,
    }
    output = capsys.readouterr().out
    assert "BTCUSDT: 1 rows, complete" in output
    assert "open_time" in output and "close" in output


@pytest.mark.parametrize(
    ("message_field", "expected_code"),
    [
        ("problems", "resource_unavailable"),
        ("errors", "unknown_pair"),
    ],
)
def test_cli_returns_one_and_renders_incomplete_results(
    message_field: str,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Confirm pair-level failures remain visible and set a failure exit code.

    Args:
        message_field: The result collection receiving the message.
        expected_code: The diagnostic code expected in output.
        monkeypatch: Pytest's replacement helper.
        capsys: Pytest's captured output streams.
    """
    message = Message(expected_code, "Representative failure.", date(2025, 1, 1))
    value = result(**{message_field: [message]})
    monkeypatch.setattr(cli, "get_results", lambda *_args, **_kwargs: value)

    exit_code = cli.main(["BTCUSDT", "--start", "2025-01-01", "--end", "2025-01-01"])

    assert exit_code == 1
    assert expected_code in capsys.readouterr().out


def test_cli_warnings_do_not_turn_a_successful_result_into_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm harmless range adjustments retain a successful exit code.

    Args:
        monkeypatch: Pytest's replacement helper.
    """
    value = result(warnings=[Message("start_trimmed", "Start adjusted.")])
    monkeypatch.setattr(cli, "get_results", lambda *_args, **_kwargs: value)

    assert cli.main(["BTCUSDT", "--start", "2025-01-01", "--end", "2025-01-01"]) == 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["BTCUSDT", "--start", "bad", "--end", "2025-01-01"],
        ["BTCUSDT", "--start", "2025-01-02", "--end", "2025-01-01"],
    ],
)
def test_cli_turns_invalid_requests_into_usage_errors(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Confirm invalid user input does not produce a Python traceback.

    Args:
        arguments: The invalid command-line request.
        capsys: Pytest's captured output streams.
    """
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)

    assert error.value.code == 2
    output = capsys.readouterr().err
    assert "error:" in output
    assert "Traceback" not in output


@pytest.mark.parametrize(
    ("rows", "expected"),
    [("-1", "must not be negative"), ("many", "must be an integer")],
)
def test_cli_rejects_invalid_preview_rows_before_calling_the_service(
    rows: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Confirm an invalid display option cannot trigger source access.

    Args:
        rows: The invalid preview limit text.
        expected: The parser error expected for the value.
        monkeypatch: Pytest's replacement helper.
        capsys: Pytest's captured output streams.
    """
    monkeypatch.setattr(
        cli,
        "get_results",
        lambda *_args, **_kwargs: pytest.fail("service must not be called"),
    )

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "BTCUSDT",
                "--start",
                "2025-01-01",
                "--end",
                "2025-01-01",
                "--rows",
                rows,
            ]
        )

    assert error.value.code == 2
    assert expected in capsys.readouterr().err


def test_cli_reports_operational_failures_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Confirm source connection failures become a concise CLI error.

    Args:
        monkeypatch: Pytest's replacement helper.
        capsys: Pytest's captured output streams.
    """
    request = httpx.Request("GET", "https://example.test")
    monkeypatch.setattr(
        cli,
        "get_results",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            httpx.ConnectError("source unavailable", request=request)
        ),
    )

    exit_code = cli.main(["BTCUSDT", "--start", "2025-01-01", "--end", "2025-01-01"])

    output = capsys.readouterr().err
    assert exit_code == 1
    assert "ERROR" in output and "source unavailable" in output
    assert "Traceback" not in output


def test_cli_debug_logging_is_scoped_and_restored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Confirm --debug shows package logs without changing caller logger state.

    Args:
        monkeypatch: Pytest's replacement helper.
        capsys: Pytest's captured output streams.
    """
    package_logger = logging.getLogger("crypto_downloader")
    handlers = tuple(package_logger.handlers)
    level = package_logger.level
    propagate = package_logger.propagate
    monkeypatch.setattr(cli, "get_results", lambda *_args, **_kwargs: result())

    assert (
        cli.main(
            [
                "BTCUSDT",
                "--start",
                "2025-01-01",
                "--end",
                "2025-01-01",
                "--debug",
            ]
        )
        == 0
    )

    debug_output = capsys.readouterr().err
    assert "CLI arguments" in debug_output and "parsed:" in debug_output
    assert tuple(package_logger.handlers) == handlers
    assert package_logger.level == level
    assert package_logger.propagate == propagate


def test_python_module_entrypoint_exits_with_cli_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm python -m delegates directly to cli.main.

    Args:
        monkeypatch: Pytest's replacement helper.
    """
    monkeypatch.delitem(sys.modules, "crypto_downloader.__main__", raising=False)
    monkeypatch.setattr(cli, "main", lambda _argv=None: 7)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("crypto_downloader.__main__", run_name="__main__")

    assert error.value.code == 7
