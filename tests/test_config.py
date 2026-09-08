"""Test TOML downloader settings."""

from datetime import date
from pathlib import Path

import pytest

from crypto_downloader._core.config import Settings, load_settings


def _settings_file(tmp_path: Path, text: str) -> Path:
    """Write one TOML settings file for a test.

    Args:
        tmp_path: The isolated temporary directory.
        text: The TOML text to write.

    Returns:
        The written TOML path.
    """
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_settings_uses_installed_defaults_without_a_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm default settings do not depend on the working directory.

    Args:
        tmp_path: The empty directory used as the process working directory.
        monkeypatch: The pytest helper used to change the working directory.
    """
    monkeypatch.chdir(tmp_path)

    assert load_settings() == Settings(date(2020, 1, 1), "1m")
    assert not (tmp_path / "config.toml").exists()


def test_load_settings_reads_a_fixed_history_cutoff(tmp_path: Path) -> None:
    """Confirm TOML settings expose a fixed configured history date.

    Args:
        tmp_path: The isolated temporary directory.
    """
    path = _settings_file(
        tmp_path,
        """[history]
earliest_date = "2020-01-01"

[klines]
base_interval = "1m"
""",
    )

    assert load_settings(path) == Settings(date(2020, 1, 1), "1m")


def test_load_settings_accepts_all_history(tmp_path: Path) -> None:
    """Confirm all-history TOML settings do not impose a date cutoff.

    Args:
        tmp_path: The isolated temporary directory.
    """
    path = _settings_file(
        tmp_path,
        """[history]
earliest_date = "all"

[klines]
base_interval = "1m"
""",
    )

    assert load_settings(path) == Settings(None, "1m")


@pytest.mark.parametrize(
    "text",
    [
        "",
        '[history]\nearliest_date = "not-a-date"\n[klines]\nbase_interval = "1m"\n',
        '[history]\nearliest_date = "2020-01-01"\n[klines]\nbase_interval = "1h"\n',
        '[history]\nearliest_date = 20200101\n[klines]\nbase_interval = "1m"\n',
    ],
)
def test_load_settings_rejects_invalid_values(tmp_path: Path, text: str) -> None:
    """Confirm malformed TOML settings fail with a value error.

    Args:
        tmp_path: The isolated temporary directory.
        text: The invalid TOML text.
    """
    with pytest.raises(ValueError):
        load_settings(_settings_file(tmp_path, text))


def test_load_settings_requires_an_existing_file(tmp_path: Path) -> None:
    """Confirm settings loading explains a missing TOML file.

    Args:
        tmp_path: The isolated temporary directory.
    """
    with pytest.raises(FileNotFoundError, match="config.toml"):
        load_settings(tmp_path / "config.toml")
