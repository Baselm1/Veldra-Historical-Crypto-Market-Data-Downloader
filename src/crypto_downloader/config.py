"""Load the downloader settings stored in TOML."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Settings:
    """Hold the small set of project-wide downloader settings."""

    earliest_date: date | None
    kline_base_interval: str


def _table(data: dict[str, object], name: str) -> dict[str, object]:
    """Return one required TOML table.

    Args:
        data: The decoded root TOML mapping.
        name: The required table name.

    Returns:
        The validated TOML table mapping.

    Raises:
        ValueError: If the table is absent or not a mapping.
    """
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"config requires a [{name}] table")
    return value


def _earliest_date(value: object) -> date | None:
    """Parse a configured cutoff date or the all-history option.

    Args:
        value: The TOML history boundary value.

    Returns:
        A fixed cutoff date, or ``None`` for all available history.

    Raises:
        ValueError: If the configured value is not an ISO date or ``all``.
    """
    if not isinstance(value, str):
        raise ValueError("history.earliest_date must be an ISO date or 'all'")
    if value == "all":
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            "history.earliest_date must be an ISO date or 'all'"
        ) from error


def _kline_base_interval(value: object) -> str:
    """Validate the currently supported cached Kline resolution.

    Args:
        value: The TOML Kline base interval value.

    Returns:
        The validated source archive interval.

    Raises:
        ValueError: If the value is not the supported one-minute interval.
    """
    if value != "1m":
        raise ValueError("klines.base_interval currently supports only '1m'")
    return "1m"


def load_settings(path: str | Path = "config.toml") -> Settings:
    """Read downloader settings from one TOML file.

    Args:
        path: The TOML file containing downloader settings.

    Returns:
        The validated immutable downloader settings.
    """
    config_path = Path(path).expanduser()
    try:
        with config_path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML in {config_path}") from error
    if not isinstance(data, dict):
        raise ValueError("config root must be a TOML table")
    history = _table(data, "history")
    klines = _table(data, "klines")
    return Settings(
        earliest_date=_earliest_date(history.get("earliest_date")),
        kline_base_interval=_kline_base_interval(klines.get("base_interval")),
    )
