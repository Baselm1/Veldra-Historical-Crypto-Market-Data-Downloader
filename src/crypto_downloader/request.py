"""Validate request values before the downloader uses disk or the network."""

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
import logging
import re
from typing import TYPE_CHECKING, cast

import pandas as pd

if TYPE_CHECKING:
    from .datasets import DatasetSpec

type TimeRange = tuple[datetime, datetime]
type ColumnSelection = dict[str, str] | None
GAP_POLICIES = frozenset({"forward", "backward", "nan", "keep", "raise"})
UTC = timezone.utc
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
LOGGER = logging.getLogger(__name__)
INTERVAL_PATTERN = re.compile(r"[1-9]\d*(?:mo|[smhdw])")


def normalize_pair(value: str) -> str:
    """Convert a pair spelling into uppercase ASCII letters and digits.

    Args:
        value: The pair spelling supplied by the caller or source.

    Returns:
        The normalized pair used to compare different spellings.
    """
    return "".join(
        character
        for character in value.upper()
        if character.isascii() and character.isalnum()
    )


def _text_timestamp(value: str) -> date | datetime:
    """Parse one ISO date or datetime string.

    Args:
        value: The ISO-formatted text to parse.

    Returns:
        A date for date-only text or a datetime for timestamp text.
    """
    try:
        if DATE_PATTERN.fullmatch(value):
            return date.fromisoformat(value)
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("dates must be ISO dates or ISO datetimes") from error


def _datetime_value(value: datetime) -> datetime:
    """Convert one datetime into UTC without changing its instant.

    Args:
        value: The exact datetime boundary to convert.

    Returns:
        A timezone-aware datetime in UTC.
    """
    if getattr(value, "nanosecond", 0):
        raise ValueError("dates must have at most microsecond precision")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _date_value(value: date, *, end: bool) -> datetime:
    """Convert one date into the start of that UTC day or the next day.

    Args:
        value: The calendar date to convert.
        end: Whether the date represents an inclusive ending day.

    Returns:
        The corresponding timezone-aware UTC boundary.
    """
    try:
        day = value + timedelta(days=1 if end else 0)
    except OverflowError as error:
        raise ValueError("date is outside the supported timestamp range") from error
    return datetime.combine(day, time.min, UTC)


def parse_timestamp(value: object, *, end: bool = False) -> datetime:
    """Convert a date-like input into a timezone-aware UTC timestamp.

    Args:
        value: A date, datetime, or ISO-formatted string.
        end: Whether a date-only value represents an inclusive ending day.

    Returns:
        The exact UTC boundary represented by the input.
    """
    if value is None or value is pd.NaT or isinstance(value, bool):
        raise TypeError("dates must be date, datetime, or ISO strings")
    parsed = _text_timestamp(value) if isinstance(value, str) else value
    if isinstance(parsed, datetime):
        return _datetime_value(parsed)
    if isinstance(parsed, date):
        return _date_value(parsed, end=end)
    raise TypeError("dates must be date, datetime, or ISO strings")


def parse_pairs(pairs: object) -> tuple[tuple[str, ...], bool]:
    """Validate one pair or an ordered list of pairs.

    Args:
        pairs: One pair string or a list of pair strings.

    Returns:
        The pairs as a tuple and whether the caller supplied one string.
    """
    single = isinstance(pairs, str)
    if single:
        values = [pairs]
    elif isinstance(pairs, list):
        values = pairs
    else:
        raise TypeError("pairs must be a string or a list of strings")
    if not values:
        raise ValueError("pairs must not be empty")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("each pair must be a string")
    typed_values = cast(list[str], values)
    if any(not normalize_pair(value) for value in typed_values):
        raise ValueError("each pair must contain ASCII letters or digits")
    return tuple(typed_values), single


def parse_range(starting_date: object, end_date: object) -> TimeRange:
    """Validate the requested start and exclusive end timestamps.

    Args:
        starting_date: The requested first date or exact timestamp.
        end_date: The requested inclusive date or exact exclusive timestamp.

    Returns:
        The validated UTC start and exclusive end timestamps.
    """
    start = parse_timestamp(starting_date)
    end = parse_timestamp(end_date, end=True)
    if start >= end:
        raise ValueError(
            "starting_date must be before end_date; equal dates mean one full day"
        )
    return start, end


def parse_identifier(value: object, *, name: str) -> str:
    """Validate a lowercase identifier such as a product or dataset name.

    Args:
        value: The identifier supplied by the caller.
        name: The field name used in validation errors.

    Returns:
        The validated identifier.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{name} must start with a lowercase letter and contain only "
            "lowercase letters, digits, or underscores"
        )
    return value


def _interval_value(value: object, *, name: str) -> str:
    """Validate one interval value and return its spelling.

    Args:
        value: The interval value to validate.
        name: The field name used in validation errors.

    Returns:
        The validated interval spelling.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if INTERVAL_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be a positive integer followed by s, m, h, d, w, or mo"
        )
    return value


def parse_interval(value: object, *, default: object) -> str:
    """Validate an interval spelling and apply its configured default.

    Args:
        value: The optional interval supplied by the caller.
        default: The interval used when the caller supplies ``None``.

    Returns:
        A positive interval with a supported unit spelling.
    """
    default_value = (
        _interval_value(default, name="base_interval") if default is not None else None
    )
    if value is None:
        if default_value is None:
            raise TypeError("base_interval must be a string")
        return default_value
    return _interval_value(value, name="interval")


def _column_text(value: str) -> bool:
    """Return whether a column name or label contains usable text.

    Args:
        value: The column name or output label to inspect.

    Returns:
        True when the text is non-empty and contains no NUL character.
    """
    return bool(value.strip()) and "\x00" not in value


def _validate_column_text(values: list[str], *, name: str) -> None:
    """Reject empty column names or labels and NUL characters.

    Args:
        values: The column names or labels to inspect.
        name: The plural field name used in validation errors.
    """
    if any(not _column_text(value) for value in values):
        raise ValueError(f"{name} must be non-empty and contain no NUL")


def _list_columns(value: list[object]) -> dict[str, str]:
    """Convert an ordered column-name list into an identity mapping.

    Args:
        value: The column-name values supplied by the caller.

    Returns:
        A source-to-output mapping that preserves caller order.
    """
    if not value:
        raise ValueError("column selection must not be empty")
    if any(not isinstance(item, str) for item in value):
        raise TypeError("column names must be strings")
    names = cast(list[str], value)
    _validate_column_text(names, name="column names")
    if len(set(names)) != len(names):
        raise ValueError("duplicate columns are not allowed")
    return {name: name for name in names}


def _mapped_columns(value: dict[object, object]) -> dict[str, str]:
    """Validate and copy a source-column to output-label mapping.

    Args:
        value: The column mapping supplied by the caller.

    Returns:
        A copied mapping that preserves caller order.
    """
    if not value:
        raise ValueError("column selection must not be empty")
    if any(
        not isinstance(name, str) or not isinstance(label, str)
        for name, label in value.items()
    ):
        raise TypeError("column names and labels must be strings")
    columns = cast(dict[str, str], value)
    _validate_column_text(list(columns), name="column names")
    _validate_column_text(list(columns.values()), name="column labels")
    if len(set(columns.values())) != len(columns):
        raise ValueError("output column labels must be unique")
    return columns.copy()


def parse_columns(value: object) -> ColumnSelection:
    """Validate optional source-column selections and output labels.

    Args:
        value: ``None``, a column-name list, or a source-to-label dictionary.

    Returns:
        A copied source-to-label dictionary, or ``None`` for dataset defaults.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return _list_columns(value)
    if isinstance(value, dict):
        return _mapped_columns(value)
    raise TypeError("desired_columns must be a list or a column-to-label dictionary")


def parse_gap_policy(value: object) -> str:
    """Validate a missing-candle policy.

    Args:
        value: The proposed policy name.

    Returns:
        A supported lowercase policy.
    """
    if not isinstance(value, str):
        raise TypeError("gap_policy must be a string")
    if value not in GAP_POLICIES:
        supported = ", ".join(sorted(GAP_POLICIES))
        raise ValueError(f"gap_policy must be one of: {supported}")
    return value


@dataclass(frozen=True)
class Request:
    """Hold one validated source-independent downloader request."""

    pairs: tuple[str, ...]
    single: bool
    start: datetime
    end: datetime
    interval: str | None
    columns: ColumnSelection
    product: str = "spot"
    dataset: str = "klines"
    gap_policy: str | None = "forward"

    @classmethod
    def parse(
        cls,
        pairs: object,
        starting_date: object,
        end_date: object,
        *,
        interval: object = None,
        desired_columns: object = None,
        base_interval: object = None,
        product: object = "spot",
        dataset: object = "klines",
        gap_policy: object = "forward",
    ) -> "Request":
        """Validate caller values and create a request.

        Args:
            pairs: One pair string or an ordered list of pair strings.
            starting_date: The requested first date or exact timestamp.
            end_date: The requested inclusive date or exact exclusive timestamp.
            interval: The optional output interval spelling.
            desired_columns: Optional column names or source-to-label mappings.
            base_interval: The interval used when no output interval is supplied.
            product: The lowercase source product identifier.
            dataset: The lowercase dataset identifier.
            gap_policy: The behavior used for internal missing candles.

        Returns:
            A validated source-independent request.
        """
        parsed_pairs, single = parse_pairs(pairs)
        start, end = parse_range(starting_date, end_date)
        parsed_product = parse_identifier(product, name="product")
        parsed_dataset = parse_identifier(dataset, name="dataset")
        parsed_interval = (
            None
            if interval is None and base_interval is None
            else parse_interval(interval, default=base_interval)
        )
        columns = parse_columns(desired_columns)
        parsed_gap_policy = None if gap_policy is None else parse_gap_policy(gap_policy)
        request = cls(
            pairs=parsed_pairs,
            single=single,
            start=start,
            end=end,
            interval=parsed_interval,
            columns=columns,
            product=parsed_product,
            dataset=parsed_dataset,
            gap_policy=parsed_gap_policy,
        )
        LOGGER.debug(
            "Request parsed: pairs=%s single=%s range=[%s, %s) product=%s "
            "dataset=%s interval=%s columns=%s gap_policy=%s",
            request.pairs,
            request.single,
            request.start,
            request.end,
            request.product,
            request.dataset,
            request.interval,
            request.columns,
            request.gap_policy,
        )
        return request

    def resolve_dataset(self, dataset: "DatasetSpec") -> "Request":
        """Apply one dataset's defaults and capability restrictions.

        Args:
            dataset: The resolved product and dataset declaration.

        Returns:
            A request with effective interval, columns, and gap policy.

        Raises:
            ValueError: If the dataset does not match this request or rejects an option.
        """
        if (dataset.product, dataset.name) != (self.product, self.dataset):
            raise ValueError("dataset declaration does not match the request")
        resolved = replace(
            self,
            interval=dataset.resolve_interval(self.interval),
            columns=dataset.resolve_columns(self.columns),
            gap_policy=dataset.resolve_gap_policy(self.gap_policy),
        )
        LOGGER.debug(
            "Request resolved: product=%s dataset=%s interval=%s columns=%s "
            "gap_policy=%s",
            resolved.product,
            resolved.dataset,
            resolved.interval,
            resolved.columns,
            resolved.gap_policy,
        )
        return resolved
