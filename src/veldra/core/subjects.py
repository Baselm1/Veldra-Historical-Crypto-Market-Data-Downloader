"""Model the native subjects addressed by historical data providers."""

from dataclasses import dataclass
from typing import Literal, cast

type SubjectKind = Literal["instrument", "instrument_family", "currency", "all"]
SUBJECT_KINDS = frozenset({"instrument", "instrument_family", "currency", "all"})


def normalize_subject(value: str) -> str:
    """Convert a native identifier into uppercase ASCII letters and digits.

    Args:
        value: The native identifier supplied by a caller or source.

    Returns:
        The normalized spelling used only for comparisons.
    """
    return "".join(
        character
        for character in value.upper()
        if character.isascii() and character.isalnum()
    )


def parse_subject_kind(value: object) -> SubjectKind:
    """Validate one historical data subject kind.

    Args:
        value: The proposed subject kind.

    Returns:
        A supported subject kind.
    """
    if not isinstance(value, str):
        raise TypeError("subject_kind must be a string")
    if value not in SUBJECT_KINDS:
        supported = ", ".join(sorted(SUBJECT_KINDS))
        raise ValueError(f"subject_kind must be one of: {supported}")
    return cast(SubjectKind, value)


@dataclass(frozen=True)
class DataSubject:
    """Identify one native instrument, family, currency, or bulk scope."""

    kind: SubjectKind
    value: str

    def __post_init__(self) -> None:
        """Reject unsupported kinds and malformed native identifiers."""
        parsed_kind = parse_subject_kind(self.kind)
        if not isinstance(self.value, str):
            raise TypeError("subject value must be a string")
        normalized = normalize_subject(self.value)
        if parsed_kind in {"instrument", "instrument_family"} and not normalized:
            raise ValueError("subject value must contain ASCII letters or digits")
        if parsed_kind == "currency" and (
            not normalized or normalized != self.value.upper()
        ):
            raise ValueError(
                "currency subjects must contain only ASCII letters or digits"
            )
        if parsed_kind == "all" and normalized != "ANY":
            raise ValueError("all subjects must use the native value 'ANY'")

    @property
    def normalized_value(self) -> str:
        """Return the source-neutral comparison spelling for this subject."""
        return normalize_subject(self.value)


def parse_subjects(
    values: object, kind: object = "instrument"
) -> tuple[tuple[DataSubject, ...], bool]:
    """Validate one subject or an ordered list of subjects.

    Args:
        values: One native value or a list of native values.
        kind: The subject kind shared by all supplied values.

    Returns:
        The validated subjects and whether the caller supplied one string.
    """
    parsed_kind = parse_subject_kind(kind)
    single = isinstance(values, str)
    if single:
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raise TypeError("subjects must be a string or a list of strings")
    if not raw_values:
        raise ValueError("subjects must not be empty")
    if any(not isinstance(value, str) for value in raw_values):
        raise TypeError("each subject must be a string")
    typed_values = cast(list[str], raw_values)
    return tuple(DataSubject(parsed_kind, value) for value in typed_values), single


def resolve_subject(
    value: object,
    kind: object,
    candidates: tuple[DataSubject, ...] | list[DataSubject],
) -> DataSubject:
    """Resolve one native subject without guessing or fuzzy substitution.

    Args:
        value: The requested native identifier.
        kind: The requested subject kind.
        candidates: The subjects available from the provider.

    Returns:
        The exact or uniquely normalized matching subject.
    """
    requested = DataSubject(parse_subject_kind(kind), value)  # type: ignore[arg-type]
    matching_kind = [item for item in candidates if item.kind == requested.kind]
    native = [item for item in matching_kind if item.value == requested.value]
    if native:
        return native[0]
    normalized = [
        item
        for item in matching_kind
        if item.normalized_value == requested.normalized_value
    ]
    if len(normalized) == 1:
        return normalized[0]
    if len(normalized) > 1:
        raise ValueError(
            f"subject {requested.value!r} is ambiguous for kind {requested.kind!r}"
        )
    raise ValueError(
        f"subject {requested.value!r} was not found for kind {requested.kind!r}"
    )
