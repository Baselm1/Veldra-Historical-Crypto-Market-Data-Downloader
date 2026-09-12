"""Discover stable OKX physical archives from expiring manifest URLs."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import math
from urllib.parse import parse_qs, urlsplit

from veldra.core.models import ArchiveKey, ArchiveObject, IntegritySpec
from veldra.core.subjects import DataSubject
from veldra.okx.client import OKXClient
from veldra.okx.datasets import Cadence, manifest_spec


@dataclass(frozen=True)
class ManifestRequest:
    """Describe one maximally packed OKX manifest request."""

    subjects: tuple[DataSubject, ...]
    cadence: Cadence
    begin: date
    end: date


def _month_start(value: date) -> date:
    """Return the first day of a date's calendar month.

    Args:
        value: Calendar date.

    Returns:
        First day in its month.
    """
    return value.replace(day=1)


def _add_months(value: date, months: int) -> date:
    """Move a first-of-month date by whole calendar months.

    Args:
        value: First day of a calendar month.
        months: Nonnegative number of months to advance.

    Returns:
        Shifted first-of-month date.
    """
    index = value.year * 12 + value.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _windows(cadence: Cadence, begin: date, end: date) -> list[tuple[date, date]]:
    """Split one inclusive range into legal ten-unit manifest windows.

    Args:
        cadence: Daily or monthly source aggregation.
        begin: First inclusive source date.
        end: Last inclusive source date.

    Returns:
        Contiguous legal manifest windows.
    """
    if begin > end:
        raise ValueError("manifest begin must not follow end")
    windows: list[tuple[date, date]] = []
    if cadence == "daily":
        current = begin
        while current <= end:
            last = min(end, current + timedelta(days=9))
            windows.append((current, last))
            current = last + timedelta(days=1)
        return windows
    current = _month_start(begin)
    final = _month_start(end)
    while current <= final:
        last = min(final, _add_months(current, 9))
        windows.append((current, last))
        current = _add_months(last, 1)
    return windows


def chunk_manifest_requests(
    subjects: Sequence[DataSubject], cadence: Cadence, begin: date, end: date
) -> list[ManifestRequest]:
    """Batch subjects and dates at OKX's observed request maxima.

    Args:
        subjects: Logical subjects of one kind.
        cadence: Daily or monthly archive grouping.
        begin: First inclusive source date.
        end: Last inclusive source date.

    Returns:
        Five-subject, ten-day/month request chunks.
    """
    if not subjects:
        raise ValueError("manifest discovery requires a subject")
    requests: list[ManifestRequest] = []
    batches = [
        tuple(subjects[index : index + 5]) for index in range(0, len(subjects), 5)
    ]
    for batch in batches:
        for first, last in _windows(cadence, begin, end):
            requests.append(ManifestRequest(batch, cadence, first, last))
    return requests


def _expiry(url: str) -> datetime | None:
    """Parse common S3 signed-URL expiry parameters without retaining secrets.

    Args:
        url: Temporary object-storage URL.

    Returns:
        UTC expiry timestamp when declared.
    """
    query = parse_qs(urlsplit(url).query)
    issued = query.get("X-Amz-Date", [None])[0]
    duration = query.get("X-Amz-Expires", [None])[0]
    if issued and duration:
        try:
            return datetime.strptime(issued, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=UTC
            ) + timedelta(seconds=int(duration))
        except ValueError, OverflowError:
            return None
    expires = query.get("Expires", [None])[0]
    if expires:
        try:
            return datetime.fromtimestamp(int(expires), UTC)
        except ValueError, OverflowError:
            return None
    return None


def _source_date(value: object, offset: int) -> date:
    """Convert one manifest epoch into its documented source calendar.

    Args:
        value: Epoch milliseconds.
        offset: Source calendar seconds east of UTC.

    Returns:
        Source calendar date.
    """
    try:
        timestamp = datetime.fromtimestamp(int(str(value)) / 1000, UTC)
    except (ValueError, OverflowError) as error:
        raise ValueError("OKX manifest data timestamp is invalid") from error
    return (timestamp + timedelta(seconds=offset)).date()


def _period(day: date, cadence: Cadence) -> tuple[date, date]:
    """Return one daily or calendar-month physical period.

    Args:
        day: Source archive date.
        cadence: Daily or monthly aggregation.

    Returns:
        Inclusive physical period bounds.
    """
    if cadence == "daily":
        return day, day
    first = day.replace(day=1)
    return first, _add_months(first, 1) - timedelta(days=1)


def _size(value: object) -> int | None:
    """Convert optional decimal megabytes into bytes.

    Args:
        value: Manifest `sizeMB` field.

    Returns:
        Rounded byte count or ``None``.
    """
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("OKX manifest sizeMB is invalid") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("OKX manifest sizeMB is invalid")
    return round(parsed * 1024 * 1024)


def _objects(
    rows: Iterable[dict[str, object]],
    *,
    product: str,
    dataset: str,
    requested: tuple[DataSubject, ...],
    cadence: Cadence,
) -> list[ArchiveObject]:
    """Parse manifest response groups into stable archive objects.

    Args:
        rows: Top-level manifest response groups.
        product: Requested Veldra product.
        dataset: Requested historical dataset.
        requested: Subjects sent to this manifest request.
        cadence: Requested aggregation cadence.

    Returns:
        Stable physical archive objects.
    """
    spec = manifest_spec(product, dataset)
    found: list[ArchiveObject] = []
    for row in rows:
        details = row.get("details")
        if not isinstance(details, list):
            raise ValueError("OKX manifest details must be a list")
        for group in details:
            if not isinstance(group, dict) or not isinstance(
                group.get("groupDetails"), list
            ):
                raise ValueError("OKX manifest groupDetails must be a list")
            scope = _scope(group, requested, spec.subject_kind)
            for item in group["groupDetails"]:
                found.append(
                    _archive(
                        item, product, dataset, scope, cadence, spec.archive_day_offset
                    )
                )
    unique = {item.key.archive_id: item for item in found}
    if len(unique) != len(found):
        raise ValueError("OKX manifest returned a duplicate archive")
    return sorted(
        unique.values(), key=lambda item: (item.key.period_start, item.key.remote_name)
    )


def _scope(
    group: dict[str, object],
    requested: tuple[DataSubject, ...],
    expected_kind: str,
) -> DataSubject:
    """Resolve the physical scope represented by one manifest group.

    Args:
        group: Manifest group metadata.
        requested: Subjects sent in the request.
        expected_kind: Dataset-specific ordinary subject kind.

    Returns:
        Instrument, family, currency, or all scope.
    """
    if requested == (DataSubject("all", "ANY"),):
        return requested[0]
    field = {
        "instrument": "instId",
        "instrument_family": "instFamily",
        "currency": "ccy",
    }[expected_kind]
    value = group.get(field)
    if not isinstance(value, str) or not value:
        if len(requested) == 1:
            return requested[0]
        raise ValueError("OKX manifest omitted a requested group identity")
    return DataSubject(expected_kind, value)  # type: ignore[arg-type]


def _archive(
    value: object,
    product: str,
    dataset: str,
    scope: DataSubject,
    cadence: Cadence,
    offset: int,
) -> ArchiveObject:
    """Parse one physical file detail.

    Args:
        value: Manifest file object.
        product: Requested Veldra product.
        dataset: Historical dataset.
        scope: Remote physical scope.
        cadence: Daily or monthly aggregation.
        offset: Source calendar seconds east of UTC.

    Returns:
        Stable archive metadata.
    """
    if not isinstance(value, dict):
        raise ValueError("OKX manifest file detail must be an object")
    filename = value.get("filename")
    url = value.get("url")
    timestamp = value.get("dataTs", value.get("dateTs"))
    if not isinstance(filename, str) or not filename:
        raise ValueError("OKX manifest filename is invalid")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ValueError("OKX manifest URL is invalid")
    first, last = _period(_source_date(timestamp, offset), cadence)
    key = ArchiveKey(
        "okx",
        product,
        dataset,
        f"module_{manifest_spec(product, dataset).module}",
        scope.kind,
        scope.value,
        cadence,
        first,
        last,
        filename,
    )
    return ArchiveObject(
        key,
        url,
        url_expires_at=_expiry(url),
        remote_size=_size(value.get("sizeMB")),
        integrity=IntegritySpec("response_header", algorithm="md5"),
    )


class OKXManifestDiscovery:
    """Batch and parse OKX historical archive manifest requests."""

    def __init__(self, client: OKXClient) -> None:
        """Retain one source-wide rate-limited public client.

        Args:
            client: Shared OKX public client.
        """
        self.client = client

    def discover(
        self,
        product: str,
        dataset: str,
        subjects: Sequence[DataSubject],
        cadence: Cadence,
        begin: date,
        end: date,
    ) -> list[ArchiveObject]:
        """Return stable archives across bounded manifest chunks.

        Args:
            product: Veldra OKX product.
            dataset: Historical dataset.
            subjects: Specific or all-market archive scopes.
            cadence: Daily or monthly source grouping.
            begin: First inclusive source date.
            end: Last inclusive source date.

        Returns:
            Deduplicated physical archive objects.
        """
        spec = manifest_spec(product, dataset)
        allowed = (
            {spec.subject_kind, "all"}
            if spec.supports_any_daily
            else {spec.subject_kind}
        )
        if any(subject.kind not in allowed for subject in subjects):
            raise ValueError("manifest subject kind does not match the dataset")
        if any(subject.kind == "all" for subject in subjects) and cadence != "daily":
            raise ValueError("ANY manifests require daily cadence")
        native = _native_type(product)
        found: list[ArchiveObject] = []
        for chunk in chunk_manifest_requests(subjects, cadence, begin, end):
            rows = self.client.get_manifest(
                spec.module,
                native,
                [subject.value for subject in chunk.subjects],
                cadence,
                chunk.begin,
                chunk.end,
            )
            found.extend(
                _objects(
                    rows,
                    product=product,
                    dataset=dataset,
                    requested=chunk.subjects,
                    cadence=cadence,
                )
            )
        unique = {item.key.archive_id: item for item in found}
        return sorted(
            unique.values(),
            key=lambda item: (item.key.period_start, item.key.remote_name),
        )


def _native_type(product: str) -> str:
    """Return the manifest instrument type for a Veldra product.

    Args:
        product: Veldra OKX product.

    Returns:
        Native manifest instrument type.
    """
    if product in {"spot", "margin"}:
        return "SPOT"
    if product in {"linear_swap", "inverse_swap"}:
        return "SWAP"
    if product in {"linear_futures", "inverse_futures"}:
        return "FUTURES"
    if product == "options":
        return "OPTION"
    raise ValueError(f"unsupported OKX product {product!r}")
