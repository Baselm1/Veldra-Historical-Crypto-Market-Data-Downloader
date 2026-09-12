"""Build deterministic physical archive plans for logical OKX requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol, Sequence

from veldra.core.models import ArchiveObject
from veldra.core.subjects import DataSubject
from veldra.okx.datasets import Cadence, manifest_spec


class ManifestDiscovery(Protocol):
    """Describe the manifest operation required by the archive planner."""

    def discover(
        self,
        product: str,
        dataset: str,
        subjects: Sequence[DataSubject],
        cadence: Cadence,
        begin: date,
        end: date,
    ) -> list[ArchiveObject]:
        """Discover physical archives for one bounded request."""
        ...


@dataclass(frozen=True)
class OKXArchivePlan:
    """Describe cached and remote archives selected for one logical request."""

    cached: tuple[ArchiveObject, ...]
    selected: tuple[ArchiveObject, ...]
    explanation: str

    @property
    def remote_files(self) -> int:
        """Return the number of physical files that require retrieval."""
        return len(self.selected)

    @property
    def remote_bytes(self) -> int:
        """Return the sum of advertised bytes for selected physical files."""
        return sum(item.remote_size or 0 for item in self.selected)


class OKXArchivePlanner:
    """Choose non-overlapping specific, family, or all-market OKX archives."""

    def __init__(
        self,
        discovery: ManifestDiscovery,
        *,
        cached: Sequence[ArchiveObject] = (),
        bulk_threshold: int = 8,
    ) -> None:
        """Create a planner over manifest discovery and known archives.

        Args:
            discovery: Service that resolves physical manifest objects.
            cached: Existing physical archive metadata.
            bulk_threshold: Subject count where automatic planning prefers ANY.
        """
        if bulk_threshold < 1:
            raise ValueError("bulk_threshold must be positive")
        self._discovery = discovery
        self._cached = tuple(cached)
        self._bulk_threshold = bulk_threshold

    def plan(
        self,
        product: str,
        dataset: str,
        subjects: Sequence[DataSubject],
        begin: date,
        end: date,
        *,
        transport: str = "auto",
    ) -> OKXArchivePlan:
        """Plan physical archives that cover one inclusive source-date range.

        Args:
            product: Veldra OKX product.
            dataset: Historical dataset.
            subjects: Instrument, family, currency, or all-market scopes.
            begin: First inclusive source date.
            end: Final inclusive source date.
            transport: Automatic, specific, or bulk selection.

        Returns:
            Cached and remote physical objects with a short explanation.
        """
        if transport not in {"auto", "specific", "bulk"}:
            raise ValueError("transport must be 'auto', 'specific', or 'bulk'")
        if begin > end:
            raise ValueError("archive plan begins after it ends")
        if not subjects:
            raise ValueError("archive plan requires at least one subject")
        spec = manifest_spec(product, dataset)
        requested = tuple(dict.fromkeys(subjects))
        use_bulk = transport == "bulk" or (
            transport == "auto"
            and (
                any(subject.kind == "all" for subject in requested)
                or len(requested) >= self._bulk_threshold
            )
        )
        if use_bulk and not spec.supports_any_daily:
            raise ValueError(
                f"OKX does not provide bulk archives for {product}/{dataset}"
            )
        physical = (DataSubject("all", "ANY"),) if use_bulk else requested
        cached = self._matching_cached(product, dataset, requested, begin, end)
        selected = self._discover_missing(
            product,
            dataset,
            physical,
            begin,
            end,
            cached,
            use_bulk=use_bulk,
            supports_monthly=spec.supports_monthly,
            monthly_only_specific=dataset == "funding_rates" and not use_bulk,
        )
        mode = "bulk daily" if use_bulk else "specific"
        cadences = {item.key.cadence for item in selected}
        if not use_bulk and "monthly" in cadences:
            mode += " monthly with daily tails"
        return OKXArchivePlan(
            cached,
            tuple(sorted(selected, key=self._sort_key)),
            f"{mode}; {len(cached)} cached and {len(selected)} remote physical files",
        )

    def _matching_cached(
        self,
        product: str,
        dataset: str,
        subjects: Sequence[DataSubject],
        begin: date,
        end: date,
    ) -> tuple[ArchiveObject, ...]:
        """Return ready cached archives that overlap a logical request."""
        values: list[ArchiveObject] = []
        for item in self._cached:
            key = item.key
            if item.status != "ready" or key.source != "okx":
                continue
            if key.product != product or key.dataset != dataset:
                continue
            if key.period_end < begin or key.period_start > end:
                continue
            if key.remote_scope_kind == "all" or key.subject in subjects:
                values.append(item)
        return tuple(sorted(values, key=self._sort_key))

    def _discover_missing(
        self,
        product: str,
        dataset: str,
        subjects: Sequence[DataSubject],
        begin: date,
        end: date,
        cached: Sequence[ArchiveObject],
        *,
        use_bulk: bool,
        supports_monthly: bool,
        monthly_only_specific: bool,
    ) -> list[ArchiveObject]:
        """Discover only source dates not covered by ready physical objects."""
        selected: list[ArchiveObject] = []
        for subject in subjects:
            covered = self._covered_days(cached, subject, begin, end)
            missing = [day for day in self._days(begin, end) if day not in covered]
            if not missing:
                continue
            if monthly_only_specific:
                monthly = sorted({day.replace(day=1) for day in missing})
                daily: list[date] = []
            else:
                monthly, daily = self._split_monthly(
                    missing, supports_monthly and not use_bulk
                )
            for first, last in self._month_segments(monthly):
                selected.extend(
                    self._discovery.discover(
                        product, dataset, [subject], "monthly", first, last
                    )
                )
            for first, last in self._segments(daily):
                selected.extend(
                    self._discovery.discover(
                        product, dataset, [subject], "daily", first, last
                    )
                )
        return list({item.key.archive_id: item for item in selected}.values())

    @staticmethod
    def _covered_days(
        cached: Sequence[ArchiveObject],
        subject: DataSubject,
        begin: date,
        end: date,
    ) -> set[date]:
        """Return requested source days covered by matching cached archives."""
        covered: set[date] = set()
        for item in cached:
            if item.key.remote_scope_kind != "all" and item.key.subject != subject:
                continue
            first = max(begin, item.key.period_start)
            last = min(end, item.key.period_end)
            covered.update(OKXArchivePlanner._days(first, last))
        return covered

    @staticmethod
    def _split_monthly(
        missing: Sequence[date], supports_monthly: bool
    ) -> tuple[list[date], list[date]]:
        """Separate complete uncovered months from daily tails and holes."""
        if not supports_monthly:
            return [], list(missing)
        missing_set = set(missing)
        monthly: list[date] = []
        daily = list(missing)
        by_month = {(day.year, day.month) for day in missing}
        for year, month in sorted(by_month):
            first = date(year, month, 1)
            next_month = (
                date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
            )
            days = list(OKXArchivePlanner._days(first, next_month - timedelta(days=1)))
            if all(day in missing_set for day in days):
                monthly.append(first)
                daily = [day for day in daily if day not in days]
        return monthly, daily

    @staticmethod
    def _segments(days: Sequence[date]) -> list[tuple[date, date]]:
        """Merge consecutive dates while keeping monthly starts separate."""
        if not days:
            return []
        values = sorted(set(days))
        segments: list[tuple[date, date]] = []
        first = previous = values[0]
        for current in values[1:]:
            if current == previous + timedelta(days=1):
                previous = current
                continue
            segments.append((first, previous))
            first = previous = current
        segments.append((first, previous))
        return segments

    @staticmethod
    def _month_segments(months: Sequence[date]) -> list[tuple[date, date]]:
        """Merge consecutive first-of-month values into manifest ranges."""
        if not months:
            return []
        values = sorted(set(months))
        segments: list[tuple[date, date]] = []
        first = previous = values[0]
        for current in values[1:]:
            next_month = (
                date(previous.year + 1, 1, 1)
                if previous.month == 12
                else date(previous.year, previous.month + 1, 1)
            )
            if current == next_month:
                previous = current
                continue
            segments.append((first, previous))
            first = previous = current
        segments.append((first, previous))
        return segments

    @staticmethod
    def _days(begin: date, end: date) -> list[date]:
        """Return every date in an inclusive range."""
        return [
            begin + timedelta(days=offset) for offset in range((end - begin).days + 1)
        ]

    @staticmethod
    def _sort_key(item: ArchiveObject) -> tuple[date, date, str, str]:
        """Return a deterministic physical archive ordering key."""
        key = item.key
        return key.period_start, key.period_end, key.remote_scope_value, key.remote_name
