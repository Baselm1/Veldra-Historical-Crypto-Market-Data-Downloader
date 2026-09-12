"""Test deterministic OKX specific and bulk archive planning."""

from datetime import UTC, date, datetime

from veldra.core.models import ArchiveKey, ArchiveObject, IntegritySpec
from veldra.core.subjects import DataSubject
from veldra.okx.planner import OKXArchivePlan, OKXArchivePlanner


def archive(
    scope: DataSubject,
    start: date,
    end: date | None = None,
    *,
    cadence: str = "daily",
    status: str = "discovered",
) -> ArchiveObject:
    """Build one planner archive fixture.

    Args:
        scope: Physical source scope.
        start: First source day.
        end: Optional final source day.
        cadence: Daily or monthly grouping.
        status: Catalog state.

    Returns:
        Physical archive fixture.
    """
    last = end or start
    key = ArchiveKey(
        "okx",
        "spot",
        "klines",
        "module_2",
        scope.kind,
        scope.value,
        cadence,
        start,
        last,
        f"{scope.value}-{start}-{cadence}.zip",
    )
    return ArchiveObject(
        key,
        "https://example/archive.zip",
        integrity=IntegritySpec("response_header", algorithm="md5"),
        status=status,  # type: ignore[arg-type]
    )


class Discovery:
    """Record planner discovery calls and return matching fixtures."""

    def __init__(self) -> None:
        """Create an empty discovery call record."""
        self.calls: list[tuple[tuple[DataSubject, ...], str, date, date]] = []

    def discover(
        self,
        product: str,
        dataset: str,
        subjects: list[DataSubject],
        cadence: str,
        begin: date,
        end: date,
    ) -> list[ArchiveObject]:
        """Return one file per requested subject and physical period.

        Args:
            product: Requested product.
            dataset: Requested dataset.
            subjects: Requested physical scopes.
            cadence: Daily or monthly grouping.
            begin: First source date.
            end: Last source date.

        Returns:
            Synthetic archives spanning requested periods.
        """
        assert product == "spot" and dataset == "klines"
        self.calls.append((tuple(subjects), cadence, begin, end))
        values: list[ArchiveObject] = []
        cursor = begin
        while cursor <= end:
            if cadence == "monthly":
                next_month = (
                    date(cursor.year + 1, 1, 1)
                    if cursor.month == 12
                    else date(cursor.year, cursor.month + 1, 1)
                )
                last = next_month.fromordinal(next_month.toordinal() - 1)
                for subject in subjects:
                    values.append(archive(subject, cursor, last, cadence=cadence))
                cursor = next_month
            else:
                for subject in subjects:
                    values.append(archive(subject, cursor))
                cursor = cursor.fromordinal(cursor.toordinal() + 1)
        return values


def planner(
    cached: list[ArchiveObject] | None = None,
) -> tuple[OKXArchivePlanner, Discovery]:
    """Build a planner over one recording discovery fixture.

    Args:
        cached: Existing physical archive metadata.

    Returns:
        Planner and its discovery fixture.
    """
    discovery = Discovery()
    return OKXArchivePlanner(discovery, cached=cached or []), discovery  # type: ignore[arg-type]


def test_auto_prefers_monthly_specific_then_daily_tail() -> None:
    """Confirm complete months use monthly objects and partial months use days."""
    value, discovery = planner()
    result = value.plan(
        "spot",
        "klines",
        [DataSubject("instrument", "BTC-USDT")],
        date(2025, 1, 1),
        date(2025, 2, 3),
    )
    assert isinstance(result, OKXArchivePlan)
    assert len(result.selected) == 4
    assert [call[1] for call in discovery.calls] == ["monthly", "daily"]
    assert "monthly" in result.explanation


def test_bulk_transport_uses_any_without_enumerating_specific_subjects() -> None:
    """Confirm explicit all-market work sends only one ANY scope."""
    value, discovery = planner()
    result = value.plan(
        "spot",
        "klines",
        [DataSubject("instrument", f"S{i}") for i in range(20)],
        date(2025, 1, 1),
        date(2025, 1, 3),
        transport="bulk",
    )
    assert len(result.selected) == 3
    assert all(call[0] == (DataSubject("all", "ANY"),) for call in discovery.calls)
    assert all(call[1] == "daily" for call in discovery.calls)


def test_cached_bulk_archive_wins_for_a_specific_request() -> None:
    """Confirm a ready shared day prevents a redundant specific download."""
    cached = archive(DataSubject("all", "ANY"), date(2025, 1, 1), status="ready")
    value, discovery = planner([cached])
    result = value.plan(
        "spot",
        "klines",
        [DataSubject("instrument", "BTC-USDT")],
        date(2025, 1, 1),
        date(2025, 1, 1),
    )
    assert result.cached == (cached,)
    assert result.selected == ()
    assert discovery.calls == []


def test_partial_cached_month_discovers_only_uncovered_daily_segments() -> None:
    """Confirm cached days are subtracted before remote discovery."""
    cached = archive(
        DataSubject("instrument", "BTC-USDT"), date(2025, 1, 2), status="ready"
    )
    value, discovery = planner([cached])
    result = value.plan(
        "spot",
        "klines",
        [DataSubject("instrument", "BTC-USDT")],
        date(2025, 1, 1),
        date(2025, 1, 3),
        transport="specific",
    )
    assert result.cached == (cached,)
    assert [(call[2], call[3]) for call in discovery.calls] == [
        (date(2025, 1, 1), date(2025, 1, 1)),
        (date(2025, 1, 3), date(2025, 1, 3)),
    ]


def test_planner_rejects_invalid_transport_and_ranges() -> None:
    """Confirm invalid planner requests fail without discovery."""
    value, discovery = planner()
    subject = [DataSubject("instrument", "BTC-USDT")]
    for transport in ("fastest", ""):
        try:
            value.plan(
                "spot",
                "klines",
                subject,
                date(2025, 1, 1),
                date(2025, 1, 2),
                transport=transport,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid transport was accepted")
    for subjects, begin, end in (
        ([], date(2025, 1, 1), date(2025, 1, 2)),
        (subject, date(2025, 1, 2), date(2025, 1, 1)),
    ):
        try:
            value.plan("spot", "klines", subjects, begin, end)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid archive request was accepted")
    try:
        value.plan(
            "spot",
            "order_book_400",
            subject,
            date(2025, 1, 1),
            date(2025, 1, 2),
            transport="bulk",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported bulk transport was accepted")
    assert discovery.calls == []


def test_consecutive_complete_months_share_one_manifest_call() -> None:
    """Confirm adjacent complete months use one bounded manifest request."""
    value, discovery = planner()
    result = value.plan(
        "spot",
        "klines",
        [DataSubject("instrument", "BTC-USDT")],
        date(2025, 1, 1),
        date(2025, 2, 28),
    )
    assert len(result.selected) == 2
    assert [(call[1], call[2], call[3]) for call in discovery.calls] == [
        ("monthly", date(2025, 1, 1), date(2025, 2, 1))
    ]


def test_archive_plan_reports_physical_bytes_and_counts() -> None:
    """Confirm plan summaries distinguish cached and remote physical files."""
    item = archive(DataSubject("instrument", "BTC-USDT"), date(2025, 1, 1))
    sized = ArchiveObject(
        item.key,
        item.url,
        remote_size=100,
        integrity=item.integrity,
    )
    plan = OKXArchivePlan((), (sized,), "specific daily")
    assert plan.remote_files == 1
    assert plan.remote_bytes == 100
