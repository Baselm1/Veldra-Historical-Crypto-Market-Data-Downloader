"""Test OKX archive manifest discovery and stable identities."""

from datetime import UTC, date, datetime

import httpx
import pytest

from veldra.core.models import ArchiveObject
from veldra.core.subjects import DataSubject
from veldra.okx.client import OKXClient
from veldra.okx.datasets import manifest_spec
from veldra.okx.manifest import OKXManifestDiscovery, chunk_manifest_requests


def manifest_row(
    *,
    filename: str = "BTC-USDT-candlesticks-2025-01-01.zip",
    subject: str = "BTC-USDT",
    timestamp: str = "1735689600000",
) -> dict[str, object]:
    """Build one compact successful manifest response group.

    Args:
        filename: Remote archive name.
        subject: Returned instrument ID.
        timestamp: Source date epoch milliseconds.

    Returns:
        One top-level manifest response row.
    """
    return {
        "dateAggrType": "daily",
        "details": [
            {
                "instType": "SPOT",
                "instId": subject,
                "instFamily": "",
                "dateRangeStart": timestamp,
                "dateRangeEnd": timestamp,
                "groupSizeMB": "0.25",
                "groupDetails": [
                    {
                        "dataTs": timestamp,
                        "filename": filename,
                        "sizeMB": "0.25",
                        "url": (
                            "https://static.okx.com/file.zip?"
                            "X-Amz-Date=20250101T000000Z&X-Amz-Expires=600&token=secret"
                        ),
                    }
                ],
            }
        ],
    }


def test_manifest_dataset_declarations_are_product_scoped() -> None:
    """Confirm native modules, calendars, and unsupported combinations are explicit."""
    assert manifest_spec("spot", "klines").module == 2
    assert manifest_spec("spot", "trades").module == 1
    assert manifest_spec("linear_swap", "funding_rates").module == 3
    assert manifest_spec("spot", "order_book_400").archive_day_offset == 0
    assert manifest_spec("spot", "klines").archive_day_offset == 8 * 3600
    with pytest.raises(ValueError, match="unsupported"):
        manifest_spec("spot", "funding_rates")


def test_chunk_manifest_requests_batches_five_subjects_and_ten_days() -> None:
    """Confirm discovery maximizes each scarce manifest request."""
    subjects = [DataSubject("instrument", f"S{i}") for i in range(7)]
    chunks = chunk_manifest_requests(
        subjects, "daily", date(2025, 1, 1), date(2025, 1, 12)
    )
    assert len(chunks) == 4
    assert [len(chunk.subjects) for chunk in chunks] == [5, 5, 2, 2]
    assert chunks[0].begin == date(2025, 1, 1)
    assert chunks[0].end == date(2025, 1, 10)
    assert chunks[1].begin == date(2025, 1, 11)


def test_monthly_chunks_use_calendar_months() -> None:
    """Confirm monthly windows never exceed ten inclusive calendar months."""
    chunks = chunk_manifest_requests(
        [DataSubject("instrument", "BTC-USDT")],
        "monthly",
        date(2024, 1, 15),
        date(2025, 2, 20),
    )
    assert [(chunk.begin, chunk.end) for chunk in chunks] == [
        (date(2024, 1, 1), date(2024, 10, 1)),
        (date(2024, 11, 1), date(2025, 2, 1)),
    ]


def test_discovery_parses_stable_specific_archive_metadata() -> None:
    """Confirm a signed URL does not enter the physical archive identity."""
    responses = [manifest_row()]

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return one successful manifest group."""
        return httpx.Response(200, json={"code": "0", "msg": "", "data": responses})

    raw = httpx.Client(transport=httpx.MockTransport(handler))
    with raw:
        discovery = OKXManifestDiscovery(OKXClient(client=raw, retries=0))
        archives = discovery.discover(
            "spot",
            "klines",
            [DataSubject("instrument", "BTC-USDT")],
            "daily",
            date(2025, 1, 1),
            date(2025, 1, 1),
        )

    assert len(archives) == 1
    archive = archives[0]
    assert archive.key.remote_scope_kind == "instrument"
    assert archive.key.remote_scope_value == "BTC-USDT"
    assert archive.key.period_start == date(2025, 1, 1)
    assert archive.key.period_end == date(2025, 1, 1)
    assert archive.remote_size == 262_144
    assert archive.integrity is not None
    assert archive.integrity.mode == "response_header"
    assert archive.url_expires_at == datetime(2025, 1, 1, 0, 10, tzinfo=UTC)
    replaced = ArchiveObject(archive.key, "https://new.example/signed")
    assert replaced.key.archive_id == archive.key.archive_id


def test_discovery_parses_any_and_family_scopes() -> None:
    """Confirm bulk and derivatives manifests retain physical scope kinds."""
    any_row = manifest_row(subject="ANY", filename="ANY-candlesticks-2025-01-01.zip")
    family_row = manifest_row(subject="", filename="BTC-USDT-swap-2025-01-01.zip")
    details = family_row["details"]
    assert isinstance(details, list) and isinstance(details[0], dict)
    details[0]["instType"] = "SWAP"
    details[0]["instFamily"] = "BTC-USDT"

    responses = iter([[any_row], [family_row]])

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return successive Spot and swap fixtures."""
        return httpx.Response(
            200, json={"code": "0", "msg": "", "data": next(responses)}
        )

    raw = httpx.Client(transport=httpx.MockTransport(handler))
    with raw:
        discovery = OKXManifestDiscovery(OKXClient(client=raw, retries=0))
        bulk = discovery.discover(
            "spot",
            "klines",
            [DataSubject("all", "ANY")],
            "daily",
            date(2025, 1, 1),
            date(2025, 1, 1),
        )
        family = discovery.discover(
            "linear_swap",
            "trades",
            [DataSubject("instrument_family", "BTC-USDT")],
            "daily",
            date(2025, 1, 1),
            date(2025, 1, 1),
        )
    assert bulk[0].key.remote_scope_kind == "all"
    assert family[0].key.remote_scope_kind == "instrument_family"


def test_empty_manifest_is_success_and_invalid_rows_fail() -> None:
    """Confirm sparse gaps stay empty while malformed source rows fail visibly."""
    responses = iter(
        [
            [{"dateAggrType": "daily", "details": []}],
            [{"dateAggrType": "daily", "details": [{"groupDetails": "bad"}]}],
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        """Return one empty then one malformed manifest."""
        return httpx.Response(
            200, json={"code": "0", "msg": "", "data": next(responses)}
        )

    raw = httpx.Client(transport=httpx.MockTransport(handler))
    with raw:
        discovery = OKXManifestDiscovery(OKXClient(client=raw, retries=0))
        subjects = [DataSubject("instrument", "BTC-USDT")]
        assert (
            discovery.discover(
                "spot",
                "klines",
                subjects,
                "daily",
                date(2025, 1, 1),
                date(2025, 1, 1),
            )
            == []
        )
        with pytest.raises(ValueError, match="groupDetails"):
            discovery.discover(
                "spot",
                "klines",
                subjects,
                "daily",
                date(2025, 1, 1),
                date(2025, 1, 1),
            )


def test_discovery_rejects_subject_kind_mismatches_before_io() -> None:
    """Confirm Spot, derivatives, currency, and bulk scopes cannot be confused."""
    raw = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("unexpected HTTP request")
        )
    )
    with raw:
        discovery = OKXManifestDiscovery(OKXClient(client=raw, retries=0))
        with pytest.raises(ValueError, match="subject"):
            discovery.discover(
                "spot",
                "klines",
                [DataSubject("instrument_family", "BTC-USDT")],
                "daily",
                date(2025, 1, 1),
                date(2025, 1, 1),
            )
