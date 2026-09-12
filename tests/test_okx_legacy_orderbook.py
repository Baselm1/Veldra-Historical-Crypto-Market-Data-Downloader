"""Test explicit OKX module 6 legacy order-book snapshots."""

from base64 import b64encode
import csv
from datetime import date
import gzip
from hashlib import md5
from io import BytesIO, StringIO
from pathlib import Path

import httpx
import pandas as pd
import pytest

from veldra import OKX
from veldra.core.models import (
    ArchiveKey,
    ArchiveObject,
    DataValidationError,
    IntegritySpec,
)
from veldra.okx.datasets import get_dataset
from veldra.okx.legacy_orderbook import (
    _count,
    _header,
    _line,
    _number,
    _record,
    _timestamp,
    materialize_legacy_order_book,
)


def columns(levels: int, *, grouped: bool = False) -> list[str]:
    """Return one observed dynamic module 6 header.

    Args:
        levels: Number of bid and ask levels.
        grouped: Whether all bids precede all asks as in Option files.

    Returns:
        Complete source header.
    """
    fields = ["timeMs", "exchTimeMs"]
    sides = ("bid", "ask")
    if grouped:
        order = [(side, number) for side in sides for number in range(1, levels + 1)]
    else:
        order = [(side, number) for number in range(1, levels + 1) for side in sides]
    for side, number in order:
        fields.extend(
            [
                f"{side}_{number}_px",
                f"{side}_{number}_qty",
                f"{side}_{number}_ordCnt",
            ]
        )
    return [*fields, "symbol"]


def row(header: list[str], *, symbol: str = "BTC-USDT.OK") -> list[str]:
    """Return one source snapshot matching a dynamic header.

    Args:
        header: Source columns to populate.
        symbol: Native `.OK` source symbol.

    Returns:
        One complete CSV record.
    """
    values: dict[str, str] = {
        "timeMs": "1735689600003",
        "exchTimeMs": "1735689600001",
        "symbol": symbol,
    }
    for name in header:
        if name.endswith("_px"):
            values[name] = "99" if name.startswith("bid") else "101"
        elif name.endswith("_qty"):
            values[name] = "2"
        elif name.endswith("_ordCnt"):
            values[name] = "3"
    return [values[name] for name in header]


def gzip_csv(header: list[str], rows: list[list[str]]) -> bytes:
    """Return deterministic GZIP/CSV source bytes.

    Args:
        header: Source column names.
        rows: Source snapshot records.

    Returns:
        Compressed archive bytes.
    """
    text = StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    output = BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as archive:
        archive.write(text.getvalue().encode())
    return output.getvalue()


def physical(
    *,
    product: str = "spot",
    scope_kind: str = "instrument",
    scope: str = "BTC-USDT",
    name: str = "BTC-USDT.OK.csv.gz",
) -> ArchiveObject:
    """Return one physical legacy archive object.

    Args:
        product: Veldra OKX product.
        scope_kind: Physical manifest subject kind.
        scope: Physical instrument or family.
        name: Remote GZIP filename.

    Returns:
        Integrity-bearing archive metadata.
    """
    return ArchiveObject(
        ArchiveKey(
            "okx",
            product,
            "legacy_order_book_50",
            "module_6",
            scope_kind,  # type: ignore[arg-type]
            scope,
            "daily",
            date(2025, 1, 1),
            date(2025, 1, 1),
            name,
        ),
        f"https://files.test/{name}",
        integrity=IntegritySpec("response_header", algorithm="md5"),
    )


def test_dynamic_headers_support_50_levels_and_sparse_option_depths() -> None:
    """Confirm observed interleaved and grouped schemas normalize identically."""
    spot_header = columns(50)
    spot = _record(
        row(spot_header),
        _header(spot_header),
        physical(),
        get_dataset("spot", "legacy_order_book_50"),
        0,
    )
    assert len(spot["bids"]) == 50
    assert spot["bids"][0]["base_quantity"] == 2.0  # type: ignore[index]

    option_header = columns(4, grouped=True)
    option = _record(
        row(option_header, symbol="BTC-USD-250102-90000-P.OK"),
        _header(option_header),
        physical(
            product="options",
            scope_kind="instrument_family",
            scope="BTC-USD",
            name="BTC-USD-250102-90000-P.OK.csv.gz",
        ),
        get_dataset("options", "legacy_order_book_50"),
        1,
    )
    assert len(option["asks"]) == 4
    assert option["asks"][0]["contract_quantity"] == 2.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (["timeMs", "exchTimeMs", "symbol", "symbol"], "duplicates"),
        (["timeMs", "symbol"], "required"),
        (["timeMs", "exchTimeMs", "other", "symbol"], "unknown"),
        (
            [
                "timeMs",
                "exchTimeMs",
                "bid_2_px",
                "bid_2_qty",
                "bid_2_ordCnt",
                "ask_2_px",
                "ask_2_qty",
                "ask_2_ordCnt",
                "symbol",
            ],
            "contiguous",
        ),
    ],
)
def test_dynamic_header_rejects_unknown_schema_versions(
    header: list[str], message: str
) -> None:
    """Confirm an unprobed module 6 schema cannot be silently misread.

    Args:
        header: Invalid source header.
        message: Expected validation detail.
    """
    with pytest.raises(DataValidationError, match=message):
        _header(header)


def test_partial_levels_and_wrong_symbols_are_rejected() -> None:
    """Confirm corrupt rows and cross-instrument rows cannot be published."""
    header = columns(2)
    partial = row(header)
    partial[header.index("bid_2_qty")] = ""
    with pytest.raises(DataValidationError, match="incomplete"):
        _record(
            partial,
            _header(header),
            physical(),
            get_dataset("spot", "legacy_order_book_50"),
            0,
        )
    with pytest.raises(DataValidationError, match="does not match"):
        _record(
            row(header, symbol="ETH-USDT.OK"),
            _header(header),
            physical(),
            get_dataset("spot", "legacy_order_book_50"),
            0,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timeMs", "bad", "milliseconds"),
        ("timeMs", "1000", "unit"),
        ("timeMs", "1735776000000", "source day"),
        ("bid_1_px", "zero", "numeric"),
        ("bid_1_px", "0", "invalid"),
        ("bid_1_qty", "-1", "invalid"),
        ("bid_1_ordCnt", "1.5", "integer"),
        ("symbol", "BTC-USDT", "symbol"),
    ],
)
def test_snapshot_values_reject_bad_units_numbers_and_symbols(
    field: str, value: str, message: str
) -> None:
    """Confirm malformed individual snapshot fields fail visibly.

    Args:
        field: Source field to corrupt.
        value: Invalid source value.
        message: Expected validation detail.
    """
    header = columns(1)
    values = row(header)
    values[header.index(field)] = value
    with pytest.raises(DataValidationError, match=message):
        _record(
            values,
            _header(header),
            physical(),
            get_dataset("spot", "legacy_order_book_50"),
            0,
        )


def test_scalar_parsers_and_bounded_csv_handle_edge_values() -> None:
    """Confirm parser primitives accept valid zeros and reject corrupt text."""
    assert _line(BytesIO(b""), 4) is None
    assert _count("0") == 0
    assert _number("0", "quantity", positive=False) == 0
    assert _timestamp("1735689600000", "timeMs").year == 2025
    with pytest.raises(DataValidationError, match="CSV"):
        _line(BytesIO(b"\xff\n"), 4)
    with pytest.raises(DataValidationError, match="invalid"):
        _number("nan", "quantity", positive=False)


def test_dynamic_header_rejects_too_many_or_incomplete_level_fields() -> None:
    """Confirm level count and field triplets cannot drift silently."""
    with pytest.raises(DataValidationError, match="schema"):
        _header(columns(51))
    incomplete = columns(1)
    incomplete.remove("ask_1_ordCnt")
    with pytest.raises(DataValidationError, match="schema"):
        _header(incomplete)


class LegacyFixture:
    """Serve one current Spot market, manifest, and tiny GZIP archive."""

    def __init__(self) -> None:
        """Create an archive request counter and source body."""
        header = columns(2)
        self.content = gzip_csv(header, [row(header)])
        self.files = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return the matching public source response.

        Args:
            request: Incoming fixture request.

        Returns:
            Valid OKX envelope or source archive.
        """
        if request.url.path.endswith("/instruments"):
            market = {
                "instType": "SPOT",
                "instId": "BTC-USDT",
                "instFamily": "",
                "baseCcy": "BTC",
                "quoteCcy": "USDT",
                "settleCcy": "",
                "ctType": "",
                "ctVal": "",
                "ctMult": "",
                "ctValCcy": "",
                "state": "live",
                "ruleType": "normal",
                "listTime": "1609459200000",
                "expTime": "",
                "stk": "",
                "optType": "",
            }
            return httpx.Response(
                200,
                json={"code": "0", "msg": "", "data": [market]},
                request=request,
            )
        if request.url.path.endswith("/market-data-history"):
            assert request.url.params["module"] == "6"
            assert request.url.params["dateAggrType"] == "daily"
            group = {
                "instId": "BTC-USDT",
                "groupDetails": [
                    {
                        "dateTs": "1735689600000",
                        "filename": "BTC-USDT.OK.csv.gz",
                        "sizeMB": "0.01",
                        "url": "https://files.test/BTC-USDT.OK.csv.gz",
                    }
                ],
            }
            return httpx.Response(
                200,
                json={"code": "0", "msg": "", "data": [{"details": [group]}]},
                request=request,
            )
        if request.url.host == "files.test":
            self.files += 1
            return httpx.Response(
                200,
                content=self.content,
                headers={"Content-MD5": b64encode(md5(self.content).digest()).decode()},
                request=request,
            )
        raise AssertionError(f"unexpected request {request.url}")


def test_legacy_facade_is_explicit_queryable_and_offline_reusable(
    tmp_path: Path,
) -> None:
    """Confirm the opt-in facade caches and returns nested snapshots."""
    fixture = LegacyFixture()
    api = OKX(
        tmp_path,
        earliest_date="all",
        retries=0,
        progress=False,
        transport=httpx.MockTransport(fixture),
    )
    frame = api.get_legacy_order_book_50("BTC-USDT", "2025-01-01", "2025-01-01")
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [
        "event_time",
        "exchange_time",
        "event_number",
        "bids",
        "asks",
    ]
    assert frame.loc[0, "bids"][0]["base_quantity"] == 2.0
    cached = api.get_legacy_order_book_50(
        "BTC-USDT", "2025-01-01", "2025-01-01", offline=True
    )
    pd.testing.assert_frame_equal(frame, cached)
    assert fixture.files == 1


def test_corrupt_gzip_is_rejected_without_partial_output(tmp_path: Path) -> None:
    """Confirm structural GZIP failure never publishes partial Parquet."""
    item = physical()

    def handler(request: httpx.Request) -> httpx.Response:
        """Return digest-valid bytes that are not a GZIP archive."""
        content = b"not gzip"
        return httpx.Response(
            200,
            content=content,
            headers={"ETag": md5(content).hexdigest()},
            request=request,
        )

    destination = tmp_path / "legacy.parquet"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataValidationError, match="GZIP"):
            materialize_legacy_order_book(
                client,
                item,
                get_dataset("spot", "legacy_order_book_50"),
                destination,
                timeout=1,
                retries=0,
                backoff=0,
                max_archive_bytes=1_000,
            )
    assert not destination.exists()
    assert not destination.with_name("legacy.parquet.part").exists()


def test_legacy_parser_rejects_unsafe_configuration_and_lines(tmp_path: Path) -> None:
    """Confirm limits and dataset identity are checked before source parsing."""
    for name, value in (("chunk_rows", 0), ("max_line_bytes", 0)):
        with httpx.Client() as client:
            with pytest.raises(ValueError, match=name):
                materialize_legacy_order_book(
                    client,
                    physical(),
                    get_dataset("spot", "legacy_order_book_50"),
                    tmp_path / "unused.parquet",
                    timeout=1,
                    retries=0,
                    backoff=0,
                    max_archive_bytes=1,
                    **{name: value},
                )
    with pytest.raises(DataValidationError, match="exceeds"):
        _line(BytesIO(b"12345"), 4)


@pytest.mark.parametrize(
    ("item", "dataset", "message"),
    [
        (physical(), ("spot", "klines"), "module 6"),
        (
            physical(name="BTC-USDT.zip"),
            ("spot", "legacy_order_book_50"),
            "CSV/GZIP",
        ),
        (
            ArchiveObject(
                physical().key,
                "https://files.test/no-integrity.csv.gz",
                integrity=None,
            ),
            ("spot", "legacy_order_book_50"),
            "integrity",
        ),
    ],
)
def test_legacy_materialization_rejects_wrong_physical_declarations(
    tmp_path: Path,
    item: ArchiveObject,
    dataset: tuple[str, str],
    message: str,
) -> None:
    """Confirm only declared module 6 CSV/GZIP objects can be parsed.

    Args:
        tmp_path: Isolated destination root.
        item: Invalid physical archive declaration.
        dataset: Product and dataset declaration to use.
        message: Expected validation detail.
    """
    with httpx.Client() as client:
        with pytest.raises((ValueError, DataValidationError), match=message):
            materialize_legacy_order_book(
                client,
                item,
                get_dataset(*dataset),
                tmp_path / "unused.parquet",
                timeout=1,
                retries=0,
                backoff=0,
                max_archive_bytes=1,
            )
