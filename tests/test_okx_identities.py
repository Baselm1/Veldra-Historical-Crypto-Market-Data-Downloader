"""Test OKX native identities and current-market discovery."""

from datetime import UTC, datetime

import httpx
import pytest

from veldra.okx.connector import OKXConnector
from veldra.okx.identities import (
    OKXInstrument,
    historical_future,
    parse_currency,
    parse_instrument,
    parse_option_id,
)


def instrument_row(
    inst_type: str, inst_id: str, **changes: object
) -> dict[str, object]:
    """Build one compact OKX public-instrument fixture.

    Args:
        inst_type: Native instrument type.
        inst_id: Native instrument ID.
        changes: Field overrides for the scenario.

    Returns:
        A fixture public-instrument object.
    """
    row: dict[str, object] = {
        "instType": inst_type,
        "instId": inst_id,
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
    row.update(changes)
    return row


@pytest.mark.parametrize(
    ("row", "product", "style"),
    [
        (instrument_row("SPOT", "BTC-USDT"), "spot", None),
        (instrument_row("MARGIN", "BTC-USDT"), "margin", None),
        (
            instrument_row(
                "SWAP",
                "BTC-USDT-SWAP",
                instFamily="BTC-USDT",
                ctType="linear",
                settleCcy="USDT",
                ctVal="0.01",
                ctMult="1",
                ctValCcy="BTC",
            ),
            "linear_swap",
            "normal",
        ),
        (
            instrument_row(
                "FUTURES",
                "BTC-USD-261225",
                instFamily="BTC-USD",
                ctType="inverse",
                settleCcy="BTC",
                ctVal="100",
                ctMult="1",
                ctValCcy="USD",
                expTime="1798156800000",
            ),
            "inverse_futures",
            "normal",
        ),
        (
            instrument_row(
                "FUTURES",
                "BTC-USD-311231",
                instFamily="BTC-USD",
                ctType="inverse",
                ruleType="xperp",
            ),
            "inverse_futures",
            "xperp",
        ),
        (
            instrument_row(
                "FUTURES",
                "BTC-USDT-311231",
                instFamily="BTC-USDT",
                ctType="linear",
                ruleType="pre_market",
            ),
            "linear_futures",
            "pre_market_xperp",
        ),
        (
            instrument_row(
                "OPTION",
                "BTC-USD-261225-100000-C",
                instFamily="BTC-USD",
                baseCcy="",
                quoteCcy="",
                settleCcy="BTC",
                ctVal="0.01",
                ctMult="1",
                ctValCcy="BTC",
                expTime="1798156800000",
                stk="100000",
                optType="C",
            ),
            "options",
            "normal",
        ),
    ],
)
def test_parse_every_current_instrument_kind(
    row: dict[str, object], product: str, style: str | None
) -> None:
    """Confirm current source variants preserve their native metadata.

    Args:
        row: Public instrument fixture.
        product: Expected Veldra product.
        style: Expected contract style.
    """
    value = parse_instrument(row)
    assert value.product == product
    assert value.contract_style == style
    assert value.market.product == product
    assert value.market.active
    assert value.market.source == "okx"
    assert value.list_time == datetime(2021, 1, 1, tzinfo=UTC)


def test_option_identity_preserves_strike_expiry_and_type() -> None:
    """Confirm an Option contract can be resolved without ambiguous splitting."""
    expiry, strike, option_type = parse_option_id("BTC-USD-261225-100000-P")
    assert expiry == datetime(2026, 12, 25, tzinfo=UTC).date()
    assert strike == 100_000
    assert option_type == "P"


@pytest.mark.parametrize(
    ("instrument", "product", "family"),
    [
        ("BTC-USDT-250103", "linear_futures", "BTC-USDT"),
        ("BTC-USD-250103", "inverse_futures", "BTC-USD"),
    ],
)
def test_expired_futures_gain_conservative_archive_identities(
    instrument: str, product: str, family: str
) -> None:
    """Confirm expired IDs retain dates without invented contract size.

    Args:
        instrument: Native historical contract ID.
        product: Expected Futures product.
        family: Expected manifest family.
    """
    identity = historical_future(instrument.lower(), product)
    assert identity.instrument_id == instrument
    assert identity.family == family
    assert identity.expiry == datetime(2025, 1, 3).date()
    assert identity.contract_value is None
    assert identity.provenance == "archive_identity"
    assert not identity.market.active


@pytest.mark.parametrize(
    ("instrument", "product"),
    [
        ("BTC-USDT-SWAP", "linear_futures"),
        ("BTC-USD-250103", "linear_futures"),
        ("BTC-USDT-991332", "linear_futures"),
    ],
)
def test_historical_futures_reject_malformed_or_mismatched_ids(
    instrument: str, product: str
) -> None:
    """Confirm archive-derived identities remain product-safe.

    Args:
        instrument: Invalid or mismatched native ID.
        product: Proposed Futures product.
    """
    with pytest.raises(ValueError):
        historical_future(instrument, product)


@pytest.mark.parametrize("value", ["", "../BTC-USDT", "BTC/USDT", 1])
def test_identity_parser_rejects_unsafe_native_values(value: object) -> None:
    """Confirm unsafe or non-text native identities never reach storage paths.

    Args:
        value: Invalid native instrument value.
    """
    row = instrument_row("SPOT", "BTC-USDT")
    row["instId"] = value
    with pytest.raises((TypeError, ValueError), match="instrument"):
        parse_instrument(row)


@pytest.mark.parametrize(
    "changes",
    [
        {"instType": "EVENTS"},
        {"state": "unknown"},
        {"ctType": "quanto", "instType": "SWAP"},
        {"ruleType": "mystery", "instType": "FUTURES", "ctType": "linear"},
        {"ctVal": "nope", "instType": "SWAP", "ctType": "linear"},
    ],
)
def test_identity_parser_rejects_unexpected_enums_and_numbers(
    changes: dict[str, object],
) -> None:
    """Confirm source schema drift fails visibly.

    Args:
        changes: Invalid source fields applied to a valid fixture.
    """
    row = instrument_row("SPOT", "BTC-USDT", **changes)
    with pytest.raises(ValueError):
        parse_instrument(row)


@pytest.mark.parametrize("value", ["USDT", "btc", " USD "])
def test_currency_subjects_are_normalized(value: str) -> None:
    """Confirm borrowing subjects become safe uppercase currencies.

    Args:
        value: User-facing currency text.
    """
    assert parse_currency(value) == value.strip().upper()


class CurrentAPI:
    """Serve instruments, Options families, and volume fixtures."""

    def __init__(self, *, duplicate: bool = False) -> None:
        """Create one fixture and optional duplicate response.

        Args:
            duplicate: Whether Spot should repeat its current instrument.
        """
        self.duplicate = duplicate

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return an OKX public endpoint envelope.

        Args:
            request: The fixture HTTP request.

        Returns:
            A valid public API response.
        """
        path = request.url.path
        inst_type = request.url.params.get("instType")
        if path.endswith("/underlying"):
            return httpx.Response(
                200, json={"code": "0", "msg": "", "data": [["BTC-USD"]]}
            )
        if path.endswith("/tickers"):
            return httpx.Response(
                200,
                json={
                    "code": "0",
                    "msg": "",
                    "data": [{"instId": "BTC-USDT", "volCcy24h": "123.5"}],
                },
            )
        if inst_type == "SPOT":
            rows = [instrument_row("SPOT", "BTC-USDT")]
            if self.duplicate:
                rows *= 2
        elif inst_type == "SWAP":
            rows = [
                instrument_row(
                    "SWAP", "BTC-USDT-SWAP", instFamily="BTC-USDT", ctType="linear"
                ),
                instrument_row(
                    "SWAP", "BTC-USD-SWAP", instFamily="BTC-USD", ctType="inverse"
                ),
            ]
        elif inst_type == "FUTURES":
            rows = [
                instrument_row(
                    "FUTURES", "BTC-USDT-261225", instFamily="BTC-USDT", ctType="linear"
                ),
                instrument_row(
                    "FUTURES", "BTC-USD-261225", instFamily="BTC-USD", ctType="inverse"
                ),
            ]
        elif inst_type == "OPTION":
            assert request.url.params["uly"] == "BTC-USD"
            rows = [
                instrument_row(
                    "OPTION",
                    "BTC-USD-261225-100000-C",
                    instFamily="BTC-USD",
                    stk="100000",
                    optType="C",
                )
            ]
        else:
            rows = []
        return httpx.Response(200, json={"code": "0", "msg": "", "data": rows})


@pytest.mark.parametrize(
    ("product", "symbols"),
    [
        ("spot", ["BTC-USDT"]),
        ("linear_swap", ["BTC-USDT-SWAP"]),
        ("inverse_swap", ["BTC-USD-SWAP"]),
        ("linear_futures", ["BTC-USDT-261225"]),
        ("inverse_futures", ["BTC-USD-261225"]),
        ("options", ["BTC-USD-261225-100000-C"]),
    ],
)
def test_connector_discovers_each_current_product(
    product: str, symbols: list[str]
) -> None:
    """Confirm current product discovery filters native endpoint rows.

    Args:
        product: Veldra product to discover.
        symbols: Expected native IDs.
    """
    with httpx.Client(transport=httpx.MockTransport(CurrentAPI())) as client:
        markets = OKXConnector(retries=0).markets(client, product)
    assert [market.symbol for market in markets] == symbols


def test_connector_enriches_quote_volume_and_rejects_duplicates() -> None:
    """Confirm current quote volume is sortable and duplicate IDs are rejected."""
    with httpx.Client(transport=httpx.MockTransport(CurrentAPI())) as client:
        connector = OKXConnector(retries=0)
        assert connector.quote_volumes(client, "spot") == {"BTC-USDT": 123.5}
    with httpx.Client(
        transport=httpx.MockTransport(CurrentAPI(duplicate=True))
    ) as client:
        with pytest.raises(ValueError, match="duplicate"):
            OKXConnector(retries=0).markets(client, "spot")


def test_product_scoped_fuzzy_suggestions_do_not_cross_products() -> None:
    """Confirm typo suggestions are ranked only within the requested product."""
    with httpx.Client(transport=httpx.MockTransport(CurrentAPI())) as client:
        connector = OKXConnector(retries=0)
        assert connector.suggest(client, "BTCSUDT", "spot") == ["BTC-USDT"]
        assert connector.suggest(client, "BTCSUDT", "linear_swap") == ["BTC-USDT-SWAP"]


def test_connector_rejects_unknown_products_before_io() -> None:
    """Confirm an unknown Veldra product cannot issue source requests."""
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("unexpected HTTP request")
        )
    ) as client:
        with pytest.raises(ValueError, match="product"):
            OKXConnector().markets(client, "events")
