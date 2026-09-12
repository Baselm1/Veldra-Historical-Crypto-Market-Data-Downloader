"""Parse current and historical OKX instrument identities."""

from dataclasses import dataclass
from datetime import UTC, date, datetime
import math
import re
from typing import Literal, cast

from veldra.core.models import Market
from veldra.core.request import normalize_pair

type OKXProduct = Literal[
    "spot",
    "margin",
    "linear_swap",
    "inverse_swap",
    "linear_futures",
    "inverse_futures",
    "options",
]
type ContractStyle = Literal["normal", "xperp", "pre_market_xperp"]
_SAFE = re.compile(r"[A-Z0-9_-]+")
_TEXT = re.compile(r"[A-Za-z0-9_-]+")
_OPTION = re.compile(r"^[A-Z0-9]+-[A-Z0-9_]+-(\d{6})-([0-9]+(?:\.[0-9]+)?)-([CP])$")
_FUTURE = re.compile(r"^([A-Z0-9]+)-(USD(?:T|C)?(?:_UM)?)-(\d{6})$")
_STATES = frozenset({"live", "suspend", "rebase", "post_only", "preopen", "test"})


def _text(value: object, name: str, *, optional: bool = False) -> str | None:
    """Validate one plain source string.

    Args:
        value: Source field value.
        name: Field name used in errors.
        optional: Whether an empty string becomes ``None``.

    Returns:
        The safe string or ``None``.
    """
    if not isinstance(value, str):
        raise TypeError(f"OKX {name} must be a string")
    if optional and not value:
        return None
    if not value or _TEXT.fullmatch(value) is None:
        raise ValueError(f"OKX {name} contains an unsafe instrument value")
    return value


def _number(value: object, name: str) -> float | None:
    """Parse one optional finite source number.

    Args:
        value: Numeric source text or an empty value.
        name: Field name used in errors.

    Returns:
        The finite number or ``None``.
    """
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"OKX {name} is not numeric") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"OKX {name} must be finite and positive")
    return parsed


def _time(value: object, name: str) -> datetime | None:
    """Parse one optional epoch-millisecond source timestamp.

    Args:
        value: Epoch milliseconds or an empty value.
        name: Field name used in errors.

    Returns:
        UTC timestamp or ``None``.
    """
    if value in {None, ""}:
        return None
    try:
        return datetime.fromtimestamp(int(str(value)) / 1000, UTC)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"OKX {name} is not an epoch timestamp") from error


def parse_currency(value: object) -> str:
    """Normalize one currency-scoped historical subject.

    Args:
        value: User currency name.

    Returns:
        Safe uppercase currency code.
    """
    if not isinstance(value, str):
        raise TypeError("currency must be a string")
    normalized = value.strip().upper()
    if not normalized or _SAFE.fullmatch(normalized) is None or "-" in normalized:
        raise ValueError("currency contains an unsafe value")
    return normalized


def parse_option_id(value: str) -> tuple[date, float, Literal["C", "P"]]:
    """Parse expiry, strike, and call/put type from an Option ID.

    Args:
        value: Native OKX Option instrument ID.

    Returns:
        Derived expiry date, strike, and Option type.
    """
    match = _OPTION.fullmatch(value)
    if match is None:
        raise ValueError("OKX Option instrument ID is malformed")
    expiry = datetime.strptime(match.group(1), "%y%m%d").date()
    option_type = "C" if match.group(3) == "C" else "P"
    return expiry, float(match.group(2)), cast(Literal["C", "P"], option_type)


def historical_future(value: object, product: object) -> OKXInstrument:
    """Build a conservative archive-derived dated Futures identity.

    Args:
        value: Native expired contract ID.
        product: Expected linear or inverse Futures product.

    Returns:
        Archive-only identity without invented contract-size metadata.
    """
    if not isinstance(value, str) or not isinstance(product, str):
        raise TypeError("historical Futures identity and product must be strings")
    instrument = value.strip().upper()
    match = _FUTURE.fullmatch(instrument)
    if match is None:
        raise ValueError("OKX historical Futures instrument ID is malformed")
    base, settlement, expiry_text = match.groups()
    expected = "inverse_futures" if settlement == "USD" else "linear_futures"
    if product != expected:
        raise ValueError("OKX historical Futures instrument does not match product")
    try:
        expiry = datetime.strptime(expiry_text, "%y%m%d").date()
    except ValueError as error:
        raise ValueError("OKX historical Futures expiry is invalid") from error
    family = f"{base}-{settlement}"
    return OKXInstrument(
        instrument,
        cast(OKXProduct, product),
        "FUTURES",
        family,
        base,
        settlement,
        settlement,
        "inverse" if expected == "inverse_futures" else "linear",
        None,
        None,
        None,
        "normal",
        "archive_only",
        None,
        None,
        expiry,
        None,
        None,
        provenance="archive_identity",
    )


def _product(inst_type: str, contract_type: str | None) -> OKXProduct:
    """Map native instrument and contract types to one Veldra product.

    Args:
        inst_type: Native instrument type.
        contract_type: Native linear or inverse contract type.

    Returns:
        The Veldra product identifier.
    """
    if inst_type == "SPOT":
        return "spot"
    if inst_type == "MARGIN":
        return "margin"
    if inst_type == "OPTION":
        return "options"
    if inst_type not in {"SWAP", "FUTURES"} or contract_type not in {
        "linear",
        "inverse",
    }:
        raise ValueError("unsupported OKX instrument or contract type")
    suffix = "swap" if inst_type == "SWAP" else "futures"
    return f"{contract_type}_{suffix}"  # type: ignore[return-value]


def _style(inst_type: str, rule_type: str) -> ContractStyle | None:
    """Map a native Futures rule into its contract style.

    Args:
        inst_type: Native instrument type.
        rule_type: Native trading rule.

    Returns:
        Contract style for derivatives or ``None`` for cash markets.
    """
    if rule_type not in {"normal", "xperp", "pre_market"}:
        raise ValueError("unsupported OKX ruleType")
    if inst_type in {"SPOT", "MARGIN"}:
        return None
    if rule_type == "xperp":
        return "xperp"
    if rule_type == "pre_market":
        return "pre_market_xperp"
    return "normal"


@dataclass(frozen=True)
class OKXInstrument:
    """Retain a parsed OKX instrument and its unit metadata."""

    instrument_id: str
    product: OKXProduct
    instrument_type: str
    family: str | None
    base_currency: str | None
    quote_currency: str | None
    settlement_currency: str | None
    contract_type: str | None
    contract_value: float | None
    contract_multiplier: float | None
    contract_value_currency: str | None
    contract_style: ContractStyle | None
    state: str
    list_time: datetime | None
    expiry_time: datetime | None
    expiry: date | None
    strike: float | None
    option_type: Literal["C", "P"] | None
    provenance: str = "current_api"
    quote_volume_24h: float | None = None

    @property
    def market(self) -> Market:
        """Return the shared market representation used by Veldra."""
        contract_size = None
        if self.contract_value is not None:
            contract_size = self.contract_value * (self.contract_multiplier or 1)
        return Market(
            symbol=self.instrument_id,
            normalized_symbol=normalize_pair(self.instrument_id),
            base_asset=self.base_currency,
            quote_asset=self.quote_currency or self.settlement_currency,
            status=self.state,
            pair=self.family or self.instrument_id,
            contract_type=(
                self.contract_style.upper() if self.contract_style is not None else None
            ),
            contract_size=contract_size,
            onboard_time=self.list_time,
            delivery_time=self.expiry_time,
            source="okx",
            product=self.product,
            quote_volume_24h=self.quote_volume_24h,
            active=self.state == "live",
        )


def parse_instrument(value: object) -> OKXInstrument:
    """Parse one public OKX instrument response object.

    Args:
        value: Source response row.

    Returns:
        A typed native instrument identity.
    """
    if not isinstance(value, dict):
        raise ValueError("OKX instrument row must be an object")
    instrument_type = _text(value.get("instType"), "instrument type")
    instrument_id = _text(value.get("instId"), "instrument ID")
    assert instrument_type is not None and instrument_id is not None
    if _SAFE.fullmatch(instrument_id) is None:
        raise ValueError("OKX instrument ID contains an unsafe instrument value")
    state = _text(value.get("state"), "state")
    if state not in _STATES:
        raise ValueError("unsupported OKX instrument state")
    contract_type = _text(value.get("ctType", ""), "contract type", optional=True)
    if contract_type not in {None, "linear", "inverse"}:
        raise ValueError("unsupported OKX contract type")
    product = _product(instrument_type, contract_type)
    family = _text(value.get("instFamily", ""), "family", optional=True)
    base = _text(value.get("baseCcy", ""), "base currency", optional=True)
    quote = _text(value.get("quoteCcy", ""), "quote currency", optional=True)
    settle = _text(value.get("settleCcy", ""), "settlement currency", optional=True)
    value_currency = _text(
        value.get("ctValCcy", ""), "contract value currency", optional=True
    )
    expiry = strike = option_type = None
    if instrument_type == "OPTION":
        expiry, strike, option_type = parse_option_id(instrument_id)
        declared_strike = _number(value.get("stk"), "strike")
        declared_type = value.get("optType")
        if declared_strike != strike or declared_type != option_type:
            raise ValueError("OKX Option metadata contradicts its instrument ID")
    rule_type = value.get("ruleType", "normal")
    if not isinstance(rule_type, str):
        raise ValueError("OKX ruleType must be a string")
    return OKXInstrument(
        instrument_id,
        product,
        instrument_type,
        family,
        base,
        quote,
        settle,
        contract_type,
        _number(value.get("ctVal"), "contract value"),
        _number(value.get("ctMult"), "contract multiplier"),
        value_currency,
        _style(instrument_type, rule_type),
        state,
        _time(value.get("listTime"), "list time"),
        _time(value.get("expTime"), "expiry time"),
        expiry,
        strike,
        option_type,
    )
