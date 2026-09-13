"""Rank normalized market symbols without silently correcting requests."""

from collections.abc import Sequence

from rapidfuzz.distance import JaroWinkler

from veldra.core.models import Market
from veldra.core.request import normalize_pair

MINIMUM_SIMILARITY = 0.90


def market_aliases(market: Market) -> tuple[str, ...]:
    """Return normalized public, native, and archive names for one market.

    Args:
        market: The source market whose identifiers are inspected.

    Returns:
        Unique nonempty identifiers in stable preference order.
    """
    aliases: list[str] = []
    semantic = normalize_pair(market.normalized_symbol)
    native = normalize_pair(market.symbol)
    values = [market.normalized_symbol]
    if native == semantic:
        values.append(market.symbol)
    if market.pair and (normalize_pair(market.pair) != native or native == semantic):
        values.append(market.pair)
    for value in values:
        if not value:
            continue
        normalized = normalize_pair(value)
        if normalized and normalized not in aliases:
            aliases.append(normalized)
    return tuple(aliases)


def exact_markets(query: str, markets: Sequence[Market]) -> list[Market]:
    """Resolve a query against native names before normalized aliases.

    Args:
        query: The caller's original market query.
        markets: Source markets eligible for matching.

    Returns:
        Native exact matches, or all matches through normalized aliases.
    """
    native = [market for market in markets if market.symbol.upper() == query.upper()]
    if native:
        return native
    normalized = normalize_pair(query)
    return [
        market
        for market in markets
        if normalized == normalize_pair(market.normalized_symbol)
        or _archive_alias_matches(normalized, market)
    ]


def _archive_alias_matches(query: str, market: Market) -> bool:
    """Return whether a normalized query identifies an archive-only alias.

    Args:
        query: The normalized caller query.
        market: The candidate source market.

    Returns:
        True for a distinct archive alias without reversing quote-first markets.
    """
    archive = market.pair
    if not archive:
        return False
    archive_alias = normalize_pair(archive)
    native_alias = normalize_pair(market.symbol)
    semantic_alias = normalize_pair(market.normalized_symbol)
    if archive_alias == native_alias and native_alias != semantic_alias:
        return False
    return query == archive_alias


def _quote_matches(query: str, market: Market) -> bool:
    """Return whether a query appears to contain the market quote asset.

    Args:
        query: The normalized market query.
        market: The candidate source market.

    Returns:
        True when the normalized query ends with the candidate quote asset.
    """
    quote = market.quote_asset
    return bool(quote and query.endswith(quote.upper()))


def _fuzzy_markets(query: str, markets: Sequence[Market]) -> list[Market]:
    """Return high-confidence fuzzy matches in deterministic score order.

    Args:
        query: The normalized market query.
        markets: Candidate markets that did not match exactly or by prefix.

    Returns:
        Candidates scoring at least ninety percent by Jaro-Winkler similarity.
    """
    scored = [
        (
            max(
                JaroWinkler.normalized_similarity(query, alias)
                for alias in market_aliases(market)
            ),
            market,
        )
        for market in markets
        if market_aliases(market)
    ]
    accepted = [item for item in scored if item[0] >= MINIMUM_SIMILARITY]
    accepted.sort(
        key=lambda item: (
            -item[0],
            not _quote_matches(query, item[1]),
            not item[1].active,
            item[1].symbol,
            item[1].product or "",
        )
    )
    return [market for _score, market in accepted]


def rank_markets(query: str, markets: Sequence[Market], limit: int) -> list[Market]:
    """Rank exact, prefix, then high-confidence fuzzy market matches.

    Args:
        query: The normalized market query.
        markets: The source markets eligible for matching.
        limit: The maximum number of results.

    Returns:
        Ranked markets without automatically selecting a fuzzy candidate.
    """
    exact = exact_markets(query, markets)
    prefix = [
        market
        for market in markets
        if market not in exact
        and any(alias.startswith(query) for alias in market_aliases(market))
    ]
    excluded = {*exact, *prefix}
    remaining = [market for market in markets if market not in excluded]
    return [*exact, *prefix, *_fuzzy_markets(query, remaining)][:limit]


def suggest_symbols(
    query: str,
    markets: Sequence[Market],
    limit: int = 3,
) -> tuple[str, ...]:
    """Return unique native symbols for a misspelled normalized query.

    Args:
        query: The normalized unknown market query.
        markets: The source markets eligible for suggestions.
        limit: The maximum number of unique symbols.

    Returns:
        Up to the requested number of high-confidence symbol suggestions.
    """
    symbols: list[str] = []
    for market in rank_markets(query, markets, len(markets)):
        if market.symbol not in symbols:
            symbols.append(market.symbol)
        if len(symbols) == limit:
            break
    return tuple(symbols)
