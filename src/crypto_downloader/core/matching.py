"""Rank normalized market symbols without silently correcting requests."""

from collections.abc import Sequence

from rapidfuzz.distance import JaroWinkler

from crypto_downloader.core.models import Market

MINIMUM_SIMILARITY = 0.90


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
        (JaroWinkler.normalized_similarity(query, market.normalized_symbol), market)
        for market in markets
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
    exact = [market for market in markets if market.normalized_symbol == query]
    prefix = [
        market
        for market in markets
        if market.normalized_symbol != query
        and market.normalized_symbol.startswith(query)
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
