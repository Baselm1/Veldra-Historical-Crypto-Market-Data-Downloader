# OKX

Veldra exposes OKX historical archives and public history endpoints through
the `OKX` facade. Source data comes from the
[OKX historical-data portal](https://www.okx.com/en-eu/historical-data),
while current instruments and supplemental history use the
[OKX API V5](https://www.okx.com/docs-v5/en/). The official
[Python SDK](https://github.com/okxapi/python-okx) is primarily a trading API
client; it does not provide Veldra's archive discovery, validation, Parquet
cache, DuckDB queries, or DataFrame reports.

## Supported historical data

| Product | Klines | Trades | Funding | Books | Other |
| --- | :---: | :---: | :---: | :---: | --- |
| Spot | ✅ | ✅ | — | ✅ | Legacy level-50 books |
| Margin | — | — | — | — | Borrow rates |
| Linear swaps | ✅ | ✅ | ✅ | ✅ | Index, mark, premium, open-interest and ratio history |
| Inverse swaps | ✅ | ✅ | ✅ | ✅ | Index, mark, premium, open-interest and ratio history |
| Linear Futures | ✅ | ✅ | X-Perp | ✅ | Chain retrieval and settlement history |
| Inverse Futures | ✅ | ✅ | X-Perp | ✅ | Chain retrieval and settlement history |
| Options | ✅ | ✅ | — | ✅ | Chain retrieval, exercise and interest/volume history |

The archive modules currently used are trades, Klines, funding rates,
400-level and 5,000-level order-book updates, legacy level-50 order books, and
margin borrowing rates. API-only series are cached separately but returned in
the same DataFrame-oriented style.

## Create the facade

```python
from veldra import OKX

okx = OKX(data_dir="data/okx")
```

Construction performs no I/O. Set `progress=False` for silent library calls.
Use `earliest_date="all"` to allow all discoverable source history.

## Retrieve instrument data

```python
frame = okx.get_klines(
    "BTC-USDT",
    "2025-01-01",
    "2025-01-07",
    product="spot",
    interval="1h",
)

trades = okx.get_trades(
    ["BTC-USDT", "ETH-USDT"],
    "2025-01-01",
    "2025-01-02",
    product="spot",
)
```

A pair string returns one `pandas.DataFrame`; a list returns DataFrames in the
same order. Supported Kline output intervals are `1m`, `3m`, `5m`, `15m`,
`30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`, and `1mo`.
The cache stores canonical one-minute Klines and DuckDB produces larger
intervals.

Other archive-backed calls are:

```python
funding = okx.get_funding_rates(
    "BTC-USDT-SWAP",
    "2025-01-01",
    "2025-01-07",
    product="linear_swap",
)

borrow = okx.get_borrow_rates("USDT", "2025-01-01", "2025-01-07")

books = okx.get_order_book_updates(
    "BTC-USDT",
    "2025-01-01",
    "2025-01-01",
    product="spot",
    depth=400,
)

legacy = okx.get_legacy_order_book_50(
    "BTC-USDT",
    "2025-01-01",
    "2025-01-01",
    product="spot",
)
```

Order-book archives can be very large. Their bid and ask levels remain nested
inside Parquet rather than being expanded into hundreds of columns. The
legacy method is explicit because module 6 uses a different, header-defined
schema and can contain very large files.

## Retrieve contract chains

Futures and Options archives may represent a whole contract family in one
physical file. Chain methods materialize that file once and query individual
contracts through logical DuckDB partitions.

```python
futures = okx.get_futures_chain_klines(
    instrument_family="BTC-USDT",
    start="2025-01-01",
    end="2025-01-07",
    product="linear_futures",
    interval="1h",
)

options = okx.get_option_chain_trades(
    instrument_family="BTC-USD",
    start="2025-01-01",
    end="2025-01-02",
    option_type="call",
    strike_min=80000,
)
```

Related methods are `get_futures_chain_trades()`,
`get_option_chain_klines()`, and `get_option_chain_trades()`.

## Cache without returning rows

Bulk archives are useful when building a research cache for many instruments.
These methods return a `CacheReport` rather than loading all rows into memory:

```python
report = okx.cache_klines(
    "2025-01-01",
    "2025-01-31",
    product="spot",
)

report = okx.cache_dataset(
    "2025-01-01",
    "2025-01-31",
    product="spot",
    dataset="trades",
)
```

There are dedicated `cache_trades()` and `cache_funding_rates()` methods too.
`transport="auto"` chooses between instrument-specific and all-market archive
objects; `"specific"` and `"bulk"` force a choice where OKX provides both.

## Inspect instruments and coverage

```python
markets = okx.get_markets(
    product="spot",
    active=True,
    quote_asset="USDT",
    sort_by="quote_volume",
    limit=20,
)

suggestions = okx.find_markets("BTCUSDT", product="spot", limit=3)

contracts = okx.get_contracts(
    product="linear_futures",
    family="BTC-USDT",
)

options = okx.get_option_contracts(
    family="BTC-USD",
    option_type="put",
)
```

`get_contracts()` and `get_option_contracts()` describe contracts returned by
the current public instrument API. Historical chain methods can still query
expired contracts present only in archives.

Local inspection never accesses the network:

```python
known = okx.get_availability(
    "BTC-USDT",
    product="spot",
    dataset="klines",
    interval="1h",
)
```

Bounded discovery updates cataloged coverage without downloading data files:

```python
remote = okx.discover_availability(
    "BTC-USDT",
    "2025-01-01",
    "2025-01-31",
    product="spot",
    dataset="klines",
    interval="1h",
)
```

## Supplemental public history

The following methods use rate-limited OKX public endpoints and cache their
normalized results:

- `get_index_price_klines()`
- `get_mark_price_klines()`
- `get_premium_history()`
- `get_recent_funding_rates()`
- `get_settlement_history()`
- `get_delivery_exercise_history()`
- `get_open_interest_history()`
- `get_taker_volume()`
- `get_long_short_ratio()`
- `get_option_interest_volume()`

For example:

```python
mark = okx.get_mark_price_klines(
    "BTC-USDT-SWAP",
    "2025-01-01",
    "2025-01-02",
    product="linear_swap",
    interval="1h",
)
```

Each endpoint has its own OKX rate bucket. Retries honor HTTP and OKX
throttling responses rather than creating unconstrained concurrent requests.

## Time boundaries and validation

Date-only calls include the entire final date. Exact datetimes use a half-open
`[start, end)` interval. Returned timestamps are UTC.

Most OKX daily archive labels use UTC+8 source-day boundaries, so a UTC request
can touch an adjacent archive label. Current order-book modules use UTC labels.
Veldra maps both calendars before planning downloads and filters the final
result to the exact UTC request.

Archive integrity follows source metadata: objects with a published MD5 are
verified, while objects without one are validated by archive structure and
schema. Unknown instruments return typed empty frames with suggestions;
invalid products, datasets, intervals, or date ranges raise before download.
