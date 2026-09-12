# KuCoin

- Public archive: [historical-data.kucoin.com](https://historical-data.kucoin.com/)
- Public APIs: [KuCoin API documentation](https://www.kucoin.com/docs-new/)

Veldra supports KuCoin Spot plus linear- and inverse-margined perpetual
Futures. Dated delivery contracts are excluded.

## Support

KuCoin does not publish an equivalent official historical-data downloader.
Its public archive exposes daily files; Veldra adds concurrent retrieval,
automatic MD5 verification, schema normalization, a Parquet cache, exact-range
DuckDB queries, and ready-to-use pandas DataFrames.

| KuCoin archive data | Public archive | Veldra |
| --- | :---: | :---: |
| Spot Klines | ✅ | ✅ |
| Spot trades | ✅ | ✅ |
| Spot level-50 order books | ✅ | ✅ |
| Linear/inverse perpetual Klines | ✅ | ✅ |
| Linear/inverse perpetual trades | ✅ | ✅ |
| Index-price Klines | ✅ | ✅ |
| Mark-price Klines | ✅ | ✅ |
| Funding rates | ✅ | ✅ |
| Perpetual level-50 order books | ✅ | ✅ |
| Dated delivery contracts | ✅ | ❌ |
| Automatic MD5 verification | Sidecars only | ✅ |
| Normalized Parquet and DuckDB queries | ❌ | ✅ |
| Exact-range pandas DataFrames | ❌ | ✅ |

## Products and datasets

| `product` | Meaning | Supported datasets |
| --- | --- | --- |
| `spot` | Spot markets | `klines`, `trades`, `order_book_snapshots` |
| `linear_futures` | USDT/USDC-margined perpetuals | `klines`, `trades`, `index_price_klines`, `mark_price_klines`, `funding_rates`, `order_book_snapshots` |
| `inverse_futures` | Coin-margined perpetuals | `klines`, `trades`, `index_price_klines`, `mark_price_klines`, `funding_rates`, `order_book_snapshots` |

KuCoin publishes native Kline archives at `1m`, `5m`, `15m`, `1h`, `8h`,
`12h`, and `1d`. Veldra stores `1m` Klines and can return:

`1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`,
`1d`, `3d`, `1w`, `1mo`.

## Create the facade

```python
from veldra import KuCoin

kucoin = KuCoin(
    data_dir="data/kucoin",
    max_workers=32,
    progress=True,
)
```

Construction is lazy: it does not access the network or create files. See
[Getting started](getting-started.md#create-an-exchange-service) for shared
constructor options.

## Retrieve data

Every retrieval method accepts one market string or an ordered list. One
market returns one DataFrame; a list returns a list in the same order.

| Method | Product | Additional options | Returns |
| --- | --- | --- | --- |
| `get_klines(pairs, start, end, ...)` | All; default `spot` | `interval`, `columns`, `gap_policy`, `refresh`, `offline` | Trading candles |
| `get_trades(pairs, start, end, ...)` | All; default `spot` | `columns`, `refresh`, `offline` | Individual trades |
| `get_index_price_klines(pairs, start, end, ...)` | Futures, required | `interval`, `columns`, `gap_policy`, `refresh`, `offline` | Index-price candles |
| `get_mark_price_klines(pairs, start, end, ...)` | Futures, required | `interval`, `columns`, `gap_policy`, `refresh`, `offline` | Mark-price candles |
| `get_funding_rates(pairs, start, end, ...)` | Futures, required | `columns`, `refresh`, `offline` | Funding observations |
| `get_order_book_snapshots(pairs, start, end, ...)` | All; default `spot` | `columns`, `refresh`, `offline` | Nested level-50 snapshots |

```python
spot = kucoin.get_klines(
    ["BTC-USDT", "ETH-USDT"],
    "2025-01-01",
    "2025-01-07",
    interval="1h",
)

linear_trades = kucoin.get_trades(
    "XBTUSDTM",
    "2025-01-01",
    "2025-01-01",
    product="linear_futures",
)

inverse_mark = kucoin.get_mark_price_klines(
    "XBTUSDM",
    "2025-01-01",
    "2025-01-02",
    product="inverse_futures",
    interval="5m",
)

funding = kucoin.get_funding_rates(
    "XBTUSDTM",
    "2025-01-01",
    "2025-01-07",
    product="linear_futures",
)
```

KuCoin's current Futures API uses `XBTUSDTM` and `XBTUSDM`, while archive
folders use `BTCUSDTM` and `BTCUSDM`. Veldra resolves these aliases internally;
users can request either known form. Spot accepts native `BTC-USDT` and compact
`BTCUSDT` forms.

## Inspect markets and coverage

| Method | Network behavior | Result |
| --- | --- | --- |
| `get_markets(...)` | Refreshes stale metadata unless offline | Filtered `list[Market]` |
| `find_markets(query, ...)` | Refreshes stale metadata unless offline | Ranked `list[Market]` |
| `get_availability(pair, ...)` | Local only | Cataloged `Availability` |
| `discover_availability(pair, start, end, ...)` | Lists a bounded remote range | Updated `Availability` |

```python
active = kucoin.get_markets(
    product="spot",
    status="TRADING",
    quote_asset="USDT",
    sort_by="quote_volume",
    limit=20,
)

matches = kucoin.find_markets("BTCSUDT", product="spot", limit=3)

coverage = kucoin.discover_availability(
    "BTC-USDT",
    "2023-01-01",
    "2025-12-31",
    product="spot",
    dataset="klines",
    interval="1m",
)
```

## Canonical columns

| Dataset | Product | Stored/queryable columns |
| --- | --- | --- |
| Klines | Spot | `open_time`, OHLC, `base_volume`, `quote_volume`, `is_synthetic` |
| Klines | Perpetuals | `open_time`, OHLC, `contract_volume`, `is_synthetic` |
| Index/mark Klines | Perpetuals | `open_time`, OHLC, `sample_count`, `is_synthetic` |
| Trades | Spot | `event_time`, `trade_id`, `price`, `base_quantity`, `quote_quantity`, `side` |
| Trades | Linear perpetuals | Spot-style fields plus `contract_quantity` |
| Trades | Inverse perpetuals | `event_time`, `trade_id`, `price`, `contract_quantity`, `base_quantity`, `quote_notional`, `side` |
| Funding rates | Perpetuals | `funding_time`, `funding_rate` |
| Order books | Spot | `event_time`, nested `bids`, nested `asks` |
| Order books | Perpetuals | `event_time`, `sequence`, nested `bids`, nested `asks` |

Each bid or ask is a list of structures. A Spot level contains `price` and
`base_quantity`; a Futures level contains `price` and `contract_quantity`.
This preserves each source snapshot as one row instead of expanding millions
of levels into a much larger flat table.

## Archive behavior

KuCoin currently publishes daily archives only. Availability differs by
market and dataset, so Veldra treats remote listings as the source of truth
instead of assuming every day exists. Level-50 files can contain observations
a few seconds outside their nominal UTC filename day; bounded discovery and
exact timestamp filtering account for that behavior.

CSV archives are parsed and normalized with Arrow. Order-book archives contain
newline-delimited JSON records behind a `data` header; Veldra validates their
timestamps and ordered price ladders, then streams them to nested Parquet.
