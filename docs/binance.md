# Binance

- Public archive: [data.binance.vision](https://data.binance.vision/)
- Official helpers: [binance/binance-public-data](https://github.com/binance/binance-public-data/)

Veldra supports Binance Spot plus USD-M and COIN-M perpetual Futures. Dated
delivery contracts and Options are outside the current scope.

## Comparison

The official repository was audited at revision `5c7f319` (2025-01-09). Its
Python scripts download ZIP files sequentially and may also download checksum
sidecars; they do not verify the digest, normalize files, maintain a queryable
cache, or return DataFrames. A separate shell script launches concurrent
monthly Futures Kline downloads.

| Binance archive data | Official helpers | Veldra |
| --- | :---: | :---: |
| Spot Klines | ✅ | ✅ |
| Spot trades | ✅ | ✅ |
| Spot aggregate trades | ✅ | ✅ |
| USD-M and COIN-M Klines | ✅ | ✅ |
| USD-M and COIN-M trades | ✅ | ✅ |
| USD-M and COIN-M aggregate trades | ✅ | ✅ |
| Index-price Klines | ✅ | ✅ |
| Mark-price Klines | ✅ | ✅ |
| Premium-index Klines | ✅ | ✅ |
| Futures metrics | ❌ | ✅ |
| Futures book depth | ❌ | ✅ |
| Automatic SHA-256 verification | ❌ | ✅ |
| Normalized Parquet and DuckDB query layer | ❌ | ✅ |
| Exact-range pandas DataFrames | ❌ | ✅ |

### Performance

In one cold benchmark covering 366 daily BTCUSDT Spot `1m` archives, Binance's
official Python downloader took 435.49 seconds to download ZIPs with checksums
disabled. Veldra took 24.85 seconds—about **17× faster**—while also downloading
and verifying checksums, converting to Parquet, updating its catalog, querying
the requested range, and returning 527,040 rows.

This is a sustained multi-file result, not a promise for every request. Fixed
discovery and processing costs mean the official helper can be faster for one
small file. Veldra greedily uses monthly archives for complete historical
months and daily archives for uncovered edges and recent data, while retaining
minute-level queries.

## Products and datasets

| `product` | Meaning | Supported datasets |
| --- | --- | --- |
| `spot` | Spot markets | `klines`, `trades`, `agg_trades` |
| `um` | USD-M perpetual Futures | `klines`, `trades`, `agg_trades`, `index_price_klines`, `mark_price_klines`, `premium_index_klines`, `metrics`, `book_depth` |
| `cm` | COIN-M perpetual Futures | `klines`, `trades`, `agg_trades`, `index_price_klines`, `mark_price_klines`, `premium_index_klines`, `metrics`, `book_depth` |

Kline methods accept these output intervals:

`1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`,
`1d`, `3d`, `1w`, `1mo`.

The cache stores `1m` Klines. DuckDB performs OHLC aggregation and sums each
dataset's declared volume and count fields for higher intervals.

## Create the facade

```python
from veldra import Binance

binance = Binance(
    data_dir="data/binance",
    max_workers=32,
    progress=True,
)
```

Construction is lazy: it performs no network or filesystem work. See
[Getting started](getting-started.md#create-an-exchange-service) for all
constructor options.

## Retrieval methods

Each method accepts one pair string or an ordered list. One pair returns one
DataFrame; a list returns a list of DataFrames.

| Method | Product | Additional options | Returns |
| --- | --- | --- | --- |
| `get_klines(pairs, start, end, ...)` | `spot`, `um`, `cm`; default `spot` | `interval`, `columns`, `gap_policy`, `refresh`, `offline` | Trading candles |
| `get_trades(pairs, start, end, ...)` | `spot`, `um`, `cm`; default `spot` | `columns`, `refresh`, `offline` | Individual trades |
| `get_agg_trades(pairs, start, end, ...)` | `spot`, `um`, `cm`; default `spot` | `columns`, `refresh`, `offline` | Aggregate trades |
| `get_index_price_klines(pairs, start, end, ...)` | `um` or `cm`, required | `interval`, `columns`, `gap_policy`, `refresh`, `offline` | Index-price candles |
| `get_mark_price_klines(pairs, start, end, ...)` | `um` or `cm`, required | `interval`, `columns`, `gap_policy`, `refresh`, `offline` | Mark-price candles |
| `get_premium_index_klines(pairs, start, end, ...)` | `um` or `cm`, required | `interval`, `columns`, `gap_policy`, `refresh`, `offline` | Premium-index candles |
| `get_metrics(pairs, start, end, ...)` | `um` or `cm`, required | `columns`, `refresh`, `offline` | Open-interest and long/short snapshots |
| `get_book_depth(pairs, start, end, ...)` | `um` or `cm`, required | `columns`, `refresh`, `offline` | Percentage-bucket depth snapshots |

```python
spot = binance.get_klines(
    ["BTCUSDT", "ETHUSDT"],
    "2025-01-01",
    "2025-01-07",
    interval="1h",
)

usd_m_trades = binance.get_trades(
    "BTCUSDT",
    "2025-01-01",
    "2025-01-01",
    product="um",
)

coin_m_mark = binance.get_mark_price_klines(
    "BTCUSD_PERP",
    "2025-01-01",
    "2025-01-02",
    product="cm",
    interval="5m",
)
```

## Inspection methods

| Method and arguments | Network behavior | Result |
| --- | --- | --- |
| `get_markets(*, product="spot", status=None, quote_asset=None, sort_by="symbol", limit=None, refresh=False, offline=False)` | Refreshes stale metadata unless offline | Filtered `list[Market]` |
| `find_markets(query, *, product=None, status=None, quote_asset=None, limit=10, refresh=False, offline=False)` | Refreshes stale metadata unless offline | Ranked `list[Market]` |
| `get_availability(pair, *, product, dataset, interval=None)` | Local-only | Cataloged `Availability` |
| `discover_availability(pair, start, end, *, product, dataset, interval=None, refresh=False)` | Lists only the bounded remote range | Updated `Availability` |

```python
active = binance.get_markets(
    product="spot",
    status="TRADING",
    quote_asset="USDT",
    sort_by="quote_volume",
    limit=50,
)

suggestions = binance.find_markets("BTCSUDT", product="spot", limit=3)

coverage = binance.discover_availability(
    "BTCUSDT",
    "2020-01-01",
    "2025-12-31",
    product="spot",
    dataset="klines",
    interval="1m",
)
```

Market matching ignores case and common separators. A typo is never silently
substituted; failed retrievals carry up to three suggestions in the DataFrame's
download report.

## Canonical columns

| Dataset | Product | Stored/queryable columns |
| --- | --- | --- |
| Klines | Spot | `open_time`, `open`, `high`, `low`, `close`, `volume`, `close_time`, `quote_volume`, `trade_count`, `taker_buy_base_volume`, `taker_buy_quote_volume`, `is_synthetic` |
| Klines | USD-M | `open_time`, OHLC, `base_volume`, `close_time`, `quote_volume`, `trade_count`, `taker_buy_base_volume`, `taker_buy_quote_volume`, `is_synthetic` |
| Klines | COIN-M | `open_time`, OHLC, `contract_volume`, `close_time`, `base_volume`, `trade_count`, `taker_buy_contract_volume`, `taker_buy_base_volume`, `is_synthetic` |
| Price Klines | USD-M/COIN-M | `open_time`, OHLC, `close_time`, `sample_count`, `is_synthetic` |
| Trades | Spot/USD-M | `trade_id`, `price`, `base_quantity`, `quote_quantity`, `event_time`, `buyer_is_maker` |
| Trades | COIN-M | `trade_id`, `price`, `contract_quantity`, `base_quantity`, `quote_notional`, `event_time`, `buyer_is_maker` |
| Aggregate trades | Spot/USD-M | `agg_trade_id`, `first_trade_id`, `last_trade_id`, `price`, `base_quantity`, `quote_quantity`, `event_time`, `buyer_is_maker` |
| Aggregate trades | COIN-M | Aggregate IDs plus `price`, `contract_quantity`, `base_quantity`, `quote_notional`, `event_time`, `buyer_is_maker` |
| Metrics | USD-M | `event_time`, base open interest, quote open-interest value, and four long/short ratios |
| Metrics | COIN-M | `event_time`, contract/base open interest, and four long/short ratios |
| Book depth | USD-M | `event_time`, `percentage_bucket`, `base_depth`, `quote_notional` |
| Book depth | COIN-M | `event_time`, `percentage_bucket`, `contract_depth`, `base_notional` |

Binance Spot timestamps from 2025 onward use microseconds; earlier Spot files
and Futures files use their native earlier precision. Veldra detects and
normalizes these timestamps to timezone-aware UTC values.
