# Bybit

Veldra retrieves Bybit history from the
[public data archives](https://public.bybit.com/) and the
[public V5 API](https://bybit-exchange.github.io/docs/v5/intro). Daily
archives supply trades and replayable order-book events. Rate-limited API
endpoints supply candles, reference prices, funding, position statistics,
volatility, and delivery prices.

## Support

| Product | Klines | Trades | Order books | Reference Klines | Funding | Analytics |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Spot | ✅ | ✅ | ✅ | — | — | — |
| Linear contracts | ✅ | ✅ | ✅ | Mark, index, premium | ✅ | Open interest, long/short ratio, delivery |
| Inverse contracts | ✅ | ✅ | ✅ | Mark, index | ✅ | Open interest, long/short ratio, delivery |
| Options | — | ✅ | ✅ | Mark | — | Volatility, delivery |

The product names accepted by the API are `spot`, `linear`, `inverse`, and
`options`. Linear and inverse products include perpetual and dated contracts;
the current market records identify their contract type and delivery time.

Bybit does not publish checksum sidecars for these archives. Veldra validates
the downloaded archive and its records before publishing normalized Parquet.

## Use

```python
from veldra import Bybit

bybit = Bybit("data")

klines = bybit.get_klines(
    "BTCUSDT",
    "2025-01-01",
    "2025-01-07",
    product="spot",
    interval="1h",
)

trades = bybit.get_trades(
    ["BTCUSDT", "ETHUSDT"],
    "2025-01-01",
    "2025-01-02",
    product="spot",
)
```

A string pair returns one `pandas.DataFrame`. A list returns DataFrames in the
same order. API history is cached as Parquet and can be reused with
`offline=True`. Archive history is discovered, validated, normalized, and
queried through the shared DuckDB catalog.

Native Kline intervals are `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`,
`6h`, `12h`, `1d`, `1w`, and `1mo`. Bybit serves each interval directly;
Veldra does not synthesize a finer interval from coarser candles.

## Derivative history

```python
mark = bybit.get_mark_price_klines(
    "BTCUSDT", "2025-01-01", "2025-01-02", product="linear"
)
index = bybit.get_index_price_klines(
    "BTCUSDT", "2025-01-01", "2025-01-02", product="linear"
)
premium = bybit.get_premium_index_klines(
    "BTCUSDT", "2025-01-01", "2025-01-02"
)
funding = bybit.get_funding_rates(
    "BTCUSDT", "2025-01-01", "2025-01-07", product="linear"
)
interest = bybit.get_open_interest(
    "BTCUSDT", "2025-01-01", "2025-01-02", product="linear", period="1h"
)
ratios = bybit.get_long_short_ratios(
    "BTCUSDT", "2025-01-01", "2025-01-02", product="linear", period="1h"
)
```

Position-history periods are `5m`, `15m`, `30m`, `1h`, `4h`, and `1d`.
Historical volatility accepts Option base coins such as `BTC` and periods of
`7`, `14`, `21`, `30`, `60`, `90`, `180`, or `270` days:

```python
volatility = bybit.get_historical_volatility(
    "BTC", "2025-01-01", "2025-01-07", period=30
)
delivery = bybit.get_delivery_prices(
    "BTCUSDZ24", "2024-12-27", "2024-12-27", product="inverse"
)
```

Reference-price candles contain prices, not executed volume. Their returned
schema therefore contains OHLC columns without fabricated volume values.

## Order books

```python
updates = bybit.get_order_book_updates(
    "BTCUSDT", "2025-01-01", "2025-01-01", product="spot"
)
```

Order-book archives are large. Each result preserves snapshots and subsequent
deltas in source order. `bids` and `asks` remain typed lists of
`{price, quantity}` levels. A quantity of zero removes a level. The method
returns the lossless event stream; it does not pretend each delta is a full
book or reconstruct a point-in-time book automatically.

Option trade and book files are shared by contract family. Veldra filters
their rows to the exact requested instrument before publishing the result.

## Inspect markets and coverage

```python
markets = bybit.get_markets(
    product="spot", status="TRADING", sort_by="quote_volume", limit=20
)
matches = bybit.find_markets("btcusdt", product="spot", limit=3)

known = bybit.get_availability(
    "BTCUSDT", product="spot", dataset="trades"
)
remote = bybit.discover_availability(
    "BTCUSDT",
    "2025-01-01",
    "2025-01-07",
    product="spot",
    dataset="trades",
)
```

`discover_availability()` is limited to the archive-backed `trades` and
`order_book_updates` datasets. It scans a bounded date range without
downloading the data files.

Call `bybit.close()` when finished, or use `Bybit` as a context manager.
