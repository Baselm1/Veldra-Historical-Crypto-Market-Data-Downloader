# Gate.io

- Historical data: [gate.com/developer/historical_quotes](https://www.gate.com/developer/historical_quotes)
- Public API: [gate.com/docs/developers/apiv4](https://www.gate.com/docs/developers/apiv4/)

Veldra supports Gate Spot plus USDT- and BTC-margined perpetual Futures. It
retrieves Gate's public archives concurrently, validates plain ETags when
available, normalizes rows into Parquet, and returns exact DuckDB-filtered
pandas DataFrames.

## Support

| Gate historical data | Public source | Veldra |
| --- | :---: | :---: |
| Spot Klines and fills | ✅ | ✅ |
| Spot order-book updates and snapshots | ✅ | ✅ |
| USDT/BTC perpetual Klines and fills | ✅ | ✅ |
| Perpetual order-book updates and snapshots | ✅ | ✅ |
| Mark-price observations | ✅ | ✅ |
| Applied and projected funding rates | ✅ | ✅ |
| TradFi data | ✅ | ❌ |
| Integrity validation | ETag/Gzip | ✅ |
| Normalized Parquet and DuckDB queries | ❌ | ✅ |
| Exact-range pandas DataFrames | ❌ | ✅ |

## Products and datasets

| `product` | Meaning | Supported datasets |
| --- | --- | --- |
| `spot` | Spot markets | `klines`, `trades`, `order_book_updates`, `order_book_snapshots` |
| `um` | USDT-margined perpetuals | Spot datasets plus `mark_prices`, `funding_rates`, `funding_rate_updates` |
| `cm` | BTC-margined perpetuals | Spot datasets plus `mark_prices`, `funding_rates`, `funding_rate_updates` |

Spot Klines are daily files. Other non-order-book datasets are monthly files.
Gate splits each order-book day into as many as 24 hourly files; Veldra
discovers available hours and stores them as one logical daily Parquet file.

## Create the facade

```python
from veldra import Gate

gate = Gate(
    data_dir="data/gate",
    max_workers=32,
    progress=True,
)
```

Construction is lazy and performs no network or filesystem I/O. See
[Getting started](getting-started.md#create-an-exchange-service) for shared
constructor behavior.

## Retrieve market data

Every retrieval method accepts one market string or an ordered list. One
market returns one DataFrame; a list returns a list in the same order.

| Method | Product | Returns |
| --- | --- | --- |
| `get_klines(pairs, start, end, ...)` | All; default `spot` | Trading candles |
| `get_trades(pairs, start, end, ...)` | All; default `spot` | Individual fills |
| `get_order_book_updates(pairs, start, end, ...)` | All; default `spot` | Changed price levels |
| `get_order_book_snapshots(pairs, start, end, ...)` | All; default `spot` | Complete nested book states |
| `get_mark_prices(pairs, start, end, ...)` | `um` or `cm`, required | Index, mark, and last prices |
| `get_funding_rates(pairs, start, end, ...)` | `um` or `cm`, required | Applied funding rates |
| `get_funding_rate_updates(pairs, start, end, ...)` | `um` or `cm`, required | Expected next-interval funding updates |

```python
spot = gate.get_klines(
    ["BTC_USDT", "ETH_USDT"],
    "2025-01-01",
    "2025-01-07",
    interval="1h",
)

linear_trades = gate.get_trades(
    "BTC_USDT",
    "2025-01-01",
    "2025-01-02",
    product="um",
)

funding = gate.get_funding_rates(
    "BTC_USDT",
    "2025-01-01",
    "2025-01-31",
    product="um",
)
```

Gate native symbols contain an underscore, such as `BTC_USDT`. Veldra also
accepts normalized compact forms such as `BTCUSDT` when the market identity is
unambiguous.

## Kline intervals

Veldra stores ordinary Klines at `1m` and returns:

`1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`,
`1d`, `3d`, `1w`, `1mo`.

Gate Futures additionally publish native `10s` archives. An explicit
`interval="10s"` request uses those files. Spot does not support `10s`.
Missing source candles remain sparse by default with `gap_policy="keep"`;
callers may request another supported gap policy explicitly.

## Order books

Updates and snapshots answer different questions:

- Updates contain only changed price levels and are compact event history.
- Snapshots contain complete bid and ask ladders at each observation and are
  easier to inspect independently.

Snapshot `bids` and `asks` are nested lists of `price` and `quantity`
structures. Update rows contain `side`, `action`, `price`, and an explicit
`base_quantity` or `contract_quantity`. Veldra preserves both source datasets;
it does not currently expose a reconstructed-book state machine.

## Canonical columns

| Dataset | Product | Stored/queryable columns |
| --- | --- | --- |
| Klines | Spot | `open_time`, OHLC, `base_volume`, `is_synthetic` |
| Klines | Perpetuals | `open_time`, OHLC, `contract_volume`, `is_synthetic` |
| Trades | Spot | `event_time`, `event_number`, `price`, `base_quantity`, `quote_quantity`, `side` |
| Trades | Perpetuals | `event_time`, `event_number`, `price`, `contract_quantity`, `side` |
| Order-book updates | Spot | `event_time`, `update_id`, `side`, `action`, `price`, `base_quantity`, `merged_count` |
| Order-book updates | Perpetuals | Spot-style fields with `contract_quantity` |
| Order-book snapshots | All | `event_time`, `update_time`, `update_id`, nested `bids`, nested `asks` |
| Mark prices | Perpetuals | `event_time`, `index_price`, `mark_price`, `last_price` |
| Applied funding | Perpetuals | `event_time`, `funding_rate` |
| Funding updates | Perpetuals | `event_time`, rates, price differences, mark/index prices, `update_count` |

## Inspect markets and availability

```python
active = gate.get_markets(
    product="spot",
    status="tradable",
    quote_asset="USDT",
    sort_by="quote_volume",
    limit=20,
)

matches = gate.find_markets("BTCSUDT", product="spot", limit=3)

coverage = gate.discover_availability(
    "BTC_USDT",
    "2023-01-01",
    "2025-12-31",
    product="spot",
    dataset="klines",
    interval="1m",
)
```

`get_availability(...)` reads only the local catalog.
`discover_availability(...)` probes a bounded remote range without downloading
archive bodies. Availability is tracked independently for every product,
dataset, symbol, interval, and archive cadence.
