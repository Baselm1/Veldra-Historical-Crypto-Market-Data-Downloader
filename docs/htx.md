# HTX

- Public archive: [htx.com/vision](https://www.htx.com/vision/)
- Official helpers: [hbdmapi/huobi_public_data](https://github.com/hbdmapi/huobi_public_data/)

Veldra supports HTX Spot plus USDT- and coin-margined perpetual swaps. It
combines the legacy Huobi archive namespace with the newer HTX namespace into
one sparse source view.

## Comparison

The official repository was audited at revision `aae97ba` (2021-06-17). It
targets the legacy `futures.huobi.com/data` archive and provides sequential
Kline and trade download loops, including daily Python entry points and daily
or monthly shell helpers. It saves ZIPs and checksum sidecars but does not
verify the digest, normalize the schemas, or return queryable data.

| HTX archive data | Official helpers | Veldra |
| --- | :---: | :---: |
| Spot Klines | ✅ | ✅ |
| Spot trades | ✅ | ✅ |
| Spot order-book updates | ❌ | ✅ |
| USDT/coin perpetual Klines | ✅ | ✅ |
| USDT/coin perpetual trades | ✅ | ✅ |
| Index-price Klines | ❌ | ✅ |
| Mark-price Klines | ❌ | ✅ |
| USDT perpetual funding rates | ❌ | ✅ |
| Perpetual order-book updates | ❌ | ✅ |
| Dated Futures and Options | ✅ | ❌ |
| Old and new archive namespaces | Legacy only | ✅ |
| Normalized Parquet and DuckDB query layer | ❌ | ✅ |
| Exact-range pandas DataFrames | ❌ | ✅ |

The current Veldra integration uses daily files from HTX's two live archive
namespaces. It retrieves independent files concurrently, verifies them, and
makes them directly queryable.

## Products and datasets

| `product` | Meaning | Supported datasets |
| --- | --- | --- |
| `spot` | Spot markets | `klines`, `trades`, `order_book_updates` |
| `linear_swap` | USDT-margined perpetual swaps | `klines`, `trades`, `index_price_klines`, `mark_price_klines`, `funding_rates`, `order_book_updates` |
| `coin_swap` | Coin-margined perpetual swaps | `klines`, `trades`, `index_price_klines`, `mark_price_klines`, `order_book_updates` |

HTX's native archive Kline intervals are `1m`, `5m`, `15m`, `30m`, `1h`,
`4h`, and `1d`. Veldra stores `1m` and can return:

`1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`,
`1d`, `3d`, `1w`, `1mo`.

## Create the facade

```python
from veldra import HTX

htx = HTX(
    data_dir="data/htx",
    max_workers=32,
    progress=True,
)
```

Construction is lazy. See
[Getting started](getting-started.md#create-an-exchange-service) for all shared
constructor options.

## Retrieval methods

Each method accepts one pair string or an ordered list. One pair returns one
DataFrame; a list returns a list of DataFrames.

| Method | Product | Additional options | Returns |
| --- | --- | --- | --- |
| `get_klines(pairs, start, end, ...)` | `spot`, `linear_swap`, `coin_swap`; default `spot` | `interval`, `columns`, `gap_policy`, `refresh`, `offline` | Trading candles |
| `get_trades(pairs, start, end, ...)` | `spot`, `linear_swap`, `coin_swap`; default `spot` | `columns`, `refresh`, `offline` | Individual trades |
| `get_index_price_klines(pairs, start, end, ...)` | `linear_swap` or `coin_swap`, required | `interval`, `columns`, `gap_policy`, `refresh`, `offline` | Index-price candles |
| `get_mark_price_klines(pairs, start, end, ...)` | `linear_swap` or `coin_swap`, required | `interval`, `columns`, `gap_policy`, `refresh`, `offline` | Mark-price candles |
| `get_funding_rates(pairs, start, end, ...)` | Fixed to `linear_swap` | `columns`, `refresh`, `offline` | Funding-rate observations |
| `get_order_book_updates(pairs, start, end, ...)` | `spot`, `linear_swap`, `coin_swap`; default `spot` | `columns`, `refresh`, `offline` | Flattened snapshots and level updates |

```python
spot = htx.get_klines(
    "BTCUSDT",
    "2025-01-01",
    "2025-01-07",
    interval="1h",
)

linear_trades = htx.get_trades(
    "BTC-USDT",
    "2025-01-01",
    "2025-01-01",
    product="linear_swap",
)

coin_index = htx.get_index_price_klines(
    "BTC-USD",
    "2026-01-01",
    "2026-01-02",
    product="coin_swap",
    interval="5m",
)

funding = htx.get_funding_rates(
    "BTC-USDT",
    "2026-01-01",
    "2026-01-07",
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
markets = htx.get_markets(
    product="spot",
    status="online",
    quote_asset="USDT",
    sort_by="quote_volume",
    limit=50,
)

matches = htx.find_markets("BTCSUDT", product="spot", limit=3)

coverage = htx.discover_availability(
    "BTCUSDT",
    "2020-01-01",
    "2025-12-31",
    product="spot",
    dataset="klines",
    interval="1m",
)
```

## Canonical columns

| Dataset | Product | Stored/queryable columns |
| --- | --- | --- |
| Klines | Spot | `open_time`, `open`, `high`, `low`, `close`, `base_volume`, `quote_volume`, `is_synthetic` |
| Klines | Perpetuals | `open_time`, OHLC, `contract_volume`, `base_volume`, `is_synthetic` |
| Reference-price Klines | Perpetuals | `open_time`, OHLC, `sample_count`, `is_synthetic` |
| Trades | Spot | `event_time`, `trade_id`, `price`, `base_quantity`, `quote_quantity`, `side` |
| Trades | USDT perpetual | `event_time`, `trade_id`, `price`, `contract_quantity`, `base_quantity`, `quote_quantity`, `side` |
| Trades | Coin perpetual | `event_time`, `trade_id`, `price`, `contract_quantity`, `base_quantity`, `quote_notional`, `side` |
| Funding rates | USDT perpetual | `funding_time`, `funding_rate` |
| Order-book updates | All products | `event_time`, `event_number`, `action`, `side`, `level_number`, `price`, `quantity` |

## Archive behavior

HTX has two live archive generations under the same public portal:

- `data/...` contains the long-running Huobi Kline and trade history.
- `historical_data/...` contains newer HTX schemas and additional datasets.

Veldra discovers both, prefers the newer archive when the same day overlaps,
and preserves genuine gaps when neither tree provides a file. It accepts both
old and new CSV schemas. Order books are TAR.GZ archives containing JSON Lines;
Veldra streams and flattens their snapshots and incremental updates.

HTX archive filenames follow UTC+8 days. Veldra records each archive's exact
UTC coverage, then DuckDB applies the caller's exact UTC timestamps. Returned
timestamp columns are timezone-aware UTC values.
