# Bitget

Veldra retrieves Bitget history from the [daily download portal](https://www.bitget.com/data-download) and the [public market API](https://www.bitget.com/api-doc/). It keeps archive requests concurrent while limiting portal metadata calls and REST history to their separate published quotas.

## Support

| Product | Klines | Trades | Best book | Depth 500 | Mark/index/premium Klines | Funding |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Spot | ✅ | ✅ | ✅ | ✅ | — | — |
| USDT Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| USDC Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Coin Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Archive dates use Bitget's UTC+8 calendar. Veldra converts timestamps to UTC and returns the exact requested range. Files with a plain CDN MD5 ETag are checked against it before they become Parquet. Large depth-500 files use multipart ETags, so Veldra instead validates their ZIP structure and member CRC. Trade days split across several files are merged into one logical daily partition.

## Use

```python
from veldra import Bitget

bitget = Bitget("data")

klines = bitget.get_klines(
    "BTCUSDT", "2025-01-01", "2025-01-07", product="spot", interval="1h"
)
trades = bitget.get_trades(
    ["BTCUSDT", "ETHUSDT"], "2025-01-01", "2025-01-02"
)
```

Order-book archives expose two different complete snapshots:

```python
best = bitget.get_best_book_snapshots(
    "BTCUSDT", "2025-01-01", "2025-01-01"
)
deep = bitget.get_order_book_snapshots(
    "BTCUSDT", "2025-01-01", "2025-01-01"
)
```

`best` has flat bid/ask prices and quantities. `deep` stores each side as a typed list of `{price, quantity}` levels; it is not a diff stream.

Futures reference data comes from rate-limited public REST endpoints. Mark, index, and premium candles contain OHLC prices; Bitget fills both volume fields with zero because these are reference prices rather than executed trades. Native intervals are `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `4h`, `6h`, `12h`, and `1d`. These reference calls use Bitget's API each time; offline reuse currently applies to the daily archive methods.

```python
from datetime import UTC, datetime, timedelta

end = datetime.now(UTC)
start = end - timedelta(days=1)

mark = bitget.get_mark_price_klines(
    "BTCUSDT", start, end, product="usdt_futures"
)
funding = bitget.get_funding_rates(
    "BTCUSDT", end - timedelta(days=7), end, product="usdt_futures"
)
```

Markets can be listed or searched without downloading history:

```python
active = bitget.get_markets(
    product="spot", status="ONLINE", sort_by="quote_volume", limit=20
)
matches = bitget.find_markets("btcusdt", product="spot")
```

Use `get_availability(...)` for coverage already in the local catalog and `discover_availability(...)` to scan a bounded archive range without downloading it.

Call `bitget.close()` when the facade is no longer needed.
