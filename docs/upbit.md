# Upbit

- Historical archive: [upbit.com/historical_data](https://www.upbit.com/historical_data/main)
- Public API reference: [docs.upbit.com](https://docs.upbit.com/kr/reference)

Veldra supports Upbit Spot Klines and individual trades from its verified
daily archives. It does not present Upbit's current REST order book as
historical data.

## Usage restriction

Upbit states that its historical market data is for personal, non-commercial
use and prohibits redistribution, processed redistribution, sale, and other
third-party or profit-generating uses. Downloading or using the data constitutes
agreement to Upbit's terms. Read the current
[Historical Market Data terms and notices](https://www.upbit.com/historical_data/main)
before using this integration.

Veldra does not redistribute Upbit data. Downloaded archives, Parquet files,
and catalog metadata remain in the user's local cache. Repository tests contain
only synthetic records.

## Support

| Upbit historical data | Public archive | Veldra |
| --- | :---: | :---: |
| Spot Klines | ✅ | ✅ |
| Spot trades | ✅ | ✅ |
| Historical order books | ❌ | ❌ |
| Digest-only SHA-256 validation | Sidecars only | ✅ |
| Concurrent multi-pair retrieval | Raw files | ✅ |
| Normalized Parquet and DuckDB queries | ❌ | ✅ |
| Exact-range pandas DataFrames | ❌ | ✅ |

Upbit publishes no Futures product in this historical archive. Current REST
order-book snapshots and recent-only REST candle/trade history are intentionally
outside this archive-backed integration.

## Create the facade

```python
from veldra import Upbit

upbit = Upbit(
    data_dir="data/upbit",
    max_workers=32,
    progress=True,
)
```

Construction is lazy and performs no network or filesystem I/O. Use
`earliest_date="all"` to permit history before Veldra's configured 2020
boundary; each market's actual archive listing still limits its usable range.

## Retrieve Klines

```python
btc = upbit.get_klines(
    "BTCUSDT",
    "2025-01-01",
    "2025-01-07",
    interval="1h",
)
```

Upbit's native symbols are quote-first. `USDT-BTC` means Bitcoin priced in
Tether, so Veldra resolves semantic `BTCUSDT` to native `USDT-BTC`. An exact
native `BTC-USDT` request remains Tether priced in Bitcoin and is never silently
reversed.

An explicit `interval="1s"` uses Upbit's one-second archives:

```python
seconds = upbit.get_klines(
    "KRW-BTC",
    "2025-01-01T00:00:00Z",
    "2025-01-01T00:10:00Z",
    interval="1s",
)
```

For every coarser request, Veldra stores native `1m` archives and queries or
resamples them to one of:

`1m`, `3m`, `5m`, `10m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`,
`12h`, `1d`, `3d`, `1w`, `1mo`.

Upbit omits candles when no trade occurred. Veldra therefore defaults to
`gap_policy="keep"` for Upbit, preserving naturally sparse source rows without
misreporting them as corrupt. Callers may explicitly request `forward`,
`backward`, `nan`, or `raise` when they need a regular grid.

Kline columns are:

```text
open_time, open, high, low, close, base_volume, quote_volume, is_synthetic
```

## Retrieve trades

```python
trades = upbit.get_trades(
    ["KRW-BTC", "USDT-BTC"],
    "2025-01-01",
    "2025-01-02",
)
```

Trade columns are:

```text
event_time, event_number, price, base_quantity, quote_quantity, side
```

`event_number` is Upbit's archive-local sequence and may restart each day; it is
not presented as a globally unique trade ID. Veldra sorts occasional source
timestamp inversions by `(event_time, event_number)` before writing Parquet.

## Structured results

The dataset methods return one DataFrame for a string pair or an ordered list
for a pair list. To receive the underlying structured report and DataFrame
together, call:

```python
result = upbit.get_results(
    "KRW-BTC",
    "2025-01-01",
    "2025-01-02",
    dataset="trades",
)
```

## Inspect markets and coverage

| Method | Network behavior | Result |
| --- | --- | --- |
| `get_markets(...)` | Refreshes stale metadata unless offline | Filtered `list[Market]` |
| `find_markets(query, ...)` | Refreshes stale metadata unless offline | Ranked `list[Market]` |
| `get_availability(pair, ...)` | Local only | Cataloged `Availability` |
| `discover_availability(pair, start, end, ...)` | Lists a bounded remote range | Updated `Availability` |

```python
markets = upbit.get_markets(
    status="TRADING",
    quote_asset="KRW",
    sort_by="quote_volume",
    limit=20,
)

matches = upbit.find_markets("BTCSUDT", limit=3)

coverage = upbit.discover_availability(
    "KRW-BTC",
    "2022-01-01",
    "2025-12-31",
    dataset="trades",
)
```

Upbit currently publishes daily archives once per day. Veldra discovers actual
files instead of assuming every market existed for the entire exchange history.
One-second candles begin later than minute candles, and trade archives begin
later still; availability is reported independently for each dataset and
physical interval.
