# Veldra

Veldra is a Python library for retrieving historical cryptocurrency market
data from multiple exchanges. It handles public archives and rate-limited
APIs, validates source integrity when verification metadata is available,
caches normalized Parquet files, queries exact ranges with DuckDB, and returns
ready-to-use pandas DataFrames.

It is designed for research, backtesting, and machine-learning datasets—not
live market streaming.

## Supported exchanges

| Exchange | Spot | Futures | Core historical data | Documentation |
| --- | :---: | :---: | --- | --- |
| Binance | ✅ | ✅ | Klines, trades, aggregate trades | [Guide](docs/binance.md) |
| Bitget | ✅ | ✅ | Klines, trades, order books | [Guide](docs/bitget.md) |
| Bybit | ✅ | ✅ | Klines, trades, order-book updates | [Guide](docs/bybit.md) |
| Gate.io | ✅ | ✅ | Klines, trades, order books | [Guide](docs/gate.md) |
| HTX | ✅ | ✅ | Klines, trades, order-book updates | [Guide](docs/htx.md) |
| KuCoin | ✅ | ✅ | Klines, trades, order books | [Guide](docs/kucoin.md) |
| OKX | ✅ | ✅ | Klines, trades, order-book updates | [Guide](docs/okx.md) |
| Upbit | ✅ | — | Klines and trades | [Guide](docs/upbit.md) |

Products, datasets, intervals, and source-specific behavior are documented in
each exchange guide.

## Why Veldra?

| Capability | Veldra | Raw exchange sources and official tools |
| --- | :---: | :---: |
| Returns exact requested ranges as DataFrames | ✅ | ❌ |
| Concurrent or rate-aware multi-pair retrieval | ✅ | Source-dependent |
| Integrity validation when published | ✅ | Manual or source-dependent |
| Normalized Parquet cache | ✅ | ❌ |
| DuckDB filtering and supported Kline resampling | ✅ | ❌ |
| Structured warnings, gaps, and pair suggestions | ✅ | ❌ |
| Reuses locally stored historical ranges | ✅ | ❌ |
| Consistent exchange-specific facades | ✅ | ❌ |

Veldra accelerates archive retrieval through concurrent downloads and uses the
maximum safe throughput for rate-limited APIs. Observed improvements range
from roughly **2–3×** for Binance monthly archives to approximately **17×** for
large daily-archive workloads. Veldra also validates available integrity
metadata, normalizes the data, caches it as Parquet, and returns query-ready
DataFrames. Results depend on the exchange, dataset, requested range, and
network conditions. The Binance benchmark and its scope are described in
[the Binance guide](docs/binance.md#performance).

Retrieval is transparent: Veldra queries matching local Parquet data when it
exists. Otherwise it obtains the missing source data, validates and stores it,
then serves the requested rows through DuckDB.

## Install

Veldra currently requires Python 3.14 or newer and is installed from a local
clone:

```bash
git clone <repository-url> veldra
cd veldra
python -m venv .venv
python -m pip install -e .
```

Activate the virtual environment before the final command if `python` does not
already point to it.

## Quick start

Each exchange has its own facade with consistent conventions and
exchange-specific datasets:

```python
from veldra import Binance, Bitget, Bybit, Gate, HTX, KuCoin, OKX, Upbit
```

For example:

```python
from veldra import Binance

binance = Binance(data_dir="data")

btc = binance.get_klines(
    "BTCUSDT",
    start="2025-01-01",
    end="2025-01-07",
    product="spot",
    interval="1h",
    columns=["open_time", "open", "high", "low", "close", "volume"],
)

print(btc.head())
print(btc.attrs["download"])
```

A string pair returns one DataFrame. A list of pairs returns a list of
DataFrames in the same order:

```python
frames = binance.get_trades(
    ["BTCUSDT", "ETHUSDT"],
    start="2025-01-01",
    end="2025-01-02",
)
```

Progress output is enabled by default. Pass `progress=False` to any exchange
facade for a silent library call. See [Getting started](docs/getting-started.md)
for shared behavior and each exchange guide for its supported methods.

## Documentation

- [Getting started](docs/getting-started.md)
- [Binance API and datasets](docs/binance.md)
- [Bitget API and datasets](docs/bitget.md)
- [Bybit API and datasets](docs/bybit.md)
- [Gate API and datasets](docs/gate.md)
- [HTX API and datasets](docs/htx.md)
- [KuCoin API and datasets](docs/kucoin.md)
- [OKX API and datasets](docs/okx.md)
- [Upbit API and datasets](docs/upbit.md)
