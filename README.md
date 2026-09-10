# Veldra

Veldra is a high-throughput Python API for downloading and using historical
cryptocurrency market data. Ask for a market and time range; Veldra discovers
the archives, verifies their SHA-256 checksums, caches normalized Parquet files,
queries the exact rows with DuckDB, and returns ready-to-use pandas DataFrames.

It is designed for research, backtesting, and machine-learning datasets—not
live market streaming.

## Supported exchanges

| Exchange | Status | Implemented products |
| --- | --- | --- |
| Binance | ✅ Supported | Spot, USD-M perpetuals, COIN-M perpetuals |
| HTX | ✅ Supported | Spot, USDT-margined swaps, coin-margined swaps |
| KuCoin | ⏳ Planned | — |
| OKX | ⏳ Planned | — |
| Upbit | ⏳ Planned | — |
| Bybit | ⏳ Planned | — |
| Gate.io | ⏳ Planned | — |

See [Binance support](docs/binance.md) and [HTX support](docs/htx.md) for the
implemented datasets and methods.

## Why Veldra?

| Capability | Veldra | Official archive helpers |
| --- | :---: | :---: |
| Returns exact requested ranges as DataFrames | ✅ | ❌ |
| Concurrent multi-pair retrieval | ✅ | Limited |
| Automatic SHA-256 verification | ✅ | ❌ |
| Normalized Parquet cache | ✅ | ❌ |
| DuckDB filtering and Kline resampling | ✅ | ❌ |
| Structured warnings, gaps, and pair suggestions | ✅ | ❌ |
| Repeat queries without network access | ✅ | ❌ |
| One API shape across exchanges | ✅ | ❌ |

In a cold one-year Binance Spot `1m` benchmark, Veldra completed the full
download-to-DataFrame pipeline about **17× faster** than Binance's sequential
Python daily-file downloader. Results vary by range, archive cadence, dataset,
network, and cache state; single-file requests can favor the smaller official
helper. The benchmark and scope are described in
[the Binance guide](docs/binance.md#performance).

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

Progress output is enabled by default. Use `Binance(progress=False)` or
`HTX(progress=False)` for a silent library call.

## Documentation

- [Getting started](docs/getting-started.md)
- [Binance API and datasets](docs/binance.md)
- [HTX API and datasets](docs/htx.md)
