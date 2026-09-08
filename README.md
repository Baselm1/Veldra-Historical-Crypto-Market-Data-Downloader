# Crypto Downloader

A Python library for downloading, verifying, caching, and querying Binance
historical market data.

This project is currently under development.

Usage:

```python
from crypto_downloader import Binance

binance = Binance(data_dir="data")
frame = binance.get_klines(
    "BTCUSDT",
    start="2025-01-01",
    end="2025-01-07",
    product="spot",
    interval="1h",
)
```
