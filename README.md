# Crypto Downloader

A Python library for downloading, verifying, caching, and querying Binance
historical market data.

This project is currently under development.

Planned usage:

```python
import crypto_downloader as crypto

frame = crypto.get_data(
    "BTCUSDT",
    "2025-01-01",
    "2025-01-07",
    product="spot",
    dataset="klines",
)
```
