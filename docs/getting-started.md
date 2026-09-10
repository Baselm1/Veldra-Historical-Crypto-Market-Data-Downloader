# Getting started

Veldra is used as an imported Python library and imported as `veldra`.

## Installation

Python 3.14 or newer is required.

```bash
git clone <repository-url> veldra
cd veldra
python -m venv .venv
python -m pip install -e .
```

For development tools and tests:

```bash
python -m pip install -e ".[dev]"
```

## Create an exchange service

Constructing a facade does not access the network or create the data directory.
The first online retrieval loads market metadata, finds the pair's source
boundary, and discovers useful archives for the requested range. It does not
crawl every dataset file for every market.

```python
from veldra import Binance, HTX

binance = Binance(data_dir="data/binance")
htx = HTX(data_dir="data/htx", progress=False)
```

Both constructors accept the same options:

| Argument | Default | Meaning |
| --- | --- | --- |
| `data_dir` | `"data"` | DuckDB catalog and Parquet cache root |
| `config_path` | `None` | Optional TOML settings file |
| `earliest_date` | `None` | Override the configured history boundary; use `"all"` for full source history |
| `max_workers` | `32` | Exchange-wide archive worker ceiling |
| `discovery_tail_days` | `7` | Recent active-market days eligible for repeated discovery |
| `market_refresh_hours` | `24.0` | Market metadata cache lifetime |
| `timeout` | `30.0` | Timeout for each HTTP attempt, in seconds |
| `retries` | `3` | Retries after the first HTTP attempt |
| `backoff` | `0.5` | Initial exponential retry delay, in seconds |
| `progress` | `True` | Show Rich status and download progress |

Each facade also exposes four read-only properties:

| Property | Value |
| --- | --- |
| `data_dir` | Resolved catalog and Parquet root |
| `earliest_date` | Configured UTC history boundary, or `None` for all history |
| `kline_base_interval` | Kline archive interval stored in the cache |
| `max_workers` | Exchange-wide worker ceiling |

## Retrieve data

```python
frame = binance.get_klines(
    "BTCUSDT",
    start="2025-01-01",
    end="2025-01-03",
    interval="5m",
)
```

Common retrieval arguments are:

| Argument | Meaning |
| --- | --- |
| `pairs` | One pair string or an ordered list of pair strings |
| `start` | Inclusive `date`, `datetime`, or ISO string |
| `end` | Inclusive when date-only; exclusive when an exact datetime |
| `product` | Exchange-specific Spot or perpetual product |
| `interval` | Kline output interval; omitted for event and snapshot datasets |
| `columns` | Canonical column list, or `{canonical_name: output_name}` mapping |
| `gap_policy` | Kline-only internal-gap behavior |
| `refresh` | Repeat remote discovery and revalidate cached source archives |
| `offline` | Forbid network access and use only cataloged, cached data |

`refresh=True` and `offline=True` cannot be combined.

Date-only requests cover whole UTC days. This request includes January 1–3 and
uses January 4 at midnight as its exclusive query boundary:

```python
frame = binance.get_klines("BTCUSDT", "2025-01-01", "2025-01-03")
```

Exact datetime boundaries remain exact and the end is exclusive:

```python
frame = binance.get_klines(
    "BTCUSDT",
    "2025-01-01T08:30:00+00:00",
    "2025-01-01T10:00:00+00:00",
)
```

Naive datetimes are interpreted as UTC. Timezone-aware datetimes are converted
to UTC without changing their instant.

## Return values

A string pair returns one `pandas.DataFrame`. A list returns a list in the same
order, and one failed pair does not prevent other pairs from completing.

```python
frames = binance.get_klines(
    ["BTCUSDT", "ETHUSDT", "BTCSUDT"],
    "2025-01-01",
    "2025-01-01",
)

for frame in frames:
    report = frame.attrs["download"]
    print(report["pair"], report["complete"], report["errors"])
```

Each returned frame stores a JSON-safe report in `frame.attrs["download"]`:

| Field | Meaning |
| --- | --- |
| `pair`, `source`, `product`, `dataset` | Resolved request identity |
| `requested_range` | Original normalized half-open UTC range |
| `available_range` | Known source archive range |
| `used_range` | Range actually queried after safe trimming |
| `complete` | Whether a usable range exists with no problems or errors |
| `warnings` | Adjustments such as configured or source range trimming |
| `problems` | Incomplete source coverage or failed resources |
| `errors` | Pair, discovery, download, or query failures |
| `gaps` | Exact internal missing-candle ranges |
| `gap_policy` | Kline gap policy used for the result |

Unknown pairs return an empty, correctly typed DataFrame with up to three
high-confidence suggestions in the report. Invalid call-level arguments, such
as an unsupported product or reversed timestamps, raise `TypeError` or
`ValueError` before downloading.

## Missing Kline candles

Missing rows are filled only inside real source coverage. Veldra never invents
rows before a listing begins, after a delisting, or for a completely unavailable
day.

| Policy | Behavior |
| --- | --- |
| `"forward"` | Carry the previous close forward with zero activity |
| `"backward"` | Carry the next open backward with zero activity |
| `"nan"` | Insert timestamped rows with missing values |
| `"keep"` | Return only source rows |
| `"raise"` | Raise `MissingCandlesError` |

Synthetic rows are identified by `is_synthetic`. A result with source gaps
remains incomplete even when a fill policy makes the DataFrame regular.

## Inspect markets and availability

```python
markets = binance.get_markets(
    product="spot",
    status="TRADING",
    quote_asset="USDT",
    sort_by="quote_volume",
    limit=20,
)

matches = binance.find_markets("BTCSUDT", product="spot", limit=3)

known = binance.get_availability(
    "BTCUSDT",
    product="spot",
    dataset="klines",
    interval="1h",
)

discovered = binance.discover_availability(
    "BTCUSDT",
    "2024-01-01",
    "2024-12-31",
    product="spot",
    dataset="klines",
    interval="1h",
)
```

`get_availability()` is local-only and requires cached market metadata.
`discover_availability()` lists the bounded remote range but does not download
market archives.

## Configuration

The default configuration is equivalent to:

```toml
[history]
earliest_date = "2020-01-01"

[klines]
base_interval = "1m"
```

Use `earliest_date = "all"` to permit each pair's full available archive
history. The source's actual listing date still bounds every request. Klines
are currently stored at `1m`; higher supported intervals are produced by
DuckDB when queried.
