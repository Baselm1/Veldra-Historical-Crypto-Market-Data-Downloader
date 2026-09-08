"""Download a small Spot Kline range through the public Binance facade."""

from crypto_downloader import Binance


def main() -> None:
    """Run one small imported-library example."""
    binance = Binance(data_dir="data")
    frames = binance.get_klines(
        ["BTCUSDT", "ETHUSDT", "NOTREALPAIR"],
        start="2025-01-01",
        end="2025-01-03",
        interval="1h",
        columns=["open_time", "open", "high", "low", "close", "volume"],
    )
    for frame in frames:
        report = frame.attrs["download"]
        print(f"{report['pair']}: {len(frame):,} rows, complete={report['complete']}")
        print(frame.head())
        print(report)


if __name__ == "__main__":
    main()
