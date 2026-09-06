"""Download a small Spot kline range and render its structured results."""

import crypto_downloader as crypto


def main() -> None:
    """Run one small imported-library example."""
    results = crypto.get_results(
        ["BTCUSDT", "ETHUSDT", "NOTREALPAIR"],
        "2025-01-01",
        "2025-01-03",
        interval="1h",
        desired_columns=["open_time", "open", "high", "low", "close", "volume"],
    )
    crypto.render_results(results)


if __name__ == "__main__":
    main()
