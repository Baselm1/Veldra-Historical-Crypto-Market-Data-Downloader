"""Download and display one day of Binance Spot candles."""

import crypto_downloader as crypto


def main() -> None:
    """Request Bitcoin Spot klines and print the result report."""
    result = crypto.get_results(
        "BTCUSDT",
        "2025-01-01",
        "2025-01-01",
        desired_columns=["open_time", "open", "high", "low", "close", "volume"],
    )
    if isinstance(result, list):
        raise RuntimeError("single-pair input unexpectedly returned multiple results")
    print(result.data.head())
    print(
        {
            "pair": result.pair,
            "complete": result.complete,
            "warnings": result.warnings,
            "problems": result.problems,
            "errors": result.errors,
        }
    )


if __name__ == "__main__":
    main()
