"""Test end-to-end Binance Spot trade-family downloads."""

from crypto_downloader.binance.datasets import get_dataset


from datetime import date
import hashlib
from io import BytesIO
from pathlib import Path
import zipfile

import httpx
import pandas as pd
import pytest

from crypto_downloader._core.datasets import DatasetSpec
from crypto_downloader.binance.datasets import SPOT_AGG_TRADES, SPOT_TRADES
from crypto_downloader._core.engine import Downloader
from crypto_downloader.binance.connector import BinanceConnector

FIXTURES = Path(__file__).parent / "fixtures"


def listing(*, keys: tuple[str, ...] = (), prefixes: tuple[str, ...] = ()) -> str:
    """Build one complete S3-style XML listing response.

    Args:
        keys: Object keys included in the listing.
        prefixes: Folder prefixes included in the listing.

    Returns:
        The XML listing body.
    """
    objects = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    folders = "".join(
        f"<CommonPrefixes><Prefix>{prefix}</Prefix></CommonPrefixes>"
        for prefix in prefixes
    )
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>false</IsTruncated>{objects}{folders}</ListBucketResult>"
    )


class EventServer:
    """Serve one deterministic Spot event archive and exchange metadata."""

    def __init__(self, dataset: DatasetSpec, fixture: str, day: date) -> None:
        """Build the archive payload and matching bucket object key.

        Args:
            dataset: The declared Spot event dataset.
            fixture: The representative headerless CSV fixture.
            day: The UTC archive date.
        """
        self.dataset = dataset
        self.day = day
        self.archive_name = f"BTCUSDT-{dataset.remote_name}-{day.isoformat()}.zip"
        self.object_key = (
            f"data/spot/daily/{dataset.remote_name}/BTCUSDT/{self.archive_name}"
        )
        output = BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                self.archive_name.removesuffix(".zip") + ".csv",
                (FIXTURES / fixture).read_bytes(),
            )
        self.payload = output.getvalue()
        self.archive_requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return the matching mocked metadata, listing, checksum, or archive.

        Args:
            request: The outgoing HTTP request.

        Returns:
            A deterministic response for the requested Binance endpoint.
        """
        if request.url.host == "api.binance.com":
            return httpx.Response(
                200,
                json={
                    "symbols": [
                        {
                            "symbol": "BTCUSDT",
                            "baseAsset": "BTC",
                            "quoteAsset": "USDT",
                            "status": "TRADING",
                        }
                    ]
                },
            )
        if request.url.host == "s3-ap-northeast-1.amazonaws.com":
            prefix = request.url.params["prefix"]
            if request.url.params.get("delimiter") == "/":
                return httpx.Response(
                    200,
                    text=listing(prefixes=("data/spot/daily/klines/BTCUSDT/",)),
                )
            if prefix != f"data/spot/daily/{self.dataset.remote_name}/BTCUSDT/":
                raise AssertionError(f"unexpected archive prefix: {prefix}")
            return httpx.Response(200, text=listing(keys=(self.object_key,)))
        if str(request.url).endswith(".CHECKSUM"):
            digest = hashlib.sha256(self.payload).hexdigest()
            return httpx.Response(200, text=f"{digest}  {self.archive_name}\n")
        if request.url.host == "data.binance.vision":
            self.archive_requests += 1
            return httpx.Response(200, content=self.payload)
        raise AssertionError(f"unexpected request: {request.url}")


@pytest.mark.parametrize(
    ("dataset", "fixture", "day", "id_column"),
    [
        (
            SPOT_TRADES,
            "binance_spot_trades_2024-01-01.csv",
            date(2024, 1, 1),
            "trade_id",
        ),
        (
            SPOT_AGG_TRADES,
            "binance_spot_agg_trades_2025-01-01.csv",
            date(2025, 1, 1),
            "agg_trade_id",
        ),
    ],
)
def test_spot_event_pipeline_downloads_queries_and_reuses_cached_parquet(
    dataset: DatasetSpec,
    fixture: str,
    day: date,
    id_column: str,
    tmp_path: Path,
) -> None:
    """Confirm Spot event requests complete without candle-only behavior.

    Args:
        dataset: The declared Spot trade-family dataset.
        fixture: The source CSV fixture returned by the mocked archive.
        day: The requested UTC archive date.
        id_column: The canonical event identifier column.
        tmp_path: The isolated downloader data directory.
    """
    server = EventServer(dataset, fixture, day)
    downloader = Downloader(
        tmp_path,
        source=BinanceConnector(retries=0),
        transport=httpx.MockTransport(server),
        dataset_resolver=get_dataset,
    )

    first = downloader.get_results(
        "BTCUSDT",
        day,
        day,
        dataset=dataset.name,
        gap_policy=None,
    )
    second = downloader.get_results(
        "BTCUSDT",
        day,
        day,
        dataset=dataset.name,
        gap_policy=None,
    )

    assert first.complete is True
    assert first.gap_policy is None
    assert first.gaps == []
    assert first.problems == []
    assert tuple(first.data.columns) == dataset.stored_columns
    assert first.data[id_column].tolist() == sorted(first.data[id_column].tolist())
    assert first.data["event_time"].dt.date.unique().tolist() == [day]
    pd.testing.assert_frame_equal(first.data, second.data)
    assert server.archive_requests == 1
