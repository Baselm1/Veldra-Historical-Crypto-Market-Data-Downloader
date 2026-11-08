"""Verify Gate's Spot archive inventory and download resource parity."""

from datetime import date
from pathlib import Path

import httpx
import pytest

from veldra import Gate
from veldra.core.models import ResourceKey
from veldra.gate.connector import GateConnector


def page(prefix: str, files: list[str], token: str = "") -> str:
    """Build a namespaced S3 response for the requested objects."""
    objects = "".join(
        f"<Contents><Key>{key}</Key><ETag>0123456789abcdef0123456789abcdef</ETag></Contents>"
        for key in files
    )
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<Prefix>{prefix}</Prefix><IsTruncated>{str(bool(token)).lower()}</IsTruncated>"
        f"<NextContinuationToken>{token}</NextContinuationToken>{objects}</ListBucketResult>"
    )


def test_list_spot_archives_includes_delisted_pairs_and_overlapping_cadences(
    tmp_path: Path,
) -> None:
    """Follow a cursor and include pre-2023 monthly files overlapping a partial range."""

    def serve(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        assert params["list-type"] == "2"
        prefix = params["prefix"]
        assert prefix == "spot/candlesticks_1m/202212/"
        if "continuation-token" not in params:
            return httpx.Response(
                200, text=page(prefix, [prefix + "OLD_USDT-202212.csv.gz"], "next")
            )
        assert params["continuation-token"] == "next"
        return httpx.Response(
            200,
            text=page(
                prefix,
                [
                    prefix + "OLD_USDT-20221201.csv.gz",
                    prefix + "OLD_USDT-20221205.csv.gz",
                    prefix + "OLD_USDT-20221130.csv.gz",
                    prefix + "s3deals/OLD_USDT-202212.csv.gz",
                ],
            ),
        )

    gate = Gate(tmp_path, retries=0)
    gate._downloader.transport = httpx.MockTransport(serve)
    found = list(gate.list_spot_kline_archives(date(2022, 12, 4), date(2022, 12, 5)))
    assert [(file.symbol, file.day, file.cadence) for file in found] == [
        ("OLD_USDT", date(2022, 12, 1), "monthly"),
        ("OLD_USDT", date(2022, 12, 5), "daily"),
    ]


def test_broad_spot_discovery_keeps_download_identity_and_checksum() -> None:
    """Convert the exact cadence and symbol returned by the listing."""

    def serve(request: httpx.Request) -> httpx.Response:
        prefix = request.url.params["prefix"]
        assert prefix.endswith("/OLD_USDT-")
        keys = (
            [prefix + "202212.csv.gz", prefix + "20221205.csv.gz"]
            if "/202212/" in prefix
            else []
        )
        return httpx.Response(200, text=page(prefix, keys))

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        key = ResourceKey("gate", "spot", "klines", "OLD_USDT", "1m", cadence="monthly")
        found = GateConnector(retries=0).resources(
            client, key, date(2022, 11, 1), date(2022, 12, 31)
        )
    assert len(found) == 1
    assert found[0].url.endswith("/202212/OLD_USDT-202212.csv.gz")
    assert (
        found[0].integrity
        and found[0].integrity.expected == "0123456789abcdef0123456789abcdef"
    )


@pytest.mark.parametrize("failure", ["prefix", "cursor", "http"])
def test_failed_bucket_pages_are_not_successful_empty_listings(failure: str) -> None:
    """A failed or unsafe page propagates instead of hiding coverage."""

    def serve(request: httpx.Request) -> httpx.Response:
        prefix = request.url.params["prefix"]
        if failure == "http":
            return httpx.Response(503)
        return httpx.Response(
            200, text=page(prefix if failure == "cursor" else "bad/", [], "repeat")
        )

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        with pytest.raises((ValueError, httpx.HTTPStatusError)):
            list(
                GateConnector(retries=0).spot_kline_archives(
                    client, date(2022, 12, 1), date(2022, 12, 31), "1m"
                )
            )
