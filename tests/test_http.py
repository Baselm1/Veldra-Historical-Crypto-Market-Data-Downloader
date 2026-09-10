"""Test retrying HTTP requests and verified streamed downloads."""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from email.utils import format_datetime
import hashlib
from pathlib import Path
import time
from urllib.parse import quote

import httpx
import pytest

from veldra.core.download import (
    ChecksumError,
    DownloadSizeError,
    archive_checksum,
    download,
    get,
    retry_delay,
)
from veldra.core.models import Resource


class Chunks(httpx.SyncByteStream):
    """Yield predefined byte chunks through an HTTPX response stream."""

    def __init__(self, *chunks: bytes) -> None:
        """Store chunks for later iteration.

        Args:
            chunks: The byte chunks emitted by the stream.
        """
        self.chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        """Yield the stored chunks in order.

        Yields:
            One stored byte chunk at a time.
        """
        yield from self.chunks


class InterruptedStream(httpx.SyncByteStream):
    """Yield one chunk and then simulate a dropped connection."""

    def __iter__(self) -> Iterator[bytes]:
        """Yield bytes before raising a transport error.

        Yields:
            The bytes received before the connection fails.
        """
        yield b"partial"
        raise httpx.ReadError("connection dropped")


def daily_resource(name: str = "BTCUSDT-1m-2025-01-01.zip") -> Resource:
    """Create a daily resource with matching archive and sidecar URLs.

    Args:
        name: The archive filename placed at the end of the URL.

    Returns:
        A resource suitable for download tests.
    """
    url = f"https://data.example/{quote(name)}"
    return Resource(date(2025, 1, 1), url, f"{url}.CHECKSUM")


def sidecar(contents: bytes, name: str = "BTCUSDT-1m-2025-01-01.zip") -> str:
    """Create a Binance-style SHA-256 sidecar line.

    Args:
        contents: The archive bytes covered by the digest.
        name: The archive filename written beside the digest.

    Returns:
        A complete checksum sidecar line.
    """
    return f"{hashlib.sha256(contents).hexdigest()}  {name}\n"


@pytest.mark.parametrize(
    ("attempt", "backoff", "expected"),
    [(0, 0.5, 0.5), (1, 0.5, 1.0), (4, 0.25, 4.0), (10_000, 1.0, 30.0)],
)
def test_retry_delay_uses_capped_exponential_backoff(
    attempt: int, backoff: float, expected: float
) -> None:
    """Confirm ordinary retries use deterministic exponential delays.

    Args:
        attempt: The zero-based retry number.
        backoff: The initial retry delay.
        expected: The expected delay after applying the cap.
    """
    assert retry_delay(None, attempt, backoff) == expected


@pytest.mark.parametrize(
    ("header", "expected"),
    [("0", 0.0), ("2.5", 2.5), ("999", 60.0)],
)
def test_retry_delay_prefers_numeric_retry_after(header: str, expected: float) -> None:
    """Confirm valid numeric Retry-After values override backoff.

    Args:
        header: The server-provided Retry-After value.
        expected: The delay after applying the server cap.
    """
    response = httpx.Response(429, headers={"Retry-After": header})

    assert retry_delay(response, 3, 10.0) == expected


def test_retry_delay_accepts_http_date_and_uses_utc() -> None:
    """Confirm an HTTP-date Retry-After value is measured from UTC now."""
    now = datetime(2025, 1, 1, tzinfo=UTC)
    header = format_datetime(now + timedelta(seconds=12), usegmt=True)
    response = httpx.Response(503, headers={"Retry-After": header})

    assert retry_delay(response, 0, 0.5, now=now) == 12.0


@pytest.mark.parametrize(
    "header", ["bad", "nan", "inf", "-1", "Wed, 01 Jan 2025 00:00:12"]
)
def test_invalid_retry_after_falls_back_to_exponential_delay(header: str) -> None:
    """Confirm malformed Retry-After values do not bypass normal backoff.

    Args:
        header: An unusable server delay.
    """
    response = httpx.Response(429, headers={"Retry-After": header})

    assert retry_delay(response, 2, 0.25) == 1.0


@pytest.mark.parametrize(("attempt", "backoff"), [(-1, 1.0), (0, -1.0)])
def test_retry_delay_rejects_negative_inputs(attempt: int, backoff: float) -> None:
    """Confirm retry arithmetic cannot receive negative settings.

    Args:
        attempt: The proposed retry number.
        backoff: The proposed initial delay.
    """
    with pytest.raises(ValueError, match="retry"):
        retry_delay(None, attempt, backoff)


def test_get_sends_params_and_timeout() -> None:
    """Confirm metadata GET requests pass query values and bounded timeouts."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record one request and return a successful response."""
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = get(
            client,
            "https://data.example/list",
            params={"prefix": "data/spot"},
            timeout=12.5,
            retries=0,
        )

    assert response.json() == {"ok": True}
    assert requests[0].url.params["prefix"] == "data/spot"
    assert set(requests[0].extensions["timeout"].values()) == {12.5}


@pytest.mark.parametrize("failure", ["transport", "408", "429", "500", "503"])
def test_get_retries_temporary_failures_then_succeeds(failure: str) -> None:
    """Confirm transport and retryable status failures consume retry attempts.

    Args:
        failure: The temporary failure returned for the first two attempts.
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Fail twice and then return success."""
        nonlocal calls
        calls += 1
        if calls < 3:
            if failure == "transport":
                raise httpx.ConnectError("offline", request=request)
            return httpx.Response(int(failure))
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert (
            get(client, "https://data.example", retries=2, backoff=0).status_code == 200
        )

    assert calls == 3


def test_get_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm a server Retry-After value controls the actual sleep."""
    waits: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one throttling response followed by success."""
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200)

    monkeypatch.setattr(time, "sleep", waits.append)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        get(client, "https://data.example", retries=1)

    assert waits == [2.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 501])
def test_get_does_not_retry_permanent_http_failures(status: int) -> None:
    """Confirm permanent HTTP responses fail after one request.

    Args:
        status: A response status that is not safe to retry.
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Count and return one permanent failure."""
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            get(client, "https://data.example", retries=3, backoff=0)

    assert calls == 1


def test_get_raises_the_last_failure_when_retries_are_exhausted() -> None:
    """Confirm bounded retries stop and expose the final HTTP failure."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Count and return a retryable failure."""
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            get(client, "https://data.example", retries=2, backoff=0)

    assert calls == 3


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"timeout": 0}, "timeout"),
        ({"retries": -1}, "retries"),
        ({"backoff": -0.1}, "backoff"),
    ],
)
def test_get_rejects_invalid_retry_settings(
    options: dict[str, float | int], message: str
) -> None:
    """Confirm invalid HTTP settings fail before making a request.

    Args:
        options: The invalid keyword argument supplied to ``get``.
        message: The invalid setting named in the expected error.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        """Return success if validation unexpectedly permits a request."""
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match=message):
            get(client, "https://data.example", **options)  # type: ignore[arg-type]


@pytest.mark.parametrize("separator", ["  ", " *"])
def test_download_streams_verifies_and_atomically_replaces_archive(
    tmp_path: Path, separator: str
) -> None:
    """Confirm valid chunks replace the destination only after verification.

    Args:
        tmp_path: The temporary download directory.
        separator: A supported checksum filename marker.
    """
    contents = b"first-second"
    digest = hashlib.sha256(contents).hexdigest()
    item = daily_resource()
    destination = tmp_path / "nested" / "archive.zip"
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a checksum or a multi-chunk archive stream."""
        calls.append(request.url.path)
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(
                200,
                text=f"{digest.upper()}{separator}BTCUSDT-1m-2025-01-01.zip\n",
            )
        return httpx.Response(200, stream=Chunks(b"first-", b"second"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        actual = download(client, item, destination, retries=0)

    assert actual == digest
    assert destination.read_bytes() == contents
    assert not destination.with_name("archive.zip.part").exists()
    assert calls == [
        "/BTCUSDT-1m-2025-01-01.zip.CHECKSUM",
        "/BTCUSDT-1m-2025-01-01.zip",
    ]


def test_download_accepts_url_encoded_unicode_filename(tmp_path: Path) -> None:
    """Confirm checksum filename matching decodes URL-encoded symbols."""
    name = "币安USDT-1m-2025-01-01.zip"
    contents = b"archive"
    item = daily_resource(name)

    def handler(request: httpx.Request) -> httpx.Response:
        """Return matching Unicode checksum and archive responses."""
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=sidecar(contents, name))
        return httpx.Response(200, content=contents)

    destination = tmp_path / "unicode.zip"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        download(client, item, destination, retries=0)

    assert destination.read_bytes() == contents


@pytest.mark.parametrize(
    "value",
    [
        "bad",
        f"{'a' * 64}  wrong.zip",
        f"{'a' * 64}  ../BTCUSDT-1m-2025-01-01.zip",
        f"{'a' * 63}  BTCUSDT-1m-2025-01-01.zip",
    ],
)
def test_invalid_checksum_sidecar_never_starts_archive_download(
    tmp_path: Path, value: str
) -> None:
    """Confirm malformed or mismatched sidecars are permanent failures.

    Args:
        tmp_path: The temporary download directory.
        value: The invalid sidecar body.
    """
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record calls and always return the invalid sidecar."""
        calls.append(request.url.path)
        return httpx.Response(200, text=value)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="sidecar"):
            download(client, daily_resource(), tmp_path / "archive.zip", retries=3)

    assert calls == ["/BTCUSDT-1m-2025-01-01.zip.CHECKSUM"]


@pytest.mark.parametrize("failure", ["checksum", "archive_status", "sidecar_status"])
def test_download_retries_temporary_failures_and_refetches_sidecar(
    tmp_path: Path, failure: str
) -> None:
    """Confirm each archive retry obtains a fresh checksum sidecar.

    Args:
        tmp_path: The temporary download directory.
        failure: The temporary first-attempt failure.
    """
    contents = b"valid archive"
    checksum_calls = 0
    archive_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Fail one sidecar or archive attempt before succeeding."""
        nonlocal checksum_calls, archive_calls
        if request.url.path.endswith(".CHECKSUM"):
            checksum_calls += 1
            if failure == "sidecar_status" and checksum_calls == 1:
                return httpx.Response(503)
            return httpx.Response(200, text=sidecar(contents))
        archive_calls += 1
        if archive_calls == 1:
            if failure == "archive_status":
                return httpx.Response(503)
            if failure == "checksum":
                return httpx.Response(200, content=b"corrupt")
        return httpx.Response(200, content=contents)

    destination = tmp_path / "archive.zip"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        download(client, daily_resource(), destination, retries=1, backoff=0)

    assert checksum_calls == 2
    assert archive_calls == (1 if failure == "sidecar_status" else 2)
    assert destination.read_bytes() == contents


def test_interrupted_stream_is_cleaned_up_and_retried(tmp_path: Path) -> None:
    """Confirm a dropped stream leaves no partial bytes before retrying."""
    contents = b"complete"
    archive_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Interrupt the first archive stream and complete the second."""
        nonlocal archive_calls
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=sidecar(contents))
        archive_calls += 1
        if archive_calls == 1:
            return httpx.Response(200, stream=InterruptedStream())
        return httpx.Response(200, content=contents)

    destination = tmp_path / "archive.zip"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        download(client, daily_resource(), destination, retries=1, backoff=0)

    assert destination.read_bytes() == contents
    assert not destination.with_name("archive.zip.part").exists()


def test_final_checksum_failure_preserves_existing_file_and_removes_partial(
    tmp_path: Path,
) -> None:
    """Confirm exhausted verification retries cannot damage a prior archive."""
    destination = tmp_path / "archive.zip"
    destination.write_bytes(b"existing")

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a valid sidecar for bytes that never arrive."""
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=sidecar(b"expected"))
        return httpx.Response(200, content=b"corrupt")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ChecksumError, match="SHA-256"):
            download(
                client,
                daily_resource(),
                destination,
                retries=1,
                backoff=0,
            )

    assert destination.read_bytes() == b"existing"
    assert not destination.with_name("archive.zip.part").exists()


@pytest.mark.parametrize("declared", [True, False])
def test_download_rejects_archives_over_the_size_limit(
    tmp_path: Path, declared: bool
) -> None:
    """Confirm both declared and streamed byte limits remove partial files.

    Args:
        tmp_path: The temporary download directory.
        declared: Whether Content-Length declares the oversized response.
    """
    contents = b"12345"
    archive_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return an archive that is larger than the configured limit."""
        nonlocal archive_calls
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=sidecar(contents))
        archive_calls += 1
        headers = {"Content-Length": "5"} if declared else {}
        return httpx.Response(200, headers=headers, stream=Chunks(b"12", b"345"))

    destination = tmp_path / "archive.zip"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DownloadSizeError, match="size"):
            download(
                client,
                daily_resource(),
                destination,
                retries=3,
                max_bytes=4,
            )

    assert archive_calls == 1
    assert not destination.exists()
    assert not destination.with_name("archive.zip.part").exists()


def test_download_ignores_malformed_content_length(tmp_path: Path) -> None:
    """Confirm streamed byte counting handles an invalid size header."""
    contents = b"archive"

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a malformed size header with otherwise valid bytes."""
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=sidecar(contents))
        return httpx.Response(
            200, headers={"Content-Length": "unknown"}, content=contents
        )

    destination = tmp_path / "archive.zip"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        download(client, daily_resource(), destination, retries=0)

    assert destination.read_bytes() == contents


def test_permanent_sidecar_http_failure_is_not_retried(tmp_path: Path) -> None:
    """Confirm a missing checksum sidecar stops before requesting the archive."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Count and return a missing sidecar response."""
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            download(client, daily_resource(), tmp_path / "archive.zip", retries=3)

    assert calls == 1


def test_download_applies_timeout_to_sidecar_and_archive(tmp_path: Path) -> None:
    """Confirm both requests in a verified download use the chosen timeout."""
    contents = b"archive"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record and satisfy both download requests."""
        requests.append(request)
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=sidecar(contents))
        return httpx.Response(200, content=contents)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        download(
            client,
            daily_resource(),
            tmp_path / "archive.zip",
            timeout=7.0,
            retries=0,
        )

    assert len(requests) == 2
    assert all(
        set(request.extensions["timeout"].values()) == {7.0} for request in requests
    )


def test_download_rejects_nonpositive_size_limit_before_request(tmp_path: Path) -> None:
    """Confirm an unusable archive limit fails without network activity."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Count an unexpected network request."""
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="max_bytes"):
            download(client, daily_resource(), tmp_path / "archive.zip", max_bytes=0)

    assert calls == 0


def test_archive_checksum_reads_a_verified_sidecar() -> None:
    """Confirm sidecar-only revalidation returns its declared archive digest."""
    item = daily_resource()
    contents = b"archive"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record one checksum request and return a matching sidecar."""
        requests.append(request)
        return httpx.Response(200, text=sidecar(contents))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        actual = archive_checksum(client, item, timeout=7.0, retries=0)

    assert actual == hashlib.sha256(contents).hexdigest()
    assert [request.url.path for request in requests] == [
        "/BTCUSDT-1m-2025-01-01.zip.CHECKSUM"
    ]
    assert set(requests[0].extensions["timeout"].values()) == {7.0}


def test_archive_checksum_retries_a_temporary_sidecar_failure() -> None:
    """Confirm a transient checksum response uses the standard retry policy."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one temporary error before a valid checksum sidecar."""
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, text=sidecar(b"archive"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        digest = archive_checksum(client, daily_resource(), retries=1, backoff=0)

    assert digest == hashlib.sha256(b"archive").hexdigest()
    assert calls == 2
