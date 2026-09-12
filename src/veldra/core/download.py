"""Make retrying HTTP requests and verified streaming downloads."""

from collections.abc import Callable, Mapping
import base64
import binascii
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import hashlib
import logging
import math
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlsplit

import httpx

from veldra.core.models import IntegritySpec, Resource

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
LOGGER = logging.getLogger(__name__)


class ChecksumError(ValueError):
    """Report that downloaded bytes do not match their checksum sidecar."""


def _algorithm_name(algorithm: str) -> str:
    """Return a checksum algorithm's conventional display name.

    Args:
        algorithm: The hashlib algorithm identifier.

    Returns:
        A human-readable algorithm name.
    """
    return "SHA-256" if algorithm == "sha256" else algorithm.upper()


class DownloadSizeError(ValueError):
    """Report that a download exceeds its configured byte limit."""


def _retry_after(value: str, now: datetime | None) -> float | None:
    """Parse one numeric or HTTP-date Retry-After value.

    Args:
        value: The Retry-After header value to parse.
        now: The current UTC time used for an HTTP date.

    Returns:
        A nonnegative delay capped at one minute, or ``None`` when invalid.
    """
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                return None
            current = now if now is not None else datetime.now(UTC)
            seconds = (retry_at - current).total_seconds()
        except TypeError, ValueError, OverflowError:
            return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, 60.0)


def retry_delay(
    response: httpx.Response | None,
    attempt: int,
    backoff: float,
    *,
    now: datetime | None = None,
) -> float:
    """Calculate the delay before another HTTP attempt.

    Args:
        response: The failed response, when one was received.
        attempt: The zero-based retry number.
        backoff: The initial exponential delay in seconds.
        now: The current UTC time used to parse an HTTP date.

    Returns:
        The number of seconds to wait.
    """
    if attempt < 0:
        raise ValueError("retry attempt cannot be negative")
    if not math.isfinite(backoff) or backoff < 0:
        raise ValueError("retry backoff must be finite and nonnegative")

    if response is not None:
        value = response.headers.get("Retry-After")
        if value is not None:
            parsed = _retry_after(value, now)
            if parsed is not None:
                return parsed
    try:
        return min(backoff * 2.0**attempt, 30.0)
    except OverflowError:
        return 30.0


def _finite_number(value: object) -> bool:
    """Return whether a value is a finite non-Boolean number.

    Args:
        value: The value to inspect.

    Returns:
        Whether the value can safely be used as a numeric HTTP setting.
    """
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _validate_settings(timeout: float, retries: int, backoff: float) -> None:
    """Reject invalid timeout and retry settings.

    Args:
        timeout: The timeout for each request in seconds.
        retries: The number of retries after the first attempt.
        backoff: The initial exponential delay in seconds.
    """
    if not _finite_number(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and greater than zero")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    if not _finite_number(backoff) or backoff < 0:
        raise ValueError("backoff must be finite and nonnegative")


def _retry[Result](
    operation: Callable[[], Result], *, retries: int, backoff: float
) -> Result:
    """Run an operation again after temporary HTTP or checksum failures.

    Args:
        operation: The request operation to run.
        retries: The number of retries after the first attempt.
        backoff: The initial exponential delay in seconds.

    Returns:
        The first successful operation result.
    """
    for attempt in range(retries + 1):
        try:
            return operation()
        except (httpx.TransportError, httpx.HTTPStatusError, ChecksumError) as error:
            response = (
                error.response if isinstance(error, httpx.HTTPStatusError) else None
            )
            if (
                response is not None
                and response.status_code not in RETRYABLE_STATUS_CODES
            ):
                raise
            if attempt == retries:
                LOGGER.error(
                    "HTTP operation exhausted retries: attempts=%d error=%s",
                    attempt + 1,
                    error,
                )
                raise
            delay = retry_delay(response, attempt, backoff)
            LOGGER.warning(
                "Retrying HTTP operation: attempt=%d/%d delay=%.3fs error=%s",
                attempt + 1,
                retries + 1,
                delay,
                error,
            )
            time.sleep(delay)
    raise RuntimeError("retry loop ended without a result")


def get(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.5,
) -> httpx.Response:
    """Make a GET request with bounded retries.

    Args:
        client: The HTTPX client used for the request.
        url: The URL to request.
        params: Optional query string values.
        timeout: The timeout for each attempt in seconds.
        retries: The number of retries after the first attempt.
        backoff: The initial exponential delay in seconds.

    Returns:
        A successful HTTP response.
    """
    _validate_settings(timeout, retries, backoff)

    def request() -> httpx.Response:
        """Make one metadata request attempt.

        Returns:
            A successful HTTP response.
        """
        response = client.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        LOGGER.debug(
            "HTTP GET complete: url=%s status=%d bytes=%d",
            response.url,
            response.status_code,
            len(response.content),
        )
        return response

    return _retry(request, retries=retries, backoff=backoff)


def _checksum(text: str, archive_url: str, algorithm: str = "sha256") -> str:
    """Read a source digest for the exact archive URL filename.

    Args:
        text: The checksum sidecar contents.
        archive_url: The archive URL whose filename must match.
        algorithm: The source checksum algorithm.

    Returns:
        The lowercase source digest.
    """
    lengths = {"md5": 32, "sha256": 64}
    try:
        length = lengths[algorithm]
    except KeyError as error:
        raise ValueError("unsupported archive checksum algorithm") from error
    match = re.fullmatch(rf"([0-9a-fA-F]{{{length}}})[ \t]+\*?([^\r\n]+)", text.strip())
    filename = unquote(Path(urlsplit(archive_url).path).name)
    if match is None or match.group(2) != filename:
        raise ValueError(
            f"invalid {_algorithm_name(algorithm)} sidecar or target filename"
        )
    return match.group(1).lower()


def archive_checksum(
    client: httpx.Client,
    resource: Resource,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.5,
) -> str:
    """Fetch and validate one archive's checksum sidecar.

    Args:
        client: The HTTPX client used for the sidecar request.
        resource: The archive resource whose sidecar is checked.
        timeout: The timeout for each request in seconds.
        retries: The number of retries after the first attempt.
        backoff: The initial exponential retry delay in seconds.

    Returns:
        The lowercase archive digest.
    """
    _validate_settings(timeout, retries, backoff)

    def request() -> str:
        """Fetch and parse one checksum sidecar.

        Returns:
            The digest declared by the checksum sidecar.
        """
        integrity = resource.integrity_spec
        if integrity.mode != "sidecar" or integrity.sidecar_url is None:
            raise ValueError("resource does not use a checksum sidecar")
        response = client.get(integrity.sidecar_url, timeout=timeout)
        response.raise_for_status()
        digest = _checksum(
            response.text,
            resource.url,
            integrity.algorithm or resource.checksum_algorithm,
        )
        LOGGER.debug(
            "Archive checksum fetched: url=%s algorithm=%s digest=%s",
            resource.url,
            integrity.algorithm or resource.checksum_algorithm,
            digest,
        )
        return digest

    return _retry(request, retries=retries, backoff=backoff)


def _declared_size(response: httpx.Response) -> int | None:
    """Read a valid Content-Length value when the server provides one.

    Args:
        response: The streamed archive response.

    Returns:
        The declared nonnegative byte count, or ``None`` when unavailable.
    """
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        size = int(value)
    except ValueError:
        return None
    return size if size >= 0 else None


def _content_md5(value: str) -> str | None:
    """Decode one base64 Content-MD5 value.

    Args:
        value: The response header value.

    Returns:
        The lowercase hexadecimal MD5 digest, or ``None`` when malformed.
    """
    try:
        digest = base64.b64decode(value, validate=True)
    except ValueError, binascii.Error:
        return None
    return digest.hex() if len(digest) == 16 else None


def _etag_md5(value: str) -> str | None:
    """Read a plain or quoted non-multipart MD5 ETag.

    Args:
        value: The response ETag header.

    Returns:
        The lowercase digest, or ``None`` for malformed and multipart ETags.
    """
    candidate = value.strip().removeprefix("W/").strip().strip('"')
    if "-" in candidate or not re.fullmatch(r"[0-9a-fA-F]{32}", candidate):
        return None
    return candidate.lower()


def _response_md5(response: httpx.Response, integrity: IntegritySpec) -> str:
    """Return the expected MD5 from metadata or one response header.

    Args:
        response: The streamed source response carrying integrity metadata.
        integrity: The response-header integrity policy.

    Returns:
        The expected lowercase MD5 digest.
    """
    if integrity.expected is not None:
        return integrity.expected
    content_md5 = response.headers.get("Content-MD5")
    digest = _content_md5(content_md5) if content_md5 is not None else None
    if digest is None:
        etag = response.headers.get("ETag")
        digest = _etag_md5(etag) if etag is not None else None
    if digest is None:
        raise ChecksumError("response does not provide a usable MD5 digest")
    return digest


def download(
    client: httpx.Client,
    resource: Resource,
    destination: Path,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 0.5,
    max_bytes: int = 64 * 1024 * 1024,
) -> str:
    """Stream and verify one resource archive.

    Args:
        client: The HTTPX client used for the requests.
        resource: The archive and checksum URLs to download.
        destination: The final archive path.
        timeout: The timeout for each request in seconds.
        retries: The number of retries after the first attempt.
        backoff: The initial exponential delay in seconds.
        max_bytes: The largest accepted archive size.

    Returns:
        The verified lowercase source digest.
    """
    _validate_settings(timeout, retries, backoff)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")

    def attempt() -> str:
        """Fetch and verify one fresh sidecar and archive attempt.

        Returns:
            The verified lowercase source digest.
        """
        partial.unlink(missing_ok=True)
        try:
            integrity = resource.integrity_spec
            algorithm = integrity.algorithm or "sha256"
            expected: str | None = None
            if integrity.mode == "sidecar":
                assert integrity.sidecar_url is not None
                sidecar = client.get(integrity.sidecar_url, timeout=timeout)
                sidecar.raise_for_status()
                expected = _checksum(
                    sidecar.text,
                    resource.url,
                    algorithm,
                )
            digest = hashlib.new(algorithm)
            size = 0

            with client.stream("GET", resource.url, timeout=timeout) as response:
                response.raise_for_status()
                if integrity.mode == "response_header":
                    expected = _response_md5(response, integrity)
                declared = _declared_size(response)
                if declared is not None and declared > max_bytes:
                    raise DownloadSizeError("download size exceeds configured limit")
                with partial.open("wb") as output:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise DownloadSizeError(
                                "download size exceeds configured limit"
                            )
                        output.write(chunk)
                        digest.update(chunk)

            actual = digest.hexdigest()
            if expected is not None and actual != expected:
                raise ChecksumError(
                    f"{_algorithm_name(algorithm)} mismatch " f"for {resource.url}"
                )
            partial.replace(destination)
            LOGGER.info(
                "Verified download complete: url=%s bytes=%d algorithm=%s "
                "digest=%s path=%s",
                resource.url,
                size,
                algorithm,
                actual,
                destination,
            )
            return actual
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    return _retry(attempt, retries=retries, backoff=backoff)
