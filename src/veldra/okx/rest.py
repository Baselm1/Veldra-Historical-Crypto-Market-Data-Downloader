"""Cache and query OKX public paginated historical REST series."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import cast

import duckdb
import pandas as pd

from veldra.core.catalog import Catalog
from veldra.core.models import (
    ArchiveKey,
    ArchiveObject,
    IntegritySpec,
    LogicalPartition,
    Materialization,
)
from veldra.core.subjects import DataSubject
from veldra.okx.client import BASE_URL, OKXClient, OKXResponseError


@dataclass(frozen=True)
class RESTSpec:
    """Declare one public historical REST dataset.

    Args:
        name: Canonical dataset name.
        path: Public OKX endpoint path.
        policy_key: Shared limiter bucket.
        page_size: Maximum source page size.
        time_column: Canonical timestamp column.
        sequence_columns: Source array fields, or empty for object records.
        mutable_hours: Duration before a recent cached request is refreshed.
        paginated: Whether the endpoint uses descending cursor pagination.
    """

    name: str
    path: str
    policy_key: str
    page_size: int
    time_column: str
    sequence_columns: tuple[str, ...] = ()
    mutable_hours: float = 6
    paginated: bool = True


REST_SPECS: Mapping[str, RESTSpec] = {
    "index_price_klines": RESTSpec(
        "index_price_klines",
        "/api/v5/market/history-index-candles",
        "history_index_candles",
        100,
        "open_time",
        ("open_time", "open", "high", "low", "close", "confirmed"),
    ),
    "mark_price_klines": RESTSpec(
        "mark_price_klines",
        "/api/v5/market/history-mark-price-candles",
        "history_mark_candles",
        100,
        "open_time",
        ("open_time", "open", "high", "low", "close", "confirmed"),
    ),
    "premium_history": RESTSpec(
        "premium_history",
        "/api/v5/public/premium-history",
        "premium_history",
        100,
        "event_time",
    ),
    "recent_funding_rates": RESTSpec(
        "recent_funding_rates",
        "/api/v5/public/funding-rate-history",
        "funding_history",
        400,
        "funding_time",
    ),
    "settlements": RESTSpec(
        "settlements",
        "/api/v5/public/settlement-history",
        "settlement_history",
        100,
        "event_time",
    ),
    "delivery_exercise": RESTSpec(
        "delivery_exercise",
        "/api/v5/public/delivery-exercise-history",
        "delivery_exercise",
        100,
        "event_time",
    ),
    "open_interest_history": RESTSpec(
        "open_interest_history",
        "/api/v5/rubik/stat/contracts/open-interest-volume",
        "open_interest_history",
        100,
        "event_time",
        ("event_time", "open_interest", "volume"),
        paginated=False,
    ),
    "taker_volume": RESTSpec(
        "taker_volume",
        "/api/v5/rubik/stat/taker-volume",
        "taker_volume",
        100,
        "event_time",
        ("event_time", "sell_volume", "buy_volume"),
        paginated=False,
    ),
    "long_short_ratio": RESTSpec(
        "long_short_ratio",
        "/api/v5/rubik/stat/contracts/long-short-account-ratio",
        "long_short_ratio",
        100,
        "event_time",
        ("event_time", "ratio"),
        paginated=False,
    ),
    "option_interest_volume": RESTSpec(
        "option_interest_volume",
        "/api/v5/rubik/stat/option/open-interest-volume",
        "option_interest_volume",
        100,
        "event_time",
        ("event_time", "open_interest", "volume"),
        paginated=False,
    ),
}


def _source_time(value: object, name: str) -> pd.Timestamp:
    """Parse one required OKX epoch-millisecond timestamp.

    Args:
        value: Source timestamp value.
        name: Field name used in errors.

    Returns:
        UTC timestamp at microsecond precision.
    """
    try:
        result = pd.to_datetime(int(str(value)), unit="ms", utc=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid OKX REST {name}") from error
    return result.as_unit("us")


def _source_number(value: object, name: str, *, optional: bool = False) -> float:
    """Parse one finite REST number, permitting missing optional fields.

    Args:
        value: Source numeric value.
        name: Field name used in errors.
        optional: Whether an empty value becomes NaN.

    Returns:
        Floating-point value or NaN.
    """
    if optional and value in {None, ""}:
        return float("nan")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid OKX REST {name}") from error
    if not pd.notna(result) or result in {float("inf"), float("-inf")}:
        raise ValueError(f"invalid OKX REST {name}")
    return result


def _sequence_frame(spec: RESTSpec, rows: Sequence[object]) -> pd.DataFrame:
    """Normalize array-shaped candle or analytical history rows.

    Args:
        spec: Historical endpoint declaration.
        rows: Raw response values.

    Returns:
        Canonical timestamped numeric frame.
    """
    values: list[list[object]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(spec.sequence_columns):
            raise OKXResponseError("invalid_data", f"{spec.name} row shape is invalid")
        values.append(row)
    frame = pd.DataFrame(values, columns=spec.sequence_columns)
    if frame.empty:
        frame = frame.astype(
            {column: "float64" for column in spec.sequence_columns[1:]}
        )
        frame[spec.time_column] = pd.Series(dtype="datetime64[us, UTC]")
        return frame
    frame[spec.time_column] = [
        _source_time(value, spec.time_column) for value in frame[spec.time_column]
    ]
    for column in spec.sequence_columns[1:]:
        frame[column] = [_source_number(value, column) for value in frame[column]]
    if "confirmed" in frame and not frame["confirmed"].eq(1).all():
        frame = frame[frame["confirmed"].eq(1)].copy()
    return frame


def _objects(rows: Sequence[object], name: str) -> list[dict[str, object]]:
    """Return object rows or raise a source-shape error.

    Args:
        rows: Raw endpoint values.
        name: Dataset name used in errors.

    Returns:
        Typed object rows.
    """
    if not all(isinstance(row, dict) for row in rows):
        raise OKXResponseError("invalid_data", f"{name} rows must be objects")
    return list(cast(Sequence[dict[str, object]], rows))


def _object_frame(spec: RESTSpec, rows: Sequence[object]) -> pd.DataFrame:
    """Normalize object-shaped rate and lifecycle history rows.

    Args:
        spec: Historical endpoint declaration.
        rows: Raw response values.

    Returns:
        Canonical timestamped frame.
    """
    records = _objects(rows, spec.name)
    if not records:
        columns = {
            "premium_history": ("event_time", "premium"),
            "recent_funding_rates": (
                "funding_time",
                "funding_rate",
                "realized_rate",
                "formula_type",
                "method",
            ),
            "settlements": ("event_time", "instrument_id", "price", "event_type"),
            "delivery_exercise": (
                "event_time",
                "instrument_id",
                "price",
                "event_type",
            ),
        }[spec.name]
        frame = pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
        frame[spec.time_column] = pd.Series(dtype="datetime64[us, UTC]")
        return frame
    if spec.name in {"settlements", "delivery_exercise"}:
        return _lifecycle_frame(spec, records)
    normalized: list[dict[str, object]] = []
    for row in records:
        if spec.name == "premium_history":
            normalized.append(
                {
                    "event_time": _source_time(row.get("ts"), "event_time"),
                    "premium": _source_number(row.get("premium"), "premium"),
                }
            )
        else:
            normalized.append(
                {
                    "funding_time": _source_time(
                        row.get("fundingTime"), "funding_time"
                    ),
                    "funding_rate": _source_number(
                        row.get("fundingRate"), "funding_rate"
                    ),
                    "realized_rate": _source_number(
                        row.get("realizedRate"), "realized_rate", optional=True
                    ),
                    "formula_type": str(row.get("formulaType", "")),
                    "method": str(row.get("method", "")),
                }
            )
    return pd.DataFrame(normalized)


def _lifecycle_frame(spec: RESTSpec, rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Flatten settlement, delivery, and exercise event groups.

    Args:
        spec: Lifecycle endpoint declaration.
        rows: Timestamped source groups.

    Returns:
        One row per affected contract.
    """
    values: list[dict[str, object]] = []
    for row in rows:
        event_time = _source_time(row.get("ts"), "event_time")
        details = row.get("details")
        if not isinstance(details, list) or not all(
            isinstance(detail, dict) for detail in details
        ):
            raise OKXResponseError("invalid_data", "lifecycle details are invalid")
        for detail in details:
            instrument = detail.get("instId", detail.get("insId"))
            price = detail.get("settlePx", detail.get("px"))
            if not isinstance(instrument, str) or not instrument:
                raise OKXResponseError(
                    "invalid_data", "lifecycle instrument is invalid"
                )
            values.append(
                {
                    "event_time": event_time,
                    "instrument_id": instrument,
                    "price": _source_number(price, "price"),
                    "event_type": (
                        "settlement"
                        if spec.name == "settlements"
                        else str(detail.get("type", ""))
                    ),
                }
            )
    return pd.DataFrame(values)


def normalize_rest(spec: RESTSpec, rows: Sequence[object]) -> pd.DataFrame:
    """Normalize one endpoint response into a deterministic frame.

    Args:
        spec: Historical endpoint declaration.
        rows: Raw source values.

    Returns:
        Sorted, deduplicated canonical rows.
    """
    frame = (
        _sequence_frame(spec, rows)
        if spec.sequence_columns
        else _object_frame(spec, rows)
    )
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(ignore_index=True)
    order = [spec.time_column]
    if "instrument_id" in frame:
        order.append("instrument_id")
    return frame.sort_values(order, kind="stable", ignore_index=True)


def _row_time(spec: RESTSpec, row: object) -> int:
    """Return one source row timestamp used for pagination.

    Args:
        spec: Endpoint declaration.
        row: Raw array or object row.

    Returns:
        Epoch-millisecond timestamp.
    """
    value = (
        row[0]
        if isinstance(row, list)
        else (
            row.get("fundingTime" if spec.name == "recent_funding_rates" else "ts")
            if isinstance(row, dict)
            else None
        )
    )
    try:
        return int(str(value))
    except (TypeError, ValueError) as error:
        raise OKXResponseError(
            "invalid_data", f"{spec.name} time is invalid"
        ) from error


class OKXRESTHistory:
    """Fetch immutable ranges once and refresh only recent REST history."""

    def __init__(self, client: OKXClient, catalog: Catalog, root: Path) -> None:
        """Retain the source client, catalog, and storage root.

        Args:
            client: Shared rate-limited OKX client.
            catalog: Open local catalog.
            root: Veldra data root.
        """
        self.client = client
        self.catalog = catalog
        self.root = root

    def get(
        self,
        name: str,
        subject: DataSubject,
        start: datetime,
        end: datetime,
        *,
        product: str,
        params: Mapping[str, str],
        interval: str | None = None,
        offline: bool = False,
    ) -> pd.DataFrame:
        """Return one cached or freshly paginated REST history range.

        Args:
            name: Registered REST dataset.
            subject: Instrument, family, or currency scope.
            start: Inclusive UTC range start.
            end: Exclusive UTC range end.
            product: Catalog product label.
            params: Valid endpoint-specific query parameters.
            interval: Optional native Kline interval.
            offline: Whether source access is forbidden.

        Returns:
            Canonical rows filtered to the exact request.
        """
        spec = REST_SPECS.get(name)
        if spec is None:
            raise ValueError(f"unsupported OKX REST dataset {name!r}")
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("REST history requires a valid aware time range")
        key = self._key(spec, subject, start, end, product, params, interval)
        partitions = self.catalog.partitions_between(
            "okx", product, name, subject, interval, start, end
        )
        exact = [
            item
            for item in partitions
            if item.coverage_start == start and item.coverage_end == end
        ]
        archive = self.catalog.archive(key)
        fresh = archive is not None and self._fresh(archive, end, spec.mutable_hours)
        if not exact or (not offline and not fresh):
            if offline:
                raise RuntimeError("offline mode requires this exact cached REST range")
            rows = self._fetch(spec, start, end, params)
            frame = normalize_rest(spec, rows)
            if not frame.empty:
                frame = frame[
                    frame[spec.time_column].ge(start) & frame[spec.time_column].lt(end)
                ].reset_index(drop=True)
            exact = [
                self._publish(
                    key, spec, subject, start, end, product, interval, frame, rows
                )
            ]
        return self._query(exact, spec, start, end)

    @staticmethod
    def _fresh(archive: ArchiveObject, end: datetime, mutable_hours: float) -> bool:
        """Return whether one ready range is immutable or within its TTL.

        Args:
            archive: Cataloged REST request batch.
            end: Exclusive requested range end.
            mutable_hours: Recent-cache lifetime.

        Returns:
            True when no refresh is currently required.
        """
        if archive.status != "ready":
            return False
        now = datetime.now(UTC)
        if end <= now - timedelta(days=3):
            return True
        checked = archive.last_attempt_at or archive.discovered_at
        return checked >= now - timedelta(hours=mutable_hours)

    @staticmethod
    def _key(
        spec: RESTSpec,
        subject: DataSubject,
        start: datetime,
        end: datetime,
        product: str,
        params: Mapping[str, str],
        interval: str | None,
    ) -> ArchiveKey:
        """Build one stable physical identity for an exact REST request.

        Args:
            spec: Endpoint declaration.
            subject: Logical source scope.
            start: Inclusive request start.
            end: Exclusive request end.
            product: Catalog product label.
            params: Endpoint-specific query parameters.
            interval: Optional native interval.

        Returns:
            Stable catalog archive key.
        """
        identity = json.dumps(
            [
                spec.path,
                sorted(params.items()),
                start.isoformat(),
                end.isoformat(),
                interval,
            ],
            separators=(",", ":"),
        )
        digest = sha256(identity.encode()).hexdigest()
        return ArchiveKey(
            "okx",
            product,
            spec.name,
            "rest_api",
            subject.kind,
            subject.value,
            "request",
            start.date(),
            (end - timedelta(microseconds=1)).date(),
            f"{digest}.json",
        )

    def _fetch(
        self,
        spec: RESTSpec,
        start: datetime,
        end: datetime,
        params: Mapping[str, str],
    ) -> list[object]:
        """Fetch one bounded endpoint range with its native pagination.

        Args:
            spec: Endpoint declaration.
            start: Inclusive request start.
            end: Exclusive request end.
            params: Endpoint-specific parameters.

        Returns:
            Raw source rows.
        """
        query = dict(params)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        policy_key = self._policy_key(spec, params)
        if not spec.paginated:
            query.update({"begin": str(start_ms), "end": str(end_ms)})
            return self.client.request(spec.path, policy_key=policy_key, params=query)
        query["after"] = str(end_ms)

        def cursor(rows: list[object]) -> str | None:
            """Stop at the request start or continue from the oldest row."""
            oldest = min(_row_time(spec, row) for row in rows)
            return None if oldest <= start_ms else str(oldest)

        return self.client.paginate(
            spec.path,
            policy_key=policy_key,
            params=query,
            cursor=cursor,
            page_size=spec.page_size,
            max_pages=10000,
        )

    @staticmethod
    def _policy_key(spec: RESTSpec, params: Mapping[str, str]) -> str:
        """Qualify quotas whose documented scope is an instrument or family.

        Args:
            spec: Endpoint declaration.
            params: Endpoint-specific query parameters.

        Returns:
            Shared or qualified limiter bucket key.
        """
        if spec.name == "recent_funding_rates":
            return f"{spec.policy_key}:{params.get('instId', '')}"
        if spec.name == "settlements":
            return f"{spec.policy_key}:{params.get('instFamily', '')}"
        if spec.name == "delivery_exercise":
            return (
                f"{spec.policy_key}:{params.get('instType', '')}:"
                f"{params.get('instFamily', '')}"
            )
        return spec.policy_key

    def _publish(
        self,
        key: ArchiveKey,
        spec: RESTSpec,
        subject: DataSubject,
        start: datetime,
        end: datetime,
        product: str,
        interval: str | None,
        frame: pd.DataFrame,
        rows: Sequence[object],
    ) -> LogicalPartition:
        """Atomically publish one normalized exact REST request.

        Args:
            key: Stable physical request identity.
            spec: Endpoint declaration.
            subject: Logical source scope.
            start: Inclusive coverage start.
            end: Exclusive coverage end.
            product: Catalog product label.
            interval: Optional native Kline interval.
            frame: Normalized response rows.
            rows: Raw response values used for a revision digest.

        Returns:
            Published logical partition.
        """
        archive = ArchiveObject(
            key,
            f"{BASE_URL}{spec.path}",
            integrity=IntegritySpec("archive_only"),
        )
        self.catalog.save_archives([archive])
        destination = (
            self.root
            / "okx"
            / "rest"
            / product
            / spec.name
            / f"{key.archive_id}.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        frame.to_parquet(partial, compression="zstd", index=False)
        partial.replace(destination)
        stat = destination.stat()
        revision = sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        first = start if frame.empty else frame[spec.time_column].min().to_pydatetime()
        last = (
            end - timedelta(microseconds=1)
            if frame.empty
            else frame[spec.time_column].max().to_pydatetime()
        )
        materialization = Materialization(
            key,
            destination,
            1,
            len(frame),
            first,
            last,
            stat.st_size,
            local_mtime_ns=stat.st_mtime_ns,
            archive_revision=revision,
        )
        partition = LogicalPartition(
            "okx",
            product,
            spec.name,
            subject,
            interval,
            start,
            end,
            destination,
            None,
            None,
            len(frame),
            source_day=start.date(),
        )
        self.catalog.publish_materialization(materialization, [partition])
        return partition

    @staticmethod
    def _query(
        partitions: Sequence[LogicalPartition],
        spec: RESTSpec,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Query exact cached REST partitions through DuckDB.

        Args:
            partitions: Exact request partitions.
            spec: Endpoint declaration.
            start: Inclusive UTC request start.
            end: Exclusive UTC request end.

        Returns:
            Deterministically ordered canonical rows.
        """
        paths = sorted({str(item.materialization_path) for item in partitions})
        if not paths:
            return pd.DataFrame()
        with duckdb.connect() as connection:
            frame = connection.execute(
                f'SELECT * FROM read_parquet(?) WHERE "{spec.time_column}" >= ? '
                f'AND "{spec.time_column}" < ? ORDER BY "{spec.time_column}"',
                [paths, start, end],
            ).df()
        frame[spec.time_column] = pd.to_datetime(
            frame[spec.time_column], utc=True
        ).astype("datetime64[us, UTC]")
        return frame
