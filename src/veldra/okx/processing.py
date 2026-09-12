"""Normalize and validate OKX historical CSV archives."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import httpx
import pandas as pd

from veldra.core.datasets import DatasetSpec
from veldra.core.download import download
from veldra.core.models import (
    ArchiveObject,
    DataValidationError,
    LogicalPartition,
    Materialization,
    Resource,
)
from veldra.core.providers import MaterializedArchive
from veldra.core.subjects import DataSubject
from veldra.okx.datasets import get_dataset


def _numbers(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Convert required source columns to finite floating-point values.

    Args:
        frame: Mutable source frame.
        columns: Source columns that must contain finite numbers.
    """
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise DataValidationError(f"invalid OKX {column} value") from error
        if frame[column].isna().any():
            raise DataValidationError(f"invalid OKX {column} value")


def _deduplicate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Remove exact duplicates and reject conflicting event keys.

    Args:
        frame: Canonical source rows.
        keys: Columns forming one logical event identity.

    Returns:
        Rows with exact duplicates removed.
    """
    result = frame.drop_duplicates(ignore_index=True)
    if result.duplicated(keys, keep=False).any():
        raise DataValidationError("OKX archive contains conflicting duplicate events")
    return result


def normalize_klines(frame: pd.DataFrame, dataset: DatasetSpec) -> pd.DataFrame:
    """Return canonical one-minute Klines from an OKX module 2 CSV.

    Args:
        frame: Source CSV rows.
        dataset: Product-specific Kline declaration.

    Returns:
        Sorted canonical rows retaining the native instrument predicate.
    """
    if tuple(frame.columns) != dataset.source_columns:
        raise DataValidationError("CSV does not match the OKX Kline schema")
    if frame.empty:
        raise DataValidationError("OKX Kline CSV cannot be empty")
    if frame["instrument_name"].isna().any():
        raise DataValidationError("OKX instrument_name cannot be empty")
    instruments = frame["instrument_name"].astype("string").str.strip()
    if instruments.eq("").any():
        raise DataValidationError("OKX instrument_name cannot be empty")
    _numbers(
        frame,
        ("open", "high", "low", "close", "vol", "vol_ccy", "vol_quote"),
    )
    try:
        times = pd.to_datetime(
            pd.to_numeric(frame["open_time"], errors="raise"), unit="ms", utc=True
        )
        confirms = pd.to_numeric(frame["confirm"], errors="raise")
    except (TypeError, ValueError) as error:
        raise DataValidationError("invalid OKX Kline time or confirmation") from error
    if not confirms.eq(1).all():
        raise DataValidationError("OKX historical Klines must be confirmed")
    result = pd.DataFrame(
        {
            "instrument_id": instruments,
            "open_time": times,
            "open": frame["open"].astype("float64"),
            "high": frame["high"].astype("float64"),
            "low": frame["low"].astype("float64"),
            "close": frame["close"].astype("float64"),
        }
    )
    if dataset.product == "spot":
        result["base_volume"] = frame["vol"].astype("float64")
    else:
        result["contract_volume"] = frame["vol"].astype("float64")
        result["base_volume"] = frame["vol_ccy"].astype("float64")
    result["quote_volume"] = frame["vol_quote"].astype("float64")
    if (result[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise DataValidationError("OKX Kline prices must be positive")
    if (result[list(dataset.resample_sum_columns)] < 0).any(axis=None):
        raise DataValidationError("OKX Kline volumes must be nonnegative")
    if (
        result["high"].lt(result[["open", "low", "close"]].max(axis=1)).any()
        or result["low"].gt(result[["open", "high", "close"]].min(axis=1)).any()
    ):
        raise DataValidationError("OKX Kline OHLC values contradict each other")
    if (
        not result["open_time"].dt.second.eq(0).all()
        or not result["open_time"].dt.microsecond.eq(0).all()
    ):
        raise DataValidationError("OKX Kline timestamps are not minute-aligned")
    result = _deduplicate(result, ["instrument_id", "open_time"])
    return result.sort_values(
        ["instrument_id", "open_time"], kind="stable", ignore_index=True
    )


def normalize_trades(
    frame: pd.DataFrame,
    dataset: DatasetSpec,
    contract_sizes: Mapping[str, float | None] | None = None,
) -> pd.DataFrame:
    """Return canonical individual trades from an OKX module 1 CSV.

    Args:
        frame: Source CSV rows.
        dataset: Product-specific trade declaration.
        contract_sizes: Native instruments mapped to contract face values.

    Returns:
        Sorted events retaining native instrument predicates and explicit units.
    """
    if tuple(frame.columns) != dataset.source_columns:
        raise DataValidationError("CSV does not match the OKX trade schema")
    if frame.empty:
        raise DataValidationError("OKX trade CSV cannot be empty")
    instruments = frame["instrument_name"].astype("string").str.strip()
    sides = frame["side"].astype("string").str.strip().str.lower()
    if instruments.isna().any() or instruments.eq("").any():
        raise DataValidationError("OKX instrument_name cannot be empty")
    if not sides.isin(["buy", "sell"]).all():
        raise DataValidationError("OKX trade side must be buy or sell")
    _numbers(frame, ("price", "size"))
    try:
        identifiers = pd.to_numeric(frame["trade_id"], errors="raise")
        times = pd.to_datetime(
            pd.to_numeric(frame["created_time"], errors="raise"), unit="ms", utc=True
        )
    except (TypeError, ValueError) as error:
        raise DataValidationError("invalid OKX trade ID or timestamp") from error
    if not identifiers.mod(1).eq(0).all() or identifiers.lt(0).any():
        raise DataValidationError("OKX trade IDs must be nonnegative integers")
    prices = frame["price"].astype("float64")
    quantities = frame["size"].astype("float64")
    if prices.le(0).any() or quantities.lt(0).any():
        raise DataValidationError("OKX trade prices and quantities are invalid")
    result = pd.DataFrame(
        {
            "instrument_id": instruments,
            "event_time": times,
            "trade_id": identifiers.astype("int64"),
            "price": prices,
            "side": sides,
        }
    )
    if dataset.product == "spot":
        result["base_quantity"] = quantities
        result["quote_quantity"] = prices * quantities
    else:
        sizes = instruments.map(contract_sizes or {})
        if dataset.product in {"linear_futures", "inverse_futures", "options"}:
            result["contract_quantity"] = quantities
        elif sizes.isna().any() or sizes.le(0).any():
            raise DataValidationError("OKX derivative contract size is unavailable")
        elif dataset.product == "linear_swap":
            result["contract_quantity"] = quantities
            result["base_quantity"] = quantities * sizes
            result["quote_quantity"] = result["base_quantity"] * prices
        else:
            result["contract_quantity"] = quantities
            result["quote_notional"] = quantities * sizes
            result["base_quantity"] = result["quote_notional"] / prices
    result = result[["instrument_id", *dataset.stored_columns]]
    result = _deduplicate(result, ["instrument_id", "trade_id"])
    return result.sort_values(
        ["instrument_id", "event_time", "trade_id"],
        kind="stable",
        ignore_index=True,
    )


def normalize(
    frame: pd.DataFrame,
    dataset: DatasetSpec,
    contract_sizes: Mapping[str, float | None] | None = None,
) -> pd.DataFrame:
    """Dispatch one OKX CSV to its product-specific normalizer.

    Args:
        frame: Source CSV rows.
        dataset: Canonical dataset declaration.
        contract_sizes: Native derivatives mapped to contract face values.

    Returns:
        Canonical rows retaining the native instrument predicate.
    """
    if dataset.name == "klines":
        return normalize_klines(frame, dataset)
    if dataset.name == "trades":
        return normalize_trades(frame, dataset, contract_sizes)
    if dataset.name == "funding_rates":
        return normalize_funding_rates(frame, dataset)
    if dataset.name == "borrow_rates":
        return normalize_borrow_rates(frame, dataset)
    raise ValueError(f"unsupported OKX normalizer {dataset.product}/{dataset.name}")


def normalize_funding_rates(frame: pd.DataFrame, dataset: DatasetSpec) -> pd.DataFrame:
    """Return canonical signed funding observations without assumed frequency.

    Args:
        frame: Source module 3 CSV rows.
        dataset: Product-specific funding declaration.

    Returns:
        Sorted actual funding timestamps and signed rates.
    """
    if tuple(frame.columns) != dataset.source_columns or frame.empty:
        raise DataValidationError("CSV does not match the OKX funding schema")
    instruments = frame["instrument_name"].astype("string").str.strip()
    if instruments.isna().any() or instruments.eq("").any():
        raise DataValidationError("OKX instrument_name cannot be empty")
    _numbers(frame, ("funding_rate",))
    try:
        times = pd.to_datetime(
            pd.to_numeric(frame["funding_time"], errors="raise"), unit="ms", utc=True
        )
    except (TypeError, ValueError) as error:
        raise DataValidationError("invalid OKX funding timestamp") from error
    result = pd.DataFrame(
        {
            "instrument_id": instruments,
            "funding_time": times,
            "funding_rate": frame["funding_rate"].astype("float64"),
        }
    )
    result = _deduplicate(result, ["instrument_id", "funding_time"])
    return result.sort_values(
        ["instrument_id", "funding_time"], kind="stable", ignore_index=True
    )


def normalize_borrow_rates(frame: pd.DataFrame, dataset: DatasetSpec) -> pd.DataFrame:
    """Return canonical hourly margin borrowing rates by currency.

    Args:
        frame: Source module 11 CSV rows.
        dataset: Margin borrowing declaration.

    Returns:
        Sorted currency-scoped observations in UTC.
    """
    if tuple(frame.columns) != dataset.source_columns or frame.empty:
        raise DataValidationError("CSV does not match the OKX borrowing schema")
    currencies = frame["currency_name"].astype("string").str.strip().str.upper()
    if currencies.isna().any() or currencies.eq("").any():
        raise DataValidationError("OKX borrowing currency cannot be empty")
    _numbers(frame, ("borrow_rate",))
    try:
        times = pd.to_datetime(
            pd.to_numeric(frame["time"], errors="raise"), unit="ms", utc=True
        )
    except (TypeError, ValueError) as error:
        raise DataValidationError("invalid OKX borrowing timestamp") from error
    result = pd.DataFrame(
        {
            "currency": currencies,
            "event_time": times,
            "borrow_rate": frame["borrow_rate"].astype("float64"),
        }
    )
    result = _deduplicate(result, ["currency", "event_time"])
    return result.sort_values(
        ["currency", "event_time"], kind="stable", ignore_index=True
    )


def _member(archive: zipfile.ZipFile, resource: ArchiveObject) -> zipfile.ZipInfo:
    """Return the one safe CSV member expected for an OKX ZIP.

    Args:
        archive: Open source ZIP.
        resource: Manifest metadata naming the physical object.

    Returns:
        Validated CSV member metadata.
    """
    expected = resource.key.remote_name.removesuffix(".zip") + ".csv"
    members = archive.infolist()
    if (
        len(members) != 1
        or members[0].is_dir()
        or members[0].filename != expected
        or "/" in expected
        or "\\" in expected
    ):
        raise DataValidationError("OKX ZIP must contain its one expected CSV")
    return members[0]


def _source_coverage(
    resource: ArchiveObject, dataset: DatasetSpec
) -> tuple[datetime, datetime]:
    """Return exact UTC bounds represented by source calendar labels.

    Args:
        resource: Physical archive period.
        dataset: Dataset source-calendar declaration.

    Returns:
        Inclusive UTC start and exclusive UTC end.
    """
    start = datetime.combine(resource.key.period_start, time.min, UTC)
    end = datetime.combine(resource.key.period_end + timedelta(days=1), time.min, UTC)
    return start - dataset.archive_day_offset, end - dataset.archive_day_offset


def _partitions(
    resource: ArchiveObject,
    destination: Path,
    frame: pd.DataFrame,
    dataset: DatasetSpec,
) -> tuple[LogicalPartition, ...]:
    """Describe every instrument stored in one specific or shared file.

    Args:
        resource: Physical manifest object.
        destination: Final Parquet path.
        frame: Canonical rows written to that path.
        dataset: Canonical dataset declaration.

    Returns:
        Logical instrument partitions sharing the materialization.
    """
    coverage_start, coverage_end = _source_coverage(resource, dataset)
    values: list[LogicalPartition] = []
    predicate = "currency" if dataset.name == "borrow_rates" else "instrument_id"
    kind = "currency" if dataset.name == "borrow_rates" else "instrument"
    for instrument, rows in frame.groupby(predicate, sort=True):
        values.append(
            LogicalPartition(
                "okx",
                resource.key.product,
                resource.key.dataset,
                DataSubject(kind, str(instrument)),  # type: ignore[arg-type]
                dataset.base_interval,
                coverage_start,
                coverage_end,
                destination,
                predicate,
                str(instrument),
                len(rows),
                source_day=resource.key.period_start,
            )
        )
    if resource.key.remote_scope_kind == "instrument_family":
        values.append(
            LogicalPartition(
                "okx",
                resource.key.product,
                resource.key.dataset,
                resource.key.subject,
                dataset.base_interval,
                coverage_start,
                coverage_end,
                destination,
                None,
                None,
                len(frame),
                source_day=resource.key.period_start,
            )
        )
    return tuple(values)


class OKXArchiveProvider:
    """Download, validate, and materialize OKX CSV history archives."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        timeout: float = 30,
        retries: int = 3,
        backoff: float = 0.5,
        max_archive_bytes: int = 8 * 1024 * 1024 * 1024,
        contract_sizes: Mapping[str, float | None] | None = None,
        allowed_instruments: set[str] | None = None,
    ) -> None:
        """Retain shared network and archive safety settings.

        Args:
            client: Shared HTTP connection pool.
            timeout: Per-attempt archive timeout.
            retries: Retries after the first attempt.
            backoff: Initial retry delay.
            max_archive_bytes: Maximum compressed object size.
            contract_sizes: Native derivatives mapped to contract face values.
            allowed_instruments: Optional product identities retained from shared files.
        """
        self.client = client
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.max_archive_bytes = max_archive_bytes
        self.contract_sizes = dict(contract_sizes or {})
        self.allowed_instruments = allowed_instruments

    def materialize(
        self, resource: ArchiveObject, destination: Path
    ) -> MaterializedArchive:
        """Download one OKX ZIP and publish canonical Parquet metadata.

        Args:
            resource: Physical archive selected by the planner.
            destination: Final local Parquet path.

        Returns:
            Materialization and all logical instrument partitions.
        """
        dataset = get_dataset(resource.key.product, resource.key.dataset)
        if dataset.name in {"order_book_400", "order_book_5000"}:
            from veldra.okx.orderbook import materialize_order_book

            return materialize_order_book(
                self.client,
                resource,
                dataset,
                destination,
                timeout=self.timeout,
                retries=self.retries,
                backoff=self.backoff,
                max_archive_bytes=self.max_archive_bytes,
                chunk_events=16 if dataset.name == "order_book_400" else 4,
            )
        if dataset.name == "legacy_order_book_50":
            from veldra.okx.legacy_orderbook import materialize_legacy_order_book

            return materialize_legacy_order_book(
                self.client,
                resource,
                dataset,
                destination,
                timeout=self.timeout,
                retries=self.retries,
                backoff=self.backoff,
                max_archive_bytes=self.max_archive_bytes,
            )
        if resource.integrity is None:
            raise DataValidationError("OKX archive has no integrity policy")
        start, end = _source_coverage(resource, dataset)
        legacy = Resource(
            resource.key.period_start,
            resource.url,
            None,
            integrity=resource.integrity,
            end_day=resource.key.period_end,
            coverage_start=start,
            coverage_end=end,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        partial.unlink(missing_ok=True)
        with TemporaryDirectory(prefix="veldra-okx-") as directory:
            archive_path = Path(directory) / resource.key.remote_name
            digest = download(
                self.client,
                legacy,
                archive_path,
                timeout=self.timeout,
                retries=self.retries,
                backoff=self.backoff,
                max_bytes=self.max_archive_bytes,
            )
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    member = _member(archive, resource)
                    with archive.open(member) as source:
                        raw = pd.read_csv(source)
            except (zipfile.BadZipFile, UnicodeError) as error:
                raise DataValidationError(
                    "OKX archive is not a readable ZIP/CSV"
                ) from error
            frame = normalize(raw, dataset, self.contract_sizes)
            if self.allowed_instruments is not None:
                frame = frame[frame["instrument_id"].isin(self.allowed_instruments)]
                if frame.empty:
                    raise DataValidationError(
                        "OKX archive contains no instruments for the requested product"
                    )
            if (
                frame[dataset.time_column].min() < start
                or frame[dataset.time_column].max() >= end
            ):
                raise DataValidationError(
                    "OKX rows fall outside the declared source period"
                )
            frame.to_parquet(partial, compression="zstd", index=False)
        partial.replace(destination)
        stat = destination.stat()
        first = frame[dataset.time_column].min().to_pydatetime()
        last = frame[dataset.time_column].max().to_pydatetime()
        materialization = Materialization(
            resource.key,
            destination,
            dataset.schema_version,
            len(frame),
            first,
            last,
            stat.st_size,
            local_mtime_ns=stat.st_mtime_ns,
            archive_revision=digest,
        )
        return MaterializedArchive(
            materialization,
            _partitions(resource, destination, frame, dataset),
        )
