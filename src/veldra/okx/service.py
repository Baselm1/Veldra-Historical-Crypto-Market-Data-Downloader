"""Coordinate OKX archive planning, caching, and DuckDB queries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import perf_counter
import logging

import httpx
import pandas as pd

from veldra.core.catalog import Catalog, catalog_lock, open_catalog
from veldra.core.config import load_settings
from veldra.core.datasets import DatasetSpec
from veldra.core.matching import exact_markets, suggest_symbols
from veldra.core.models import ArchiveObject, LogicalPartition, Market, Message, Result
from veldra.core.providers import MaterializedArchive
from veldra.core.query import ParquetInput, empty_frame, query_parquet
from veldra.core.reporting import Reporter
from veldra.core.request import Request, parse_range, parse_timestamp
from veldra.core.subjects import DataSubject
from veldra.okx.client import OKXClient
from veldra.okx.chain import OptionChainFilter, query_chain
from veldra.okx.connector import OKXConnector
from veldra.okx.datasets import get_dataset, manifest_spec
from veldra.okx.manifest import OKXManifestDiscovery
from veldra.okx.identities import historical_future, historical_option, parse_currency
from veldra.okx.planner import OKXArchivePlanner
from veldra.okx.processing import OKXArchiveProvider
from veldra.okx.reports import CacheReport
from veldra.okx.rest import OKXRESTHistory

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MaterializeOutcome:
    """Collect physical successes and isolated archive problems."""

    completed: tuple[MaterializedArchive, ...]
    problems: tuple[Message, ...]


@dataclass(frozen=True)
class _ChainCache:
    """Collect markets, family partitions, and retrieval problems."""

    markets: tuple[Market, ...]
    partitions: tuple[LogicalPartition, ...]
    problems: tuple[Message, ...]


class OKXService:
    """Run declarative OKX requests over remote archives and local Parquet."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        config_path: str | Path | None,
        earliest_date: object,
        max_workers: int,
        market_refresh_hours: float,
        timeout: float,
        retries: int,
        backoff: float,
        progress: bool,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create an OKX request service without performing I/O.

        Args:
            data_dir: Catalog and Parquet root.
            config_path: Optional installed-settings override.
            earliest_date: Optional usable-history boundary or ``"all"``.
            max_workers: Maximum concurrent physical archive workers.
            market_refresh_hours: Hours current instruments remain fresh.
            timeout: Per-attempt source timeout.
            retries: Retries after a transient failure.
            backoff: Initial retry delay.
            progress: Whether Rich progress is shown.
            transport: Optional HTTPX transport used by tests or callers.
        """
        if not isinstance(data_dir, (str, Path)) or not str(data_dir).strip():
            raise ValueError("data_dir must be a nonempty path")
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")
        if market_refresh_hours <= 0:
            raise ValueError("market_refresh_hours must be positive")
        if not isinstance(progress, bool):
            raise TypeError("progress must be a Boolean")
        settings = load_settings(config_path)
        selected = settings.earliest_date if earliest_date is None else earliest_date
        self.earliest_date = self._history_date(selected)
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.max_workers = max_workers
        self.market_refresh_hours = float(market_refresh_hours)
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.progress = progress
        self.transport = transport
        self.connector = OKXConnector(timeout=timeout, retries=retries, backoff=backoff)

    @staticmethod
    def _history_date(value: object) -> date | None:
        """Parse a configured history boundary.

        Args:
            value: ISO date, date object, ``None``, or ``"all"``.

        Returns:
            UTC date boundary or ``None`` for all source history.
        """
        if value is None or value == "all":
            return None
        parsed = parse_timestamp(value)
        if parsed.time() != datetime.min.time():
            raise ValueError("earliest_date must be a UTC day or 'all'")
        return parsed.date()

    def _client(self, *, offline: bool) -> httpx.Client:
        """Create one request-scoped HTTP connection pool.

        Args:
            offline: Whether every attempted remote request must fail.

        Returns:
            Configured shared HTTPX client.
        """
        if offline:

            def reject(request: httpx.Request) -> httpx.Response:
                """Reject accidental offline network access."""
                raise RuntimeError(f"offline mode attempted {request.url}")

            transport: httpx.BaseTransport = httpx.MockTransport(reject)
        elif self.transport is not None:
            transport = self.transport
        else:
            transport = httpx.HTTPTransport(retries=0)
        limits = httpx.Limits(
            max_connections=self.max_workers,
            max_keepalive_connections=self.max_workers,
        )
        return httpx.Client(transport=transport, follow_redirects=True, limits=limits)

    def _markets(
        self,
        catalog: Catalog,
        client: httpx.Client,
        product: str,
        reporter: Reporter,
        *,
        refresh: bool,
        offline: bool,
    ) -> list[Market]:
        """Load fresh current instruments or their cached snapshot.

        Args:
            catalog: Open local catalog.
            client: Shared request client.
            product: Requested OKX product.
            reporter: Optional Rich reporter.
            refresh: Whether current metadata must be refreshed.
            offline: Whether network access is forbidden.

        Returns:
            Product-scoped current and cached instruments.
        """
        markets = catalog.markets("okx", product)
        snapshot = catalog.market_snapshot_at("okx", product)
        fresh_after = datetime.now(UTC) - timedelta(hours=self.market_refresh_hours)
        stale = not markets or snapshot is None or snapshot < fresh_after
        if not offline and (refresh or stale):
            with reporter.status(f"Refreshing OKX {product} markets"):
                markets = self.connector.markets(client, product)
            catalog.save_markets("okx", product, markets)
            reporter.market_summary(markets, refreshed=True)
        elif not markets:
            raise RuntimeError("offline mode requires cached OKX market metadata")
        else:
            reporter.market_summary(markets, refreshed=False)
        return markets

    @staticmethod
    def _resolve(
        pair: str, markets: list[Market]
    ) -> tuple[Market | None, Message | None]:
        """Resolve one exact market or return an error with suggestions.

        Args:
            pair: Caller-provided instrument spelling.
            markets: Product-scoped current instruments.

        Returns:
            Exact market or a structured resolution error.
        """
        matches = exact_markets(pair, markets)
        if len(matches) == 1:
            return matches[0], None
        suggestions = (
            tuple(sorted(item.symbol for item in matches))
            if matches
            else suggest_symbols(pair, markets)
        )
        code = "ambiguous_pair" if matches else "unknown_pair"
        message = (
            f"Pair '{pair}' matches more than one OKX instrument."
            if matches
            else f"Pair '{pair}' was not found."
        )
        return None, Message(code, message, suggestions=suggestions)

    @staticmethod
    def _historical_market(pair: str, product: str) -> Market | None:
        """Return an archive-derived expired Futures market when valid.

        Args:
            pair: Caller-provided native contract ID.
            product: Requested product.

        Returns:
            Conservative archive-only market or ``None``.
        """
        try:
            if product in {"linear_futures", "inverse_futures"}:
                return historical_future(pair, product).market
            if product == "options":
                return historical_option(pair).market
            return None
        except TypeError, ValueError:
            return None

    def _effective_start(self, request: Request) -> datetime:
        """Apply the optional configured history boundary.

        Args:
            request: Validated logical request.

        Returns:
            Effective inclusive UTC start.
        """
        if self.earliest_date is None:
            return request.start
        boundary = datetime.combine(self.earliest_date, datetime.min.time(), UTC)
        return max(request.start, boundary)

    @staticmethod
    def _source_days(
        start: datetime, end: datetime, dataset: DatasetSpec
    ) -> tuple[date, date]:
        """Convert an exact UTC request to inclusive OKX source dates.

        Args:
            start: Inclusive UTC timestamp.
            end: Exclusive UTC timestamp.
            dataset: Source-calendar declaration.

        Returns:
            First and last source calendar dates.
        """
        final = end - timedelta(microseconds=1)
        return (
            (start + dataset.archive_day_offset).date(),
            (final + dataset.archive_day_offset).date(),
        )

    def _destination(self, archive_id: str, product: str, dataset: str) -> Path:
        """Return a stable local Parquet path for one physical archive.

        Args:
            archive_id: Stable physical archive identifier.
            product: Source product.
            dataset: Historical dataset.

        Returns:
            Local materialization path.
        """
        return self.data_dir / "okx" / product / dataset / f"{archive_id}.parquet"

    def _materialize(
        self,
        catalog: Catalog,
        client: httpx.Client,
        selected: list[ArchiveObject],
        dataset: DatasetSpec,
        markets: list[Market],
        reporter: Reporter,
    ) -> _MaterializeOutcome:
        """Materialize selected archives concurrently and publish successes.

        Args:
            catalog: Open catalog receiving completed files.
            client: Shared request connection pool.
            selected: Remote physical archives selected by the planner.
            dataset: Canonical dataset declaration.
            markets: Current product markets carrying contract metadata.
            reporter: Optional Rich progress reporter.

        Returns:
            Completed materializations and isolated archive failures.
        """
        if not selected:
            return _MaterializeOutcome((), ())
        catalog.save_archives(selected)
        provider = OKXArchiveProvider(
            client,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
            contract_sizes={market.symbol: market.contract_size for market in markets},
            allowed_instruments=(
                {market.symbol for market in markets}
                if dataset.product in {"spot", "linear_swap", "inverse_swap"}
                else None
            ),
        )

        def process(resource: ArchiveObject) -> MaterializedArchive:
            """Materialize one physical object outside the catalog transaction."""
            path = self._destination(
                resource.key.archive_id, resource.key.product, resource.key.dataset
            )
            return provider.materialize(resource, path)

        workers = min(self.max_workers, dataset.max_concurrency, len(selected))
        with reporter.downloads("OKX", len(selected)) as advance:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process, item) for item in selected]
                outcomes: list[
                    tuple[
                        ArchiveObject,
                        MaterializedArchive | None,
                        BaseException | None,
                    ]
                ] = []
                for resource, future in zip(selected, futures):
                    try:
                        outcomes.append((resource, future.result(), None))
                        advance(resource.key.period_start, True)
                    except BaseException as caught_error:
                        outcomes.append((resource, None, caught_error))
                        advance(resource.key.period_start, False)
        problems: list[Message] = []
        completed: list[MaterializedArchive] = []
        for resource, value, error in outcomes:
            if error is not None:
                catalog.mark_archive_failed(resource.key, str(error))
                problems.append(
                    Message(
                        "archive_failed",
                        f"{resource.key.remote_name}: {error}",
                        resource.key.period_start,
                    )
                )
            else:
                assert value is not None
                catalog.publish_materialization(value.materialization, value.partitions)
                completed.append(value)
        return _MaterializeOutcome(tuple(completed), tuple(problems))

    @staticmethod
    def _inputs(partitions: list[LogicalPartition]) -> list[ParquetInput]:
        """Deduplicate physical file predicates for one logical query.

        Args:
            partitions: Cataloged logical partitions.

        Returns:
            Unique Parquet inputs in coverage order.
        """
        values: dict[tuple[Path, str | None, str | None], ParquetInput] = {}
        for item in partitions:
            key = (
                item.materialization_path,
                item.predicate_column,
                item.predicate_value,
            )
            values[key] = ParquetInput(*key)
        return list(values.values())

    def _pair(
        self,
        catalog: Catalog,
        client: httpx.Client,
        market: Market,
        markets: list[Market],
        original_pair: str,
        request: Request,
        dataset: DatasetSpec,
        reporter: Reporter,
        *,
        transport: str,
        offline: bool,
    ) -> Result:
        """Retrieve one resolved OKX instrument.

        Args:
            catalog: Open local catalog.
            client: Shared request client.
            market: Resolved current OKX instrument.
            markets: Current product markets carrying shared archive metadata.
            original_pair: Caller spelling retained in errors.
            request: Validated logical request.
            dataset: Canonical dataset declaration.
            reporter: Optional Rich reporter.
            transport: Automatic, specific, or bulk archive selection.
            offline: Whether all network access is forbidden.

        Returns:
            Data and structured retrieval report.
        """
        result = Result(
            market.symbol,
            empty_frame(dataset, request.columns or {}),
            (request.start, request.end),
            source="okx",
            product=request.product,
            dataset=request.dataset,
            gap_policy=request.gap_policy,
        )
        start = self._effective_start(request)
        if start >= request.end:
            result.warnings.append(
                Message("configured_start", "Request ends before configured history.")
            )
            return result
        subject = DataSubject("instrument", market.symbol)
        physical_subject = (
            subject
            if manifest_spec(request.product, request.dataset).subject_kind
            == "instrument"
            else DataSubject("instrument_family", market.pair or market.symbol)
        )
        first_day, last_day = self._source_days(start, request.end, dataset)
        if not offline:
            cached = catalog.ready_archives_between(
                "okx", request.product, request.dataset, first_day, last_day
            )
            discovery = OKXManifestDiscovery(
                OKXClient(
                    client=client,
                    limiter=self.connector.limiter,
                    timeout=self.timeout,
                    retries=self.retries,
                    backoff=self.backoff,
                )
            )
            plan = OKXArchivePlanner(discovery, cached=cached).plan(
                request.product,
                request.dataset,
                [physical_subject],
                first_day,
                last_day,
                transport=transport,
            )
            LOGGER.info("OKX archive plan: %s", plan.explanation)
            outcome = self._materialize(
                catalog, client, list(plan.selected), dataset, markets, reporter
            )
            result.problems.extend(outcome.problems)
        partitions = catalog.partitions_between(
            "okx",
            request.product,
            request.dataset,
            subject,
            dataset.base_interval,
            start,
            request.end,
        )
        result.data = query_parquet(
            catalog.connection,
            self._inputs(partitions),
            dataset,
            start,
            request.end,
            request.columns or {},
            gap_policy=request.gap_policy,
            interval=request.interval,
        )
        if partitions:
            result.available_range = (
                min(item.coverage_start for item in partitions),
                max(item.coverage_end for item in partitions),
            )
        if not result.data.empty:
            result.used_range = (start, request.end)
            if "is_synthetic" in result.data and result.data["is_synthetic"].any():
                count = int(result.data["is_synthetic"].sum())
                result.problems.append(
                    Message("missing_candles", f"Filled {count} missing OKX candle(s).")
                )
        elif not result.problems:
            result.problems.append(
                Message("no_data", f"No OKX data was found for '{original_pair}'.")
            )
        return result

    def get_results(
        self,
        pairs: object,
        start: object,
        end: object,
        *,
        product: object,
        dataset: object,
        interval: object,
        columns: object,
        gap_policy: object,
        transport: object,
        refresh: object,
        offline: object,
    ) -> Result | list[Result]:
        """Return structured OKX results for one or several instruments.

        Args:
            pairs: One instrument or ordered instrument list.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: OKX product.
            dataset: Historical dataset.
            interval: Optional output Kline interval.
            columns: Optional selected or renamed columns.
            gap_policy: Missing-candle behavior.
            transport: Automatic, specific, or bulk archive selection.
            refresh: Whether current markets must refresh.
            offline: Whether source access is forbidden.

        Returns:
            One result or a list matching input shape and order.
        """
        if not isinstance(transport, str) or transport not in {
            "auto",
            "specific",
            "bulk",
        }:
            raise ValueError("transport must be 'auto', 'specific', or 'bulk'")
        if not isinstance(refresh, bool) or not isinstance(offline, bool):
            raise TypeError("refresh and offline must be Booleans")
        if refresh and offline:
            raise ValueError("refresh and offline cannot both be enabled")
        request = Request.parse(
            pairs,
            start,
            end,
            product=product,
            dataset=dataset,
            interval=interval,
            desired_columns=columns,
            gap_policy=gap_policy,
        )
        specification = get_dataset(request.product, request.dataset)
        request = request.resolve_dataset(specification)
        reporter = Reporter(self.progress)
        reporter.request(
            "okx",
            request.product,
            request.dataset,
            len(request.pairs),
            request.start,
            request.end,
        )
        catalog_path = self.data_dir / "catalog.duckdb"
        started = perf_counter()
        with catalog_lock(catalog_path):
            with self._client(offline=offline) as client:
                with open_catalog(catalog_path) as catalog:
                    markets = self._markets(
                        catalog,
                        client,
                        request.product,
                        reporter,
                        refresh=refresh,
                        offline=offline,
                    )
                    results: list[Result] = []
                    for pair in request.pairs:
                        market, error = self._resolve(pair, markets)
                        if market is None:
                            market = self._historical_market(pair, request.product)
                        if market is None:
                            result = Result(
                                pair,
                                empty_frame(specification, request.columns or {}),
                                (request.start, request.end),
                                source="okx",
                                product=request.product,
                                dataset=request.dataset,
                                gap_policy=request.gap_policy,
                            )
                            assert error is not None
                            result.errors.append(error)
                        else:
                            result = self._pair(
                                catalog,
                                client,
                                market,
                                markets,
                                pair,
                                request,
                                specification,
                                reporter,
                                transport=transport,
                                offline=offline,
                            )
                        results.append(result)
        LOGGER.info("OKX request completed in %.3fs", perf_counter() - started)
        return results[0] if request.single else results

    def get_currency_results(
        self,
        currencies: object,
        start: object,
        end: object,
        *,
        transport: object,
        offline: object,
    ) -> Result | list[Result]:
        """Return currency-scoped historical margin borrowing rates.

        Args:
            currencies: One currency or an ordered currency list.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            transport: Automatic, specific, or bulk archive selection.
            offline: Whether source access is forbidden.

        Returns:
            One structured result or a list matching the input shape.
        """
        if transport not in {"auto", "specific", "bulk"}:
            raise ValueError("transport must be 'auto', 'specific', or 'bulk'")
        if not isinstance(offline, bool):
            raise TypeError("offline must be a Boolean")
        normalized: object
        if isinstance(currencies, str):
            normalized = parse_currency(currencies)
        elif isinstance(currencies, list):
            normalized = [parse_currency(value) for value in currencies]
        else:
            normalized = currencies
        request = Request.parse(
            normalized,
            start,
            end,
            product="margin",
            dataset="borrow_rates",
            interval=None,
            desired_columns=None,
            gap_policy=None,
            subject_kind="currency",
        )
        specification = get_dataset("margin", "borrow_rates")
        request = request.resolve_dataset(specification)
        catalog_path = self.data_dir / "catalog.duckdb"
        with catalog_lock(catalog_path):
            with self._client(offline=offline) as client:
                with open_catalog(catalog_path) as catalog:
                    results = [
                        self._currency_result(
                            catalog,
                            client,
                            subject,
                            request,
                            specification,
                            transport=str(transport),
                            offline=offline,
                        )
                        for subject in request.subjects
                    ]
        return results[0] if request.single else results

    def _currency_result(
        self,
        catalog: Catalog,
        client: httpx.Client,
        subject: DataSubject,
        request: Request,
        dataset: DatasetSpec,
        *,
        transport: str,
        offline: bool,
    ) -> Result:
        """Materialize and query one currency-scoped borrowing series.

        Args:
            catalog: Open source catalog.
            client: Shared request client.
            subject: Requested margin currency.
            request: Resolved logical request.
            dataset: Borrowing-rate declaration.
            transport: Archive transport selection.
            offline: Whether source access is forbidden.

        Returns:
            Structured borrowing-rate result.
        """
        result = Result(
            subject.value,
            empty_frame(dataset, request.columns or {}),
            (request.start, request.end),
            source="okx",
            product="margin",
            dataset="borrow_rates",
            gap_policy=None,
        )
        start = self._effective_start(request)
        if start >= request.end:
            result.warnings.append(
                Message("configured_start", "Request ends before configured history.")
            )
            return result
        first_day, last_day = self._source_days(start, request.end, dataset)
        if not offline:
            cached = catalog.ready_archives_between(
                "okx", "margin", "borrow_rates", first_day, last_day
            )
            api = OKXClient(
                client=client,
                limiter=self.connector.limiter,
                timeout=self.timeout,
                retries=self.retries,
                backoff=self.backoff,
            )
            plan = OKXArchivePlanner(OKXManifestDiscovery(api), cached=cached).plan(
                "margin",
                "borrow_rates",
                [subject],
                first_day,
                last_day,
                transport=transport,
            )
            result.problems.extend(
                self._materialize(
                    catalog,
                    client,
                    list(plan.selected),
                    dataset,
                    [],
                    Reporter(self.progress),
                ).problems
            )
        partitions = catalog.partitions_between(
            "okx", "margin", "borrow_rates", subject, None, start, request.end
        )
        result.data = query_parquet(
            catalog.connection,
            self._inputs(partitions),
            dataset,
            start,
            request.end,
            request.columns or {},
            gap_policy=None,
            interval=None,
        )
        if partitions:
            result.available_range = (
                min(item.coverage_start for item in partitions),
                max(item.coverage_end for item in partitions),
            )
        if not result.data.empty:
            result.used_range = (start, request.end)
        elif not result.problems:
            result.problems.append(
                Message(
                    "no_data", f"No OKX borrowing data was found for '{subject.value}'."
                )
            )
        return result

    def get_chain(
        self,
        family: object,
        start: object,
        end: object,
        *,
        product: object,
        dataset: object,
        interval: object,
        columns: object,
        contract_style: object,
        option_filter: OptionChainFilter | None,
        refresh: object,
        offline: object,
    ) -> Result:
        """Return one derivative-family history retaining exact contracts.

        Args:
            family: Native OKX Futures family.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Futures or Options product.
            dataset: Klines or trades.
            interval: Optional Kline output interval.
            columns: Optional selected or renamed canonical columns.
            contract_style: Optional normal, X-Perp, or pre-market filter.
            option_filter: Optional expiry, strike, and call/put constraints.
            refresh: Whether current instruments must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Structured family result whose frame retains ``instrument_id``.
        """
        request, specification, native_family = self._chain_request(
            family,
            start,
            end,
            product=product,
            dataset=dataset,
            interval=interval,
            columns=columns,
            contract_style=contract_style,
            option_filter=option_filter,
            refresh=refresh,
            offline=offline,
        )
        start_time = self._effective_start(request)
        result = Result(
            native_family,
            pd.DataFrame(),
            (request.start, request.end),
            source="okx",
            product=request.product,
            dataset=request.dataset,
            gap_policy=request.gap_policy,
        )
        catalog_path = self.data_dir / "catalog.duckdb"
        reporter = Reporter(self.progress)
        with catalog_lock(catalog_path):
            with self._client(offline=bool(offline)) as client:
                with open_catalog(catalog_path) as catalog:
                    cached = self._prepare_chain(
                        catalog,
                        client,
                        request,
                        specification,
                        native_family,
                        start_time,
                        reporter,
                        refresh=bool(refresh),
                        offline=bool(offline),
                    )
                    result.problems.extend(cached.problems)
                    result.data = query_chain(
                        catalog.connection,
                        [item.materialization_path for item in cached.partitions],
                        specification,
                        start_time,
                        request.end,
                        request.columns or {},
                        interval=request.interval,
                        option_filter=option_filter,
                    )
        self._filter_chain_style(result.data, cached.markets, contract_style)
        self._finish_chain_result(
            result, cached.partitions, native_family, start_time, request.end
        )
        return result

    @staticmethod
    def _chain_request(
        family: object,
        start: object,
        end: object,
        *,
        product: object,
        dataset: object,
        interval: object,
        columns: object,
        contract_style: object,
        option_filter: OptionChainFilter | None,
        refresh: object,
        offline: object,
    ) -> tuple[Request, DatasetSpec, str]:
        """Validate and resolve one derivative-family request.

        Args:
            family: Native Futures family.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: Futures or Options product.
            dataset: Klines or trades.
            interval: Optional Kline output interval.
            columns: Optional selected or renamed columns.
            contract_style: Optional normal or X-Perp filter.
            option_filter: Optional Option chain constraints.
            refresh: Whether current instruments must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Resolved request, dataset, and normalized native family.
        """
        OKXService._validate_chain_options(product, contract_style, option_filter)
        if not isinstance(refresh, bool) or not isinstance(offline, bool):
            raise TypeError("refresh and offline must be Booleans")
        if refresh and offline:
            raise ValueError("refresh and offline cannot both be enabled")
        request = Request.parse(
            family,
            start,
            end,
            product=product,
            dataset=dataset,
            interval=interval,
            desired_columns=columns,
            gap_policy="keep" if dataset == "klines" else None,
            subject_kind="instrument_family",
        )
        specification = get_dataset(request.product, request.dataset)
        request = request.resolve_dataset(specification)
        native_family = request.subjects[0].value.strip().upper()
        expected_suffix = OKXService._chain_suffix(product)
        if not native_family.endswith(expected_suffix):
            raise ValueError("chain family does not match the requested product")
        return request, specification, native_family

    @staticmethod
    def _validate_chain_options(
        product: object,
        contract_style: object,
        option_filter: OptionChainFilter | None,
    ) -> None:
        """Reject incompatible derivative-family options.

        Args:
            product: Requested Futures or Options product.
            contract_style: Optional Futures contract style.
            option_filter: Optional Option chain constraints.
        """
        if product not in {"linear_futures", "inverse_futures", "options"}:
            raise ValueError("chain product must be Futures or Options")
        if contract_style not in {None, "normal", "xperp", "pre_market_xperp"}:
            raise ValueError("contract_style is unsupported")
        if product == "options" and contract_style is not None:
            raise ValueError("contract_style does not apply to Options")
        if product != "options" and option_filter is not None:
            raise ValueError("Option filters require the options product")

    @staticmethod
    def _chain_suffix(product: object) -> str | tuple[str, ...]:
        """Return valid native family suffixes for one derivative product.

        Args:
            product: Validated Futures or Options product.

        Returns:
            Accepted family suffix or suffixes.
        """
        if product == "inverse_futures":
            return "-USD"
        if product == "linear_futures":
            return "-USDT", "-USDC", "-USD_UM"
        return "-USD", "-USDT"

    def _prepare_chain(
        self,
        catalog: Catalog,
        client: httpx.Client,
        request: Request,
        specification: DatasetSpec,
        native_family: str,
        start_time: datetime,
        reporter: Reporter,
        *,
        refresh: bool,
        offline: bool,
    ) -> _ChainCache:
        """Materialize missing family archives and return query partitions.

        Args:
            catalog: Open source catalog.
            client: Shared request-scoped HTTP client.
            request: Resolved chain request.
            specification: Product-specific schema.
            native_family: Normalized manifest family.
            start_time: Configured inclusive request start.
            reporter: Optional Rich reporter.
            refresh: Whether current instruments must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Current metadata, family partitions, and materialization problems.
        """
        first_day, last_day = self._source_days(start_time, request.end, specification)
        markets = self._markets(
            catalog,
            client,
            request.product,
            reporter,
            refresh=refresh,
            offline=offline,
        )
        subject = DataSubject("instrument_family", native_family)
        problems: tuple[Message, ...] = ()
        if not offline:
            cached = catalog.ready_archives_between(
                "okx", request.product, request.dataset, first_day, last_day
            )
            api = OKXClient(
                client=client,
                limiter=self.connector.limiter,
                timeout=self.timeout,
                retries=self.retries,
                backoff=self.backoff,
            )
            plan = OKXArchivePlanner(OKXManifestDiscovery(api), cached=cached).plan(
                request.product,
                request.dataset,
                [subject],
                first_day,
                last_day,
                transport="specific",
            )
            problems = self._materialize(
                catalog,
                client,
                list(plan.selected),
                specification,
                markets,
                reporter,
            ).problems
        partitions = catalog.partitions_between(
            "okx",
            request.product,
            request.dataset,
            subject,
            specification.base_interval,
            start_time,
            request.end,
        )
        return _ChainCache(tuple(markets), tuple(partitions), problems)

    @staticmethod
    def _filter_chain_style(
        frame: pd.DataFrame, markets: tuple[Market, ...], contract_style: object
    ) -> None:
        """Filter a family frame in place to one current contract style.

        Args:
            frame: Queried family rows.
            markets: Current contracts with native rule types.
            contract_style: Optional requested style.
        """
        if contract_style is None or frame.empty:
            return
        known = {
            market.symbol: (market.contract_type or "NORMAL").lower()
            for market in markets
        }
        styles = frame["instrument_id"].map(known).fillna("normal")
        frame.drop(frame.index[~styles.eq(contract_style)], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    @staticmethod
    def _finish_chain_result(
        result: Result,
        partitions: tuple[LogicalPartition, ...],
        family: str,
        start: datetime,
        end: datetime,
    ) -> None:
        """Attach availability and empty-data information to a chain result.

        Args:
            result: Mutable structured result.
            partitions: Family partitions used by the query.
            family: Native family used in user-facing errors.
            start: Effective inclusive query start.
            end: Exclusive query end.
        """
        if not result.data.empty:
            result.used_range = (start, end)
        elif not result.problems:
            result.problems.append(
                Message("no_data", f"No OKX data was found for '{family}'.")
            )
        if partitions:
            result.available_range = (
                min(item.coverage_start for item in partitions),
                max(item.coverage_end for item in partitions),
            )

    def cache_all(
        self,
        start: object,
        end: object,
        *,
        product: object,
        dataset: object,
        dry_run: object,
        refresh: object,
        offline: object,
    ) -> CacheReport:
        """Cache one supported all-market archive dataset without returning rows.

        Args:
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            product: OKX product.
            dataset: Bulk-capable historical dataset.
            dry_run: Whether to discover but not download remote files.
            refresh: Whether current instrument metadata must refresh.
            offline: Whether source access is forbidden.

        Returns:
            Physical plan, completion, and local cache totals.
        """
        if (
            not isinstance(dry_run, bool)
            or not isinstance(refresh, bool)
            or not isinstance(offline, bool)
        ):
            raise TypeError("dry_run, refresh and offline must be Booleans")
        if refresh and offline:
            raise ValueError("refresh and offline cannot both be enabled")
        request = Request.parse(
            "ANY",
            start,
            end,
            product=product,
            dataset=dataset,
            interval=None,
            desired_columns=None,
            gap_policy="keep" if dataset == "klines" else None,
        )
        specification = get_dataset(request.product, request.dataset)
        request = request.resolve_dataset(specification)
        start_time = self._effective_start(request)
        if start_time >= request.end:
            raise ValueError("request ends before configured OKX history")
        first_day, last_day = self._source_days(start_time, request.end, specification)
        reporter = Reporter(self.progress)
        catalog_path = self.data_dir / "catalog.duckdb"
        with catalog_lock(catalog_path):
            with self._client(offline=offline) as client:
                with open_catalog(catalog_path) as catalog:
                    markets = self._markets(
                        catalog,
                        client,
                        request.product,
                        reporter,
                        refresh=refresh,
                        offline=offline,
                    )
                    cached = catalog.ready_archives_between(
                        "okx", request.product, request.dataset, first_day, last_day
                    )
                    if offline:
                        selected: tuple[ArchiveObject, ...] = ()
                        explanation = f"offline; {len(cached)} cached physical files"
                        outcome = _MaterializeOutcome((), ())
                    else:
                        discovery = OKXManifestDiscovery(
                            OKXClient(
                                client=client,
                                limiter=self.connector.limiter,
                                timeout=self.timeout,
                                retries=self.retries,
                                backoff=self.backoff,
                            )
                        )
                        plan = OKXArchivePlanner(discovery, cached=cached).plan(
                            request.product,
                            request.dataset,
                            [DataSubject("all", "ANY")],
                            first_day,
                            last_day,
                            transport="bulk",
                        )
                        selected = plan.selected
                        explanation = plan.explanation
                        outcome = (
                            _MaterializeOutcome((), ())
                            if dry_run
                            else self._materialize(
                                catalog,
                                client,
                                list(selected),
                                specification,
                                markets,
                                reporter,
                            )
                        )
                    subjects, rows, local_bytes = catalog.partition_totals_between(
                        "okx",
                        request.product,
                        request.dataset,
                        start_time,
                        request.end,
                    )
        downloaded = len(outcome.completed)
        failures = len(outcome.problems)
        remaining = len(selected) if dry_run else failures
        return CacheReport(
            "okx",
            request.product,
            request.dataset,
            (start_time, request.end),
            "bulk",
            explanation,
            len(cached),
            remaining,
            sum(item.remote_size or 0 for item in selected),
            downloaded,
            failures,
            subjects,
            rows,
            local_bytes,
            dry_run,
            offline,
        )

    def get_rest_history(
        self,
        subject_value: object,
        start: object,
        end: object,
        *,
        subject_kind: object,
        product: object,
        dataset: object,
        params: dict[str, str],
        interval: str | None,
        offline: object,
    ) -> pd.DataFrame:
        """Return one cached public OKX historical REST series.

        Args:
            subject_value: Native instrument, family, or currency scope.
            start: Inclusive request start.
            end: Inclusive date or exclusive timestamp end.
            subject_kind: Native logical subject kind.
            product: Catalog product label.
            dataset: Registered REST dataset.
            params: Validated endpoint query parameters.
            interval: Optional native Kline interval.
            offline: Whether source access is forbidden.

        Returns:
            Canonical historical rows from DuckDB-backed Parquet.
        """
        if not isinstance(subject_value, str):
            raise TypeError("REST history subject must be a string")
        if not isinstance(subject_kind, str) or not isinstance(product, str):
            raise TypeError("REST history kind and product must be strings")
        if not isinstance(dataset, str):
            raise TypeError("REST history dataset must be a string")
        if not isinstance(offline, bool):
            raise TypeError("offline must be a Boolean")
        range_start, range_end = parse_range(start, end)
        subject = DataSubject(subject_kind, subject_value.strip().upper())  # type: ignore[arg-type]
        catalog_path = self.data_dir / "catalog.duckdb"
        with catalog_lock(catalog_path):
            with self._client(offline=offline) as client:
                api = OKXClient(
                    client=client,
                    limiter=self.connector.limiter,
                    timeout=self.timeout,
                    retries=self.retries,
                    backoff=self.backoff,
                )
                with open_catalog(catalog_path) as catalog:
                    return OKXRESTHistory(api, catalog, self.data_dir).get(
                        dataset,
                        subject,
                        range_start,
                        range_end,
                        product=product,
                        params=params,
                        interval=interval,
                        offline=offline,
                    )
