"""Test the declarative Binance Spot kline dataset contract."""

import pytest

from crypto_downloader.datasets import (
    CM_INDEX_PRICE_KLINES,
    CM_MARK_PRICE_KLINES,
    CM_KLINES,
    CM_AGG_TRADES,
    CM_TRADES,
    UM_INDEX_PRICE_KLINES,
    UM_MARK_PRICE_KLINES,
    UM_KLINES,
    UM_AGG_TRADES,
    UM_TRADES,
    DatasetSpec,
    get_dataset,
)
from crypto_downloader.request import Request

SOURCE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
STORED_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)
OUTPUT_INTERVALS = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1mo",
)


def minimal_snapshot(**changes: object) -> DatasetSpec:
    """Build a valid interval-less schema with optional field overrides.

    Args:
        changes: Field values replacing the valid defaults.

    Returns:
        A compact valid dataset declaration unless an override is invalid.
    """
    values: dict[str, object] = {
        "product": "spot",
        "name": "snapshot",
        "remote_name": "snapshot",
        "source_columns": ("event_time", "value"),
        "stored_columns": ("event_time", "value"),
        "time_column": "event_time",
        "base_interval": None,
        "output_intervals": (),
        "aliases": {},
    }
    values.update(changes)
    return DatasetSpec(**values)  # type: ignore[arg-type]


def spot_klines() -> DatasetSpec:
    """Return the Spot kline specification used by these tests.

    Returns:
        The registered Binance Spot kline specification.
    """
    return get_dataset("spot", "klines")


def test_spot_kline_schema_matches_the_daily_archive_and_cache() -> None:
    """Confirm the exact source and canonical cached kline schemas."""
    spec = spot_klines()

    assert spec.product == "spot"
    assert spec.name == "klines"
    assert spec.remote_name == "klines"
    assert spec.source_columns == SOURCE_COLUMNS
    assert spec.stored_columns == STORED_COLUMNS
    assert spec.time_column == "open_time"
    assert spec.base_interval == "1m"
    assert spec.output_intervals == OUTPUT_INTERVALS
    assert spec.max_concurrency == 32
    assert spec.csv_header == "absent"
    assert spec.schema_version == 1
    assert spec.supports_resampling is True
    assert spec.supports_gap_policy is True
    assert spec.resample_sum_columns == (
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    )
    assert spec.ordering_columns == ("open_time",)
    assert spec.timestamp_columns == ("open_time", "close_time")
    assert spec.integer_columns == ("trade_count",)
    assert spec.boolean_columns == ()
    assert spec.needs_interval is True
    assert spec.csv_header_row is None
    assert spec.storage_interval == "1m"
    assert spec.output_columns == (*STORED_COLUMNS, "is_synthetic")
    assert "ignore" not in spec.stored_columns
    assert "ignore" not in spec.output_columns


def test_interval_less_snapshot_capabilities_are_declared_without_a_subclass() -> None:
    """Confirm a declarative spec can describe a raw header-bearing dataset."""
    snapshot = minimal_snapshot(
        csv_header="present",
        schema_version=3,
        ordering_columns=("event_time", "value"),
    )

    assert snapshot.needs_interval is False
    assert snapshot.csv_header_row == 0
    assert snapshot.storage_interval == "raw"
    assert snapshot.supports_resampling is False
    assert snapshot.supports_gap_policy is False
    assert snapshot.output_columns == ("event_time", "value")
    assert snapshot.timestamp_columns == ("event_time",)


def test_capabilities_apply_dataset_specific_interval_and_gap_defaults() -> None:
    """Confirm raw datasets reject candle-only request options."""
    snapshot = minimal_snapshot()

    assert spot_klines().resolve_interval(None) == "1m"
    assert spot_klines().resolve_gap_policy(None) == "forward"
    assert snapshot.resolve_interval(None) is None
    assert snapshot.resolve_gap_policy(None) is None
    with pytest.raises(ValueError, match="does not accept an interval"):
        snapshot.resolve_interval("1m")
    with pytest.raises(ValueError, match="does not accept gap_policy"):
        snapshot.resolve_gap_policy("forward")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"csv_header": "unexpected"}, "csv_header"),
        ({"source_columns": ()}, "columns"),
        ({"stored_columns": ()}, "columns"),
        ({"time_column": "missing"}, "time_column"),
        ({"base_interval": ""}, "base_interval"),
        ({"base_interval": "1m", "output_intervals": ()}, "interval datasets"),
        ({"output_intervals": ("1m",)}, "interval-less"),
        ({"supports_resampling": True}, "raw datasets"),
        ({"supports_gap_policy": True}, "raw datasets"),
        ({"resample_sum_columns": ("value",)}, "non-resampled"),
        ({"ordering_columns": ("missing",)}, "ordering columns"),
        ({"timestamp_columns": ("missing",)}, "timestamp columns"),
        ({"integer_columns": ("missing",)}, "integer columns"),
        ({"boolean_columns": ("missing",)}, "boolean columns"),
        ({"timestamp_columns": ("value",)}, "time_column"),
        ({"schema_version": 0}, "schema_version"),
        ({"schema_version": True}, "schema_version"),
    ],
)
def test_invalid_dataset_capabilities_are_rejected(
    changes: dict[str, object], message: str
) -> None:
    """Confirm contradictory declarative capability values fail at construction.

    Args:
        changes: The invalid fields applied to a valid raw dataset.
        message: The fragment expected in the validation error.
    """
    with pytest.raises(ValueError, match=message):
        minimal_snapshot(**changes)


@pytest.mark.parametrize("interval", OUTPUT_INTERVALS)
def test_supported_output_intervals_are_returned(interval: str) -> None:
    """Confirm every declared Spot kline output interval is accepted.

    Args:
        interval: One supported output interval.
    """
    assert spot_klines().resolve_interval(interval) == interval


@pytest.mark.parametrize(
    "interval",
    [1, True, [], "", "1s", "7m", "1M", "1month"],
)
def test_unsupported_output_intervals_are_rejected(interval: object) -> None:
    """Confirm malformed, finer, and arbitrary output intervals fail.

    Args:
        interval: An interval unavailable for stored one-minute Spot klines.
    """
    with pytest.raises((TypeError, ValueError), match="interval"):
        spot_klines().resolve_interval(interval)


def test_finer_than_stored_intervals_explain_the_constraint() -> None:
    """Confirm second intervals report why they cannot be returned."""
    with pytest.raises(ValueError, match="finer than stored 1m data"):
        spot_klines().resolve_interval("30s")


def test_default_columns_use_the_complete_canonical_output_schema() -> None:
    """Confirm omitted column selections return every available output column."""
    spec = spot_klines()

    assert spec.resolve_columns(None) == {
        column: column for column in spec.output_columns
    }


def test_selected_columns_aliases_and_output_labels_are_resolved() -> None:
    """Confirm aliases select canonical data while retaining requested labels."""
    request = Request.parse(
        "BTCUSDT",
        "2025-01-01",
        "2025-01-01",
        desired_columns={"base_volume": "size", "close": "closing price"},
    )

    columns = spot_klines().resolve_columns(request.columns)

    assert columns == {"volume": "size", "close": "closing price"}


def test_unknown_and_duplicate_resolved_columns_are_rejected() -> None:
    """Confirm unknown names and two names for one stored column fail."""
    spec = spot_klines()

    with pytest.raises(ValueError, match="unknown column"):
        spec.resolve_columns({"ignore": "ignore"})
    with pytest.raises(ValueError, match="duplicate column"):
        spec.resolve_columns({"volume": "volume", "base_volume": "base_volume"})


def test_perpetual_kline_schemas_declare_product_specific_quantity_units() -> None:
    """Confirm USD-M and COIN-M Klines do not expose ambiguous volume fields."""
    assert get_dataset("um", "klines") is UM_KLINES
    assert get_dataset("cm", "klines") is CM_KLINES
    assert UM_KLINES.csv_header == "present"
    assert CM_KLINES.csv_header == "present"
    assert UM_KLINES.stored_columns[5] == "base_volume"
    assert CM_KLINES.stored_columns[5] == "contract_volume"
    assert "volume" not in UM_KLINES.stored_columns
    assert "volume" not in CM_KLINES.stored_columns


def test_perpetual_mark_price_schemas_keep_only_price_and_sample_fields() -> None:
    """Confirm Futures mark-price candles discard structural volume fields."""
    for product, specification in (
        ("um", UM_MARK_PRICE_KLINES),
        ("cm", CM_MARK_PRICE_KLINES),
    ):
        assert get_dataset(product, "mark_price_klines") is specification
        assert specification.remote_name == "markPriceKlines"
        assert specification.csv_header == "present"
        assert specification.stored_columns == (
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "close_time",
            "sample_count",
        )
        assert specification.resample_sum_columns == ("sample_count",)
        assert specification.resolve_columns({"count": "samples"}) == {
            "sample_count": "samples"
        }


def test_perpetual_index_price_schemas_declare_their_archive_identifiers() -> None:
    """Confirm CM index archives use the pair while UM uses the contract symbol."""
    assert get_dataset("um", "index_price_klines") is UM_INDEX_PRICE_KLINES
    assert get_dataset("cm", "index_price_klines") is CM_INDEX_PRICE_KLINES
    assert UM_INDEX_PRICE_KLINES.remote_name == "indexPriceKlines"
    assert CM_INDEX_PRICE_KLINES.remote_name == "indexPriceKlines"
    assert UM_INDEX_PRICE_KLINES.archive_symbol_attribute == "symbol"
    assert CM_INDEX_PRICE_KLINES.archive_symbol_attribute == "pair"
    assert CM_INDEX_PRICE_KLINES.stored_columns == UM_INDEX_PRICE_KLINES.stored_columns


def test_perpetual_trade_schemas_declare_native_and_derived_quantity_units() -> None:
    """Confirm Futures trade schemas describe their distinct source semantics."""
    assert get_dataset("um", "trades") is UM_TRADES
    assert get_dataset("cm", "trades") is CM_TRADES
    assert UM_TRADES.csv_header == "present"
    assert CM_TRADES.csv_header == "present"
    assert UM_TRADES.requires_contract_size is False
    assert CM_TRADES.requires_contract_size is True
    assert UM_TRADES.stored_columns == (
        "trade_id",
        "price",
        "base_quantity",
        "quote_quantity",
        "event_time",
        "buyer_is_maker",
    )
    assert CM_TRADES.stored_columns == (
        "trade_id",
        "price",
        "contract_quantity",
        "base_quantity",
        "quote_notional",
        "event_time",
        "buyer_is_maker",
    )


def test_perpetual_aggregate_trade_schemas_preserve_component_identifiers() -> None:
    """Confirm Futures aggregate schemas retain IDs and explicit quantity units."""
    assert get_dataset("um", "agg_trades") is UM_AGG_TRADES
    assert get_dataset("cm", "agg_trades") is CM_AGG_TRADES
    assert UM_AGG_TRADES.integer_columns == (
        "agg_trade_id",
        "first_trade_id",
        "last_trade_id",
    )
    assert CM_AGG_TRADES.integer_columns == UM_AGG_TRADES.integer_columns
    assert UM_AGG_TRADES.stored_columns[4:6] == ("base_quantity", "quote_quantity")
    assert CM_AGG_TRADES.stored_columns[4:7] == (
        "contract_quantity",
        "base_quantity",
        "quote_notional",
    )
    assert CM_AGG_TRADES.requires_contract_size is True


@pytest.mark.parametrize(
    ("product", "dataset"),
    [
        ("spot", "not_a_dataset"),
        ("unknown", "unknown"),
    ],
)
def test_unimplemented_product_dataset_combinations_are_rejected(
    product: str, dataset: str
) -> None:
    """Confirm unsupported product and dataset combinations are rejected.

    Args:
        product: An unimplemented product identifier.
        dataset: An unimplemented dataset identifier.
    """
    with pytest.raises(ValueError, match=f"{product}/{dataset}"):
        get_dataset(product, dataset)


@pytest.mark.parametrize(
    ("product", "dataset", "field"),
    [(None, "klines", "product"), ("spot", None, "dataset"), (1, "klines", "product")],
)
def test_dataset_lookup_rejects_non_string_identifiers(
    product: object, dataset: object, field: str
) -> None:
    """Confirm direct registry calls reject non-string identifiers.

    Args:
        product: The product value supplied to the registry.
        dataset: The dataset value supplied to the registry.
        field: The invalid field named in the expected error.
    """
    with pytest.raises(TypeError, match=field):
        get_dataset(product, dataset)
