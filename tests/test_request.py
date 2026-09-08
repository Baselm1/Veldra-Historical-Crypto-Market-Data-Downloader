"""Test source-independent downloader request validation."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from crypto_downloader._core.datasets import DatasetSpec
from crypto_downloader.binance.datasets import get_dataset
from crypto_downloader._core.request import Request, normalize_pair, parse_timestamp

UTC = timezone.utc


def parse_request(
    pairs: object = "BTCUSDT",
    starting_date: object = "2024-01-01",
    end_date: object = "2024-01-01",
    *,
    interval: object = None,
    desired_columns: object = None,
    base_interval: object = "1m",
    product: object = "spot",
    dataset: object = "klines",
) -> Request:
    """Create a request with useful defaults for validation tests.

    Args:
        pairs: One pair string or a list of pair strings.
        starting_date: The requested first date or timestamp.
        end_date: The requested inclusive date or exclusive timestamp.
        interval: The optional requested output interval.
        desired_columns: Optional column names or output labels.
        base_interval: The configured default interval.
        product: The source product identifier.
        dataset: The dataset identifier.

    Returns:
        The parsed request.
    """
    return Request.parse(
        pairs,
        starting_date,
        end_date,
        interval=interval,
        desired_columns=desired_columns,
        base_interval=base_interval,
        product=product,
        dataset=dataset,
    )


def test_defaults_and_single_pair_shape() -> None:
    """Confirm the default request values and one-pair return shape."""
    request = parse_request()

    assert request.pairs == ("BTCUSDT",)
    assert request.single is True
    assert request.start == datetime(2024, 1, 1, tzinfo=UTC)
    assert request.end == datetime(2024, 1, 2, tzinfo=UTC)
    assert request.interval == "1m"
    assert request.columns is None
    assert request.product == "spot"
    assert request.dataset == "klines"


def test_request_resolution_applies_kline_defaults_after_dataset_lookup() -> None:
    """Confirm unspecified kline options resolve from the declared dataset."""
    request = Request.parse(
        "BTCUSDT",
        "2024-01-01",
        "2024-01-01",
        gap_policy=None,
    )

    resolved = request.resolve_dataset(get_dataset("spot", "klines"))

    assert request.interval is None
    assert request.gap_policy is None
    assert resolved.interval == "1m"
    assert resolved.gap_policy == "forward"
    assert resolved.columns == {
        column: column for column in get_dataset("spot", "klines").output_columns
    }


def test_request_resolution_rejects_options_unsupported_by_raw_data() -> None:
    """Confirm a raw dataset rejects interval and gap-policy inputs."""
    raw = DatasetSpec(
        product="spot",
        name="snapshot",
        remote_name="snapshot",
        source_columns=("event_time", "value"),
        stored_columns=("event_time", "value"),
        time_column="event_time",
        base_interval=None,
        output_intervals=(),
        aliases={},
    )
    request = Request.parse(
        "BTCUSDT",
        "2024-01-01",
        "2024-01-01",
        product="spot",
        dataset="snapshot",
        gap_policy=None,
    )

    assert request.resolve_dataset(raw).interval is None
    with pytest.raises(ValueError, match="does not accept an interval"):
        Request.parse(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-01",
            product="spot",
            dataset="snapshot",
            interval="1m",
            gap_policy=None,
        ).resolve_dataset(raw)
    with pytest.raises(ValueError, match="does not accept gap_policy"):
        Request.parse(
            "BTCUSDT",
            "2024-01-01",
            "2024-01-01",
            product="spot",
            dataset="snapshot",
            gap_policy="keep",
        ).resolve_dataset(raw)


def test_request_resolution_rejects_a_mismatched_dataset() -> None:
    """Confirm product and dataset identity cannot be resolved accidentally."""
    request = parse_request(product="spot", dataset="klines")

    with pytest.raises(ValueError, match="does not match"):
        request.resolve_dataset(
            DatasetSpec(
                product="spot",
                name="snapshot",
                remote_name="snapshot",
                source_columns=("event_time",),
                stored_columns=("event_time",),
                time_column="event_time",
                base_interval=None,
                output_intervals=(),
                aliases={},
            )
        )


def test_pair_lists_preserve_order_duplicates_and_caller_input() -> None:
    """Confirm that a pair list is copied without sorting or deduplication."""
    pairs = ["ETHUSDT", "BTCUSDT", "ETHUSDT"]

    request = parse_request(pairs=pairs)
    pairs.append("ADAUSDT")

    assert request.pairs == ("ETHUSDT", "BTCUSDT", "ETHUSDT")
    assert request.single is False


def test_pair_normalization_keeps_only_uppercase_ascii_letters_and_digits() -> None:
    """Confirm that common source separators and surrounding text are removed."""
    assert normalize_pair("btc-usdt") == "BTCUSDT"
    assert normalize_pair(" btc_usdt ") == "BTCUSDT"
    assert normalize_pair("BTC/USDT") == "BTCUSDT"
    assert normalize_pair("1000sats-usdt") == "1000SATSUSDT"
    assert normalize_pair("é-💰") == ""


@pytest.mark.parametrize(
    "pairs",
    [None, 1, True, (), {}, {"BTCUSDT"}, iter(["BTCUSDT"])],
)
def test_invalid_pair_containers_are_rejected(pairs: object) -> None:
    """Confirm that only one string or a list of strings is accepted.

    Args:
        pairs: An unsupported pair container.
    """
    with pytest.raises(TypeError, match="string or a list"):
        parse_request(pairs=pairs)


@pytest.mark.parametrize(
    "pairs",
    [[], "", "  ", "---", [""], ["---"], [None], [1], [True], ["BTCUSDT", []]],
)
def test_empty_and_invalid_pair_entries_are_rejected(pairs: object) -> None:
    """Confirm that every requested pair is a non-empty string.

    Args:
        pairs: An empty request or a request containing an invalid pair.
    """
    with pytest.raises((TypeError, ValueError)):
        parse_request(pairs=pairs)


@pytest.mark.parametrize("field", ["starting_date", "end_date"])
@pytest.mark.parametrize(
    "value",
    [None, 3, True, [], {}, pd.NaT, "", "bad-date", "2024-02-30"],
)
def test_invalid_date_inputs_are_rejected(field: str, value: object) -> None:
    """Confirm that malformed or unsupported date inputs fail clearly.

    Args:
        field: The request boundary receiving the invalid value.
        value: The invalid date-like value.
    """
    with pytest.raises((TypeError, ValueError), match="date"):
        if field == "starting_date":
            parse_request(starting_date=value)
        else:
            parse_request(end_date=value)


def test_date_and_datetime_boundaries_use_utc_and_clear_end_semantics() -> None:
    """Confirm whole-day dates and exact datetimes become UTC boundaries."""
    whole_day = parse_request(starting_date=date(2024, 1, 1), end_date="2024-01-01")
    exact = parse_request(
        starting_date=datetime(2024, 1, 1, 2, tzinfo=timezone(timedelta(hours=2))),
        end_date="2024-01-01T00:03:00",
    )

    assert whole_day.end - whole_day.start == timedelta(days=1)
    assert exact.start == datetime(2024, 1, 1, tzinfo=UTC)
    assert exact.end == datetime(2024, 1, 1, 0, 3, tzinfo=UTC)
    assert exact.end - exact.start == timedelta(minutes=3)
    assert parse_timestamp("2024-01-01T00:00:00Z") == datetime(2024, 1, 1, tzinfo=UTC)


def test_invalid_precision_bounds_and_range_order_are_rejected() -> None:
    """Confirm unsupported precision, overflow, and empty ranges fail."""
    with pytest.raises(ValueError, match="microsecond"):
        parse_timestamp(pd.Timestamp("2024-01-01T00:00:00.000000001Z"))
    with pytest.raises(ValueError, match="supported"):
        parse_timestamp(date.max, end=True)
    with pytest.raises(ValueError, match="before"):
        parse_request(starting_date="2024-01-02", end_date="2024-01-01")
    with pytest.raises(ValueError, match="before"):
        parse_request(
            starting_date="2024-01-01T00:00:00",
            end_date="2024-01-01T00:00:00",
        )


def test_identifiers_are_safe_but_source_support_is_deferred() -> None:
    """Confirm identifiers are lowercase while capability checks remain separate."""
    request = parse_request(product="future_product", dataset="custom_data")

    assert request.product == "future_product"
    assert request.dataset == "custom_data"

    for field in ("product", "dataset"):
        for value in (None, 1, True, "", " ", "Spot", "bad-name", "1name"):
            with pytest.raises((TypeError, ValueError), match=field):
                if field == "product":
                    parse_request(product=value)
                else:
                    parse_request(dataset=value)


def test_interval_syntax_and_configured_default_are_parsed() -> None:
    """Confirm positive interval spellings are parsed before capability checks."""
    assert parse_request(base_interval="1s").interval == "1s"
    assert parse_request(interval="7m").interval == "7m"
    assert parse_request(interval="23h").interval == "23h"
    assert parse_request(interval="3mo").interval == "3mo"

    invalid_intervals: tuple[object, ...] = (
        "",
        "0m",
        "01m",
        "1M",
        "m",
        "1month",
        1,
        True,
        [],
    )
    for value in invalid_intervals:
        with pytest.raises((TypeError, ValueError), match="interval"):
            parse_request(interval=value)

    with pytest.raises(ValueError, match="base_interval"):
        parse_request(interval="1m", base_interval="bad")


def test_column_lists_and_mappings_are_copied_in_caller_order() -> None:
    """Confirm optional column selections retain order and custom labels."""
    names = ["close", "open"]
    labels = {"open_time": "time", "close": 'closing "price"'}

    selected = parse_request(desired_columns=names)
    renamed = parse_request(desired_columns=labels)
    names.append("volume")
    labels["volume"] = "size"

    assert selected.columns == {"close": "close", "open": "open"}
    assert renamed.columns == {
        "open_time": "time",
        "close": 'closing "price"',
    }
    assert parse_request().columns is None


@pytest.mark.parametrize(
    "columns",
    [
        (),
        "close",
        1,
        True,
        {"close"},
        [],
        {},
        ["open", "open"],
        [""],
        ["\0"],
        [1],
        {1: "number"},
        {"": "empty"},
        {" ": "blank"},
        {"\0": "nul"},
        {"close": 1},
        {"close": ""},
        {"close": " "},
        {"close": "\0"},
        {"close": "price", "open": "price"},
    ],
)
def test_invalid_column_selections_are_rejected(columns: object) -> None:
    """Confirm malformed, empty, and ambiguous column selections fail.

    Args:
        columns: An unsupported column selection or mapping.
    """
    with pytest.raises((TypeError, ValueError), match="column"):
        parse_request(desired_columns=columns)
