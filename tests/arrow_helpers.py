"""Adapt existing table mutations to Arrow while preserving their assertions."""

from datetime import date, datetime
import pandas as pd
import pyarrow as pa

from crypto_downloader.core.datasets import DatasetSpec
from crypto_downloader.binance.processing import DataValidationError
from crypto_downloader.binance import processing


def normalize_chunk(
    frame: pd.DataFrame, dataset: DatasetSpec, *, contract_size: float | None = None
) -> pd.DataFrame:
    """Run Arrow normalization on a pandas fixture and return its output for assertions."""
    return processing.normalize_chunk(
        pa.Table.from_pandas(frame, preserve_index=False), dataset, contract_size
    ).to_pandas()


def validate_chunk(
    frame: pd.DataFrame,
    dataset: DatasetSpec,
    day: date,
    previous_timestamp: datetime | None = None,
) -> datetime:
    """Run Arrow validation against the existing corrupted-table test cases."""
    return processing.validate_chunk(
        pa.Table.from_pandas(frame, preserve_index=False),
        dataset,
        day,
        previous_timestamp,
    )
