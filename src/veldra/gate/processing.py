"""Normalize and validate Gate historical market rows."""

from datetime import date, datetime
from typing import Any, Never

from veldra.core.datasets import DatasetSpec


def normalize_chunk(
    table: Any, dataset: DatasetSpec, contract_size: float | None
) -> Never:
    """Reserve Gate row normalization for the following dataset commits.

    Args:
        table: The source Arrow table.
        dataset: The canonical Gate schema.
        contract_size: The optional Futures contract multiplier.
    """
    raise NotImplementedError("Gate dataset normalization is not implemented yet")


def validate_chunk(
    table: Any,
    dataset: DatasetSpec,
    resource_day: date,
    previous_timestamp: datetime | None,
    resource_end_day: date | None,
) -> Never:
    """Reserve Gate row validation for the following dataset commits.

    Args:
        table: The normalized Arrow table.
        dataset: The canonical Gate schema.
        resource_day: The first archive calendar day.
        previous_timestamp: The preceding chunk's final timestamp.
        resource_end_day: The archive's inclusive final calendar day.
    """
    raise NotImplementedError("Gate dataset validation is not implemented yet")
