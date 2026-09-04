"""Small validation helpers for tabular input data."""

from __future__ import annotations

import pandas as pd


def require_unique_values(
    frame: pd.DataFrame,
    column: str,
    *,
    context: str,
    sample_size: int = 8,
) -> None:
    """Raise if a coordinate column contains duplicate values.

    Several input series are later interpolated onto event-bin centers. For
    those series, silently averaging duplicate ages would change the data.
    Duplicate coordinates should therefore be fixed at the source or handled
    explicitly by the caller.
    """

    duplicated = frame[frame.duplicated(column, keep=False)]
    if duplicated.empty:
        return

    duplicate_counts = duplicated[column].value_counts(dropna=False)
    examples = duplicate_counts.head(sample_size)
    sample = ", ".join(f"{value!r} (n={count})" for value, count in examples.items())
    raise ValueError(
        f"{context} contains duplicate {column!r} values. "
        f"Duplicate rows: {len(duplicated)}; "
        f"duplicate coordinates: {duplicate_counts.size}. "
        f"Examples: {sample}."
    )
