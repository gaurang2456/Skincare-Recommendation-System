"""
utils/history.py

Functions for persisting and loading recommendation history to/from a CSV file.

Requirements: 6.1, 6.2, 6.4
"""

from __future__ import annotations

import os

import pandas as pd

# Canonical column order for the history CSV.
HISTORY_COLUMNS = [
    "timestamp",
    "age",
    "skin_type",
    "concerns",
    "avg_severity",
    "skin_health_score",
    "cleanser",
    "moisturizer",
    "serum",
    "sunscreen",
    "confidence_score",
]


def append_history(record: dict, path: str = "data/recommendation_history.csv") -> None:
    """Append a single recommendation record to the history CSV.

    - Creates the file with a header row if it does not exist.
    - Appends the record (without writing the header again) if the file exists.

    Args:
        record: A dict whose keys match HISTORY_COLUMNS.
        path:   Path to the CSV file (relative to cwd or absolute).
    """
    row = pd.DataFrame([record], columns=HISTORY_COLUMNS)

    file_exists = os.path.exists(path)

    if file_exists:
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, mode="w", header=True, index=False)


def load_history(path: str = "data/recommendation_history.csv") -> pd.DataFrame:
    """Load recommendation history from a CSV file.

    - Returns an empty DataFrame with the correct column schema if the file
      does not exist.
    - Returns rows sorted by ``timestamp`` descending (most recent first).

    Args:
        path: Path to the CSV file (relative to cwd or absolute).

    Returns:
        pd.DataFrame with columns matching HISTORY_COLUMNS.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    df = pd.read_csv(path)

    # Ensure all expected columns are present (guard against partial files).
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Sort by timestamp descending (most recent first).
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    return df
