"""
tests/test_history.py

Unit tests for utils.history.append_history and utils.history.load_history.

Requirements: 6.1, 6.2, 6.4
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from utils.history import append_history, load_history, HISTORY_COLUMNS


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_record(timestamp: str = "2024-01-01 12:00:00") -> dict:
    """Return a minimal history record with all expected columns."""
    return {
        "timestamp": timestamp,
        "age": 30,
        "skin_type": "Oily",
        "concerns": "Acne",
        "avg_severity": 5.0,
        "skin_health_score": 75,
        "cleanser": "Gentle Foaming Cleanser",
        "moisturizer": "Oil-Free Hydrator",
        "serum": "Niacinamide Brightening Serum",
        "sunscreen": "Lightweight SPF 50 Fluid",
        "confidence_score": 0.85,
    }


# ---------------------------------------------------------------------------
# append_history tests
# ---------------------------------------------------------------------------

def test_append_history_creates_file_with_header(tmp_path):
    """append_history must create the CSV with a header row when the file does not exist."""
    path = str(tmp_path / "history.csv")
    assert not os.path.exists(path)

    append_history(_make_record(), path=path)

    assert os.path.exists(path)

    # Read back and check header
    df = pd.read_csv(path)
    for col in HISTORY_COLUMNS:
        assert col in df.columns


def test_append_history_does_not_duplicate_header(tmp_path):
    """Calling append_history twice must not add a second header row."""
    path = str(tmp_path / "history.csv")
    append_history(_make_record("2024-01-01 12:00:00"), path=path)
    append_history(_make_record("2024-01-02 12:00:00"), path=path)

    df = pd.read_csv(path)
    # The header should not appear as a data row
    assert not (df["timestamp"] == "timestamp").any()
    assert len(df) == 2


def test_append_history_increments_record_count_by_1(tmp_path):
    """Each call to append_history must add exactly 1 row."""
    path = str(tmp_path / "history.csv")

    append_history(_make_record("2024-01-01 10:00:00"), path=path)
    df1 = pd.read_csv(path)
    count_after_first = len(df1)

    append_history(_make_record("2024-01-01 11:00:00"), path=path)
    df2 = pd.read_csv(path)
    count_after_second = len(df2)

    assert count_after_first == 1
    assert count_after_second == 2
    assert count_after_second - count_after_first == 1


def test_append_history_three_calls_increments_correctly(tmp_path):
    """Three successive calls must produce exactly 3 rows."""
    path = str(tmp_path / "history.csv")
    for i in range(3):
        append_history(_make_record(f"2024-01-0{i+1} 12:00:00"), path=path)

    df = pd.read_csv(path)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# load_history tests
# ---------------------------------------------------------------------------

def test_load_history_returns_empty_dataframe_when_file_absent(tmp_path):
    """load_history must return an empty DataFrame when the CSV does not exist."""
    path = str(tmp_path / "nonexistent.csv")
    df = load_history(path=path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_load_history_empty_dataframe_has_correct_columns(tmp_path):
    """The empty DataFrame returned for a missing file must have all expected columns."""
    path = str(tmp_path / "nonexistent.csv")
    df = load_history(path=path)

    for col in HISTORY_COLUMNS:
        assert col in df.columns


def test_load_history_returns_rows_sorted_by_timestamp_descending(tmp_path):
    """load_history must return rows sorted by timestamp descending."""
    path = str(tmp_path / "history.csv")

    # Append records with out-of-order timestamps
    timestamps = [
        "2024-01-03 10:00:00",
        "2024-01-01 10:00:00",
        "2024-01-02 10:00:00",
    ]
    for ts in timestamps:
        append_history(_make_record(ts), path=path)

    df = load_history(path=path)
    assert len(df) == 3

    # Most recent should be first
    actual_timestamps = df["timestamp"].tolist()
    assert actual_timestamps == sorted(actual_timestamps, reverse=True)
    assert actual_timestamps[0] == "2024-01-03 10:00:00"
    assert actual_timestamps[-1] == "2024-01-01 10:00:00"


def test_load_history_single_record(tmp_path):
    """load_history must return a DataFrame with exactly 1 row after 1 append."""
    path = str(tmp_path / "history.csv")
    record = _make_record()
    append_history(record, path=path)

    df = load_history(path=path)
    assert len(df) == 1
    assert df.iloc[0]["skin_type"] == "Oily"
