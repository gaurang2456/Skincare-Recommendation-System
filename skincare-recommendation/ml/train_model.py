"""
ml/train_model.py

One-time script that trains the skincare recommendation model and saves the
fitted artifact to models/skincare_model.pkl.

Run from the project root (skincare-recommendation/):
    python ml/train_model.py

Outputs:
    models/skincare_model.pkl    — serialised model artifact
    data/model_metrics.json      — per-target evaluation metrics
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths (relative to the project root, i.e., skincare-recommendation/)
# ---------------------------------------------------------------------------

DATASET_PATH = os.path.join("data", "skincare_dataset.csv")
MODEL_PATH   = os.path.join("models", "skincare_model.pkl")
METRICS_PATH = os.path.join("data", "model_metrics.json")

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

FEATURE_COLS = ["Age", "Skin_Type", "Concern", "Severity_Score"]
TARGET_COLS  = [
    "Recommended_Cleanser",
    "Recommended_Moisturizer",
    "Recommended_Serum",
    "Recommended_Sunscreen",
]

# Short names used as keys in metrics / label-encoder dicts
TARGET_KEYS = ["Cleanser", "Moisturizer", "Serum", "Sunscreen"]


# ---------------------------------------------------------------------------
# Task 4.1 — Data loading and cleaning
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    """Load the CSV and apply data-quality filters.

    Steps:
      1. Load CSV with pandas.
      2. Drop rows with nulls in any feature column.
      3. Remove rows where Age < 10 or Age > 100.

    Returns the cleaned DataFrame.
    """
    df = pd.read_csv(path)

    # Drop nulls in feature columns
    df = df.dropna(subset=FEATURE_COLS)

    # Remove Age outliers
    df = df[(df["Age"] >= 10) & (df["Age"] <= 100)]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task 4.2 — Feature encoding
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame):
    """Fit label encoders and build feature / target matrices.

    Returns:
        X        — numpy array of shape (n, 4) with columns
                   [Age, encoded_Skin_Type, encoded_Concern, Severity_Score]
        y        — numpy array of shape (n, 4) with encoded target columns
        encoders — dict mapping column names to fitted LabelEncoder objects
    """
    encoders: dict[str, LabelEncoder] = {}

    # --- Feature encoders ---
    le_skin_type = LabelEncoder()
    le_concern   = LabelEncoder()

    encoded_skin_type = le_skin_type.fit_transform(df["Skin_Type"])
    encoded_concern   = le_concern.fit_transform(df["Concern"])

    encoders["Skin_Type"] = le_skin_type
    encoders["Concern"]   = le_concern

    # --- Feature matrix ---
    X = np.column_stack([
        df["Age"].values,
        encoded_skin_type,
        encoded_concern,
        df["Severity_Score"].values,
    ]).astype(float)

    # --- Target encoders + target matrix ---
    y_cols = []
    for col, key in zip(TARGET_COLS, TARGET_KEYS):
        le = LabelEncoder()
        encoded_col = le.fit_transform(df[col])
        encoders[key] = le
        y_cols.append(encoded_col)

    y = np.column_stack(y_cols)

    return X, y, encoders


# ---------------------------------------------------------------------------
# Task 4.3 — Train-test split, model training, and serialization
# ---------------------------------------------------------------------------

def train(X, y, df: pd.DataFrame):
    """Split data, train the multi-output classifier, and return it.

    Split: 80/20, random_state=42, stratify on encoded Skin_Type
    (column index 1 in X).

    Returns:
        clf       — fitted MultiOutputClassifier
        X_train, X_test, y_train, y_test
    """
    stratify_col = X[:, 1].astype(int)   # encoded Skin_Type

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_col,
    )

    clf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    clf.fit(X_train, y_train)

    return clf, X_train, X_test, y_train, y_test


def save_artifact(clf, encoders: dict, metrics: dict, path: str) -> None:
    """Serialize the model artifact to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    artifact = {
        "model": clf,
        "label_encoders": {
            "Skin_Type":   encoders["Skin_Type"],
            "Concern":     encoders["Concern"],
            "Cleanser":    encoders["Cleanser"],
            "Moisturizer": encoders["Moisturizer"],
            "Serum":       encoders["Serum"],
            "Sunscreen":   encoders["Sunscreen"],
        },
        "feature_names": ["Age", "Skin_Type", "Concern", "Severity_Score"],
        "metrics": metrics,
    }

    joblib.dump(artifact, path)
    print(f"Model artifact saved → {path}")


# ---------------------------------------------------------------------------
# Task 4.4 — Evaluation and metrics persistence
# ---------------------------------------------------------------------------

def evaluate(clf, X_test, y_test, encoders: dict) -> dict:
    """Compute per-target accuracy, precision, recall, and F1 on the test split.

    Returns a dict with the schema:
        {
            "Cleanser":    {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
            "Moisturizer": {...},
            "Serum":       {...},
            "Sunscreen":   {...},
        }
    """
    y_pred = clf.predict(X_test)
    metrics: dict[str, dict] = {}

    for idx, key in enumerate(TARGET_KEYS):
        true_col = y_test[:, idx]
        pred_col = y_pred[:, idx]

        metrics[key] = {
            "accuracy":  round(float(accuracy_score(true_col, pred_col)), 4),
            "precision": round(float(precision_score(true_col, pred_col, average="weighted", zero_division=0)), 4),
            "recall":    round(float(recall_score(true_col, pred_col, average="weighted", zero_division=0)), 4),
            "f1":        round(float(f1_score(true_col, pred_col, average="weighted", zero_division=0)), 4),
        }

    return metrics


def save_metrics(metrics: dict, path: str) -> None:
    """Write metrics dict to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved       → {path}")


def print_metrics_table(metrics: dict) -> None:
    """Print a formatted summary table of per-target metrics to stdout."""
    col_width = 13
    header = (
        f"{'Target':<{col_width}}"
        f"{'Accuracy':>{col_width}}"
        f"{'Precision':>{col_width}}"
        f"{'Recall':>{col_width}}"
        f"{'F1':>{col_width}}"
    )
    separator = "-" * len(header)

    print()
    print("=== Model Evaluation (Test Split) ===")
    print(separator)
    print(header)
    print(separator)

    for key, m in metrics.items():
        row = (
            f"{key:<{col_width}}"
            f"{m['accuracy']:>{col_width}.4f}"
            f"{m['precision']:>{col_width}.4f}"
            f"{m['recall']:>{col_width}.4f}"
            f"{m['f1']:>{col_width}.4f}"
        )
        print(row)

    # Summary row (averages)
    avg = {
        stat: round(sum(metrics[k][stat] for k in TARGET_KEYS) / len(TARGET_KEYS), 4)
        for stat in ("accuracy", "precision", "recall", "f1")
    }
    print(separator)
    avg_row = (
        f"{'Average':<{col_width}}"
        f"{avg['accuracy']:>{col_width}.4f}"
        f"{avg['precision']:>{col_width}.4f}"
        f"{avg['recall']:>{col_width}.4f}"
        f"{avg['f1']:>{col_width}.4f}"
    )
    print(avg_row)
    print(separator)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading dataset from {DATASET_PATH} …")
    df = load_and_clean(DATASET_PATH)
    print(f"Rows after cleaning : {len(df)}")

    print("Encoding features …")
    X, y, encoders = encode_features(df)
    print(f"Feature matrix      : {X.shape}")
    print(f"Target matrix       : {y.shape}")

    print("Training model …")
    clf, X_train, X_test, y_train, y_test = train(X, y, df)
    print(f"Training samples    : {X_train.shape[0]}")
    print(f"Test samples        : {X_test.shape[0]}")

    print("Evaluating model …")
    metrics = evaluate(clf, X_test, y_test, encoders)

    print_metrics_table(metrics)
    save_metrics(metrics, METRICS_PATH)

    save_artifact(clf, encoders, metrics, MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
