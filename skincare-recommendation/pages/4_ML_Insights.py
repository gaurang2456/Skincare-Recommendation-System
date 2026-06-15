"""
pages/4_ML_Insights.py

ML Insights page.

Displays:
  Section 1 — Dataset overview
  Section 2 — Model metrics (loaded from data/model_metrics.json)
  Section 3 — Feature importance (averaged across MultiOutputClassifier estimators)
  Section 4 — Confusion matrix for Recommended_Cleanser on the test split
  Section 5 — Skin type class distribution

Requirements: 8.1–8.7
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ML Insights",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_PATH   = os.path.join("models", "skincare_model.pkl")
DATA_PATH    = os.path.join("data", "skincare_dataset.csv")
METRICS_PATH = os.path.join("data", "model_metrics.json")

FEATURE_NAMES = ["Age", "Skin_Type", "Concern", "Severity_Score"]
TARGET_KEYS   = ["Cleanser", "Moisturizer", "Serum", "Sunscreen"]

COLUMN_DESCRIPTIONS = [
    ("Age",                     "Numeric",      "User age (10–100)"),
    ("Skin_Type",               "Categorical",  "One of 5 skin types"),
    ("Concern",                 "Categorical",  "Primary skin concern"),
    ("Severity_Score",          "Numeric",      "Mean concern severity (1–10)"),
    ("Recommended_Cleanser",    "Categorical",  "Target: recommended cleanser product"),
    ("Recommended_Moisturizer", "Categorical",  "Target: recommended moisturizer product"),
    ("Recommended_Serum",       "Categorical",  "Target: recommended serum product"),
    ("Recommended_Sunscreen",   "Categorical",  "Target: recommended sunscreen product"),
]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_artifact() -> dict:
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Compute confusion matrix data (cached so it only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def compute_confusion_matrix_data() -> tuple[np.ndarray, list[str]]:
    """Re-encode the dataset and compute the confusion matrix for Cleanser."""
    from sklearn.preprocessing import LabelEncoder

    df = load_dataset()

    # Drop nulls and age outliers (mirrors train_model.py)
    df = df.dropna(subset=["Age", "Skin_Type", "Concern", "Severity_Score"])
    df = df[(df["Age"] >= 10) & (df["Age"] <= 100)].reset_index(drop=True)

    artifact = load_artifact()
    label_encoders = artifact["label_encoders"]

    # Encode features using the saved encoders
    encoded_skin_type = label_encoders["Skin_Type"].transform(df["Skin_Type"])
    encoded_concern   = label_encoders["Concern"].transform(df["Concern"])

    X = np.column_stack([
        df["Age"].values,
        encoded_skin_type,
        encoded_concern,
        df["Severity_Score"].values,
    ]).astype(float)

    # Encode Cleanser target
    y_cleanser = label_encoders["Cleanser"].transform(df["Recommended_Cleanser"])

    # Reproduce the same train/test split used in train_model.py
    stratify_col = X[:, 1].astype(int)
    _, X_test, _, y_test = train_test_split(
        X,
        y_cleanser,
        test_size=0.2,
        random_state=42,
        stratify=stratify_col,
    )

    model = artifact["model"]
    # Predict only the Cleanser target (index 0)
    # model.predict returns shape (n, 4); take column 0
    # We need the full y to do multi-output predict — reconstruct y temporarily
    # Use a simpler approach: predict on X_test and take column 0
    y_pred_all = model.predict(X_test)
    y_pred = y_pred_all[:, 0]

    # Build confusion matrix using pandas crosstab for label alignment
    class_names = list(label_encoders["Cleanser"].classes_)
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(y_test, y_pred):
        cm[int(true), int(pred)] += 1

    return cm, class_names


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🔬 ML Insights")

    # Guard: check required files exist
    missing = [p for p in [MODEL_PATH, DATA_PATH] if not os.path.exists(p)]
    if missing:
        st.error(
            f"Missing required files: {', '.join(missing)}\n\n"
            "Please run the setup scripts first:\n\n"
            "```\npython ml/generate_dataset.py\npython ml/train_model.py\n```"
        )
        st.stop()

    # -----------------------------------------------------------------------
    # Section 1 — Dataset Overview
    # -----------------------------------------------------------------------
    st.subheader("📂 Dataset Overview")

    df = load_dataset()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features / Columns", len(df.columns))

    st.markdown("**First 5 rows:**")
    st.dataframe(df.head(5), use_container_width=True)

    st.markdown("**Column descriptions:**")
    desc_df = pd.DataFrame(COLUMN_DESCRIPTIONS, columns=["Column", "Type", "Description"])
    st.table(desc_df)

    st.divider()

    # -----------------------------------------------------------------------
    # Section 2 — Model Metrics
    # -----------------------------------------------------------------------
    st.subheader("📈 Model Metrics")
    st.caption("Metrics are evaluated on the held-out 20% test split and averaged across all four prediction targets.")

    metrics: dict = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)
    else:
        # Fallback: read from model artifact
        artifact = load_artifact()
        metrics = artifact.get("metrics", {})

    if metrics:
        avg_metrics = {
            stat: round(
                sum(metrics[k][stat] for k in TARGET_KEYS if k in metrics) / len(TARGET_KEYS),
                4,
            )
            for stat in ("accuracy", "precision", "recall", "f1")
        }

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Accuracy",  f"{avg_metrics['accuracy']:.1%}")
        with m_col2:
            st.metric("Precision", f"{avg_metrics['precision']:.1%}")
        with m_col3:
            st.metric("Recall",    f"{avg_metrics['recall']:.1%}")
        with m_col4:
            st.metric("F1 Score",  f"{avg_metrics['f1']:.1%}")

        with st.expander("Per-target breakdown"):
            rows = [
                {
                    "Target":    key,
                    "Accuracy":  f"{v['accuracy']:.1%}",
                    "Precision": f"{v['precision']:.1%}",
                    "Recall":    f"{v['recall']:.1%}",
                    "F1":        f"{v['f1']:.1%}",
                }
                for key, v in metrics.items()
                if key in TARGET_KEYS
            ]
            st.table(pd.DataFrame(rows))
    else:
        st.warning("Metrics data not available. Please run `python ml/train_model.py`.")

    st.divider()

    # -----------------------------------------------------------------------
    # Section 3 — Feature Importance
    # -----------------------------------------------------------------------
    st.subheader("📊 Feature Importance")
    st.caption(
        "Average feature importance across all four Random Forest estimators "
        "in the MultiOutputClassifier, sorted by importance."
    )

    artifact = load_artifact()
    model = artifact["model"]

    # MultiOutputClassifier exposes .estimators_ — each is a RandomForestClassifier
    all_importances = np.array(
        [est.feature_importances_ for est in model.estimators_]
    )
    avg_importance = all_importances.mean(axis=0)

    # Sort descending
    sorted_idx = np.argsort(avg_importance)
    sorted_features = [FEATURE_NAMES[i] for i in sorted_idx]
    sorted_values = avg_importance[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(sorted_features, sorted_values, color="#2C5F8A")
    ax.set_xlabel("Average Feature Importance")
    ax.set_title("Feature Importance (averaged across targets)")
    ax.set_xlim(0, sorted_values.max() * 1.15)
    for i, val in enumerate(sorted_values):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # -----------------------------------------------------------------------
    # Section 4 — Confusion Matrix (Recommended_Cleanser)
    # -----------------------------------------------------------------------
    st.subheader("🔲 Confusion Matrix — Cleanser Target")
    st.caption(
        "Computed on the 20% held-out test split for the Recommended_Cleanser target."
    )

    with st.spinner("Computing confusion matrix…"):
        cm, class_names = compute_confusion_matrix_data()

    # Abbreviate long class names for readability
    short_names = [n[:20] for n in class_names]

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax2,
        linewidths=0.5,
    )
    ax2.set_xlabel("Predicted Label", fontsize=11)
    ax2.set_ylabel("True Label", fontsize=11)
    ax2.set_title("Confusion Matrix — Recommended Cleanser", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    # -----------------------------------------------------------------------
    # Section 5 — Class Distribution
    # -----------------------------------------------------------------------
    st.subheader("🏷️ Skin Type Distribution")
    st.caption("Distribution of records across the five skin types in the training dataset.")

    st.bar_chart(df["Skin_Type"].value_counts())


main()
