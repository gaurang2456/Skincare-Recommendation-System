"""
pages/1_Assessment.py

Skin Assessment form page.

Collects user input (age, skin type, concerns, severity, ingredient exclusions),
validates it, calls the recommendation engine, stores results in session state,
appends to history, and navigates to the Recommendations page.

Requirements: 1.1–1.9, 2.3, 6.1
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import joblib
import streamlit as st

from ml.predictor import AssessmentInput, get_recommendation
from utils.history import append_history

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Skin Assessment",
    page_icon="📋",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONCERNS_LIST = [
    "Acne",
    "Dark Spots",
    "Hyperpigmentation",
    "Dryness",
    "Redness",
    "Uneven Skin Tone",
    "Wrinkles",
    "Fine Lines",
    "Sun Damage",
    "Large Pores",
]

SKIN_TYPES = ["Oily", "Dry", "Combination", "Normal", "Sensitive"]

EXCLUDABLE_INGREDIENTS = ["Fragrance", "Alcohol", "Parabens", "Sulfates"]


# ---------------------------------------------------------------------------
# Model loader (re-uses the same cached function via joblib direct load)
# We load the model here using st.cache_resource so it is cached once per
# session even when the user navigates to this page directly.
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model() -> dict:
    """Load and cache the trained ML model artifact."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "skincare_model.pkl")

    if not os.path.exists(model_path):
        st.error(
            "⚠️ Model file not found. Please run:\n\n"
            "```\npython ml/train_model.py\n```"
        )
        st.stop()

    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        st.error(f"⚠️ Failed to load model: {exc}")
        st.stop()

    return artifact  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Product catalog loader
# ---------------------------------------------------------------------------

@st.cache_data
def load_product_catalog() -> list[dict]:
    """Load the product catalog JSON file once and cache it."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(BASE_DIR, "products", "product_catalog.json")
    with open(catalog_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📋 Skin Assessment")
    st.markdown(
        "Fill in the form below to receive personalised skincare recommendations. "
        "All fields marked with \\* are required."
    )

    artifact = load_model()
    product_catalog = load_product_catalog()

    with st.form("assessment_form"):
        st.subheader("Your Skin Profile")

        age = st.number_input(
            "Age *",
            min_value=10,
            max_value=100,
            value=25,
            step=1,
            help="Enter your age (10–100).",
        )

        skin_type = st.selectbox(
            "Skin Type *",
            options=[""] + SKIN_TYPES,
            index=0,
            help="Select the skin type that best describes your skin.",
        )

        concerns = st.multiselect(
            "Skin Concerns *",
            options=CONCERNS_LIST,
            help="Select one or more skin concerns you want to address.",
        )

        # Dynamic per-concern severity sliders
        severity_scores: dict[str, int] = {}
        if concerns:
            st.markdown("**Severity of each concern** (1 = mild, 10 = severe)")
            for concern in concerns:
                severity_scores[concern] = st.slider(
                    f"{concern} Severity",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key=f"severity_{concern}",
                )

        excluded_ingredients = st.multiselect(
            "Exclude Ingredients (optional)",
            options=EXCLUDABLE_INGREDIENTS,
            help="Select any ingredients you want to avoid (e.g. due to sensitivity).",
        )

        submitted = st.form_submit_button("🔍 Get Recommendations")

    # -----------------------------------------------------------------------
    # Validation and processing (outside the form block)
    # -----------------------------------------------------------------------
    if submitted:
        errors: list[str] = []

        if not skin_type:
            errors.append("Skin type is required. Please select a skin type.")
        if not concerns:
            errors.append("At least one skin concern is required.")
        if age < 10 or age > 100:
            errors.append("Age must be between 10 and 100.")

        if errors:
            for msg in errors:
                st.error(msg)
            st.stop()

        # Valid submission — call the recommendation engine
        with st.spinner("Generating your personalised skincare recommendations…"):
            model = artifact["model"]
            label_encoders = artifact["label_encoders"]

            result = get_recommendation(
                model=model,
                label_encoders=label_encoders,
                age=int(age),
                skin_type=skin_type,
                concerns=concerns,
                severity_scores=severity_scores,
                excluded_ingredients=excluded_ingredients,
                product_catalog=product_catalog,
            )

        # Build the AssessmentInput for PDF generation and session state
        assessment_input = AssessmentInput(
            age=int(age),
            skin_type=skin_type,
            concerns=concerns,
            severity_scores=severity_scores,
            excluded_ingredients=excluded_ingredients,
        )

        # Store in session state so the Recommendations page can access them
        st.session_state["recommendation_result"] = result
        st.session_state["assessment_input"] = assessment_input

        # Persist to history CSV
        avg_severity = (
            sum(severity_scores.values()) / len(severity_scores)
            if severity_scores
            else 5.0
        )
        mean_confidence = (
            result.cleanser.confidence_score
            + result.moisturizer.confidence_score
            + result.serum.confidence_score
            + result.sunscreen.confidence_score
        ) / 4.0

        history_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "age": int(age),
            "skin_type": skin_type,
            "concerns": ";".join(concerns),
            "avg_severity": round(avg_severity, 2),
            "skin_health_score": result.skin_health_score,
            "cleanser": result.cleanser.name,
            "moisturizer": result.moisturizer.name,
            "serum": result.serum.name,
            "sunscreen": result.sunscreen.name,
            "confidence_score": round(mean_confidence, 4),
        }

        try:
            append_history(history_record)
        except Exception:
            pass  # History append failure should not block the user flow

        st.switch_page("pages/2_Recommendations.py")


main()
