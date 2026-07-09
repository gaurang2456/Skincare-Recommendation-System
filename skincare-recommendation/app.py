"""
app.py

Entry point for the AI Skincare Recommendation System Streamlit application.

Responsibilities:
  - Define and cache `load_model()` with `@st.cache_resource`.
  - Render the Home page: title, intro paragraph, "Start Assessment" button.

Run with:
    streamlit run app.py

Requirements: 2.6, 9.1, 9.3, 9.4
"""

from __future__ import annotations

import os

import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Corium - AI Skincare Recommendation System",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto",
)


# ---------------------------------------------------------------------------
# Model loader — cached for the lifetime of the Streamlit session
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model() -> dict:
    """Load and cache the trained skincare ML model artifact.

    The artifact is a dict with keys: "model", "label_encoders",
    "feature_names", and "metrics".

    Stops the app with an error message if the file is missing or corrupted.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "skincare_model.pkl")

    if not os.path.exists(model_path):
        st.error(
            "⚠️ Model file not found. Please run the training script first:\n\n"
            "```\npython ml/train_model.py\n```"
        )
        st.stop()

    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        st.error(
            f"⚠️ Failed to load the model file: {exc}\n\n"
            "Please regenerate the model:\n\n"
            "```\npython ml/generate_dataset.py\npython ml/train_model.py\n```"
        )
        st.stop()

    return artifact  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

def main() -> None:
    # Load model on startup (cached — no performance cost on rerun)
    load_model()

    st.title("✨ Corium - Skincare, made personal. ")

    st.markdown(
        """
       Build a skincare routine around your skin type, concerns, and ingredient preferences. Corium provides personalized product recommendations and a structured daily routine designed around your individual profile.

       Complete a short assessment to receive product recommendations across essential skincare categories, along with a personalized morning and evening routine.
        """
    )

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("Your routine starts with your skin")
        st.markdown(
            "Complete a quick assessment and receive recommendations tailored to your profile. "
        )
        st.page_link(
            "pages/1_Assessment.py",
            label="🚀 Start Assessment",
            icon="🚀",
        )

    with col_right:
        st.metric(
            """
            **What you'll get:**
            - ✅ 4 product recommendations
            - 📊 Skin Health Score
            - ☀️ Morning routine
            - 🌙 Night routine
            - 📄 Downloadable PDF report
            """
        )

    st.divider()

    st.caption(
        "Navigate using the sidebar to jump to Recommendations, History, or ML Insights."
    )


if __name__ == "__main__":
    main()
else:
    # Streamlit executes this file as a script; run main() at module level.
    main()
