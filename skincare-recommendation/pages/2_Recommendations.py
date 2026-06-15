"""
pages/2_Recommendations.py

Recommendations results page.

Displays:
  Section 1 — Skin Health Score with color-coded badge
  Section 2 — 2×2 product cards grid
  Section 3 — Morning routine steps
  Section 4 — Night routine steps
  Section 5 — PDF download button

Requirements: 3.1–3.4, 4.3, 5.3, 7.1–7.4
"""

from __future__ import annotations

import streamlit as st

from utils.pdf_report import generate_pdf

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Your Recommendations",
    page_icon="💆",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Color-coded badge helpers
# ---------------------------------------------------------------------------

_BADGE_COLORS = {
    "Excellent":       ("#1a7a4a", "#d4edda"),   # dark green text, light green bg
    "Good":            ("#856404", "#fff3cd"),   # amber text, light yellow bg
    "Fair":            ("#cc5500", "#fde8d8"),   # orange text, light orange bg
    "Needs Attention": ("#842029", "#f8d7da"),   # red text, light red bg
}

_SCORE_EXPLANATIONS = {
    "Excellent":       "Your skin is in great shape — keep up the routine!",
    "Good":            "Your skin shows some minor concerns that are manageable with the right routine.",
    "Fair":            "Your skin has a few concerns; a targeted routine will make a noticeable difference.",
    "Needs Attention": "Your skin needs focused care — the recommended products are specially chosen to help.",
}

_CATEGORY_ICONS = {
    "Cleanser":    "🧴",
    "Moisturizer": "💧",
    "Serum":       "✨",
    "Sunscreen":   "☀️",
}


def _badge_html(label: str) -> str:
    text_color, bg_color = _BADGE_COLORS.get(label, ("#333", "#eee"))
    return (
        f'<span style="'
        f'background-color:{bg_color};'
        f'color:{text_color};'
        f'padding:4px 12px;'
        f'border-radius:12px;'
        f'font-weight:bold;'
        f'font-size:0.95rem;'
        f'">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("💆 Your Skincare Recommendations")

    # Guard: session state check
    if "recommendation_result" not in st.session_state:
        st.info(
            "No recommendation available yet. Please complete the skin assessment first."
        )
        st.page_link("pages/1_Assessment.py", label="Go to Assessment →")
        return

    result = st.session_state["recommendation_result"]
    profile = st.session_state.get("assessment_input", None)

    # -----------------------------------------------------------------------
    # Section 1 — Skin Health Score
    # -----------------------------------------------------------------------
    st.subheader("📊 Skin Health Score")

    score = result.skin_health_score
    label = result.skin_health_label

    col_metric, col_badge = st.columns([1, 2])
    with col_metric:
        st.metric("Skin Health Score", f"{score}/100")
    with col_badge:
        st.markdown(_badge_html(label), unsafe_allow_html=True)
        explanation = _SCORE_EXPLANATIONS.get(
            label, "Consistent skincare will improve your skin health over time."
        )
        st.caption(explanation)

    st.divider()

    # -----------------------------------------------------------------------
    # Section 2 — Product Cards (2×2 grid)
    # -----------------------------------------------------------------------
    st.subheader("🛍️ Recommended Products")

    products = [
        ("Cleanser",    result.cleanser),
        ("Moisturizer", result.moisturizer),
        ("Serum",       result.serum),
        ("Sunscreen",   result.sunscreen),
    ]

    # Row 1
    row1_cols = st.columns(2)
    for col, (category, rec) in zip(row1_cols, products[:2]):
        with col:
            _render_product_card(category, rec)

    # Row 2
    row2_cols = st.columns(2)
    for col, (category, rec) in zip(row2_cols, products[2:]):
        with col:
            _render_product_card(category, rec)

    st.divider()

    # -----------------------------------------------------------------------
    # Section 3 — Morning Routine
    # -----------------------------------------------------------------------
    st.subheader("☀️ Morning Routine")
    for step in result.morning_routine:
        st.markdown(f"**Step {step.step}** — {step.category}: {step.product}")

    st.divider()

    # -----------------------------------------------------------------------
    # Section 4 — Night Routine
    # -----------------------------------------------------------------------
    st.subheader("🌙 Night Routine")
    for step in result.night_routine:
        st.markdown(f"**Step {step.step}** — {step.category}: {step.product}")

    st.divider()

    # -----------------------------------------------------------------------
    # Section 5 — PDF Download
    # -----------------------------------------------------------------------
    st.subheader("📄 Download Report")

    if "pdf_error" not in st.session_state:
        st.session_state["pdf_error"] = False

    if st.session_state["pdf_error"]:
        if st.button("🔄 Retry PDF Generation"):
            st.session_state["pdf_error"] = False
            st.rerun()
    else:
        try:
            pdf_bytes = generate_pdf(result, profile)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name="skincare_report.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")
            st.session_state["pdf_error"] = True
            if st.button("🔄 Retry"):
                st.session_state["pdf_error"] = False
                st.rerun()


def _render_product_card(category: str, rec) -> None:
    """Render a single product recommendation card."""
    icon = _CATEGORY_ICONS.get(category, "🔹")

    if rec is None or rec.name == "No compatible product":
        st.warning(
            f"No compatible {category} found. "
            "Consider removing some ingredient exclusions."
        )
        return

    with st.container(border=True):
        st.markdown(f"#### {icon} {category}")
        st.markdown(f"**{rec.name}**")

        if rec.key_ingredients:
            st.caption("Key ingredients: " + ", ".join(rec.key_ingredients))
        else:
            st.caption("Key ingredients: —")

        confidence_pct = rec.confidence_score * 100
        st.markdown(f"**Confidence:** {confidence_pct:.1f}%")
        st.progress(min(max(rec.confidence_score, 0.0), 1.0))

        st.markdown(f"*{rec.reason}*")


main()
