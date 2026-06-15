"""
pages/3_History.py

Recommendation History page.

Loads the recommendation history CSV, sorts by timestamp descending,
and renders it as an interactive Streamlit dataframe.

Requirements: 6.2–6.4
"""

from __future__ import annotations

import streamlit as st

from utils.history import load_history

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Recommendation History",
    page_icon="📜",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📜 Recommendation History")
    st.markdown(
        "Your past skincare assessments and recommendations are displayed below, "
        "sorted from most recent to oldest."
    )

    df = load_history()

    if df.empty:
        st.info(
            "No recommendation history yet. "
            "Complete an assessment to get started."
        )
        st.stop()

    # load_history() already sorts descending by timestamp, but sort again
    # to be defensive.
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    st.markdown(f"**Total records:** {len(df)}")
    st.dataframe(df, use_container_width=True)


main()
