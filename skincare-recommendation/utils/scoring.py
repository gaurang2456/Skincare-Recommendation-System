"""
utils/scoring.py

Pure function for computing the Skin Health Score.

Implemented fully in Task 6.1. This module is imported by ml/predictor.py.
"""

from __future__ import annotations


def compute_skin_health_score(
    concerns: list[str],
    severity_scores: dict[str, int],
) -> tuple[int, str]:
    """Return (score: int [0-100], label: str).

    Formula:
        concern_penalty  = (n / 10) * 50
        severity_penalty = (avg_sev / 10) * 50
        score = max(0, min(100, round(100 - concern_penalty - severity_penalty)))

    Labels:
        Excellent       score >= 80
        Good            score >= 60
        Fair            score >= 40
        Needs Attention score < 40

    Boundary conditions:
        0 concerns  → 100, "Excellent"
        10 concerns each at severity 10 → 0, "Needs Attention"
    """
    if not concerns:
        return 100, "Excellent"

    n = len(concerns)
    avg_sev = sum(severity_scores.get(c, 5) for c in concerns) / n

    concern_penalty = (n / 10) * 50
    severity_penalty = (avg_sev / 10) * 50
    raw = 100 - concern_penalty - severity_penalty
    score = max(0, min(100, round(raw)))

    label = (
        "Excellent"       if score >= 80 else
        "Good"            if score >= 60 else
        "Fair"            if score >= 40 else
        "Needs Attention"
    )

    return score, label
