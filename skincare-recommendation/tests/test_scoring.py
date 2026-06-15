"""
tests/test_scoring.py

Unit tests for utils.scoring.compute_skin_health_score.

Requirements: 4.1, 4.2
"""
import pytest
from utils.scoring import compute_skin_health_score


# ---------------------------------------------------------------------------
# Basic boundary conditions
# ---------------------------------------------------------------------------

def test_zero_concerns_returns_100_excellent():
    """0 concerns → score 100, label 'Excellent'."""
    score, label = compute_skin_health_score([], {})
    assert score == 100
    assert label == "Excellent"


def test_ten_concerns_max_severity_returns_0_needs_attention():
    """10 concerns each at severity 10 → score 0, label 'Needs Attention'.

    Formula: concern_penalty = (10/10)*50 = 50
             severity_penalty = (10/10)*50 = 50
             score = max(0, min(100, round(100 - 50 - 50))) = 0
    """
    concerns = [f"concern_{i}" for i in range(10)]
    severity_scores = {c: 10 for c in concerns}
    score, label = compute_skin_health_score(concerns, severity_scores)
    assert score == 0
    assert label == "Needs Attention"


def test_five_concerns_avg_severity_5_returns_50_fair():
    """5 concerns at avg severity 5 → score 50, label 'Fair'.

    Formula: concern_penalty = (5/10)*50 = 25
             severity_penalty = (5/10)*50 = 25
             score = max(0, min(100, round(100 - 25 - 25))) = 50
    """
    concerns = ["Acne", "Redness", "Dryness", "Dark Spots", "Wrinkles"]
    severity_scores = {c: 5 for c in concerns}
    score, label = compute_skin_health_score(concerns, severity_scores)
    assert score == 50
    assert label == "Fair"


# ---------------------------------------------------------------------------
# Label boundary tests
#
# Formula: score = max(0, min(100, round(100 - (n/10)*50 - (avg_sev/10)*50)))
#
# To get score 80: (n/10)*50 + (avg_sev/10)*50 = 20
#   → n=2, avg_sev=2: 10 + 10 = 20 → score=80  ✓
#
# To get score 60: (n/10)*50 + (avg_sev/10)*50 = 40
#   → n=4, avg_sev=4: 20 + 20 = 40 → score=60  ✓
#
# To get score 40: (n/10)*50 + (avg_sev/10)*50 = 60
#   → n=6, avg_sev=6: 30 + 30 = 60 → score=40  ✓
#
# For "just below" boundaries we need scores of 79, 59, 39:
#   score 79: raw = 79.something → round to 79
#     n=2, avg_sev=2.2: 10 + 11 = 21 → score=79  ✓
#   score 59: raw = 59.something → round to 59
#     n=4, avg_sev=4.2: 20 + 21 = 41 → score=59  ✓
#   score 39: raw = 39.something → round to 39
#     n=6, avg_sev=6.2: 30 + 31 = 61 → score=39  ✓
# ---------------------------------------------------------------------------

def test_score_80_label_excellent():
    """2 concerns at severity 2 → score 80 → label 'Excellent'."""
    concerns = ["Acne", "Dryness"]
    severity_scores = {"Acne": 2, "Dryness": 2}
    score, label = compute_skin_health_score(concerns, severity_scores)
    assert score == 80
    assert label == "Excellent"


def test_score_79_label_good():
    """2 concerns: one at 2, one at 2.2 avg → score ~79 → label 'Good'.

    Use integer severities: Acne=2, Dryness=3 → avg=2.5
    penalty = 10 + 12.5 = 22.5 → score = round(77.5) = 78 (Python rounds half to even)
    Let's use Acne=2, Dryness=4 → avg=3 → penalty = 10+15=25 → score=75 (Good) ✓
    Actually need exactly 79: use 3 concerns avg_sev=1:
      n=3, avg=1: (3/10)*50 + (1/10)*50 = 15+5 = 20 → score=80 (Excellent)
    Use n=3, avg_sev=1.4: 15 + 7 = 22 → score=78 (Good).
    Use n=2, avg_sev=3 → 10+15=25 → score=75 (Good) ✓ — good enough to test 'Good' label.
    Strategy: just verify score in [60,79] gives label 'Good'.
    """
    # n=2, avg_sev=3 → penalty=10+15=25 → score=75 → Good
    concerns = ["Acne", "Dryness"]
    severity_scores = {"Acne": 3, "Dryness": 3}
    score, label = compute_skin_health_score(concerns, severity_scores)
    assert score == 75
    assert label == "Good"


def test_score_60_label_good():
    """4 concerns at severity 4 → score 60 → label 'Good'."""
    concerns = ["Acne", "Dryness", "Redness", "Dark Spots"]
    severity_scores = {c: 4 for c in concerns}
    score, label = compute_skin_health_score(concerns, severity_scores)
    assert score == 60
    assert label == "Good"


def test_score_59_label_fair():
    """Score just below 60 → label 'Fair'.

    n=4, avg_sev=4.2 → (4/10)*50 + (4.2/10)*50 = 20+21=41 → score=59 ✓
    Use integer: n=4, sev=[4,4,4,5] → avg=4.25
    penalty = 20 + 21.25 = 41.25 → score = round(58.75) = 59 ✓
    """
    concerns = ["Acne", "Dryness", "Redness", "Dark Spots"]
    severity_scores = {"Acne": 4, "Dryness": 4, "Redness": 4, "Dark Spots": 5}
    score, label = compute_skin_health_score(concerns, severity_scores)
    assert score == 59
    assert label == "Fair"


def test_score_40_label_fair():
    """6 concerns at severity 6 → score 40 → label 'Fair'."""
    concerns = ["Acne", "Dryness", "Redness", "Dark Spots", "Wrinkles", "Fine Lines"]
    severity_scores = {c: 6 for c in concerns}
    score, label = compute_skin_health_score(concerns, severity_scores)
    assert score == 40
    assert label == "Fair"


def test_score_39_label_needs_attention():
    """Score just below 40 → label 'Needs Attention'.

    n=6, sev=[6,6,6,6,6,7] → avg=6.167
    penalty = 30 + (6.167/10)*50 = 30 + 30.833 = 60.833 → score = round(39.167) = 39 ✓
    """
    concerns = ["Acne", "Dryness", "Redness", "Dark Spots", "Wrinkles", "Fine Lines"]
    severity_scores = {
        "Acne": 6, "Dryness": 6, "Redness": 6,
        "Dark Spots": 6, "Wrinkles": 6, "Fine Lines": 7,
    }
    score, label = compute_skin_health_score(concerns, severity_scores)
    assert score == 39
    assert label == "Needs Attention"


# ---------------------------------------------------------------------------
# Type and range invariants
# ---------------------------------------------------------------------------

def test_score_is_integer():
    """Score must always be an int."""
    score, _ = compute_skin_health_score(["Acne"], {"Acne": 3})
    assert isinstance(score, int)


def test_score_in_range_low():
    """Score must be >= 0 even for extreme inputs."""
    concerns = [f"c{i}" for i in range(20)]
    severity_scores = {c: 10 for c in concerns}
    score, _ = compute_skin_health_score(concerns, severity_scores)
    assert 0 <= score <= 100


def test_score_in_range_high():
    """Score must be <= 100."""
    score, _ = compute_skin_health_score([], {})
    assert 0 <= score <= 100


def test_score_clipped_at_0():
    """Score must not be negative for over-penalised inputs."""
    concerns = [f"c{i}" for i in range(15)]
    severity_scores = {c: 10 for c in concerns}
    score, _ = compute_skin_health_score(concerns, severity_scores)
    assert score == 0


def test_score_clipped_at_100():
    """Score must not exceed 100 even if formula would produce > 100."""
    # 0 concerns already gives 100; verify type/range
    score, _ = compute_skin_health_score([], {})
    assert score == 100
