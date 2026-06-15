"""
tests/test_pdf.py

Unit tests for utils.pdf_report.generate_pdf.

Requirements: 7.1, 7.2
"""
import pytest
from ml.predictor import (
    AssessmentInput,
    ProductRecommendation,
    RecommendationResult,
)
from utils.routine import RoutineStep, generate_routines
from utils.pdf_report import generate_pdf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def profile() -> AssessmentInput:
    """Minimal AssessmentInput for testing."""
    return AssessmentInput(
        age=30,
        skin_type="Oily",
        concerns=["Acne", "Large Pores"],
        severity_scores={"Acne": 6, "Large Pores": 4},
        excluded_ingredients=[],
    )


@pytest.fixture
def cleanser_rec() -> ProductRecommendation:
    return ProductRecommendation(
        name="Gentle Foaming Cleanser",
        category="Cleanser",
        key_ingredients=["Niacinamide", "Aloe Vera"],
        confidence_score=0.88,
        reason="Controls excess sebum and helps prevent breakouts.",
    )


@pytest.fixture
def moisturizer_rec() -> ProductRecommendation:
    return ProductRecommendation(
        name="Oil-Free Hydrator",
        category="Moisturizer",
        key_ingredients=["Hyaluronic Acid", "Niacinamide"],
        confidence_score=0.82,
        reason="Controls excess sebum and helps prevent breakouts.",
    )


@pytest.fixture
def serum_rec() -> ProductRecommendation:
    return ProductRecommendation(
        name="Niacinamide Brightening Serum",
        category="Serum",
        key_ingredients=["Niacinamide", "Zinc"],
        confidence_score=0.79,
        reason="Controls excess sebum and helps prevent breakouts.",
    )


@pytest.fixture
def sunscreen_rec() -> ProductRecommendation:
    return ProductRecommendation(
        name="Lightweight SPF 50 Fluid",
        category="Sunscreen",
        key_ingredients=["Zinc Oxide", "Niacinamide"],
        confidence_score=0.91,
        reason="Controls excess sebum and helps prevent breakouts.",
    )


@pytest.fixture
def result(cleanser_rec, moisturizer_rec, serum_rec, sunscreen_rec) -> RecommendationResult:
    """Minimal RecommendationResult built from fixture products."""
    morning, night = generate_routines(
        cleanser=cleanser_rec.name,
        moisturizer=moisturizer_rec.name,
        serum=serum_rec.name,
        sunscreen=sunscreen_rec.name,
    )
    return RecommendationResult(
        cleanser=cleanser_rec,
        moisturizer=moisturizer_rec,
        serum=serum_rec,
        sunscreen=sunscreen_rec,
        skin_health_score=72,
        skin_health_label="Good",
        morning_routine=morning,
        night_routine=night,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_pdf_returns_bytes(result, profile):
    """generate_pdf must return a bytes object."""
    pdf_bytes = generate_pdf(result, profile)
    assert isinstance(pdf_bytes, bytes)


def test_generate_pdf_returns_non_empty_bytes(result, profile):
    """generate_pdf must return non-empty bytes."""
    pdf_bytes = generate_pdf(result, profile)
    assert len(pdf_bytes) > 0


def test_generate_pdf_starts_with_pdf_magic_bytes(result, profile):
    """The returned bytes must start with the PDF magic bytes b'%PDF'."""
    pdf_bytes = generate_pdf(result, profile)
    assert pdf_bytes[:4] == b"%PDF"


def test_generate_pdf_with_excluded_ingredients(result):
    """generate_pdf must work when profile includes excluded ingredients."""
    profile_with_exclusions = AssessmentInput(
        age=25,
        skin_type="Sensitive",
        concerns=["Redness"],
        severity_scores={"Redness": 7},
        excluded_ingredients=["Fragrance", "Alcohol"],
    )
    pdf_bytes = generate_pdf(result, profile_with_exclusions)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes[:4] == b"%PDF"


def test_generate_pdf_with_needs_attention_score(cleanser_rec, moisturizer_rec, serum_rec, sunscreen_rec):
    """generate_pdf must handle 'Needs Attention' label correctly."""
    morning, night = generate_routines(
        cleanser=cleanser_rec.name,
        moisturizer=moisturizer_rec.name,
        serum=serum_rec.name,
        sunscreen=sunscreen_rec.name,
    )
    low_score_result = RecommendationResult(
        cleanser=cleanser_rec,
        moisturizer=moisturizer_rec,
        serum=serum_rec,
        sunscreen=sunscreen_rec,
        skin_health_score=15,
        skin_health_label="Needs Attention",
        morning_routine=morning,
        night_routine=night,
    )
    profile = AssessmentInput(
        age=45,
        skin_type="Oily",
        concerns=["Acne", "Redness", "Dark Spots"],
        severity_scores={"Acne": 9, "Redness": 8, "Dark Spots": 7},
        excluded_ingredients=[],
    )
    pdf_bytes = generate_pdf(low_score_result, profile)
    assert pdf_bytes[:4] == b"%PDF"
