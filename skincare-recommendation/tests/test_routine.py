"""
tests/test_routine.py

Unit tests for utils.routine.generate_routines.

Requirements: 5.1, 5.2, 5.4
"""
import pytest
from utils.routine import generate_routines, MORNING_ORDER, NIGHT_ORDER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def routines():
    """Return (morning, night) routines for a known set of products."""
    return generate_routines(
        cleanser="Gentle Foaming Cleanser",
        moisturizer="Oil-Free Hydrator",
        serum="Niacinamide Brightening Serum",
        sunscreen="Lightweight SPF 50 Fluid",
    )


@pytest.fixture
def morning(routines):
    return routines[0]


@pytest.fixture
def night(routines):
    return routines[1]


# ---------------------------------------------------------------------------
# Morning routine tests
# ---------------------------------------------------------------------------

def test_morning_has_exactly_4_steps(morning):
    """Morning routine must contain exactly 4 steps."""
    assert len(morning) == 4


def test_morning_step_order(morning):
    """Morning routine steps must be in order: Cleanser → Serum → Moisturizer → Sunscreen."""
    categories = [step.category for step in morning]
    assert categories == MORNING_ORDER
    assert categories == ["Cleanser", "Serum", "Moisturizer", "Sunscreen"]


def test_morning_step_numbers_are_1_indexed_sequential(morning):
    """Morning routine step numbers must be 1, 2, 3, 4."""
    step_numbers = [step.step for step in morning]
    assert step_numbers == [1, 2, 3, 4]


def test_morning_cleanser_product(morning):
    """Morning routine step 1 must be the provided cleanser."""
    assert morning[0].product == "Gentle Foaming Cleanser"


def test_morning_serum_product(morning):
    """Morning routine step 2 must be the provided serum."""
    assert morning[1].product == "Niacinamide Brightening Serum"


def test_morning_moisturizer_product(morning):
    """Morning routine step 3 must be the provided moisturizer."""
    assert morning[2].product == "Oil-Free Hydrator"


def test_morning_sunscreen_product(morning):
    """Morning routine step 4 must be the provided sunscreen."""
    assert morning[3].product == "Lightweight SPF 50 Fluid"


# ---------------------------------------------------------------------------
# Night routine tests
# ---------------------------------------------------------------------------

def test_night_has_exactly_3_steps(night):
    """Night routine must contain exactly 3 steps."""
    assert len(night) == 3


def test_night_step_order(night):
    """Night routine steps must be in order: Cleanser → Treatment Serum → Moisturizer."""
    categories = [step.category for step in night]
    assert categories == NIGHT_ORDER
    assert categories == ["Cleanser", "Treatment Serum", "Moisturizer"]


def test_night_step_numbers_are_1_indexed_sequential(night):
    """Night routine step numbers must be 1, 2, 3."""
    step_numbers = [step.step for step in night]
    assert step_numbers == [1, 2, 3]


def test_night_treatment_serum_uses_serum_product(night):
    """The 'Treatment Serum' slot in the night routine must use the serum product name."""
    treatment_step = next(s for s in night if s.category == "Treatment Serum")
    assert treatment_step.product == "Niacinamide Brightening Serum"


def test_night_cleanser_product(night):
    """Night routine step 1 must be the provided cleanser."""
    assert night[0].product == "Gentle Foaming Cleanser"


def test_night_moisturizer_product(night):
    """Night routine step 3 must be the provided moisturizer."""
    assert night[2].product == "Oil-Free Hydrator"


# ---------------------------------------------------------------------------
# Step number invariants across routines
# ---------------------------------------------------------------------------

def test_step_numbers_are_integers(morning, night):
    """All step numbers must be integers."""
    for step in morning + night:
        assert isinstance(step.step, int)


def test_morning_step_numbers_start_at_1(morning):
    """Morning routine first step number must be 1."""
    assert morning[0].step == 1


def test_night_step_numbers_start_at_1(night):
    """Night routine first step number must be 1."""
    assert night[0].step == 1
