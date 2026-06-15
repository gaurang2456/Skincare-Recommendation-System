"""
tests/test_predictor.py

Unit tests for ml.predictor.apply_ingredient_filter.

Requirements: 3.3, 3.4
"""
import pytest
from ml.predictor import apply_ingredient_filter


# ---------------------------------------------------------------------------
# Minimal product catalog fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def catalog():
    """A minimal in-memory product catalog with two Cleanser products.

    Product A has no excluded ingredients.
    Product B contains 'Fragrance' — should be filtered out when Fragrance is excluded.
    """
    return [
        {
            "name": "Safe Cleanser",
            "category": "Cleanser",
            "key_ingredients": ["Aloe Vera", "Glycerin"],
            "suitable_for": ["Oily", "Combination", "Normal"],
            "exclude_if": [],
        },
        {
            "name": "Fragrant Cleanser",
            "category": "Cleanser",
            "key_ingredients": ["Ceramides", "Fragrance"],
            "suitable_for": ["Dry", "Normal"],
            "exclude_if": ["Fragrance"],
        },
        {
            "name": "Soothing Cleanser",
            "category": "Cleanser",
            "key_ingredients": ["Oat Extract", "Chamomile"],
            "suitable_for": ["Sensitive"],
            "exclude_if": [],
        },
        {
            "name": "Alcohol Moisturizer",
            "category": "Moisturizer",
            "key_ingredients": ["Alcohol", "Vitamin C"],
            "suitable_for": ["Oily", "Normal"],
            "exclude_if": ["Alcohol"],
        },
        {
            "name": "Paraben Moisturizer",
            "category": "Moisturizer",
            "key_ingredients": ["Parabens", "Retinol"],
            "suitable_for": ["Normal", "Dry"],
            "exclude_if": ["Parabens"],
        },
    ]


# ---------------------------------------------------------------------------
# Test: no conflict — return original product unchanged
# ---------------------------------------------------------------------------

def test_no_conflict_returns_original_product(catalog):
    """When excluded ingredients do not appear in the predicted product,
    apply_ingredient_filter must return the original product dict unchanged."""
    product, conf = apply_ingredient_filter(
        predicted_name="Safe Cleanser",
        category="Cleanser",
        excluded_ingredients=["Fragrance"],
        product_catalog=catalog,
        skin_type="Oily",
        base_confidence=0.90,
    )
    assert product is not None
    assert product["name"] == "Safe Cleanser"
    assert conf == 0.90


def test_no_exclusions_returns_original_product(catalog):
    """When the excluded_ingredients list is empty, the original product is returned."""
    product, conf = apply_ingredient_filter(
        predicted_name="Fragrant Cleanser",
        category="Cleanser",
        excluded_ingredients=[],
        product_catalog=catalog,
        skin_type="Dry",
        base_confidence=0.75,
    )
    assert product is not None
    assert product["name"] == "Fragrant Cleanser"
    assert conf == 0.75


# ---------------------------------------------------------------------------
# Test: conflict — return a different (alternative) product
# ---------------------------------------------------------------------------

def test_conflict_returns_different_product(catalog):
    """When the predicted product contains an excluded ingredient,
    apply_ingredient_filter must return a different product from the same category."""
    product, conf = apply_ingredient_filter(
        predicted_name="Fragrant Cleanser",
        category="Cleanser",
        excluded_ingredients=["Fragrance"],
        product_catalog=catalog,
        skin_type="Oily",
        base_confidence=0.80,
    )
    assert product is not None
    assert product["name"] != "Fragrant Cleanser"
    assert product["category"] == "Cleanser"
    # The returned product must not contain the excluded ingredient
    assert "Fragrance" not in product["key_ingredients"]


def test_conflict_prefers_skin_type_match(catalog):
    """When multiple alternatives exist, the one matching skin_type is preferred."""
    # 'Safe Cleanser' suits ['Oily', 'Combination', 'Normal']
    # 'Soothing Cleanser' suits ['Sensitive']
    product, conf = apply_ingredient_filter(
        predicted_name="Fragrant Cleanser",
        category="Cleanser",
        excluded_ingredients=["Fragrance"],
        product_catalog=catalog,
        skin_type="Oily",
        base_confidence=0.80,
    )
    assert product is not None
    assert product["name"] == "Safe Cleanser"


# ---------------------------------------------------------------------------
# Test: confidence penalty when fallback is selected
# ---------------------------------------------------------------------------

def test_confidence_penalized_by_0_85_on_fallback(catalog):
    """When a fallback product is selected due to a conflict,
    the confidence score must be base_confidence * 0.85."""
    base_confidence = 0.80
    product, conf = apply_ingredient_filter(
        predicted_name="Fragrant Cleanser",
        category="Cleanser",
        excluded_ingredients=["Fragrance"],
        product_catalog=catalog,
        skin_type="Oily",
        base_confidence=base_confidence,
    )
    assert product is not None
    assert product["name"] != "Fragrant Cleanser"  # fallback was chosen
    assert abs(conf - base_confidence * 0.85) < 1e-9


def test_no_fallback_confidence_unchanged(catalog):
    """When no conflict occurs, the confidence score must be exactly base_confidence."""
    base_confidence = 0.92
    product, conf = apply_ingredient_filter(
        predicted_name="Safe Cleanser",
        category="Cleanser",
        excluded_ingredients=["Parabens"],
        product_catalog=catalog,
        skin_type="Oily",
        base_confidence=base_confidence,
    )
    assert product is not None
    assert conf == base_confidence


# ---------------------------------------------------------------------------
# Test: all products in category contain excluded ingredient → (None, 0.0)
# ---------------------------------------------------------------------------

def test_all_products_excluded_returns_none_zero(catalog):
    """When ALL products in the category contain at least one excluded ingredient,
    apply_ingredient_filter must return (None, 0.0)."""
    # Both Moisturizer products in the catalog contain excluded ingredients:
    # 'Alcohol Moisturizer' has Alcohol; 'Paraben Moisturizer' has Parabens.
    product, conf = apply_ingredient_filter(
        predicted_name="Alcohol Moisturizer",
        category="Moisturizer",
        excluded_ingredients=["Alcohol", "Parabens"],
        product_catalog=catalog,
        skin_type="Oily",
        base_confidence=0.88,
    )
    assert product is None
    assert conf == 0.0


def test_predicted_not_in_catalog_returns_none_zero(catalog):
    """When the predicted product name is not found in the catalog,
    apply_ingredient_filter must return (None, 0.0)."""
    product, conf = apply_ingredient_filter(
        predicted_name="Unknown Product XYZ",
        category="Cleanser",
        excluded_ingredients=[],
        product_catalog=catalog,
        skin_type="Oily",
        base_confidence=0.70,
    )
    assert product is None
    assert conf == 0.0
