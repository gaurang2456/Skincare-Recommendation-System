"""
ml/predictor.py

Runtime recommendation engine.

Encodes assessment input → runs ML prediction → applies ingredient filter →
computes Skin Health Score → generates routines → returns RecommendationResult.

Run from the project root (skincare-recommendation/):
    python -c "from ml.predictor import get_recommendation, apply_ingredient_filter; print('OK')"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.scoring import compute_skin_health_score
from utils.routine import generate_routines, RoutineStep  # RoutineStep defined in utils.routine

# ---------------------------------------------------------------------------
# Task 5.1 — Data classes
# ---------------------------------------------------------------------------

@dataclass
class AssessmentInput:
    age: int
    skin_type: str
    concerns: list[str]
    severity_scores: dict[str, int]
    excluded_ingredients: list[str]


@dataclass
class ProductRecommendation:
    name: str
    category: str
    key_ingredients: list[str]
    confidence_score: float
    reason: str


# RoutineStep is imported from utils.routine (defined there to avoid circular imports)
# Re-export it so callers can import RoutineStep from ml.predictor if needed.
__all__ = [
    "AssessmentInput",
    "ProductRecommendation",
    "RoutineStep",
    "RecommendationResult",
    "apply_ingredient_filter",
    "get_recommendation",
    "REASON_TEMPLATES",
]


@dataclass
class RecommendationResult:
    cleanser: ProductRecommendation
    moisturizer: ProductRecommendation
    serum: ProductRecommendation
    sunscreen: ProductRecommendation
    skin_health_score: int
    skin_health_label: str
    morning_routine: list[RoutineStep]
    night_routine: list[RoutineStep]


# ---------------------------------------------------------------------------
# Task 5.4 — Reason templates
# ---------------------------------------------------------------------------

REASON_TEMPLATES: dict[tuple[str, str], str] = {
    ("Oily", "Acne"):                 "Controls excess sebum and helps prevent breakouts.",
    ("Oily", "Large Pores"):          "Minimises pore appearance while keeping skin matte.",
    ("Oily", "Hyperpigmentation"):    "Brightens oily-prone skin and fades dark spots.",
    ("Oily", "Dark Spots"):           "Targets discolouration while controlling oil production.",
    ("Oily", "Uneven Skin Tone"):     "Evens skin tone and manages excess shine.",
    ("Oily", "Redness"):              "Calms inflammation on oily skin.",
    ("Oily", "Wrinkles"):             "Fights signs of ageing without clogging pores.",
    ("Oily", "Fine Lines"):           "Smooths fine lines on oily skin.",
    ("Oily", "Sun Damage"):           "Repairs sun damage while regulating sebum.",
    ("Oily", "Dryness"):              "Provides lightweight hydration for oil-prone skin.",
    ("Dry", "Dryness"):               "Provides deep hydration for dry, flaky skin.",
    ("Dry", "Fine Lines"):            "Plumps fine lines with intense moisture.",
    ("Dry", "Wrinkles"):              "Nourishes and firms mature dry skin.",
    ("Dry", "Redness"):               "Soothes and hydrates sensitised dry skin.",
    ("Dry", "Dark Spots"):            "Brightens dry skin while restoring moisture.",
    ("Dry", "Hyperpigmentation"):     "Fades hyperpigmentation while deeply hydrating.",
    ("Dry", "Uneven Skin Tone"):      "Evens tone and replenishes lipid barrier.",
    ("Dry", "Sun Damage"):            "Repairs sun damage with nourishing actives.",
    ("Dry", "Acne"):                  "Treats breakouts without stripping dry skin.",
    ("Dry", "Large Pores"):           "Tightens pores while maintaining moisture balance.",
    ("Combination", "Acne"):          "Targets breakouts across oily and dry zones.",
    ("Combination", "Large Pores"):   "Refines pores while balancing combination skin.",
    ("Combination", "Dryness"):       "Hydrates dry areas without overloading the T-zone.",
    ("Combination", "Dark Spots"):    "Brightens combination skin with balanced actives.",
    ("Combination", "Hyperpigmentation"): "Evens discolouration across combination skin.",
    ("Combination", "Uneven Skin Tone"):  "Balances tone across oily and dry zones.",
    ("Combination", "Redness"):       "Calms reactive areas on combination skin.",
    ("Combination", "Wrinkles"):      "Addresses ageing on combination skin.",
    ("Combination", "Fine Lines"):    "Smooths fine lines on combination skin.",
    ("Combination", "Sun Damage"):    "Repairs sun damage on combination skin.",
    ("Normal", "Sun Damage"):         "Shields and repairs normal skin from UV damage.",
    ("Normal", "Wrinkles"):           "Prevents and reduces wrinkles on normal skin.",
    ("Normal", "Fine Lines"):         "Smooths fine lines and maintains skin balance.",
    ("Normal", "Hyperpigmentation"):  "Brightens and evens normal skin tone.",
    ("Normal", "Dark Spots"):         "Fades dark spots on well-balanced skin.",
    ("Normal", "Acne"):               "Clears occasional breakouts on normal skin.",
    ("Normal", "Uneven Skin Tone"):   "Keeps normal skin radiant and even-toned.",
    ("Normal", "Redness"):            "Reduces occasional redness on normal skin.",
    ("Normal", "Dryness"):            "Maintains hydration balance for normal skin.",
    ("Normal", "Large Pores"):        "Minimises pores on normal skin.",
    ("Sensitive", "Redness"):         "Calms redness and strengthens the skin barrier.",
    ("Sensitive", "Dryness"):         "Soothes and hydrates sensitive, reactive skin.",
    ("Sensitive", "Acne"):            "Gently treats breakouts on sensitive skin.",
    ("Sensitive", "Dark Spots"):      "Fades spots with gentle, non-irritating actives.",
    ("Sensitive", "Hyperpigmentation"): "Brightens sensitive skin without irritation.",
    ("Sensitive", "Uneven Skin Tone"):  "Evens tone while respecting sensitive skin.",
    ("Sensitive", "Wrinkles"):        "Addresses ageing gently for sensitive skin.",
    ("Sensitive", "Fine Lines"):      "Smooths fine lines without causing flare-ups.",
    ("Sensitive", "Sun Damage"):      "Repairs sun damage with calming ingredients.",
    ("Sensitive", "Large Pores"):     "Minimises pores without irritating sensitive skin.",
}


# ---------------------------------------------------------------------------
# Task 5.1 — Input encoding helpers
# ---------------------------------------------------------------------------

def _encode_input(
    age: int,
    skin_type: str,
    severity_scores: dict[str, int],
    label_encoders: dict,
) -> tuple[list[float], str]:
    """Encode the assessment input into a feature vector.

    Returns:
        feature_vector  — [age, encoded_skin_type, encoded_concern, avg_severity]
        primary_concern — the concern with the highest severity score
    """
    # Determine primary concern: most severe by severity_scores
    if severity_scores:
        primary_concern = max(severity_scores, key=lambda c: severity_scores[c])
    else:
        # Fallback: should not happen given form validation, but handle gracefully
        primary_concern = list(severity_scores.keys())[0] if severity_scores else "Dryness"

    # Encode skin_type
    encoded_skin_type = int(
        label_encoders["Skin_Type"].transform([skin_type])[0]
    )

    # Encode primary concern
    encoded_concern = int(
        label_encoders["Concern"].transform([primary_concern])[0]
    )

    # Average severity
    avg_severity = float(np.mean(list(severity_scores.values()))) if severity_scores else 5.0

    feature_vector = [float(age), float(encoded_skin_type), float(encoded_concern), avg_severity]

    return feature_vector, primary_concern


# ---------------------------------------------------------------------------
# Task 5.3 — Ingredient filter
# ---------------------------------------------------------------------------

def apply_ingredient_filter(
    predicted_name: str,
    category: str,
    excluded_ingredients: list[str],
    product_catalog: list[dict],
    skin_type: str,
    base_confidence: float,
) -> tuple[dict | None, float]:
    """Check the predicted product for ingredient conflicts and find alternatives.

    Returns:
        (product_dict, confidence) where product_dict is the chosen product or
        None if no compatible product exists for this category.
    """
    # Look up the predicted product in the catalog
    predicted = next(
        (p for p in product_catalog if p["name"] == predicted_name),
        None,
    )
    if predicted is None:
        # Predicted product not in catalog — return no result
        return None, 0.0

    # Check for conflicts
    conflicts = [i for i in excluded_ingredients if i in predicted["key_ingredients"]]

    if not conflicts:
        return predicted, base_confidence  # no filter needed

    # Find compatible alternatives: same category, no excluded ingredients
    alternatives = [
        p for p in product_catalog
        if p["category"] == category
        and not any(excl in p["key_ingredients"] for excl in excluded_ingredients)
    ]

    if not alternatives:
        return None, 0.0  # no compatible product in this category

    # Prefer products suitable for the user's skin type
    skin_matches = [p for p in alternatives if skin_type in p.get("suitable_for", [])]
    chosen = skin_matches[0] if skin_matches else alternatives[0]

    return chosen, base_confidence * 0.85  # slight confidence penalty for fallback


# ---------------------------------------------------------------------------
# Task 5.2 — Prediction and confidence scoring
# ---------------------------------------------------------------------------

_CATEGORIES = ["Cleanser", "Moisturizer", "Serum", "Sunscreen"]


def _predict(
    model,
    label_encoders: dict,
    feature_vector: list[float],
) -> tuple[list[str], list[float]]:
    """Run the model and return decoded product names and confidence scores.

    Returns:
        predicted_names     — list of 4 product name strings
        confidence_scores   — list of 4 floats in [0.0, 1.0]
    """
    fv = [feature_vector]  # shape (1, 4)

    # Raw class predictions — shape (1, 4)
    raw_preds = model.predict(fv)

    # Per-target probability arrays — list of 4 arrays, each shape (1, n_classes)
    proba_list = model.predict_proba(fv)

    predicted_names: list[str] = []
    confidence_scores: list[float] = []

    for i, category in enumerate(_CATEGORIES):
        # Decode class index → product name
        encoded_label = int(raw_preds[0][i])
        product_name = label_encoders[category].inverse_transform([encoded_label])[0]
        predicted_names.append(str(product_name))

        # Confidence = max probability in the probability distribution for this target
        conf = float(np.max(proba_list[i][0]))
        confidence_scores.append(conf)

    return predicted_names, confidence_scores


# ---------------------------------------------------------------------------
# Task 5.4 — get_recommendation: full wiring
# ---------------------------------------------------------------------------

def get_recommendation(
    model,
    label_encoders: dict,
    age: int,
    skin_type: str,
    concerns: list[str],
    severity_scores: dict[str, int],
    excluded_ingredients: list[str],
    product_catalog: list[dict],
) -> RecommendationResult:
    """Generate a full RecommendationResult for the given assessment inputs.

    Steps:
      1. Encode input → feature vector
      2. Predict → product names + confidence scores
      3. Apply ingredient filter per category
      4. Compute Skin Health Score
      5. Generate routines
      6. Build and return RecommendationResult
    """
    # --- Step 1: Encode input ---
    feature_vector, primary_concern = _encode_input(
        age, skin_type, severity_scores, label_encoders
    )

    # --- Step 2: Predict ---
    predicted_names, confidence_scores = _predict(model, label_encoders, feature_vector)

    # --- Step 3: Ingredient filter per category ---
    filtered_products: list[dict | None] = []
    filtered_confidences: list[float] = []

    for i, category in enumerate(_CATEGORIES):
        product, conf = apply_ingredient_filter(
            predicted_name=predicted_names[i],
            category=category,
            excluded_ingredients=excluded_ingredients,
            product_catalog=product_catalog,
            skin_type=skin_type,
            base_confidence=confidence_scores[i],
        )
        filtered_products.append(product)
        filtered_confidences.append(conf)

    # --- Step 4: Skin Health Score ---
    skin_health_score, skin_health_label = compute_skin_health_score(
        concerns, severity_scores
    )

    # --- Step 5: Generate routines ---
    # Product names (or placeholder if filter exhausted a category)
    def _name(product: dict | None) -> str:
        return product["name"] if product is not None else "No compatible product"

    cleanser_name = _name(filtered_products[0])
    moisturizer_name = _name(filtered_products[1])
    serum_name = _name(filtered_products[2])
    sunscreen_name = _name(filtered_products[3])

    morning_routine, night_routine = generate_routines(
        cleanser_name, moisturizer_name, serum_name, sunscreen_name
    )

    # --- Step 6: Build ProductRecommendation objects ---
    reason = REASON_TEMPLATES.get(
        (skin_type, primary_concern),
        f"Formulated for {skin_type} skin with {primary_concern} concerns.",
    )

    def _make_rec(category: str, product: dict | None, conf: float) -> ProductRecommendation:
        if product is None:
            return ProductRecommendation(
                name="No compatible product",
                category=category,
                key_ingredients=[],
                confidence_score=0.0,
                reason=f"No compatible {category} found after applying ingredient exclusions.",
            )
        return ProductRecommendation(
            name=product["name"],
            category=category,
            key_ingredients=product.get("key_ingredients", []),
            confidence_score=conf,
            reason=reason,
        )

    cleanser_rec = _make_rec("Cleanser", filtered_products[0], filtered_confidences[0])
    moisturizer_rec = _make_rec("Moisturizer", filtered_products[1], filtered_confidences[1])
    serum_rec = _make_rec("Serum", filtered_products[2], filtered_confidences[2])
    sunscreen_rec = _make_rec("Sunscreen", filtered_products[3], filtered_confidences[3])

    return RecommendationResult(
        cleanser=cleanser_rec,
        moisturizer=moisturizer_rec,
        serum=serum_rec,
        sunscreen=sunscreen_rec,
        skin_health_score=skin_health_score,
        skin_health_label=skin_health_label,
        morning_routine=morning_routine,
        night_routine=night_routine,
    )
