"""
ml/generate_dataset.py

One-time script that generates data/skincare_dataset.csv (1,200 rows).
Run from the project root (skincare-recommendation/):
    python ml/generate_dataset.py

Output columns:
    Age, Skin_Type, Concern, Severity_Score,
    Recommended_Cleanser, Recommended_Moisturizer,
    Recommended_Serum, Recommended_Sunscreen
"""

import csv
import os
import random

# ---------------------------------------------------------------------------
# Mapping table: (skin_type, primary_concern) → (cleanser, moisturizer, serum, sunscreen)
# Product names come directly from products/product_catalog.json.
# ---------------------------------------------------------------------------

MAPPING = {
    # Oily skin
    ("Oily", "Acne"):               ("Gentle Foaming Cleanser",      "Oil-Free Hydrator",           "Salicylic Acid Acne Serum",         "Lightweight SPF 50 Fluid"),
    ("Oily", "Large Pores"):        ("Balancing Micellar Cleanser",   "Oil-Free Hydrator",           "Niacinamide Brightening Serum",     "Lightweight SPF 50 Fluid"),
    ("Oily", "Uneven Skin Tone"):   ("Gentle Foaming Cleanser",      "Lightweight Daily Lotion",    "Niacinamide Brightening Serum",     "Mineral Sunscreen SPF 30"),
    ("Oily", "Hyperpigmentation"):  ("Balancing Micellar Cleanser",   "Oil-Free Hydrator",           "Niacinamide Brightening Serum",     "Lightweight SPF 50 Fluid"),
    ("Oily", "Dark Spots"):         ("Gentle Foaming Cleanser",      "Lightweight Daily Lotion",    "Salicylic Acid Acne Serum",         "Lightweight SPF 50 Fluid"),
    ("Oily", "Redness"):            ("Gentle Foaming Cleanser",      "Lightweight Daily Lotion",    "Niacinamide Brightening Serum",     "Mineral Sunscreen SPF 30"),
    ("Oily", "Dryness"):            ("Balancing Micellar Cleanser",   "Lightweight Daily Lotion",    "Niacinamide Brightening Serum",     "Lightweight SPF 50 Fluid"),
    ("Oily", "Sun Damage"):         ("Gentle Foaming Cleanser",      "Oil-Free Hydrator",           "Niacinamide Brightening Serum",     "Lightweight SPF 50 Fluid"),
    ("Oily", "Wrinkles"):           ("Balancing Micellar Cleanser",   "Oil-Free Hydrator",           "Retinol Renewal Serum",             "Mineral Sunscreen SPF 30"),
    ("Oily", "Fine Lines"):         ("Gentle Foaming Cleanser",      "Oil-Free Hydrator",           "Retinol Renewal Serum",             "Lightweight SPF 50 Fluid"),

    # Dry skin
    ("Dry", "Dryness"):             ("Hydrating Cream Cleanser",     "Rich Repair Cream",           "Hyaluronic Acid Hydration Serum",   "Hydrating Sunscreen SPF 50"),
    ("Dry", "Fine Lines"):          ("Soothing Milk Cleanser",       "Rich Repair Cream",           "Retinol Renewal Serum",             "Hydrating Sunscreen SPF 50"),
    ("Dry", "Wrinkles"):            ("Hydrating Cream Cleanser",     "Anti-Aging Firming Cream",    "Retinol Renewal Serum",             "Tinted Mineral SPF 35"),
    ("Dry", "Redness"):             ("Soothing Milk Cleanser",       "Barrier Restore Moisturizer", "Centella Calm Serum",               "Mineral Sunscreen SPF 30"),
    ("Dry", "Dark Spots"):          ("Hydrating Cream Cleanser",     "Rich Repair Cream",           "Vitamin C Glow Serum",              "Hydrating Sunscreen SPF 50"),
    ("Dry", "Hyperpigmentation"):   ("Hydrating Cream Cleanser",     "Rich Repair Cream",           "Vitamin C Glow Serum",              "Tinted Mineral SPF 35"),
    ("Dry", "Sun Damage"):          ("Soothing Milk Cleanser",       "Rich Repair Cream",           "Vitamin C Glow Serum",              "Hydrating Sunscreen SPF 50"),
    ("Dry", "Acne"):                ("Soothing Milk Cleanser",       "Barrier Restore Moisturizer", "Hyaluronic Acid Hydration Serum",   "Mineral Sunscreen SPF 30"),
    ("Dry", "Uneven Skin Tone"):    ("Hydrating Cream Cleanser",     "Barrier Restore Moisturizer", "Vitamin C Glow Serum",              "Tinted Mineral SPF 35"),
    ("Dry", "Large Pores"):         ("Soothing Milk Cleanser",       "Rich Repair Cream",           "Hyaluronic Acid Hydration Serum",   "Hydrating Sunscreen SPF 50"),

    # Combination skin
    ("Combination", "Acne"):        ("Balancing Micellar Cleanser",  "Lightweight Daily Lotion",    "Salicylic Acid Acne Serum",         "Lightweight SPF 50 Fluid"),
    ("Combination", "Large Pores"): ("Balancing Micellar Cleanser",  "Oil-Free Hydrator",           "Niacinamide Brightening Serum",     "Daily Defense SPF 40"),
    ("Combination", "Dryness"):     ("Brightening Gel Cleanser",     "Lightweight Daily Lotion",    "Hyaluronic Acid Hydration Serum",   "Mineral Sunscreen SPF 30"),
    ("Combination", "Dark Spots"):  ("Balancing Micellar Cleanser",  "Lightweight Daily Lotion",    "Niacinamide Brightening Serum",     "Daily Defense SPF 40"),
    ("Combination", "Hyperpigmentation"): ("Brightening Gel Cleanser", "Oil-Free Hydrator",         "Niacinamide Brightening Serum",     "Lightweight SPF 50 Fluid"),
    ("Combination", "Uneven Skin Tone"): ("Balancing Micellar Cleanser", "Lightweight Daily Lotion", "Niacinamide Brightening Serum",    "Mineral Sunscreen SPF 30"),
    ("Combination", "Redness"):     ("Balancing Micellar Cleanser",  "Lightweight Daily Lotion",    "Centella Calm Serum",               "Mineral Sunscreen SPF 30"),
    ("Combination", "Sun Damage"):  ("Brightening Gel Cleanser",     "Oil-Free Hydrator",           "Niacinamide Brightening Serum",     "Daily Defense SPF 40"),
    ("Combination", "Wrinkles"):    ("Balancing Micellar Cleanser",  "Lightweight Daily Lotion",    "Retinol Renewal Serum",             "Mineral Sunscreen SPF 30"),
    ("Combination", "Fine Lines"):  ("Brightening Gel Cleanser",     "Lightweight Daily Lotion",    "Retinol Renewal Serum",             "Daily Defense SPF 40"),

    # Normal skin
    ("Normal", "Acne"):             ("Balancing Micellar Cleanser",  "Lightweight Daily Lotion",    "Salicylic Acid Acne Serum",         "Mineral Sunscreen SPF 30"),
    ("Normal", "Dark Spots"):       ("Brightening Gel Cleanser",     "Lightweight Daily Lotion",    "Vitamin C Glow Serum",              "Daily Defense SPF 40"),
    ("Normal", "Hyperpigmentation"):("Brightening Gel Cleanser",     "Lightweight Daily Lotion",    "Vitamin C Glow Serum",              "Tinted Mineral SPF 35"),
    ("Normal", "Dryness"):          ("Hydrating Cream Cleanser",     "Barrier Restore Moisturizer", "Hyaluronic Acid Hydration Serum",   "Tinted Mineral SPF 35"),
    ("Normal", "Wrinkles"):         ("Brightening Gel Cleanser",     "Anti-Aging Firming Cream",    "Retinol Renewal Serum",             "Daily Defense SPF 40"),
    ("Normal", "Fine Lines"):       ("Balancing Micellar Cleanser",  "Anti-Aging Firming Cream",    "Retinol Renewal Serum",             "Tinted Mineral SPF 35"),
    ("Normal", "Sun Damage"):       ("Brightening Gel Cleanser",     "Lightweight Daily Lotion",    "Vitamin C Glow Serum",              "Daily Defense SPF 40"),
    ("Normal", "Redness"):          ("Balancing Micellar Cleanser",  "Barrier Restore Moisturizer", "Centella Calm Serum",               "Mineral Sunscreen SPF 30"),
    ("Normal", "Uneven Skin Tone"): ("Brightening Gel Cleanser",     "Lightweight Daily Lotion",    "Niacinamide Brightening Serum",     "Daily Defense SPF 40"),
    ("Normal", "Large Pores"):      ("Balancing Micellar Cleanser",  "Lightweight Daily Lotion",    "Niacinamide Brightening Serum",     "Mineral Sunscreen SPF 30"),

    # Sensitive skin
    ("Sensitive", "Redness"):       ("Soothing Milk Cleanser",       "Barrier Restore Moisturizer", "Centella Calm Serum",               "Mineral Sunscreen SPF 30"),
    ("Sensitive", "Dryness"):       ("Soothing Milk Cleanser",       "Barrier Restore Moisturizer", "Hyaluronic Acid Hydration Serum",   "Tinted Mineral SPF 35"),
    ("Sensitive", "Acne"):          ("Soothing Milk Cleanser",       "Barrier Restore Moisturizer", "Centella Calm Serum",               "Mineral Sunscreen SPF 30"),
    ("Sensitive", "Dark Spots"):    ("Soothing Milk Cleanser",       "Barrier Restore Moisturizer", "Centella Calm Serum",               "Tinted Mineral SPF 35"),
    ("Sensitive", "Hyperpigmentation"): ("Soothing Milk Cleanser",   "Barrier Restore Moisturizer", "Centella Calm Serum",               "Mineral Sunscreen SPF 30"),
    ("Sensitive", "Uneven Skin Tone"):  ("Soothing Milk Cleanser",   "Barrier Restore Moisturizer", "Hyaluronic Acid Hydration Serum",   "Tinted Mineral SPF 35"),
    ("Sensitive", "Fine Lines"):    ("Soothing Milk Cleanser",       "Barrier Restore Moisturizer", "Centella Calm Serum",               "Mineral Sunscreen SPF 30"),
    ("Sensitive", "Wrinkles"):      ("Soothing Milk Cleanser",       "Barrier Restore Moisturizer", "Centella Calm Serum",               "Tinted Mineral SPF 35"),
    ("Sensitive", "Sun Damage"):    ("Soothing Milk Cleanser",       "Barrier Restore Moisturizer", "Centella Calm Serum",               "Mineral Sunscreen SPF 30"),
    ("Sensitive", "Large Pores"):   ("Soothing Milk Cleanser",       "Barrier Restore Moisturizer", "Hyaluronic Acid Hydration Serum",   "Tinted Mineral SPF 35"),
}

# ---------------------------------------------------------------------------
# Perturbation pools per category — alternate products introduced by
# small random perturbation so that multiple products appear per skin-type /
# concern pair (satisfying requirement 10.5).
# ---------------------------------------------------------------------------

CLEANSER_POOL     = [
    "Gentle Foaming Cleanser",
    "Hydrating Cream Cleanser",
    "Balancing Micellar Cleanser",
    "Soothing Milk Cleanser",
    "Brightening Gel Cleanser",
]
MOISTURIZER_POOL  = [
    "Oil-Free Hydrator",
    "Rich Repair Cream",
    "Barrier Restore Moisturizer",
    "Lightweight Daily Lotion",
    "Anti-Aging Firming Cream",
]
SERUM_POOL        = [
    "Niacinamide Brightening Serum",
    "Vitamin C Glow Serum",
    "Hyaluronic Acid Hydration Serum",
    "Retinol Renewal Serum",
    "Salicylic Acid Acne Serum",
    "Centella Calm Serum",
]
SUNSCREEN_POOL    = [
    "Lightweight SPF 50 Fluid",
    "Hydrating Sunscreen SPF 50",
    "Mineral Sunscreen SPF 30",
    "Daily Defense SPF 40",
    "Tinted Mineral SPF 35",
]

# Probability of applying a perturbation to each column (≈20 % of rows)
PERTURB_PROB = 0.20


def _perturb(value: str, pool: list[str]) -> str:
    """Return a random alternate from the pool (never the same as value)."""
    alternates = [p for p in pool if p != value]
    return random.choice(alternates) if alternates else value


# ---------------------------------------------------------------------------
# Age sampling — realistic distribution
#   Tails  : 10–17  (weight 5 %)  and  66–100 (weight 10 %)
#   Core   : 18–65  (weight 85 %)
# ---------------------------------------------------------------------------

def _sample_age() -> int:
    draw = random.random()
    if draw < 0.05:
        return random.randint(10, 17)
    elif draw < 0.15:
        return random.randint(66, 100)
    else:
        return random.randint(18, 65)


# ---------------------------------------------------------------------------
# Stratified pool construction
#   5 skin types, max 40 % each → max 480 rows per type for N=1200.
#   To keep all types represented, split 1200 evenly (240 per type).
# ---------------------------------------------------------------------------

SKIN_TYPES = ["Oily", "Dry", "Combination", "Normal", "Sensitive"]
CONCERNS   = [
    "Acne", "Dark Spots", "Hyperpigmentation", "Dryness",
    "Redness", "Uneven Skin Tone", "Wrinkles", "Fine Lines",
    "Sun Damage", "Large Pores",
]

TOTAL_ROWS = 1_200
ROWS_PER_SKIN_TYPE = TOTAL_ROWS // len(SKIN_TYPES)   # 240 each


def generate() -> list[dict]:
    """Return a list of row dicts for the dataset."""
    rows: list[dict] = []

    for skin_type in SKIN_TYPES:
        for _ in range(ROWS_PER_SKIN_TYPE):
            concern = random.choice(CONCERNS)

            key = (skin_type, concern)
            cleanser, moisturizer, serum, sunscreen = MAPPING[key]

            # Small random perturbation (~20 % chance per column)
            if random.random() < PERTURB_PROB:
                cleanser    = _perturb(cleanser,    CLEANSER_POOL)
            if random.random() < PERTURB_PROB:
                moisturizer = _perturb(moisturizer, MOISTURIZER_POOL)
            if random.random() < PERTURB_PROB:
                serum       = _perturb(serum,       SERUM_POOL)
            if random.random() < PERTURB_PROB:
                sunscreen   = _perturb(sunscreen,   SUNSCREEN_POOL)

            rows.append({
                "Age":                    _sample_age(),
                "Skin_Type":              skin_type,
                "Concern":                concern,
                "Severity_Score":         round(random.uniform(1.0, 10.0), 2),
                "Recommended_Cleanser":   cleanser,
                "Recommended_Moisturizer": moisturizer,
                "Recommended_Serum":      serum,
                "Recommended_Sunscreen":  sunscreen,
            })

    # Shuffle so consecutive rows are not ordered by skin type
    random.shuffle(rows)
    return rows


COLUMNS = [
    "Age",
    "Skin_Type",
    "Concern",
    "Severity_Score",
    "Recommended_Cleanser",
    "Recommended_Moisturizer",
    "Recommended_Serum",
    "Recommended_Sunscreen",
]

OUTPUT_PATH = os.path.join("data", "skincare_dataset.csv")


def main() -> None:
    rows = generate()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rows written : {len(rows)}")
    print(f"Column names : {', '.join(COLUMNS)}")
    print(f"Output file  : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
