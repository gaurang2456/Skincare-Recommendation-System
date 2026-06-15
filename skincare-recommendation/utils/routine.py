"""
utils/routine.py

Pure function for generating morning and night skincare routines.

Implemented fully in Task 6.2. This module is imported by ml/predictor.py.
"""

from __future__ import annotations

# Import RoutineStep from predictor to avoid circular imports — use a local
# definition here and re-export. predictor.py owns the canonical dataclass.
from dataclasses import dataclass


@dataclass
class RoutineStep:
    step: int
    category: str
    product: str


MORNING_ORDER = ["Cleanser", "Serum", "Moisturizer", "Sunscreen"]
NIGHT_ORDER = ["Cleanser", "Treatment Serum", "Moisturizer"]


def generate_routines(
    cleanser: str,
    moisturizer: str,
    serum: str,
    sunscreen: str,
) -> tuple[list[RoutineStep], list[RoutineStep]]:
    """Return (morning_routine, night_routine) as lists of RoutineStep.

    Morning order: Cleanser → Serum → Moisturizer → Sunscreen
    Night order:   Cleanser → Treatment Serum → Moisturizer

    The serum doubles as the Treatment Serum in the night routine.
    """
    product_map = {
        "Cleanser": cleanser,
        "Serum": serum,
        "Moisturizer": moisturizer,
        "Sunscreen": sunscreen,
        "Treatment Serum": serum,
    }

    morning = [
        RoutineStep(step=i + 1, category=cat, product=product_map[cat])
        for i, cat in enumerate(MORNING_ORDER)
    ]
    night = [
        RoutineStep(step=i + 1, category=cat, product=product_map[cat])
        for i, cat in enumerate(NIGHT_ORDER)
    ]

    return morning, night
