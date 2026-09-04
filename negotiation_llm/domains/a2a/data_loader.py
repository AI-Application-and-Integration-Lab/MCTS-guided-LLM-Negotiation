"""
Agent2Agent Consumer Negotiation dataset loader.

Loads products.json / products_mini.json and creates negotiation scenarios
where the buyer targets the wholesale price and the seller starts at retail.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional


def _parse_price(s: str) -> Optional[float]:
    """Parse '$1,234.56' → 1234.56."""
    try:
        cleaned = str(s).replace(",", "").replace("$", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def load_a2a_scenarios(
    json_path: str,
    start_index: int = 0,
    end_index: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    buyer_target_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Load A2A products JSON and return negotiation scenarios.

    Each scenario:
    {
        "scenario_id": int,
        "buyer":  {"role": "buyer",  "target_price": float},
        "seller": {"role": "seller", "target_price": float},  # wholesale price
        "item": {
            "name": str,
            "type": str,          # Vehicle | Electronics | Real Estate
            "features": str,
            "listing_price": float,   # retail price
            "wholesale_price": float,
            "reference": str,
        },
    }

    Buyer target is set as:
        buyer_target = wholesale + (retail - wholesale) * buyer_target_ratio

    Default ratio 0.0  → buyer targets the wholesale price exactly.
    Ratio 0.2 → buyer targets 20% of the spread above wholesale.

    Args:
        json_path: Path to products.json or products_mini.json
        start_index: First scenario index after optional shuffle
        end_index: One-past-last index (None = all)
        shuffle: Shuffle before slicing
        seed: RNG seed
        buyer_target_ratio: How far above wholesale the buyer targets [0, 1]
    """
    with open(json_path) as f:
        products = json.load(f)

    scenarios = []
    for product in products:
        retail = _parse_price(product.get("Retail Price"))
        wholesale = _parse_price(product.get("Wholesale Price"))

        if retail is None or wholesale is None or retail <= 0 or wholesale <= 0:
            continue

        buyer_target = wholesale + (retail - wholesale) * buyer_target_ratio

        scenarios.append({
            "scenario_id": len(scenarios),
            "buyer": {
                "role": "buyer",
                "target_price": round(buyer_target, 2),
            },
            "seller": {
                "role": "seller",
                "target_price": round(wholesale, 2),   # seller's floor
            },
            "item": {
                "name": str(product.get("Product Name", "Item")),
                "type": str(product.get("Type", "misc")),
                "features": str(product.get("Features", "")),
                "listing_price": round(retail, 2),
                "wholesale_price": round(wholesale, 2),
                "reference": str(product.get("Reference", "")),
            },
        })

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(scenarios)

    return scenarios[start_index:end_index]


# ------------------------------------------------------------------ #
# Seller persona generation                                            #
# ------------------------------------------------------------------ #

_DEFAULT_PERSONAS = [
    "a motivated seller who wants a fair price for their product",
    "a professional seller with in-depth knowledge of the product's value",
]

_PERSONALITY_TRAITS = {
    "openness":          "creative and flexible, willing to explore unconventional deal structures",
    "conscientiousness": "organized and firm, keeps detailed records and sticks to fair pricing",
    "extraversion":      "outgoing and talkative, enjoys the social aspect of bargaining",
    "agreeableness":     "cooperative and accommodating, prefers to reach a mutually satisfying deal",
    "neuroticism":       "cautious and anxious about being lowballed, needs reassurance about the deal",
}

_DECISION_STYLES = {
    "directive":  "decisive and action-oriented, preferring quick deals with clear outcomes",
    "analytical": "data-driven and thorough, carefully comparing market prices before deciding",
    "conceptual": "big-picture focused, considering the buyer's situation and creative trade-offs",
    "behavioral": "people-oriented and collaborative, valuing rapport and a friendly exchange",
}


def generate_seller_persona(
    scenario: Dict[str, Any],
    seed: Optional[int] = None,
    return_components: bool = False,
) -> "str | dict":
    """
    Generate a seller persona string for an A2A scenario.

    Combines Big Five personality trait and decision-making style,
    sampled deterministically from scenario_id.

    Args:
        scenario: A2A scenario dict with scenario_id
        seed: Optional random seed (defaults to scenario_id)
        return_components: If True, return dict with persona, big_five_personality,
            and decision_making_style instead of just the persona string.
    """
    scenario_id = scenario.get("scenario_id", 0)
    rng = random.Random(seed if seed is not None else scenario_id)

    # Personality trait (cycle uniformly by scenario_id)
    trait_names = list(_PERSONALITY_TRAITS.keys())
    trait_desc = _PERSONALITY_TRAITS[trait_names[scenario_id % len(trait_names)]]

    # Decision-making style (offset by 2 to decouple from personality)
    style_names = list(_DECISION_STYLES.keys())
    style_desc = _DECISION_STYLES[style_names[(scenario_id + 2) % len(style_names)]]

    background = rng.choice(_DEFAULT_PERSONAS)

    persona = (
        f"{background} who is {trait_desc}. "
        f"In decision-making, this seller is {style_desc}."
    )
    if return_components:
        return {
            "persona": persona,
            "big_five_personality": trait_desc,
            "decision_making_style": style_desc,
        }
    return persona
