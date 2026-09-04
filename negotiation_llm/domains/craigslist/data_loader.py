"""
Craigslist Bargains dataset loader.

Parses CB CSV files (train/test/validation) into negotiation scenarios
suitable for MCTS evaluation.
"""

import ast
import re
import random
from pathlib import Path
from typing import Dict, Any, List, Optional


def _parse_numpy_str_array(s: str) -> List[str]:
    """Parse numpy-repr string array: ['str1'\\n 'str2'] → ['str1', 'str2']."""
    # Insert commas between adjacent quoted strings
    s2 = re.sub(r"'\s*'", "', '", s)
    return ast.literal_eval(s2)


def _parse_agent_turn(s: str) -> List[int]:
    """Parse '[0 1 0 0]' → [0, 1, 0, 0]."""
    return [int(n) for n in re.findall(r'\d+', s)]


def _safe_parse_dict(s: str) -> Dict[str, Any]:
    """Parse dict that may contain numpy array repr."""
    s2 = re.sub(
        r"array\(\[([^\]]*)\],\s*dtype=\w+\)",
        lambda m: '[' + m.group(1) + ']',
        s
    )
    return ast.literal_eval(s2)


def load_cb_scenarios(
    csv_path: str,
    start_index: int = 0,
    end_index: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load CB CSV and return list of negotiation scenarios.

    Each scenario has the structure:
    {
        "scenario_id": int,
        "buyer": {"role": "buyer", "target_price": float},
        "seller": {"role": "seller", "target_price": float},
        "item": {
            "title": str,
            "category": str,
            "description": str,
            "listing_price": float
        },
        "human_utterances": List[str],   # original dialogue turns
        "human_agent_turns": List[int],  # 0=buyer, 1=seller per turn
    }

    Args:
        csv_path: Path to train.csv / test.csv / validation.csv
        start_index: First scenario index to include (0-based, after shuffle)
        end_index: One-past-last scenario index (None = all remaining)
        shuffle: Whether to shuffle before slicing
        seed: Random seed for shuffling

    Returns:
        List of scenario dicts
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for loading CB scenarios: pip install pandas")

    df = pd.read_csv(csv_path)
    scenarios = []

    for _, row in df.iterrows():
        try:
            agent_info = _safe_parse_dict(row['agent_info'])
            items = _safe_parse_dict(row['items'])

            roles = list(agent_info.get('Role', []))
            if 'buyer' not in roles or 'seller' not in roles:
                continue

            buyer_idx = roles.index('buyer')
            seller_idx = roles.index('seller')

            targets = agent_info.get('Target', [None, None])
            buyer_target = float(targets[buyer_idx]) if targets[buyer_idx] is not None else None
            seller_target = float(targets[seller_idx]) if targets[seller_idx] is not None else None

            listing_price = float(items['Price'][0]) if items.get('Price') else None
            if not listing_price or listing_price <= 0:
                continue

            title = str(items['Title'][0]) if items.get('Title') else "Item"
            category = str(items['Category'][0]) if items.get('Category') else "misc"
            description = str(items['Description'][0]) if items.get('Description') else ""

            # Parse human dialogue
            try:
                utterances = _parse_numpy_str_array(row['utterance'])
                utterances = [u for u in utterances if u.strip()]  # remove empty turns
            except Exception:
                utterances = []

            try:
                agent_turns = _parse_agent_turn(row['agent_turn'])
            except Exception:
                agent_turns = []

            scenarios.append({
                "scenario_id": len(scenarios),
                "buyer": {
                    "role": "buyer",
                    "target_price": buyer_target,
                },
                "seller": {
                    "role": "seller",
                    "target_price": seller_target,
                },
                "item": {
                    "title": title,
                    "category": category,
                    "description": description,
                    "listing_price": listing_price,
                },
                "human_utterances": utterances,
                "human_agent_turns": agent_turns,
            })

        except (ValueError, KeyError, IndexError, SyntaxError, TypeError):
            continue

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(scenarios)

    scenarios = scenarios[start_index:end_index]

    return scenarios


# ------------------------------------------------------------------ #
# Seller persona generation                                            #
# ------------------------------------------------------------------ #

# Big Five personality traits and their seller negotiation descriptions
_PERSONALITY_TRAITS = {
    "openness": "creative and flexible, willing to explore unconventional deal structures",
    "conscientiousness": "organized and firm, keeps detailed records and sticks to fair pricing",
    "extraversion": "outgoing and talkative, enjoys the social aspect of bargaining",
    "agreeableness": "cooperative and accommodating, prefers to reach a mutually satisfying deal",
    "neuroticism": "cautious and anxious about being lowballed, needs reassurance about the deal",
}

# Decision-making styles
_DECISION_STYLES = {
    "directive": "decisive and action-oriented, preferring quick deals with clear outcomes",
    "analytical": "data-driven and thorough, carefully comparing market prices before deciding",
    "conceptual": "big-picture focused, considering the buyer's situation and creative trade-offs",
    "behavioral": "people-oriented and collaborative, valuing rapport and a friendly exchange",
}

_DEFAULT_BACKGROUNDS = [
    "a private seller looking to declutter",
    "someone who bought it but never really used it",
    "a person selling to make room for something new",
]


def generate_seller_persona(
    scenario: Dict[str, Any],
    seed: Optional[int] = None,
    return_components: bool = False,
) -> "str | dict":
    """
    Generate a seller persona string from a CB scenario.

    Combines:
    - Big Five personality trait (uniform rotation by scenario_id)
    - Decision-making style (uniform rotation, offset from personality)
    - Generic background

    Args:
        scenario: CB scenario dict with seller, item, scenario_id
        seed: Optional random seed (defaults to scenario_id)

    Returns:
        Persona string for the seller prompt
    """
    scenario_id = scenario.get("scenario_id", 0)
    rng = random.Random(seed if seed is not None else scenario_id)

    # 1. Personality trait (cycle uniformly)
    trait_names = list(_PERSONALITY_TRAITS.keys())
    trait_name = trait_names[scenario_id % len(trait_names)]
    trait_desc = _PERSONALITY_TRAITS[trait_name]

    # 2. Decision-making style (cycle uniformly, offset by 2 to decouple from personality)
    style_names = list(_DECISION_STYLES.keys())
    style_name = style_names[(scenario_id + 2) % len(style_names)]
    style_desc = _DECISION_STYLES[style_name]

    # 3. Background
    background = rng.choice(_DEFAULT_BACKGROUNDS)

    persona = (
        f"{background} who is {trait_desc}. "
        f"In decision-making, this seller is {style_desc}."
    )
    if return_components:
        return {
            "persona": persona,
            "big_five_personality": trait_name,
            "decision_making_style": style_name,
        }
    return persona


def format_cb_history(
    history: List[Dict[str, Any]],
    buyer_label: str = "Buyer",
    seller_label: str = "Seller"
) -> str:
    """Format negotiation history for CB (price-based) domain as readable text."""
    if not history:
        return "(No conversation yet)\n"

    lines = []
    for i, turn in enumerate(history, 1):
        speaker = turn.get('speaker', 'unknown')
        dialogue = turn.get('dialogue', '')
        offer = turn.get('offer', {})
        price = offer.get('price') if offer else None

        label = buyer_label if speaker == 'buyer' else seller_label
        line = f"Turn {i} - {label}: {dialogue}"
        if price is not None:
            line += f"  [Price: ${price}]"
        lines.append(line)

    return "\n".join(lines) + "\n"
