"""
Reward function for the A2A consumer negotiation buyer agent.

Same tanh-based structure as the CB reward. The baseline expected
discount is calibrated to typical consumer negotiation margins:
  - Vehicles:     5-15% below retail
  - Electronics:  10-25% below retail
  - Real Estate:  5-20% below retail
  → blended average ~15% (same constant as CB).
"""

from math import tanh
from typing import Dict, Any

_AVG_EXPECTED_DISCOUNT = 0.15


def calculate_a2a_reward(
    final_price: float,
    listing_price: float,
    buyer_target: float,
    outcome: str,
    num_rounds: int,
    max_rounds: int,
    failure_penalty: float = -1.0,
    max_rounds_penalty: float = -0.5,
) -> float:
    """
    Reward for the A2A buyer agent, normalised to [-1, 1].

    Accepted:   tanh-blend of discount-from-retail and closeness to target
                + small efficiency bonus for fewer rounds.
    Timeout:    max_rounds_penalty (default -0.5)
    Failure:    failure_penalty    (default -1.0)
    """
    if outcome in ("max_rounds", "max_rounds_reached"):
        return max_rounds_penalty

    if outcome in ("accepted", "accept", "agreement"):
        if listing_price <= 0:
            return 0.0

        discount_rate = (listing_price - final_price) / listing_price
        discount_reward = tanh(discount_rate / _AVG_EXPECTED_DISCOUNT)

        if buyer_target is not None and buyer_target > 0:
            price_range = listing_price - buyer_target
            if price_range > 0:
                target_ratio = (listing_price - final_price) / price_range
                target_reward = tanh(max(0.0, target_ratio))
            else:
                target_reward = 1.0 if final_price <= buyer_target else 0.0
            base_reward = 0.5 * discount_reward + 0.5 * target_reward
        else:
            base_reward = discount_reward

        rounds_saved = max(0, max_rounds - num_rounds)
        efficiency_bonus = (rounds_saved / max_rounds) * 0.10

        return min(1.0, base_reward + efficiency_bonus)

    return 0.0


def calculate_a2a_reward_from_terms(
    final_terms: Dict[str, Any],
    agent_profile: Dict[str, Any],
    item_config: Dict[str, Any],
    outcome: str,
    num_rounds: int,
    max_rounds: int,
    failure_penalty: float = -1.0,
    max_rounds_penalty: float = -0.5,
) -> float:
    final_price = final_terms.get("price")
    listing_price = item_config.get("listing_price", 0)
    buyer_target = agent_profile.get("target_price", 0)

    if final_price is None:
        if outcome in ("accepted", "accept"):
            return 0.0
        return calculate_a2a_reward(
            final_price=listing_price,
            listing_price=listing_price,
            buyer_target=buyer_target,
            outcome=outcome,
            num_rounds=num_rounds,
            max_rounds=max_rounds,
            failure_penalty=failure_penalty,
            max_rounds_penalty=max_rounds_penalty,
        )

    return calculate_a2a_reward(
        final_price=final_price,
        listing_price=listing_price,
        buyer_target=buyer_target,
        outcome=outcome,
        num_rounds=num_rounds,
        max_rounds=max_rounds,
        failure_penalty=failure_penalty,
        max_rounds_penalty=max_rounds_penalty,
    )
