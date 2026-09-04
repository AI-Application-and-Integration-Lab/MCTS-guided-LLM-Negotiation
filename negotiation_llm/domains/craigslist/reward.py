"""
Reward function for the Craigslist Bargains buyer agent.

The buyer's goal is to purchase the item at the lowest possible price.
Reward is based on how much discount the buyer achieved relative to
the listing price.
"""

from math import tanh
from typing import Dict, Any


# Average expected discount rate in CB dataset (calibration constant).
# CB negotiations typically end ~15-25% below listing price.
_AVG_EXPECTED_DISCOUNT = 0.15


def calculate_cb_reward(
    final_price: float,
    listing_price: float,
    buyer_target: float,
    outcome: str,
    num_rounds: int,
    max_rounds: int,
    failure_penalty: float = -0.5,
    max_rounds_penalty: float = -0.5,
) -> float:
    """
    Calculate MCTS reward for the CB buyer agent.

    Reward is normalized to [-1.0, 1.0]:
      - accepted: tanh(discount_rate / avg_expected_discount) + efficiency_bonus
      - rejected:   failure_penalty  (default -0.5)
      - max_rounds: max_rounds_penalty (default -0.5)

    Where discount_rate = (listing_price - final_price) / listing_price.

    A 15% discount yields reward ≈ 0.76 (tanh(1.0)).
    A 30% discount yields reward ≈ 0.96 (tanh(2.0)).
    A 0%  discount yields reward ≈ 0.0.

    Args:
        final_price: Agreed final price
        listing_price: Original listing price (denominator for normalization)
        buyer_target: Buyer's target price (for context; not used in main reward)
        outcome: "accepted", "rejected", "max_rounds", or "max_rounds_reached"
        num_rounds: Rounds taken to reach this state
        max_rounds: Maximum allowed rounds
        failure_penalty: Penalty for explicit rejection
        max_rounds_penalty: Penalty for exhausting rounds without a deal

    Returns:
        Reward in [failure_penalty, 1.0]
    """

    if outcome in ("max_rounds", "max_rounds_reached"):
        return max_rounds_penalty

    if outcome in ("accepted", "accept", "agreement"):
        if listing_price <= 0:
            return 0.0

        discount_rate = (listing_price - final_price) / listing_price

        # Normalize via tanh so reward is bounded in (-1, 1)
        discount_reward = tanh(discount_rate / _AVG_EXPECTED_DISCOUNT)

        # Target achievement: how close is final_price to buyer_target?
        # 1.0 if final_price <= buyer_target, drops toward 0 as final_price rises above target
        if buyer_target is not None and buyer_target > 0:
            price_range = listing_price - buyer_target
            if price_range > 0:
                target_ratio = (listing_price - final_price) / price_range
                target_reward = tanh(max(0.0, target_ratio))
            else:
                target_reward = 1.0 if final_price <= buyer_target else 0.0
            # Blend: 50% discount from listing, 50% closeness to target
            base_reward = 0.5 * discount_reward + 0.5 * target_reward
        else:
            base_reward = discount_reward

        # Efficiency bonus: up to +0.10 for fast negotiations
        rounds_saved = max(0, max_rounds - num_rounds)
        efficiency_bonus = (rounds_saved / max_rounds) * 0.10

        return min(1.0, base_reward + efficiency_bonus)

    # Unknown outcome
    return 0.0


def calculate_cb_reward_from_terms(
    final_terms: Dict[str, Any],
    agent_profile: Dict[str, Any],
    item_config: Dict[str, Any],
    outcome: str,
    num_rounds: int,
    max_rounds: int,
    failure_penalty: float = -1.0,
    max_rounds_penalty: float = -0.5,
) -> float:
    """
    Convenience wrapper that extracts values from dicts.

    Args:
        final_terms: Last agent (buyer) offer dict, must contain "price"
        agent_profile: Buyer profile with "target_price"
        item_config: Item config with "listing_price"
        outcome: Negotiation outcome string
        num_rounds: Rounds elapsed
        max_rounds: Maximum rounds allowed
    """
    final_price = final_terms.get("price")
    listing_price = item_config.get("listing_price", 0)
    buyer_target = item_config.get("buyer_target") or agent_profile.get("target_price", 0)

    if final_price is None:
        # No price in final terms — treat as failure
        if outcome in ("accepted", "accept"):
            return 0.0
        return calculate_cb_reward(
            final_price=listing_price,  # worst case
            listing_price=listing_price,
            buyer_target=buyer_target,
            outcome=outcome,
            num_rounds=num_rounds,
            max_rounds=max_rounds,
            failure_penalty=failure_penalty,
            max_rounds_penalty=max_rounds_penalty,
        )

    return calculate_cb_reward(
        final_price=final_price,
        listing_price=listing_price,
        buyer_target=buyer_target,
        outcome=outcome,
        num_rounds=num_rounds,
        max_rounds=max_rounds,
        failure_penalty=failure_penalty,
        max_rounds_penalty=max_rounds_penalty,
    )
