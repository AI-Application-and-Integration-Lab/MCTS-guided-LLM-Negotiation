"""
Craigslist Bargains negotiation domain.

MCTS agent = buyer (tries to minimize purchase price).
LLM opponent = seller (tries to maximize sale price).

History uses "buyer"/"seller" speaker labels (matching
MCTSNegotiator's conventions) while prompts use "Buyer"/"Seller".
"""

import sys
import os
from typing import Dict, Any, List, Optional


from negotiation_llm.mcts.domain import NegotiationDomain
from .strategies import (
    CraigslistBuyerStrategy,
    STRATEGY_CONFIGS,
    get_default_cb_strategies,
    get_cb_strategy_prior,
)
from .prompts import (
    create_cb_buyer_prompt,
    create_cb_buyer_strategy_selection_prompt,
    create_cb_seller_prompt,
    create_cb_seller_action_prompt,
    parse_cb_buyer_response,
    parse_cb_seller_response,
)
from .reward import calculate_cb_reward_from_terms


class CraigslistNegotiationDomain(NegotiationDomain):
    """
    Domain implementation for Craigslist Bargains price negotiations.

    The MCTS agent plays the buyer role. The LLM simulates the seller.

    Usage:
        domain = CraigslistNegotiationDomain(
            item_config=scenario["item"],
            seller_persona="a motivated seller who wants a fair price",
        )
        mcts = MCTSNegotiator(model=model, domain_config=scenario["item"], domain=domain)
    """

    def __init__(
        self,
        item_config: Dict[str, Any],
        seller_persona: str = "a private seller who wants a fair price for their item",
        seller_target: Optional[float] = None,
    ):
        """
        Args:
            item_config: Item metadata: {title, category, description, listing_price}
            seller_persona: Short persona description for the LLM-simulated seller
            seller_target: Seller's minimum acceptable price (included in seller prompt)
        """
        self.item_config = item_config
        self.seller_persona = seller_persona
        self.seller_target = seller_target

    # ------------------------------------------------------------------ #
    # Strategies                                                           #
    # ------------------------------------------------------------------ #

    def get_strategies(self) -> List[CraigslistBuyerStrategy]:
        return get_default_cb_strategies()

    def get_strategy_prior(
        self, strategy: CraigslistBuyerStrategy, context: Dict[str, Any] = None
    ) -> float:
        return get_cb_strategy_prior(strategy, context)

    # ------------------------------------------------------------------ #
    # Prompts                                                              #
    # ------------------------------------------------------------------ #

    def create_strategy_selection_prompt(
        self,
        agent_profile: Dict[str, Any],
        strategies: List[CraigslistBuyerStrategy],
        history: List[Dict[str, Any]]
    ) -> str:
        return create_cb_buyer_strategy_selection_prompt(
            buyer_profile=agent_profile,
            strategies=strategies,
            history=history,
            item_config=self.item_config,
        )

    def create_strategy_prompt(
        self,
        strategy: CraigslistBuyerStrategy,
        agent_profile: Dict[str, Any],
        domain_config: Dict[str, Any],
        history: List[Dict[str, Any]],
        current_opponent_offer: Optional[Dict[str, Any]] = None,
        current_opponent_dialogue: Optional[str] = None,
        previous_agent_offer: Optional[Dict[str, Any]] = None
    ) -> str:
        item = domain_config if domain_config else self.item_config
        return create_cb_buyer_prompt(
            strategy=strategy,
            buyer_profile=agent_profile,
            item_config=item,
            history=history,
            current_seller_offer=current_opponent_offer,
            current_seller_dialogue=current_opponent_dialogue,
            previous_buyer_offer=previous_agent_offer,
        )

    def create_opponent_prompt(
        self,
        opponent_profile: Dict[str, Any],
        history: List[Dict[str, Any]],
        current_agent_offer: Optional[Dict[str, Any]] = None,
        current_agent_dialogue: Optional[str] = None
    ) -> str:
        return create_cb_seller_prompt(
            seller_persona=self.seller_persona,
            item_config=self.item_config,
            history=history,
            current_buyer_offer=current_agent_offer,
            current_buyer_dialogue=current_agent_dialogue,
            seller_target=self.seller_target,
        )

    def create_opponent_action_prompt(
        self,
        opponent_profile: Dict[str, Any],
        history: List[Dict[str, Any]],
        current_agent_offer: Optional[Dict[str, Any]] = None,
        current_agent_dialogue: Optional[str] = None
    ) -> str:
        return create_cb_seller_action_prompt(
            seller_persona=self.seller_persona,
            item_config=self.item_config,
            history=history,
            current_buyer_offer=current_agent_offer,
            current_buyer_dialogue=current_agent_dialogue,
            seller_target=self.seller_target,
        )

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #

    def parse_agent_response(self, response: str) -> Dict[str, Any]:
        return parse_cb_buyer_response(response)

    def parse_opponent_response(self, response: str) -> Dict[str, Any]:
        return parse_cb_seller_response(response)

    # ------------------------------------------------------------------ #
    # Offer construction                                                   #
    # ------------------------------------------------------------------ #

    def build_agent_offer(
        self, parsed: Dict[str, Any], agent_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build buyer offer dict from parsed buyer response."""
        price = parsed.get("price")
        action_type = parsed.get("action_type", "propose")

        # If buyer agreed but no explicit price, use seller's last price
        # (the caller will fill this in from history if needed)
        return {
            "price": price,
            "action_type": action_type,
        }

    def build_opponent_offer(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Build seller offer dict from parsed seller response."""
        return {
            "price": parsed.get("price"),
            "action_type": parsed.get("action_type", "counter"),
        }

    # ------------------------------------------------------------------ #
    # Reward                                                               #
    # ------------------------------------------------------------------ #

    def calculate_reward(
        self,
        final_terms: Dict[str, Any],
        agent_profile: Dict[str, Any],
        outcome: str,
        num_rounds: int,
        max_rounds: int,
    ) -> float:
        return calculate_cb_reward_from_terms(
            final_terms=final_terms,
            agent_profile=agent_profile,
            item_config=self.item_config,
            outcome=outcome,
            num_rounds=num_rounds,
            max_rounds=max_rounds,
        )

    # ------------------------------------------------------------------ #
    # Strategy parsing override                                            #
    # ------------------------------------------------------------------ #

    def parse_strategy_from_response(
        self, response: str, available_strategies: List[CraigslistBuyerStrategy]
    ) -> Optional[CraigslistBuyerStrategy]:
        """
        Parse strategy from '[Strategy Name]response' format.

        CB strategy values have spaces and mixed case, so we override
        the default snake_case-only regex.
        """
        import re
        match = re.match(r'^\[([^\]]+)\]', response.strip())
        if match:
            found = match.group(1).strip()
            for strategy in available_strategies:
                if strategy.value.lower() == found.lower():
                    return strategy
        # Substring fallback
        response_lower = response.lower()
        for strategy in available_strategies:
            if strategy.value.lower() in response_lower:
                return strategy
        return None
