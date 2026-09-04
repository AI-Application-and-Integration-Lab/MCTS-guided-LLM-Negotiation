"""
A2A consumer negotiation domain.

MCTS agent = buyer (minimise purchase price).
LLM opponent = seller (professional, knows cost floor).
"""

import sys
import os
from typing import Dict, Any, List, Optional


from negotiation_llm.mcts.domain import NegotiationDomain
from .strategies import A2ABuyerStrategy, STRATEGY_CONFIGS, get_default_a2a_strategies, get_a2a_strategy_prior
from .prompts import (
    create_a2a_buyer_prompt,
    create_a2a_buyer_strategy_selection_prompt,
    create_a2a_seller_prompt,
    create_a2a_seller_action_prompt,
    parse_a2a_buyer_response,
    parse_a2a_seller_response,
)
from .reward import calculate_a2a_reward_from_terms


class A2ANegotiationDomain(NegotiationDomain):
    """
    Domain for A2A consumer negotiations (Vehicles, Electronics, Real Estate).

    Usage:
        domain = A2ANegotiationDomain(
            item_config=scenario["item"],
            seller_persona="a car dealer who rarely moves far from the asking price",
            seller_target=scenario["seller"]["target_price"],
        )
    """

    def __init__(
        self,
        item_config: Dict[str, Any],
        seller_persona: str = "a professional seller who wants a fair price",
        seller_target: Optional[float] = None,
    ):
        self.item_config = item_config
        self.seller_persona = seller_persona
        self.seller_target = seller_target
        # Analyzer's latest predictions of the seller's next move, used to make
        # the simulated opponent in MCTS rollouts more realistic. Updated per turn.
        self.predicted_action: Optional[str] = None
        self.predicted_counter: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Strategies                                                           #
    # ------------------------------------------------------------------ #

    def get_strategies(self) -> List[A2ABuyerStrategy]:
        return get_default_a2a_strategies()

    def get_strategy_prior(
        self, strategy: A2ABuyerStrategy, context: Dict[str, Any] = None
    ) -> float:
        return get_a2a_strategy_prior(strategy, context)

    # ------------------------------------------------------------------ #
    # Prompts                                                              #
    # ------------------------------------------------------------------ #

    def create_strategy_selection_prompt(
        self,
        agent_profile: Dict[str, Any],
        strategies: List[A2ABuyerStrategy],
        history: List[Dict[str, Any]],
    ) -> str:
        return create_a2a_buyer_strategy_selection_prompt(
            buyer_profile=agent_profile,
            strategies=strategies,
            history=history,
            item_config=self.item_config,
        )

    def create_strategy_prompt(
        self,
        strategy: A2ABuyerStrategy,
        agent_profile: Dict[str, Any],
        domain_config: Dict[str, Any],
        history: List[Dict[str, Any]],
        current_opponent_offer: Optional[Dict[str, Any]] = None,
        current_opponent_dialogue: Optional[str] = None,
        previous_agent_offer: Optional[Dict[str, Any]] = None,
    ) -> str:
        item = domain_config if domain_config else self.item_config
        return create_a2a_buyer_prompt(
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
        current_agent_dialogue: Optional[str] = None,
    ) -> str:
        return create_a2a_seller_prompt(
            seller_persona=self.seller_persona,
            item_config=self.item_config,
            history=history,
            current_buyer_offer=current_agent_offer,
            current_buyer_dialogue=current_agent_dialogue,
            seller_target=self.seller_target,
            predicted_action=self.predicted_action,
            predicted_counter=self.predicted_counter,
        )

    def create_opponent_action_prompt(
        self,
        opponent_profile: Dict[str, Any],
        history: List[Dict[str, Any]],
        current_agent_offer: Optional[Dict[str, Any]] = None,
        current_agent_dialogue: Optional[str] = None,
    ) -> str:
        return create_a2a_seller_action_prompt(
            seller_persona=self.seller_persona,
            item_config=self.item_config,
            history=history,
            current_buyer_offer=current_agent_offer,
            current_buyer_dialogue=current_agent_dialogue,
            seller_target=self.seller_target,
            predicted_action=self.predicted_action,
            predicted_counter=self.predicted_counter,
        )

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #

    def parse_agent_response(self, response: str) -> Dict[str, Any]:
        return parse_a2a_buyer_response(response)

    def parse_opponent_response(self, response: str) -> Dict[str, Any]:
        return parse_a2a_seller_response(response)

    # ------------------------------------------------------------------ #
    # Offer construction                                                   #
    # ------------------------------------------------------------------ #

    def build_agent_offer(
        self, parsed: Dict[str, Any], agent_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"price": parsed.get("price"), "action_type": parsed.get("action_type", "propose")}

    def build_opponent_offer(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        return {"price": parsed.get("price"), "action_type": parsed.get("action_type", "counter")}

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
        return calculate_a2a_reward_from_terms(
            final_terms=final_terms,
            agent_profile=agent_profile,
            item_config=self.item_config,
            outcome=outcome,
            num_rounds=num_rounds,
            max_rounds=max_rounds,
        )

    # ------------------------------------------------------------------ #
    # Strategy parsing                                                     #
    # ------------------------------------------------------------------ #

    def parse_strategy_from_response(
        self, response: str, available_strategies: List[A2ABuyerStrategy]
    ) -> Optional[A2ABuyerStrategy]:
        import re
        match = re.match(r'^\[([^\]]+)\]', response.strip())
        if match:
            found = match.group(1).strip()
            for s in available_strategies:
                if s.value.lower() == found.lower():
                    return s
        response_lower = response.lower()
        for s in available_strategies:
            if s.value.lower() in response_lower:
                return s
        return None
