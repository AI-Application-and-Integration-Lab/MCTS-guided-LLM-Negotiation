"""
Abstract base class for domain-specific negotiation logic.

Decouples the MCTS core algorithm from domain-specific concerns:
- Prompt generation (agent and opponent)
- Strategy definitions and priors
- Response parsing
- Reward calculation

New datasets only need to implement this interface; the MCTS core
(mcts_negotiator.py, mcts_node.py) stays unchanged.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple


class NegotiationDomain(ABC):
    """
    Abstract interface for domain-specific negotiation logic.

    Implement this class to support a new negotiation dataset.
    Pass an instance to MCTSNegotiator(domain=...).
    """

    # ------------------------------------------------------------------ #
    # Strategies                                                           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_strategies(self) -> List[Any]:
        """Return list of available strategy enum values for the agent."""
        ...

    @abstractmethod
    def get_strategy_prior(self, strategy: Any, context: Dict[str, Any] = None) -> float:
        """Return prior probability for a strategy (0-1)."""
        ...

    # ------------------------------------------------------------------ #
    # Prompt generation                                                    #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def create_strategy_selection_prompt(
        self,
        agent_profile: Dict[str, Any],
        strategies: List[Any],
        history: List[Dict[str, Any]]
    ) -> str:
        """Prompt asking LLM to select the best strategy for the current state."""
        ...

    @abstractmethod
    def create_strategy_prompt(
        self,
        strategy: Any,
        agent_profile: Dict[str, Any],
        domain_config: Dict[str, Any],
        history: List[Dict[str, Any]],
        current_opponent_offer: Optional[Dict[str, Any]] = None,
        current_opponent_dialogue: Optional[str] = None,
        previous_agent_offer: Optional[Dict[str, Any]] = None
    ) -> str:
        """Prompt for the agent (MCTS-controlled) to generate a response using the given strategy."""
        ...

    @abstractmethod
    def create_opponent_prompt(
        self,
        opponent_profile: Dict[str, Any],
        history: List[Dict[str, Any]],
        current_agent_offer: Optional[Dict[str, Any]] = None,
        current_agent_dialogue: Optional[str] = None
    ) -> str:
        """Prompt for the opponent (LLM-simulated) to generate a response."""
        ...

    @abstractmethod
    def create_opponent_action_prompt(
        self,
        opponent_profile: Dict[str, Any],
        history: List[Dict[str, Any]],
        current_agent_offer: Optional[Dict[str, Any]] = None,
        current_agent_dialogue: Optional[str] = None
    ) -> str:
        """
        Prompt for estimating opponent acceptance likelihood.

        Should end at "Action:" so logprob estimation can be applied
        to action token probabilities (accept/reject/ask).
        """
        ...

    # ------------------------------------------------------------------ #
    # Response parsing                                                     #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def parse_agent_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the agent's (MCTS-controlled party's) LLM response.

        Must return a dict with at minimum:
          - "dialogue": str
          - "action_type": str  ("ask"/"propose"/"accept"/"reject")
          - domain-specific offer fields (e.g., price)
        """
        ...

    @abstractmethod
    def parse_opponent_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the opponent's (LLM-simulated party's) response.

        Must return a dict with at minimum:
          - "dialogue": str
          - "action_type": str  ("accept"/"reject"/"ask"/"counter"/...)
          - domain-specific offer fields
        """
        ...

    # ------------------------------------------------------------------ #
    # Offer construction                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build_agent_offer(
        self,
        parsed_response: Dict[str, Any],
        agent_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build the structured agent offer dict from a parsed response.

        The returned dict is stored in history as: {"speaker": "buyer", "offer": <this>}
        It must include "action_type" so terminal state detection works.
        """
        ...

    @abstractmethod
    def build_opponent_offer(
        self,
        parsed_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build the structured opponent offer dict from a parsed response.

        The returned dict is stored in history as: {"speaker": "seller", "offer": <this>}
        It must include "action_type" so terminal state detection works.
        """
        ...

    # ------------------------------------------------------------------ #
    # Reward                                                               #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def calculate_reward(
        self,
        final_terms: Dict[str, Any],
        agent_profile: Dict[str, Any],
        outcome: str,
        num_rounds: int,
        max_rounds: int
    ) -> float:
        """
        Calculate MCTS reward for a terminal state.

        Args:
            final_terms: Last agent offer dict (from history)
            agent_profile: Agent's profile
            outcome: "accepted", "rejected", or "max_rounds"
            num_rounds: Number of rounds taken
            max_rounds: Maximum allowed rounds

        Returns:
            Reward value (typically in [-2.0, 1.0] range for PUCT compatibility)
        """
        ...

    # ------------------------------------------------------------------ #
    # Optional overrides                                                   #
    # ------------------------------------------------------------------ #

    def get_terminal_actions(self) -> Dict[str, Any]:
        """
        Return domain-specific terminal action types and logprob tokens.

        The default below is what both bundled domains (CB and A2A) actually
        use — neither overrides it. Do not change the token list or its order
        without re-running the evaluation: it drives the logprob-based
        acceptance estimate.

        Returns:
            {
                "accept_actions": list of action_type strings meaning "deal accepted",
                "reject_actions": list of action_type strings meaning "deal rejected",
                "action_tokens":  list of token strings for logprob-based acceptance
                                  estimation (first = accept, second = reject, rest = neutral)
            }
        """
        return {
            "accept_actions": ["accept"],
            "reject_actions": ["reject"],
            "action_tokens": [" accept", " reject", " ask"],
        }

    def parse_strategy_from_response(
        self,
        response: str,
        available_strategies: List[Any]
    ) -> Optional[Any]:
        """
        Parse strategy selection from LLM response.

        Default implementation looks for [strategy_value] in brackets.
        Override for domains with non-snake_case strategy values.
        """
        import re
        # Try exact match with brackets
        match = re.search(r'\[([^\]]+)\]', response.strip())
        if match:
            found = match.group(1).strip()
            for strategy in available_strategies:
                if strategy.value.lower() == found.lower():
                    return strategy
        # Fallback: substring match
        response_lower = response.lower()
        for strategy in available_strategies:
            if strategy.value.lower() in response_lower:
                return strategy
        return None
