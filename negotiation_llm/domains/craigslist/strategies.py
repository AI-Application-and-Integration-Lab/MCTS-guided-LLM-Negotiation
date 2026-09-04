"""
Negotiation strategies for the Craigslist Bargains buyer agent.

Uses the 11 CB dialogue act types as MCTS strategies, matching the
annotation schema from the original Craigslist Bargains dataset paper.

Reference:
  He et al., "Decoupling Strategy and Generation in Negotiation Dialogues" (2018)
  https://arxiv.org/abs/1808.09637
"""

from enum import Enum
from typing import Dict, Any, Optional


class CraigslistBuyerStrategy(Enum):
    """CB dialogue act types used as buyer MCTS strategies."""
    GREETINGS           = "Greetings"                 # Say hello or chat casually
    ASK_QUESTION        = "Ask a question"            # Ask about product, year, price, usage
    ANSWER_QUESTION     = "Answer a question"         # Provide information if seller asks
    PROPOSE_FIRST_PRICE = "Propose the first price"   # Initiate a price or price range
    PROPOSE_COUNTER     = "Propose a counter price"   # Propose a new price in response to seller
    USE_COMPARATIVES    = "Use comparatives"          # Vague price reference via comparatives
    CONFIRM_INFO        = "Confirm information"       # Ask about info that needs confirmation
    AFFIRM_CONFIRM      = "Affirm confirmation"       # Affirmative response to a confirmation
    DENY_CONFIRM        = "Deny confirmation"         # Negative response to a confirmation
    AGREE               = "Agree with the proposal"   # Accept the seller's price
    DISAGREE            = "Disagree with a proposal"  # Reject/push back on seller's price


# Strategy configurations: prior probability and description
STRATEGY_CONFIGS: Dict[CraigslistBuyerStrategy, Dict[str, Any]] = {
    CraigslistBuyerStrategy.GREETINGS: {
        "description": "Say hello or chat randomly to build rapport.",
        "typical_prior": 0.4,
    },
    CraigslistBuyerStrategy.ASK_QUESTION: {
        "description": "Ask about the product's condition, year, usage, or any other detail.",
        "typical_prior": 0.6,
    },
    CraigslistBuyerStrategy.ANSWER_QUESTION: {
        "description": "Provide information about yourself or clarify your position when asked.",
        "typical_prior": 0.45,
    },
    CraigslistBuyerStrategy.PROPOSE_FIRST_PRICE: {
        "description": "Initiate a price offer or suggest a price range for the product.",
        "typical_prior": 0.85,
    },
    CraigslistBuyerStrategy.PROPOSE_COUNTER: {
        "description": "Propose a new price or price range in response to the seller's offer.",
        "typical_prior": 0.9,
    },
    CraigslistBuyerStrategy.USE_COMPARATIVES: {
        "description": "Use comparatives to suggest a vague price (e.g., 'a bit less than that').",
        "typical_prior": 0.6,
    },
    CraigslistBuyerStrategy.CONFIRM_INFO: {
        "description": "Ask the seller to confirm specific information about the item or price.",
        "typical_prior": 0.5,
    },
    CraigslistBuyerStrategy.AFFIRM_CONFIRM: {
        "description": "Affirmatively confirm the seller's statement or question.",
        "typical_prior": 0.45,
    },
    CraigslistBuyerStrategy.DENY_CONFIRM: {
        "description": "Deny or disagree with the seller's statement or question.",
        "typical_prior": 0.5,
    },
    CraigslistBuyerStrategy.AGREE: {
        "description": "Agree with the seller's price proposal and accept the deal.",
        "typical_prior": 0.7,
    },
    CraigslistBuyerStrategy.DISAGREE: {
        "description": "Disagree with the seller's price and push for a lower one.",
        "typical_prior": 0.8,
    },
}


def get_default_cb_strategies():
    """Return the full set of CB buyer strategies for MCTS exploration."""
    return list(CraigslistBuyerStrategy)


def get_cb_strategy_prior(
    strategy: CraigslistBuyerStrategy,
    context: Optional[Dict[str, Any]] = None
) -> float:
    """Return prior probability for a CB buyer strategy."""
    return STRATEGY_CONFIGS[strategy]["typical_prior"]
