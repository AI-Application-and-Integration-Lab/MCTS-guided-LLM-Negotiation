"""
Negotiation strategies for the A2A consumer buyer agent.

Reuses the CB dialogue act taxonomy — these strategies are generic enough
to apply to Vehicle, Electronics, and Real Estate negotiations.
"""

from enum import Enum
from typing import Dict, Any, Optional


class A2ABuyerStrategy(Enum):
    GREETINGS           = "Greetings"
    ASK_QUESTION        = "Ask a question"
    ANSWER_QUESTION     = "Answer a question"
    PROPOSE_FIRST_PRICE = "Propose the first price"
    PROPOSE_COUNTER     = "Propose a counter price"
    USE_COMPARATIVES    = "Use comparatives"
    CONFIRM_INFO        = "Confirm information"
    AFFIRM_CONFIRM      = "Affirm confirmation"
    DENY_CONFIRM        = "Deny confirmation"
    AGREE               = "Agree with the proposal"
    DISAGREE            = "Disagree with a proposal"


STRATEGY_CONFIGS: Dict[A2ABuyerStrategy, Dict[str, Any]] = {
    A2ABuyerStrategy.GREETINGS: {
        "description": "Say hello or chat casually to open the conversation.",
        "typical_prior": 0.4,
    },
    A2ABuyerStrategy.ASK_QUESTION: {
        "description": "Ask about the product's features, condition, warranty, or history.",
        "typical_prior": 0.6,
    },
    A2ABuyerStrategy.ANSWER_QUESTION: {
        "description": "Provide information about yourself or clarify your position when asked.",
        "typical_prior": 0.45,
    },
    A2ABuyerStrategy.PROPOSE_FIRST_PRICE: {
        "description": "Initiate a price offer or suggest a price range for the product.",
        "typical_prior": 0.85,
    },
    A2ABuyerStrategy.PROPOSE_COUNTER: {
        "description": "Propose a new price in response to the seller's counter-offer.",
        "typical_prior": 0.9,
    },
    A2ABuyerStrategy.USE_COMPARATIVES: {
        "description": "Reference comparable products or market prices to justify a lower offer.",
        "typical_prior": 0.6,
    },
    A2ABuyerStrategy.CONFIRM_INFO: {
        "description": "Ask the seller to confirm specific details about the product or terms.",
        "typical_prior": 0.5,
    },
    A2ABuyerStrategy.AFFIRM_CONFIRM: {
        "description": "Affirmatively confirm the seller's statement or question.",
        "typical_prior": 0.45,
    },
    A2ABuyerStrategy.DENY_CONFIRM: {
        "description": "Deny or push back on the seller's statement.",
        "typical_prior": 0.5,
    },
    A2ABuyerStrategy.AGREE: {
        "description": "Agree with the seller's price and accept the deal.",
        "typical_prior": 0.7,
    },
    A2ABuyerStrategy.DISAGREE: {
        "description": "Disagree with the seller's price and push for a lower one.",
        "typical_prior": 0.8,
    },
}


def get_default_a2a_strategies():
    return list(A2ABuyerStrategy)


def get_a2a_strategy_prior(
    strategy: A2ABuyerStrategy,
    context: Optional[Dict[str, Any]] = None,
) -> float:
    return STRATEGY_CONFIGS[strategy]["typical_prior"]
