"""
Strategy-level Monte Carlo Tree Search for negotiation dialogue.

The tree searches over high-level negotiation strategies (dialogue acts); an
LLM realizes the selected strategy into a concrete utterance and offer. All
dataset-specific behaviour — strategies, prompts, parsing, reward — lives
behind the `NegotiationDomain` interface, so the search core stays domain
agnostic.
"""

from .node import MCTSNode
from .negotiator import MCTSNegotiator
from .domain import NegotiationDomain

__all__ = [
    'MCTSNode',
    'MCTSNegotiator',
    'NegotiationDomain',
]
