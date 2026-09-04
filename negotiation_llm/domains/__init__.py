"""
Negotiation domains.

Each domain contributes two things:

  * a `NegotiationDomain` (search side) — strategies, prompts, parsing, reward,
    consumed by MCTSNegotiator;
  * a `DomainSpec` (evaluation side) — scenario loading, the analyzer, the
    ground-truth opponent prompt, and final accounting.

Adding a dataset means implementing both and adding one line to
DOMAIN_REGISTRY below. See the README.
"""

from typing import Dict

from .base import DomainSpec
from .a2a.spec import A2ASpec
from .craigslist.spec import CBSpec

DOMAIN_REGISTRY: Dict[str, DomainSpec] = {
    "cb": CBSpec(),
    "a2a": A2ASpec(),
}


def get_domain(name: str) -> DomainSpec:
    """Look up a DomainSpec by its --domain key."""
    try:
        return DOMAIN_REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"unknown --domain {name!r}; choose from {sorted(DOMAIN_REGISTRY)}"
        ) from None


__all__ = ["DomainSpec", "CBSpec", "A2ASpec", "DOMAIN_REGISTRY", "get_domain"]
