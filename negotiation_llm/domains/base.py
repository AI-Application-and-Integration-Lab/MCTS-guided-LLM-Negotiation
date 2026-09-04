"""
DomainSpec — the evaluation-side adapter for a negotiation dataset.

`NegotiationDomain` (see `negotiation_llm.mcts.domain`) is the *search* side
contract: strategies, prompts, parsing and reward, consumed by MCTSNegotiator.
Its instances are per-scenario and mutable — the analyzer writes its estimate
of the opponent onto them mid-episode.

`DomainSpec` is the *evaluation harness* side: loading scenarios, building the
per-scenario item config, running the analyzer, prompting the ground-truth
opponent, and doing the final accounting. Its instances are per-run and
stateless.

The two are kept apart deliberately. Bolting scenario loaders and CLI
defaults onto NegotiationDomain would force every future search domain to
implement harness concerns, and NegotiationDomain cannot be instantiated
without an item_config, so it cannot own run-scoped behaviour.

Almost every genuine CB/A2A behavioural difference lives behind one of the
methods below — `money`, `create_analyzer_prompt`, `apply_analysis`,
`finalize`. That is on purpose: the differences are semantic, not cosmetic,
and the single biggest silent-failure risk in this codebase is someone
deciding the two domains are "the same" and merging them. See
the README.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class DomainSpec(ABC):
    """Per-run, stateless adapter binding a dataset to the evaluation loop."""

    #: registry key, matches --domain
    name: str
    #: the NegotiationDomain subclass used for search
    domain_cls: type
    #: the buyer strategy enum for this domain
    strategy_enum: type

    # ------------------------------------------------------------------ #
    # CLI                                                                  #
    # ------------------------------------------------------------------ #

    def add_arguments(self, parser) -> None:
        """Register domain-only CLI flags. Default: none."""

    @abstractmethod
    def cli_defaults(self, method_name: str) -> Dict[str, Any]:
        """Per-domain (and optionally per-method) argparse defaults."""

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_scenarios(self, args) -> List[Dict[str, Any]]:
        """Load and slice scenarios according to the parsed CLI args."""

    @abstractmethod
    def build_item_config(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the per-scenario item config handed to NegotiationDomain.

        Must embed the buyer's target so prompts and reward can reach it
        without the buyer profile being passed as the opponent profile.
        """

    @abstractmethod
    def describe_item(self, item_config: Dict[str, Any]) -> str:
        """One-line human-readable item label for --verbose output."""

    # ------------------------------------------------------------------ #
    # Formatting                                                           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def money(self, value: Optional[float]) -> str:
        """
        Format a price for display AND for prompt bodies.

        CB uses '$120' and A2A uses '$1,200' — this reaches the LLM inside
        the analyzer prompt, so it is semantic, not cosmetic. Do not unify.
        """

    # ------------------------------------------------------------------ #
    # Analyzer (used by the mcts method)                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def create_analyzer_prompt(
        self,
        negotiation_history: List[Dict[str, Any]],
        buyer_offer: Dict[str, Any],
        buyer_dialogue: str,
        seller_response: str,
    ) -> str:
        """Prompt asking the analyzer LoRA to characterise the seller."""

    @abstractmethod
    def parse_analyzer_output(self, text: str, listing_price: float) -> Dict[str, Any]:
        """Parse the analyzer's free-text prediction into a dict."""

    @abstractmethod
    def apply_analysis(self, domain, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write the analyzer's estimate onto the live NegotiationDomain and
        return the payload to store as turn_record["analyzer_output"].

        CB sets seller_target from predicted_target and stores a 3-key
        subset. A2A prefers predicted_cost_floor, additionally feeds
        predicted_action / predicted_counter back into the simulated seller,
        and stores the full analysis.
        """

    def analyzer_domain_state(self, domain) -> Dict[str, Any]:
        """Payload stored as turn_record["analyzer_domain_state"]."""
        return {
            "seller_target": getattr(domain, "seller_target", None),
            "seller_persona": getattr(domain, "seller_persona", None),
        }

    @abstractmethod
    def seller_components(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Ground-truth seller persona + trait components for the result record."""

    # ------------------------------------------------------------------ #
    # Ground-truth opponent                                                #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def create_ground_truth_seller_prompt(
        self,
        scenario: Dict[str, Any],
        item_config: Dict[str, Any],
        history: List[Dict[str, Any]],
        buyer_offer: Optional[Dict[str, Any]],
        buyer_dialogue: Optional[str],
    ) -> str:
        """
        Prompt the *real* simulated seller with ground-truth persona/target.

        The analyzer-driven methods need this because their NegotiationDomain
        is holding the analyzer's estimate and serves as the rollout opponent,
        so it cannot also stand in for the true seller.
        """

    # ------------------------------------------------------------------ #
    # Final accounting — where CB and A2A genuinely diverge                #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def finalize(
        self,
        scenario: Dict[str, Any],
        item_config: Dict[str, Any],
        domain,
        history: List[Dict[str, Any]],
        outcome: str,
        max_rounds: int,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract the final price, apply the domain's clamp policy, count
        rounds, and compute reward / discount / sale-to-list.

        Returns the core per-episode result fields.
        """

    def result_extras(self, domain) -> Dict[str, Any]:
        """Domain-specific fields appended to each episode result."""
        return {}
