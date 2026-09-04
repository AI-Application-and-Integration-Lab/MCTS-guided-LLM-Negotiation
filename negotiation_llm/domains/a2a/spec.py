"""
Agent2Agent consumer-negotiation evaluation adapter.

Ported verbatim from experiments/a2a/run_negotiation_a2a_mcts.py. The places
where A2A differs from CB are deliberate and load-bearing:

  * prices are formatted with a thousands separator, and that string reaches
    the LLM inside the analyzer prompt;
  * the analyzer is asked for a "Seller Cost Floor" and prefers it over the
    target price when updating the domain's seller model;
  * the analyzer's predicted_action / predicted_counter are fed back into the
    simulated seller, which create_a2a_seller_prompt actually consumes;
  * the full analysis dict is stored per turn, and the final analyzer state is
    stored per episode.

Do not "unify" any of this with CB.
"""

import re
from typing import Any, Dict, List, Optional

from ..base import DomainSpec
from .data_loader import load_a2a_scenarios, generate_seller_persona
from .domain import A2ANegotiationDomain
from .prompts import create_a2a_seller_prompt
from .strategies import A2ABuyerStrategy


class A2ASpec(DomainSpec):
    name = "a2a"
    domain_cls = A2ANegotiationDomain
    strategy_enum = A2ABuyerStrategy

    # ------------------------------------------------------------------ #
    # CLI                                                                  #
    # ------------------------------------------------------------------ #

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--buyer-target-ratio",
            type=float,
            default=0.2,
            help="Buyer target as a fraction of the retail-wholesale spread above wholesale",
        )

    def cli_defaults(self, method_name: str) -> Dict[str, Any]:
        return {
            "data_file": "Agent2Agent-Negotiation-in-Consumer-Setting-Dataset/products.json",
            "num_simulations": 5,
            # No A2A-specific checkpoints exist. Evaluation reuses the
            # CB-trained adapters as cross-domain transfer; this matches the
            # defaults every reported A2A run used. See README.
            "actor_adapter": "./checkpoints/actor",
            "analyzer_adapter": "./checkpoints/analyzer",
        }

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #

    def load_scenarios(self, args) -> List[Dict[str, Any]]:
        return load_a2a_scenarios(
            json_path=args.data_file,
            start_index=args.start_index,
            end_index=args.end_index,
            shuffle=args.shuffle,
            seed=args.seed,
            buyer_target_ratio=getattr(args, "buyer_target_ratio", 0.2),
        )

    def build_item_config(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        return {**scenario["item"], "buyer_target": scenario["buyer"].get("target_price")}

    def describe_item(self, item_config: Dict[str, Any]) -> str:
        return f"{item_config['name']} ({self.money(item_config.get('listing_price'))})"

    def seller_components(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        return generate_seller_persona(scenario, return_components=True)

    # ------------------------------------------------------------------ #
    # Formatting                                                           #
    # ------------------------------------------------------------------ #

    def money(self, value: Optional[float]) -> str:
        # NOTE: thousands separator. CB has none. Reaches the LLM inside the
        # analyzer prompt, so this is semantic, not cosmetic.
        return f"${value:,.0f}" if value is not None else "N/A"

    # ------------------------------------------------------------------ #
    # Analyzer                                                             #
    # ------------------------------------------------------------------ #

    def create_analyzer_prompt(
        self,
        negotiation_history: List[Dict[str, Any]],
        buyer_offer: Dict[str, Any],
        buyer_dialogue: str,
        seller_response: str,
    ) -> str:
        if not negotiation_history:
            history_text = "Negotiation Start\n"
        else:
            lines = ["Negotiation History:"]
            for turn in negotiation_history:
                speaker = turn.get("speaker", "unknown")
                dialogue = turn.get("dialogue", "")
                offer = turn.get("offer", {}) or {}
                price = offer.get("price")
                label = "Buyer" if speaker == "buyer" else "Seller"
                if price is not None:
                    lines.append(f"  {label}: {dialogue} [${price:,.0f}]")
                else:
                    lines.append(f"  {label}: {dialogue}")
            history_text = "\n".join(lines) + "\n"

        buyer_price = buyer_offer.get("price")
        buyer_price_str = f"${buyer_price:,.0f}" if buyer_price is not None else "N/A"

        return f"""Given the following price negotiation context, predict the seller's target price and next action.

{history_text}
Buyer's Current Offer: {buyer_price_str}
Buyer: {buyer_dialogue}

Seller Response: {seller_response}

Task: Based on the seller's response and the negotiation context, predict the seller's target price and likely next action.

Prediction:
- Seller Target Price:
- Seller Cost Floor:
- Seller Action:
- Seller Counter Price:
- Big Five Personality:
- Decision Making Style:"""

    def parse_analyzer_output(self, text: str, listing_price: float) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        match = re.search(r'Seller Target Price:\s*\$?([\d,]+(?:\.\d+)?)', text)
        if match:
            try:
                result["predicted_target"] = float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        match = re.search(r'Seller Cost Floor:\s*\$?([\d,]+(?:\.\d+)?)', text)
        if match:
            try:
                result["predicted_cost_floor"] = float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        match = re.search(r'Seller Action:\s*(\w+)', text)
        if match:
            result["predicted_action"] = match.group(1).lower()

        match = re.search(r'Seller Counter Price:\s*\$?([\d,]+(?:\.\d+)?)', text)
        if match:
            try:
                result["predicted_counter"] = float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        match = re.search(r'Big Five Personality:\s*(.+)', text)
        if match:
            result["predicted_big_five"] = match.group(1).strip()

        match = re.search(r'Decision Making Style:\s*(.+)', text)
        if match:
            result["predicted_decision_style"] = match.group(1).strip()

        return result

    def apply_analysis(self, domain, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """A2A: cost floor wins over target; predictions feed the seller."""
        cost_floor = analysis.get("predicted_cost_floor")
        if cost_floor is not None:
            domain.seller_target = cost_floor
        elif analysis.get("predicted_target") is not None:
            domain.seller_target = analysis["predicted_target"]

        predicted_big_five = analysis.get("predicted_big_five")
        predicted_style = analysis.get("predicted_decision_style")
        if predicted_big_five or predicted_style:
            persona_parts = []
            if predicted_big_five:
                persona_parts.append(f"a seller who is {predicted_big_five}")
            if predicted_style:
                persona_parts.append(f"In decision-making, this seller is {predicted_style}")
            domain.seller_persona = ". ".join(persona_parts) + "."

        # Feed the analyzer's next-move predictions into the simulated seller
        # so MCTS rollouts model a more realistic opponent. CB does not do this.
        domain.predicted_action = analysis.get("predicted_action")
        domain.predicted_counter = analysis.get("predicted_counter")

        # A2A stores the full analysis, CB stores a 3-key subset.
        return analysis


    # ------------------------------------------------------------------ #
    # Ground-truth opponent                                                #
    # ------------------------------------------------------------------ #

    def create_ground_truth_seller_prompt(
        self,
        scenario: Dict[str, Any],
        item_config: Dict[str, Any],
        history: List[Dict[str, Any]],
        buyer_offer: Optional[Dict[str, Any]],
        buyer_dialogue: Optional[str],
        seller_persona: Optional[str] = None,
        seller_target: Optional[float] = None,
    ) -> str:
        # Deliberately passes no predicted_* args: the real seller must not see
        # the analyzer's guesses, only the rollout seller does.
        return create_a2a_seller_prompt(
            seller_persona=seller_persona,
            item_config=item_config,
            history=history,
            current_buyer_offer=buyer_offer,
            current_buyer_dialogue=buyer_dialogue,
            seller_target=seller_target,
        )

    # ------------------------------------------------------------------ #
    # Final accounting                                                     #
    # ------------------------------------------------------------------ #

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
        buyer_profile = scenario["buyer"]
        listing_price = scenario["item"]["listing_price"]

        final_offer: Dict[str, Any] = {}
        final_price = None

        if outcome == "accepted":
            last_turn = history[-1] if history else {}
            last_speaker = last_turn.get("speaker")

            if last_speaker == "buyer":
                final_offer = last_turn.get("offer") or {}
                final_price = final_offer.get("price")
            elif last_speaker == "seller":
                buyer_turns = [h for h in history if h.get("speaker") == "buyer"]
                if buyer_turns:
                    final_offer = buyer_turns[-1].get("offer") or {}
                    final_price = final_offer.get("price")

            if final_price is None:
                final_price = listing_price
                final_offer = {"price": final_price, "action_type": "accept"}

            if buyer_profile.get("target_price") is not None and final_price > listing_price:
                if verbose:
                    print(f"  [Warning] Final price ${final_price:,.0f} exceeds listing "
                          f"${listing_price:,.0f} — clamping")
                final_price = listing_price
                final_offer["price"] = final_price
            # A2A has no below-target diagnostic; CB does.

        num_rounds = len([t for t in history if t.get("speaker") == "buyer"])

        reward = domain.calculate_reward(
            final_terms=final_offer,
            agent_profile=buyer_profile,
            outcome=outcome,
            num_rounds=num_rounds,
            max_rounds=max_rounds,
        )

        discount_pct = 0.0
        if outcome == "accepted" and final_price is not None and listing_price > 0:
            discount_pct = (listing_price - final_price) / listing_price * 100

        sale_to_list = (final_price / listing_price) if (
            final_price is not None and listing_price) else None

        return {
            "outcome": outcome,
            "final_offer": final_offer,
            "final_price": final_price,
            "listing_price": listing_price,
            "discount_pct": discount_pct,
            "sale_to_list": sale_to_list,
            "num_rounds": num_rounds,
            "reward": reward,
        }

    def result_extras(self, domain) -> Dict[str, Any]:
        return {
            "final_analyzer_state": {
                "seller_target": domain.seller_target,
                "seller_persona": domain.seller_persona,
            }
        }
