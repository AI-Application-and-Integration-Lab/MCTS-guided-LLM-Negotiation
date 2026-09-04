"""
CraigslistBargain evaluation adapter.

Every prompt string and numeric policy here is carried over verbatim from
experiments/cb/run_negotiation_cb_mcts.py so that collapsing the runners does
not perturb published numbers. Where CB differs from A2A the difference is
kept, not unified — see the README.
"""

import re
from typing import Any, Dict, List, Optional

from ..base import DomainSpec
from .data_loader import load_cb_scenarios, generate_seller_persona
from .domain import CraigslistNegotiationDomain
from .prompts import create_cb_seller_prompt
from .strategies import CraigslistBuyerStrategy


class CBSpec(DomainSpec):
    name = "cb"
    domain_cls = CraigslistNegotiationDomain
    strategy_enum = CraigslistBuyerStrategy

    # ------------------------------------------------------------------ #
    # CLI                                                                  #
    # ------------------------------------------------------------------ #

    def cli_defaults(self, method_name: str) -> Dict[str, Any]:
        return {
            "data_file": "CB/test.csv",
            "num_simulations": 6,
            "actor_adapter": "./checkpoints/actor",
            "analyzer_adapter": "./checkpoints/analyzer",
        }

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #

    def load_scenarios(self, args) -> List[Dict[str, Any]]:
        return load_cb_scenarios(
            csv_path=args.data_file,
            start_index=args.start_index,
            end_index=args.end_index,
            shuffle=args.shuffle,
            seed=args.seed,
        )

    def build_item_config(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        return {**scenario["item"], "buyer_target": scenario["buyer"].get("target_price")}

    def describe_item(self, item_config: Dict[str, Any]) -> str:
        return f"{item_config['title']} ({self.money(item_config.get('listing_price'))})"

    def seller_components(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        return generate_seller_persona(scenario, return_components=True)

    # ------------------------------------------------------------------ #
    # Formatting                                                           #
    # ------------------------------------------------------------------ #

    def money(self, value: Optional[float]) -> str:
        # NOTE: no thousands separator. A2A uses one. This string reaches the
        # LLM inside the analyzer prompt, so the difference is semantic.
        return f"${value:.0f}" if value is not None else "N/A"

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
                speaker = turn.get('speaker', 'unknown')
                dialogue = turn.get('dialogue', '')
                offer = turn.get('offer', {}) or {}
                price = offer.get('price')
                label = "Buyer" if speaker == 'buyer' else "Seller"
                if price is not None:
                    lines.append(f"  {label}: {dialogue} [${price:.0f}]")
                else:
                    lines.append(f"  {label}: {dialogue}")
            history_text = "\n".join(lines) + "\n"

        buyer_price = buyer_offer.get("price")
        buyer_price_str = f"${buyer_price:.0f}" if buyer_price is not None else "N/A"

        return f"""Given the following price negotiation context, predict the seller's target price and next action.

{history_text}
Buyer's Current Offer: {buyer_price_str}
Buyer: {buyer_dialogue}

Seller Response: {seller_response}

Task: Based on the seller's response and the negotiation context, predict the seller's target price and likely next action.

Prediction:
- Seller Target Price:
- Seller Action:
- Seller Counter Price:
- Big Five Personality:
- Decision Making Style:"""

    def parse_analyzer_output(self, text: str, listing_price: float) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        match = re.search(r'Seller Target Price:\s*\$?([\d,]+(?:\.\d+)?)', text)
        if match:
            try:
                result["predicted_target"] = float(match.group(1).replace(',', ''))
            except ValueError:
                pass

        match = re.search(r'Seller Action:\s*(\w+)', text)
        if match:
            result["predicted_action"] = match.group(1).lower()

        match = re.search(r'Seller Counter Price:\s*\$?([\d,]+(?:\.\d+)?)', text)
        if match:
            try:
                result["predicted_counter"] = float(match.group(1).replace(',', ''))
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
        """CB: seller_target from predicted_target; store a 3-key subset."""
        if analysis.get("predicted_target") is not None:
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

        # CB deliberately does NOT feed predicted_action / predicted_counter
        # back into the simulated seller; A2A does.
        return {
            "predicted_target": analysis.get("predicted_target"),
            "predicted_big_five": predicted_big_five,
            "predicted_decision_style": predicted_style,
        }

    def analyzer_domain_state(self, domain) -> Dict[str, Any]:
        """turn_record["analyzer_domain_state"] payload."""
        return {
            "seller_target": domain.seller_target,
            "seller_persona": domain.seller_persona,
        }

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
        # NOTE: create_cb_seller_prompt accepts seller_target but never reads
        # it — the CB simulated seller has no reservation price. That is a
        # real bug, but fixing it would change every published CB number.
        #
        return create_cb_seller_prompt(
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
            last_speaker = last_turn.get('speaker')

            if last_speaker == 'buyer':
                final_offer = last_turn.get('offer') or {}
                final_price = final_offer.get('price')
            elif last_speaker == 'seller':
                buyer_turns = [h for h in history if h.get('speaker') == 'buyer']
                if buyer_turns:
                    final_offer = buyer_turns[-1].get('offer') or {}
                    final_price = final_offer.get('price')

            if final_price is None:
                final_price = listing_price
                final_offer = {"price": final_price, "action_type": "accept"}

            if buyer_profile.get('target_price') is not None and final_price > listing_price:
                if verbose:
                    print(f"  [Warning] Final price ${final_price:.0f} exceeds listing "
                          f"${listing_price:.0f} — clamping")
                final_price = listing_price
                final_offer['price'] = final_price
            # CB-only diagnostic: warns but does NOT clamp.
            if (buyer_profile.get('target_price') is not None
                    and final_price < buyer_profile['target_price'] * 0.5):
                if verbose:
                    print(f"  [Warning] Final price ${final_price:.0f} is suspiciously "
                          f"below target — possible parse error")

        num_rounds = len([t for t in history if t.get('speaker') == 'buyer'])

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
