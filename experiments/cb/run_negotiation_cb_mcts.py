"""
Run MCTS negotiation on the Craigslist Bargains (CB) dataset.

MCTS acts as the BUYER and tries to purchase items at the lowest price.
A LLM-simulated seller responds to the buyer's offers.

At each turn:
1. MCTS selects a buyer strategy (from 11 CB dialogue acts)
2. LLM generates buyer's message based on the selected strategy
3. LLM-simulated seller responds
4. Negotiation ends when: seller accepts, seller rejects, or max_rounds reached

Metrics reported:
  - Acceptance rate
  - Average discount achieved (% below listing price)
  - Average discount vs buyer's target price
  - Average rounds
"""

import argparse
import json
import re
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm


from negotiation_llm.models import MockModel, HuggingFaceModel, VLLMModel, OpenAIModel, BaseModel
from negotiation_llm.mcts import MCTSNegotiator
from negotiation_llm.config import MCTS_CONFIG, MODEL_CONFIG
from negotiation_llm.domains.craigslist import CraigslistNegotiationDomain, load_cb_scenarios, generate_seller_persona
from negotiation_llm.domains.craigslist.prompts import create_cb_seller_prompt


class TokenCountingWrapper(BaseModel):
    """
    Wraps any BaseModel to count prompt and completion tokens for the buyer model.

    Token counts are estimated from the tokenizer if available (vLLM/HuggingFace),
    otherwise fall back to whitespace-split word count as a rough proxy.
    """

    def __init__(self, model: BaseModel):
        self._model = model
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def _count(self, text: str) -> int:
        tok = getattr(self._model, "tokenizer", None)
        if tok is not None:
            try:
                return len(tok.encode(text))
            except Exception:
                pass
        return len(text.split())

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompt_tokens += self._count(prompt)
        response = self._model.generate(prompt, **kwargs)
        self.completion_tokens += self._count(response)
        return response

    def generate_batch(self, prompt: str, num_samples: int = 3, **kwargs) -> list:
        self.prompt_tokens += self._count(prompt) * num_samples
        responses = self._model.generate_batch(prompt, num_samples=num_samples, **kwargs)
        for r in responses:
            self.completion_tokens += self._count(r)
        return responses

    # Delegate everything else to the wrapped model
    def __getattr__(self, name):
        return getattr(self._model, name)

    def cleanup(self):
        self._model.cleanup()

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0


def create_cb_analyzer_prompt(
    negotiation_history: List[Dict[str, Any]],
    buyer_offer: Dict[str, Any],
    buyer_dialogue: str,
    seller_response: str,
) -> str:
    """Create analyzer prompt to predict seller's target price and next action."""
    # Format history (all turns before the latest buyer-seller exchange)
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


def parse_cb_analyzer_output(text: str, listing_price: float) -> Dict[str, Any]:
    """Parse analyzer output to extract predicted seller target price, action, and persona traits."""
    import re
    result = {}

    # Extract target price
    match = re.search(r'Seller Target Price:\s*\$?([\d,]+(?:\.\d+)?)', text)
    if match:
        try:
            result["predicted_target"] = float(match.group(1).replace(',', ''))
        except ValueError:
            pass

    # Extract action
    match = re.search(r'Seller Action:\s*(\w+)', text)
    if match:
        result["predicted_action"] = match.group(1).lower()

    # Extract counter price
    match = re.search(r'Seller Counter Price:\s*\$?([\d,]+(?:\.\d+)?)', text)
    if match:
        try:
            result["predicted_counter"] = float(match.group(1).replace(',', ''))
        except ValueError:
            pass

    # Extract Big Five personality
    match = re.search(r'Big Five Personality:\s*(.+)', text)
    if match:
        result["predicted_big_five"] = match.group(1).strip()

    # Extract decision-making style
    match = re.search(r'Decision Making Style:\s*(.+)', text)
    if match:
        result["predicted_decision_style"] = match.group(1).strip()

    return result


def run_cb_negotiation(
    model,
    mcts_agent: MCTSNegotiator,
    scenario: Dict[str, Any],
    max_rounds: int = 10,
    verbose: bool = False,
    seller_model=None,
    use_analyzer: bool = False,
    token_counter: Optional["TokenCountingWrapper"] = None,
) -> Tuple[Dict[str, Any], str, float]:
    """
    Run a single CB negotiation.

    The MCTSNegotiator acts as the buyer; an LLM simulates the seller.

    Args:
        model: LLM model instance (used for MCTS buyer)
        mcts_agent: MCTSNegotiator configured with CraigslistNegotiationDomain
        scenario: CB scenario dict from load_cb_scenarios()
        max_rounds: Maximum negotiation rounds
        verbose: Print turn-by-turn output
        seller_model: Separate model for seller responses (defaults to model if None)
        use_analyzer: Use analyzer adapter to predict seller target price

    Returns:
        Tuple of (result_dict, outcome_str, reward)
    """
    if seller_model is None:
        seller_model = model
    buyer_profile = scenario["buyer"]
    listing_price = scenario["item"]["listing_price"]

    # Embed buyer_target in item_config so prompts and reward can find it
    # without the buyer profile being passed as seller_profile
    item_config = {**scenario["item"], "buyer_target": buyer_profile.get("target_price")}

    # Ground-truth seller info — used ONLY for the real seller simulation.
    # The MCTS (buyer) must not receive these values; it learns about the
    # seller incrementally through the analyzer.
    _seller_components = generate_seller_persona(scenario, return_components=True)
    real_seller_persona = _seller_components["persona"]
    real_seller_target = scenario["seller"].get("target_price")

    # Update domain's item_config for this scenario (MCTSNegotiator uses self.domain_config)
    mcts_agent.domain_config = item_config
    mcts_agent.domain = CraigslistNegotiationDomain(
        item_config=item_config,
        # Start with no knowledge of the seller — the analyzer will fill
        # seller_persona and seller_target in after the first exchange.
    )

    negotiation_history = []
    turn_records = []
    negotiation_outcome = "max_rounds_reached"
    scenario_start = time.perf_counter()

    if verbose:
        print(f"\n{'='*70}")
        print(f"CB SCENARIO: {item_config['title']} (${listing_price:.0f})")
        print(f"Buyer target: ${buyer_profile.get('target_price', 'N/A')}")
        print(f"Category: {item_config['category']}")
        print(f"{'='*70}")

    for round_num in range(max_rounds):
        turn_number = round_num + 1

        if verbose:
            print(f"\n--- Turn {turn_number} ---")

        # Merge buyer goals with the analyzer's current estimate of the seller.
        # Rebuilt every turn so MCTS picks up the latest analyzer update.
        # Only predicted (not ground-truth) seller info is included here.
        mcts_profile = {
            **buyer_profile,
            "analyzed_seller_target": mcts_agent.domain.seller_target,
            "analyzed_seller_persona": mcts_agent.domain.seller_persona,
        }

        _t0 = time.perf_counter()
        buyer_offer, buyer_dialogue, root = mcts_agent.select_action(
            agent_profile=mcts_profile,
            negotiation_history=negotiation_history,
            max_negotiation_rounds=max_rounds,
        )
        turn_time = time.perf_counter() - _t0

        if verbose:
            price_str = f"${buyer_offer.get('price'):.0f}" if buyer_offer.get('price') is not None else "N/A"
            print(f"Buyer: {buyer_dialogue}  [{buyer_offer.get('action_type', '?')} @ {price_str}]")

        # Build turn record
        ranked_strategies = []
        for child in root.children:
            if child.visit_count > 0:
                ranked_strategies.append({
                    "strategy": child.strategy.value if child.strategy else "unknown",
                    "avg_reward": child.avg_reward,
                    "visit_count": child.visit_count,
                })
        ranked_strategies.sort(key=lambda x: x["avg_reward"], reverse=True)

        turn_record = {
            "turn": turn_number,
            "strategies_explored": ranked_strategies,
            "selected_strategy": max(ranked_strategies, key=lambda x: (x["visit_count"], x["avg_reward"]))["strategy"] if ranked_strategies else "unknown",
            "buyer_action": {"offer": buyer_offer, "dialogue": buyer_dialogue},
            "turn_time_sec": round(turn_time, 3),
        }

        # Add buyer turn to history
        negotiation_history.append({
            "speaker": "buyer",
            "offer": buyer_offer,
            "dialogue": buyer_dialogue,
        })

        # Only seller can close — buyer "Agree" signals willingness but seller must confirm
        if buyer_offer.get("action_type") == "accept":
            buyer_offer["action_type"] = "propose"

        # Clear MCTS cache
        if hasattr(mcts_agent, 'terminal_cache'):
            mcts_agent.terminal_cache.clear()

        # Generate seller response using ground-truth seller info.
        # This bypasses domain.create_opponent_prompt so that the real seller
        # always uses the true persona/target, independent of what the analyzer
        # has (or hasn't) predicted yet.
        seller_prompt = create_cb_seller_prompt(
            seller_persona=real_seller_persona,
            item_config=item_config,
            history=negotiation_history,
            current_buyer_offer=buyer_offer,
            current_buyer_dialogue=buyer_dialogue,
            seller_target=real_seller_target,
        )

        seller_response = seller_model.generate(seller_prompt)
        parsed_seller = mcts_agent.domain.parse_opponent_response(seller_response)

        seller_offer = mcts_agent.domain.build_opponent_offer(parsed_seller)
        seller_dialogue = parsed_seller.get("dialogue", seller_response)
        seller_action = parsed_seller.get("action_type", "counter")

        if verbose:
            price_str = f"${seller_offer.get('price'):.0f}" if seller_offer.get('price') is not None else "N/A"
            print(f"Seller: {seller_dialogue}  [{seller_action} @ {price_str}]")

        turn_record["seller_response"] = {
            "offer": seller_offer,
            "dialogue": seller_dialogue,
            "action_type": seller_action,
        }
        turn_records.append(turn_record)

        # Add seller turn to history
        negotiation_history.append({
            "speaker": "seller",
            "offer": seller_offer,
            "dialogue": seller_dialogue,
        })

        # Check termination
        if seller_action == "accept" or parsed_seller.get("accepts"):
            negotiation_outcome = "accepted"
            # Pin final price to the buyer's offer the seller just accepted
            buyer_turns = [h for h in negotiation_history if h.get('speaker') == 'buyer']
            if buyer_turns:
                accepted_price = (buyer_turns[-1].get('offer') or {}).get('price')
                if accepted_price is not None:
                    seller_offer['price'] = accepted_price
            if verbose:
                print("  → Seller accepted buyer's offer!")
            break
        elif seller_action == "reject" or parsed_seller.get("rejects"):
            if verbose:
                print("  → Seller rejected offer (negotiation continues)")

        # Run analyzer to refine MCTS domain's seller model for the next turn
        if use_analyzer and len(negotiation_history) >= 2:
            analyzer_prompt = create_cb_analyzer_prompt(
                negotiation_history=negotiation_history[:-2],  # history before this exchange
                buyer_offer=buyer_offer,
                buyer_dialogue=buyer_dialogue,
                seller_response=seller_dialogue,
            )
            analysis_text = model.generate(analyzer_prompt, lora_adapter='analyzer', max_tokens=256, temperature=0.7)
            analysis = parse_cb_analyzer_output(analysis_text, listing_price)

            # Update the domain's seller model so MCTS rollouts use refined estimates
            if analysis.get("predicted_target") is not None:
                mcts_agent.domain.seller_target = analysis["predicted_target"]

            predicted_big_five = analysis.get("predicted_big_five")
            predicted_style = analysis.get("predicted_decision_style")
            if predicted_big_five or predicted_style:
                persona_parts = []
                if predicted_big_five:
                    persona_parts.append(f"a seller who is {predicted_big_five}")
                if predicted_style:
                    persona_parts.append(f"In decision-making, this seller is {predicted_style}")
                mcts_agent.domain.seller_persona = ". ".join(persona_parts) + "."

            turn_record["analyzer_output"] = {
                "predicted_target": analysis.get("predicted_target"),
                "predicted_big_five": predicted_big_five,
                "predicted_decision_style": predicted_style,
            }
            turn_record["analyzer_domain_state"] = {
                "seller_target": mcts_agent.domain.seller_target,
                "seller_persona": mcts_agent.domain.seller_persona,
            }

            if verbose:
                pred_target = analysis.get("predicted_target")
                pred_action = analysis.get("predicted_action", "?")
                pred_counter = analysis.get("predicted_counter")
                print(f"  [Analyzer] Predicted seller target: ${pred_target:.0f}" if pred_target else "  [Analyzer] No target predicted")
                if pred_counter:
                    print(f"  [Analyzer] Predicted counter: ${pred_counter:.0f}, action: {pred_action}")
                if predicted_big_five:
                    print(f"  [Analyzer] Big Five: {predicted_big_five}")
                if predicted_style:
                    print(f"  [Analyzer] Decision Style: {predicted_style}")

    # Extract final agreed price
    final_offer = {}
    final_price = None

    if negotiation_outcome == "accepted":
        last_turn = negotiation_history[-1] if negotiation_history else {}
        last_speaker = last_turn.get('speaker')

        if last_speaker == 'buyer':
            # Buyer accepted → final price is the buyer's pinned offer
            final_offer = last_turn.get('offer') or {}
            final_price = final_offer.get('price')
        elif last_speaker == 'seller':
            # Seller accepted → final price is the buyer's offer seller agreed to
            buyer_turns = [h for h in negotiation_history if h.get('speaker') == 'buyer']
            if buyer_turns:
                final_offer = buyer_turns[-1].get('offer') or {}
                final_price = final_offer.get('price')

        if final_price is None:
            final_price = listing_price
            final_offer = {"price": final_price, "action_type": "accept"}

        # Sanity check
        if buyer_profile.get('target_price') is not None and final_price > listing_price:
            if verbose:
                print(f"  [Warning] Final price ${final_price:.0f} exceeds listing ${listing_price:.0f} — clamping")
            final_price = listing_price
            final_offer['price'] = final_price
        if buyer_profile.get('target_price') is not None and final_price < buyer_profile['target_price'] * 0.5:
            if verbose:
                print(f"  [Warning] Final price ${final_price:.0f} is suspiciously below target — possible parse error")

    num_rounds = len([t for t in negotiation_history if t.get('speaker') == 'buyer'])

    # Calculate reward
    reward = mcts_agent.domain.calculate_reward(
        final_terms=final_offer,
        agent_profile=buyer_profile,
        outcome=negotiation_outcome,
        num_rounds=num_rounds,
        max_rounds=max_rounds,
    )

    # Calculate metrics
    discount_pct = 0.0
    if negotiation_outcome == "accepted" and final_price is not None and listing_price > 0:
        discount_pct = (listing_price - final_price) / listing_price * 100

    if verbose:
        print(f"\n{'='*70}")
        print(f"Outcome: {negotiation_outcome}")
        if negotiation_outcome == "accepted":
            print(f"Final price: ${final_price:.0f} / ${listing_price:.0f} ({discount_pct:.1f}% discount)")
        print(f"Rounds: {num_rounds} / {max_rounds}")
        print(f"Reward: {reward:.4f}")

    elapsed_sec = time.perf_counter() - scenario_start
    buyer_prompt_tokens = token_counter.prompt_tokens if token_counter is not None else None
    buyer_completion_tokens = token_counter.completion_tokens if token_counter is not None else None
    buyer_total_tokens = token_counter.total_tokens if token_counter is not None else None

    result = {
        "scenario_id": scenario.get("scenario_id", 0),
        "buyer_profile": buyer_profile,
        "seller_profile": {
            "target_price": real_seller_target,
            "persona": real_seller_persona,
            "big_five_personality": _seller_components["big_five_personality"],
            "decision_making_style": _seller_components["decision_making_style"],
        },
        "item_config": item_config,
        "turn_records": turn_records,
        "negotiation_history": negotiation_history,
        "outcome": negotiation_outcome,
        "final_offer": final_offer,
        "final_price": final_price,
        "listing_price": listing_price,
        "discount_pct": discount_pct,
        "num_rounds": num_rounds,
        "reward": reward,
        "elapsed_sec": round(elapsed_sec, 3),
        "buyer_prompt_tokens": buyer_prompt_tokens,
        "buyer_completion_tokens": buyer_completion_tokens,
        "buyer_total_tokens": buyer_total_tokens,
        "timestamp": datetime.now().isoformat(),
    }

    return result, negotiation_outcome, reward


def save_results(
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_file: str,
    verbose: bool = False,
):
    """Save aggregated results to JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    accepted = sum(1 for r in results if r['outcome'] == 'accepted')
    rejected = sum(1 for r in results if r['outcome'] == 'rejected')
    max_rounds_count = sum(1 for r in results if r['outcome'] == 'max_rounds_reached')
    avg_reward = sum(r['reward'] for r in results) / total if total else 0.0
    avg_rounds = sum(r['num_rounds'] for r in results) / total if total else 0.0
    avg_discount = (
        sum(r['discount_pct'] for r in results) / total
        if total else 0.0
    )
    avg_elapsed = sum(r.get('elapsed_sec', 0) for r in results) / total if total else 0.0
    total_prompt_tokens = sum(r.get('buyer_prompt_tokens') or 0 for r in results)
    total_completion_tokens = sum(r.get('buyer_completion_tokens') or 0 for r in results)
    total_buyer_tokens = sum(r.get('buyer_total_tokens') or 0 for r in results)

    output = {
        "config": config,
        "summary": {
            "total_scenarios": total,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "max_rounds_count": max_rounds_count,
            "acceptance_rate": accepted / total if total else 0.0,
            "avg_reward": avg_reward,
            "avg_rounds": avg_rounds,
            "avg_discount_pct": avg_discount,
            "avg_elapsed_sec": round(avg_elapsed, 3),
            "total_buyer_prompt_tokens": total_prompt_tokens,
            "total_buyer_completion_tokens": total_completion_tokens,
            "total_buyer_tokens": total_buyer_tokens,
            "avg_buyer_tokens_per_scenario": round(total_buyer_tokens / total, 1) if total else 0,
        },
        "results": results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    if verbose:
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run MCTS buyer negotiation on Craigslist Bargains dataset"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="CB/test.csv",
        help="Path to CB CSV file (train/test/validation)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/cb_mcts",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_CONFIG["model_name"],
        help="Model name or path"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First scenario index to run (0-based)"
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="One-past-last scenario index (default: all)"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum negotiation rounds per scenario"
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=6,
        help="MCTS simulations per turn"
    )
    parser.add_argument(
        "--exploration-weight",
        type=float,
        default=2.5,
        help="PUCT exploration constant"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle scenarios before selecting"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Deprecated no-op: vLLM is already the backend unless --use-mock "
             "is passed. Kept so existing commands still parse."
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock model (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print turn-by-turn output"
    )
    parser.add_argument(
        "--actor-adapter",
        type=str,
        default="./checkpoints/actor",
        help="Path to actor LoRA adapter for MCTS buyer strategy selection"
    )
    parser.add_argument(
        "--analyzer-adapter",
        type=str,
        default="./checkpoints/analyzer",
        help="Path to analyzer LoRA adapter for seller target prediction"
    )
    parser.add_argument(
        "--use-openai-seller",
        action="store_true",
        help="Use OpenAI API model for seller (otherwise seller uses the same model as buyer)"
    )
    parser.add_argument(
        "--seller-model",
        type=str,
        default="gpt-4o",
        help="OpenAI model name for seller (default: gpt-4o, requires --use-openai-seller)"
    )
    parser.add_argument(
        "--seller-api-key",
        type=str,
        default=None,
        help="OpenAI API key for seller model (defaults to OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--seller-base-url",
        type=str,
        default=None,
        help="Custom base URL for seller model API"
    )

    args = parser.parse_args()

    # Load scenarios
    print(f"Loading CB scenarios from: {args.data_file}")
    scenarios = load_cb_scenarios(
        csv_path=args.data_file,
        start_index=args.start_index,
        end_index=args.end_index,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(f"Loaded {len(scenarios)} scenarios")

    if not scenarios:
        print("No valid scenarios found. Check the CSV file.")
        return

    # Initialize buyer model (MCTS)
    use_analyzer = args.analyzer_adapter is not None
    lora_adapter_name = None

    if args.use_mock:
        print("Using MockModel for buyer")
        model = MockModel()
    else:
        model_config = MODEL_CONFIG.copy()
        model_config['model_name'] = args.model_name

        # Enable multi-LoRA if adapters are provided
        if args.actor_adapter or args.analyzer_adapter:
            model_config['enable_lora'] = True
            model_config['max_lora_rank'] = 64
            lora_modules = {}
            # Fail loudly on a bad path. A missing adapter otherwise degrades
            # silently to the base model and still reports itself as the full
            # system — which is exactly what a stale default path once did.
            for _name, _path in (("actor", args.actor_adapter),
                                 ("analyzer", args.analyzer_adapter)):
                if _path and not os.path.isdir(_path):
                    raise SystemExit(
                        f"--{_name}-adapter path does not exist: {_path}\n"
                        f"Pass --{_name}-adapter explicitly, or place the adapter "
                        f"at ./checkpoints/{_name}/."
                    )
            if args.actor_adapter:
                lora_modules['actor'] = args.actor_adapter
                lora_adapter_name = 'actor'
            if args.analyzer_adapter:
                lora_modules['analyzer'] = args.analyzer_adapter
            model_config['lora_modules'] = lora_modules
            print(f"Loading VLLMModel with LoRA: {args.model_name}")
            if args.actor_adapter:
                print(f"  Actor adapter: {args.actor_adapter}")
            if args.analyzer_adapter:
                print(f"  Analyzer adapter: {args.analyzer_adapter}")
        else:
            print(f"Loading VLLMModel: {args.model_name}")

        model = VLLMModel(**model_config)

    # Initialize seller model
    if args.use_openai_seller and not args.use_mock:
        print(f"Loading OpenAI seller model: {args.seller_model}")
        seller_model = OpenAIModel(
            model_name=args.seller_model,
            api_key=args.seller_api_key,
            base_url=args.seller_base_url,
        )
    else:
        seller_model = model  # Same model as buyer

    # Initialize MCTS (domain will be set per-scenario)
    mcts_config = MCTS_CONFIG.copy()
    mcts_config['num_simulations'] = args.num_simulations
    mcts_config['exploration_weight'] = args.exploration_weight

    # Create a placeholder domain for initialization
    first_item = scenarios[0]["item"]
    initial_domain = CraigslistNegotiationDomain(item_config=first_item)

    # Wrap buyer model for token counting
    buyer_model = TokenCountingWrapper(model)

    mcts_agent = MCTSNegotiator(
        model=buyer_model,
        domain_config=first_item,
        domain=initial_domain,
        lora_adapter=lora_adapter_name,
        **mcts_config,
    )

    print(f"MCTS config: {args.num_simulations} simulations, max_rounds={args.max_rounds}")

    # Run negotiations
    all_results = []

    for i, scenario in enumerate(tqdm(scenarios, desc="Scenarios"), 1):
        buyer_model.reset()
        result, outcome, reward = run_cb_negotiation(
            model=buyer_model,
            mcts_agent=mcts_agent,
            scenario=scenario,
            max_rounds=args.max_rounds,
            verbose=args.verbose,
            seller_model=seller_model,
            use_analyzer=use_analyzer,
            token_counter=buyer_model,
        )
        all_results.append(result)

        # Print running metrics every 50 scenarios
        if i % 25 == 0:
            n = len(all_results)
            acc = sum(1 for r in all_results if r['outcome'] == 'accepted')
            rew = sum(r['reward'] for r in all_results) / n
            rnds = sum(r['num_rounds'] for r in all_results) / n
            disc = sum(r['discount_pct'] for r in all_results) / n
            tok = sum(r.get('buyer_total_tokens') or 0 for r in all_results)
            elapsed = sum(r.get('elapsed_sec', 0) for r in all_results)
            print(f"\n[{i}/{len(scenarios)}] Acceptance: {acc}/{n} ({acc/n*100:.1f}%) | "
                  f"Avg Reward: {rew:.4f} | Avg Discount: {disc:.1f}% | Avg Rounds: {rnds:.2f} | "
                  f"Tokens: {tok} | Elapsed: {elapsed:.1f}s")

    # Summary
    total = len(all_results)
    accepted = sum(1 for r in all_results if r['outcome'] == 'accepted')
    avg_reward = sum(r['reward'] for r in all_results) / total if total else 0
    avg_rounds = sum(r['num_rounds'] for r in all_results) / total if total else 0
    avg_discount = (
        sum(r['discount_pct'] for r in all_results) / total
        if total else 0.0
    )
    total_tokens = sum(r.get('buyer_total_tokens') or 0 for r in all_results)
    total_elapsed = sum(r.get('elapsed_sec', 0) for r in all_results)

    print(f"\n{'='*70}")
    print(f"CB MCTS RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total Scenarios:    {total}")
    print(f"Accepted (deals):   {accepted} ({accepted/total*100:.1f}%)")
    print(f"Avg Reward:         {avg_reward:.4f}")
    print(f"Avg Discount:       {avg_discount:.1f}%")
    print(f"Avg Rounds:         {avg_rounds:.2f}")
    print(f"Total Buyer Tokens: {total_tokens}  (avg {total_tokens//total if total else 0}/scenario)")
    print(f"Total Elapsed:      {total_elapsed:.1f}s  (avg {total_elapsed/total:.1f}s/scenario)")
    print(f"{'='*70}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"cb_mcts_results_{timestamp}.json")

    config = {
        "script": "run_negotiation_cb_mcts",
        "data_file": args.data_file,
        "model_name": args.model_name,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "max_rounds": args.max_rounds,
        "num_simulations": args.num_simulations,
        "exploration_weight": args.exploration_weight,
        "use_mock": args.use_mock,
        "use_vllm": args.use_vllm,
        # Provenance: without these a results file cannot tell you whether the
        # adapters were loaded, or whether this was really a full-system run.
        "actor_adapter": args.actor_adapter,
        "analyzer_adapter": args.analyzer_adapter,
        "use_actor_adapter": bool(args.actor_adapter),
        "use_analyzer_adapter": bool(args.analyzer_adapter),
        "use_analyzer": args.analyzer_adapter is not None,
    }

    save_results(
        results=all_results,
        config=config,
        output_file=output_file,
        verbose=True,
    )

    print(f"\nCB MCTS negotiation complete!")


if __name__ == "__main__":
    main()
