"""
Collect DPO training data for buyer strategy selection, for any registered domain.

At each buyer turn:
1. Run MCTS to explore different strategies
2. Rank strategies by avg_reward
3. Select best realization (highest value) for each strategy
4. Create pairwise DPO triples: (prompt, best_strategy, other_strategy)

Output: JSON files per turn with {prompt, chosen, rejected} triples.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm


from negotiation_llm.models import MockModel, HuggingFaceModel, VLLMModel
from negotiation_llm.mcts import MCTSNegotiator
from negotiation_llm.config import MCTS_CONFIG, MODEL_CONFIG
from negotiation_llm.domains import DOMAIN_REGISTRY, get_domain


def extract_best_expression(child_node) -> Tuple[Dict[str, Any], str, bool]:
    """
    Extract the best expression (realization) from a child node.

    Best expression = realization with highest average value estimate.
    Extracts the last buyer turn from that realization.

    Returns:
        Tuple of (best_offer, best_dialogue, has_realizations)
    """
    has_realizations = (
        hasattr(child_node, 'realizations') and
        hasattr(child_node, 'realization_values') and
        hasattr(child_node, 'realization_visits') and
        child_node.realizations and
        len(child_node.realizations) > 0
    )

    if has_realizations:
        best_idx = -1
        best_avg_value = -float('inf')

        for i in range(len(child_node.realization_values)):
            visits = child_node.realization_visits[i]
            if visits > 0:
                avg_value = child_node.realization_values[i] / visits
                if avg_value > best_avg_value:
                    best_avg_value = avg_value
                    best_idx = i

        if best_idx >= 0:
            best_history = child_node.realizations[best_idx]
            for turn in reversed(best_history):
                if turn.get('speaker') == 'buyer':
                    best_offer = turn.get('offer')
                    best_dialogue = turn.get('dialogue')
                    if best_offer and best_dialogue:
                        return best_offer, best_dialogue, True

    # Fallback: first realization
    if has_realizations and len(child_node.realizations) > 0:
        first_history = child_node.realizations[0]
        for turn in reversed(first_history):
            if turn.get('speaker') == 'buyer':
                return turn.get('offer') or {}, turn.get('dialogue') or "", False

    return {}, "", False


def min_seller_price(history: List[Dict[str, Any]]) -> Optional[float]:
    """
    Lowest price the seller quoted across a transcript.

    Only the seller's own quotes count; the buyer's lowballs are ignored.
    """
    prices = [
        (turn.get("offer") or {}).get("price")
        for turn in history
        if turn.get("speaker") == "seller"
    ]
    prices = [float(p) for p in prices if isinstance(p, (int, float))]
    return min(prices) if prices else None


def collect_turn_dpo(
    root,
    buyer_profile: Dict[str, Any],
    negotiation_history: List[Dict[str, Any]],
    turn_number: int,
    domain,
) -> List[Dict[str, Any]]:
    """
    Extract DPO preference pairs from MCTS root node for one buyer turn.

    Creates N-1 pairwise comparisons where N = number of visited strategies.
    Ranking is by avg_reward (higher = better for buyer).

    Returns:
        List of DPO preference pairs
    """
    if not root.children or len(root.children) < 2:
        return []

    children_with_rewards = [
        (child, child.avg_reward)
        for child in root.children
        if child.visit_count > 0
    ]

    if len(children_with_rewards) < 2:
        return []

    children_with_rewards.sort(key=lambda x: x[1], reverse=True)

    best_child, best_reward = children_with_rewards[0]
    best_offer, best_dialogue, best_has_real = extract_best_expression(best_child)
    best_strategy = best_child.strategy.value if hasattr(best_child, 'strategy') and best_child.strategy else "unknown"

    # Build the strategy selection prompt from the SAME strategy list the search
    # actually expanded, so the training prompt matches the one scored at
    # inference. MCTSNegotiator caps the list at max_children (10 of 11), and
    # reading it back off the node keeps the two in step automatically if that
    # cap ever changes. Falls back to the domain's full list only if the root
    # was never expanded.
    strategies = getattr(root, "available_strategies", None) or domain.get_strategies()
    prompt = domain.create_strategy_selection_prompt(
        agent_profile=buyer_profile,
        strategies=strategies,
        history=negotiation_history,
    )

    listing_price = domain.item_config.get("listing_price", 0)
    buyer_target = buyer_profile.get("target_price")

    dpo_examples = []
    for rejected_child, rejected_reward in children_with_rewards[1:]:
        rejected_offer, rejected_dialogue, rejected_has_real = extract_best_expression(rejected_child)
        rejected_strategy = (
            rejected_child.strategy.value
            if hasattr(rejected_child, 'strategy') and rejected_child.strategy
            else "unknown"
        )

        dpo_examples.append({
            "prompt": prompt,
            "chosen": {
                "strategy": best_strategy,
                "offer": best_offer,
                "dialogue": best_dialogue,
                "avg_reward": best_reward,
                "visit_count": best_child.visit_count,
            },
            "rejected": {
                "strategy": rejected_strategy,
                "offer": rejected_offer,
                "dialogue": rejected_dialogue,
                "avg_reward": rejected_reward,
                "visit_count": rejected_child.visit_count,
            },
            "metadata": {
                "turn": turn_number,
                "reward_margin": best_reward - rejected_reward,
                "chosen_has_realizations": best_has_real,
                "rejected_has_realizations": rejected_has_real,
                "listing_price": listing_price,
                "buyer_target": buyer_target,
                "depth": root.depth,
            },
        })

    return dpo_examples


def save_dpo_examples(
    domain_name: str,
    dpo_examples: List[Dict[str, Any]],
    buyer_profile: Dict[str, Any],
    item_config: Dict[str, Any],
    output_dir: str,
    prefix: str = "",
) -> Path:
    """Save DPO examples to a JSON file, tagged with the domain."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix_str = f"{prefix}_" if prefix else ""
    filepath = output_path / f"{prefix_str}{domain_name}_dpo_{timestamp}.json"

    data = {
        "metadata": {
            "component": f"{domain_name}_strategy_selection",
            "format": "dpo_pairwise",
            "item_title": item_config.get("title", ""),
            "listing_price": item_config.get("listing_price", 0),
            "buyer_target": buyer_profile.get("target_price"),
            "num_examples": len(dpo_examples),
            "generated_at": datetime.now().isoformat(),
            "description": f"DPO preference pairs for {domain_name} buyer strategy selection (best vs others, ranked by avg_reward)",
        },
        "buyer_profile": buyer_profile,
        "item_config": item_config,
        "examples": dpo_examples,
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    return filepath


def run_scenario_with_dpo(
    model,
    mcts_agent: MCTSNegotiator,
    spec,
    scenario: Dict[str, Any],
    max_rounds: int = 10,
    verbose: bool = False,
    output_dir: str = "dpo_data",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """
    Run one negotiation and collect DPO preference pairs at each buyer turn.

    Returns:
        Tuple of (negotiation_history, all_dpo_examples, reward)
    """
    buyer_profile = scenario["buyer"]
    item_config = scenario["item"]
    scenario_id = scenario.get("scenario_id", 0)
    listing_price = item_config["listing_price"]

    # Re-instantiate domain for this scenario
    seller_persona = buyer_profile.get("seller_persona") or spec.seller_components(scenario)["persona"]
    mcts_agent.domain_config = item_config
    mcts_agent.domain = spec.domain_cls(
        item_config=item_config,
        seller_persona=seller_persona,
        seller_target=scenario["seller"].get("target_price"),
    )
    domain = mcts_agent.domain

    negotiation_history = []
    all_dpo_examples = []
    negotiation_outcome = "max_rounds_reached"

    if verbose:
        print(f"\n{'='*70}")
        print(f"{spec.name.upper()} SCENARIO {scenario_id}: {spec.describe_item(item_config)}")
        target = buyer_profile.get('target_price')
        print(f"Buyer target: ${target:.0f}" if target else "Buyer target: N/A")
        print(f"{'='*70}")

    for round_num in range(max_rounds):
        turn_number = round_num + 1

        if verbose:
            print(f"\n--- Turn {turn_number} ---")

        # MCTS selects buyer action
        buyer_offer, buyer_dialogue, root = mcts_agent.select_action(
            agent_profile=buyer_profile,
            negotiation_history=negotiation_history,
            max_negotiation_rounds=max_rounds,
        )

        if verbose:
            price_str = f"${buyer_offer.get('price'):.0f}" if buyer_offer.get('price') is not None else "N/A"
            print(f"Buyer: {buyer_dialogue}  [{buyer_offer.get('action_type', '?')} @ {price_str}]")
            print("  Strategy visits:")
            for child in sorted(root.children, key=lambda c: c.avg_reward, reverse=True):
                if child.visit_count > 0:
                    strat = child.strategy.value if child.strategy else "unknown"
                    print(f"    {strat}: visits={child.visit_count}, avg_reward={child.avg_reward:.3f}")

        # Extract DPO preferences from this turn's MCTS search
        turn_dpo = collect_turn_dpo(
            root=root,
            buyer_profile=buyer_profile,
            negotiation_history=negotiation_history,
            turn_number=turn_number,
            domain=domain,
        )
        all_dpo_examples.extend(turn_dpo)

        # Save per-turn (incremental, survives interruption)
        if turn_dpo:
            saved_path = save_dpo_examples(
                domain_name=spec.name,
                dpo_examples=turn_dpo,
                buyer_profile=buyer_profile,
                item_config=item_config,
                output_dir=output_dir,
                prefix=f"scenario{scenario_id}_turn{turn_number}",
            )
            if verbose:
                print(f"  → Saved {len(turn_dpo)} DPO pairs to {saved_path.name}")
        elif verbose:
            print(f"  → No DPO pairs (need ≥2 visited strategies)")

        # Add buyer turn to history
        negotiation_history.append({
            "speaker": "buyer",
            "offer": buyer_offer,
            "dialogue": buyer_dialogue,
        })

        # Buyer accepted seller's last price
        if buyer_offer.get("action_type") == "accept":
            negotiation_outcome = "accepted"
            if verbose:
                print("  → Buyer accepted seller's price!")
            break

        # Clear MCTS cache
        if hasattr(mcts_agent, 'terminal_cache'):
            mcts_agent.terminal_cache.clear()

        # Generate seller response
        seller_prompt = domain.create_opponent_prompt(
            opponent_profile=scenario["seller"],
            history=negotiation_history,
            current_agent_offer=buyer_offer,
            current_agent_dialogue=buyer_dialogue,
        )
        seller_response = model.generate(seller_prompt)
        parsed_seller = domain.parse_opponent_response(seller_response)

        seller_offer = domain.build_opponent_offer(parsed_seller)
        seller_dialogue = parsed_seller.get("dialogue", seller_response)
        seller_action = parsed_seller.get("action_type", "counter")

        if verbose:
            price_str = f"${seller_offer.get('price'):.0f}" if seller_offer.get('price') is not None else "N/A"
            print(f"Seller: {seller_dialogue}  [{seller_action} @ {price_str}]")

        # Add seller turn to history
        negotiation_history.append({
            "speaker": "seller",
            "offer": seller_offer,
            "dialogue": seller_dialogue,
        })

        if seller_action == "accept" or parsed_seller.get("accepts"):
            negotiation_outcome = "accepted"
            if verbose:
                print("  → Seller accepted buyer's offer!")
            break
        elif seller_action == "reject" or parsed_seller.get("rejects"):
            negotiation_outcome = "rejected"
            if verbose:
                print("  → Seller rejected!")
            break

    # Find final agreed price
    final_offer = {}
    final_price = None
    if negotiation_outcome == "accepted":
        for turn in reversed(negotiation_history):
            offer = turn.get('offer', {}) or {}
            if offer.get('price') is not None:
                final_price = offer['price']
                final_offer = offer
                break
        if final_price is None:
            final_price = listing_price
            final_offer = {"price": final_price, "action_type": "accept"}

    num_rounds = len([t for t in negotiation_history if t.get('speaker') == 'buyer'])

    reward = domain.calculate_reward(
        final_terms=final_offer,
        agent_profile=buyer_profile,
        outcome=negotiation_outcome,
        num_rounds=num_rounds,
        max_rounds=max_rounds,
    )

    if verbose:
        discount_pct = 0.0
        if negotiation_outcome == "accepted" and final_price and listing_price > 0:
            discount_pct = (listing_price - final_price) / listing_price * 100
        print(f"\nOutcome: {negotiation_outcome} | Rounds: {num_rounds} | Reward: {reward:.4f}")
        if negotiation_outcome == "accepted":
            print(f"Final price: ${final_price:.0f} / ${listing_price:.0f} ({discount_pct:.1f}% discount)")
        print(f"Total DPO pairs this scenario: {len(all_dpo_examples)}")

    return negotiation_history, all_dpo_examples, reward


def main():
    parser = argparse.ArgumentParser(
        description="Collect buyer strategy DPO data with MCTS, for any registered domain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test with mock model
  python -m negotiation_llm.data.collect_dpo --domain cb --max-scenarios 3 \
      --num-simulations 3 --max-rounds 5 --use-mock --verbose

  # CraigslistBargain training set
  python -m negotiation_llm.data.collect_dpo --domain cb --data-file CB/train.csv \
      --num-simulations 30 --output-dir ./cb_dpo_train

  # A2A — no A2A-specific adapter existed before this script became
  # domain-general; evaluation fell back to the CB-trained one.
  python -m negotiation_llm.data.collect_dpo --domain a2a \
      --num-simulations 30 --output-dir ./a2a_dpo_train
        """,
    )

    parser.add_argument("--domain", type=str, default="cb", choices=sorted(DOMAIN_REGISTRY),
                        help="Negotiation domain to collect for (default: cb)")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Scenario file; defaults to the domain's standard path")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Max scenarios to process (default: all)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle scenarios before selecting")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for DPO JSON files (default: <domain>_dpo_data)")
    parser.add_argument("--num-simulations", type=int, default=20,
                        help="MCTS simulations per buyer turn (default: 20)")
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="Maximum negotiation rounds per scenario (default: 10)")
    parser.add_argument("--exploration-weight", type=float, default=2.5,
                        help="PUCT exploration constant (default: 2.5)")
    parser.add_argument("--model-name", type=str, default=MODEL_CONFIG["model_name"],
                        help="Model name or path")
    parser.add_argument("--use-mock", action="store_true",
                        help="Use mock model (for testing)")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for inference")
    parser.add_argument("--verbose", action="store_true",
                        help="Print turn-by-turn output")

    args = parser.parse_args()

    spec = get_domain(args.domain)
    if args.output_dir is None:
        args.output_dir = f"{spec.name}_dpo_data"
    if args.data_file is None:
        args.data_file = spec.cli_defaults("mcts")["data_file"]

    # DomainSpec.load_scenarios slices with start/end; this script exposes
    # --max-scenarios instead, so translate.
    args.start_index = 0
    args.end_index = args.max_scenarios
    args.buyer_target_ratio = getattr(args, "buyer_target_ratio", 0.2)

    print(f"Loading {spec.name} scenarios from: {args.data_file}")
    scenarios = spec.load_scenarios(args)
    print(f"Loaded {len(scenarios)} scenarios")

    if not scenarios:
        print("No valid scenarios found.")
        return

    # Initialize model
    if args.use_mock:
        print("Using MockModel for testing")
        model = MockModel()
    else:
        model_config = MODEL_CONFIG.copy()
        model_config['model_name'] = args.model_name
        print(f"Loading VLLMModel: {args.model_name}")
        model = VLLMModel(**model_config)

    # Initialize MCTS agent (domain is overridden per scenario)
    first_item = scenarios[0]["item"]
    analyzer_dir = str(Path(args.output_dir) / "analyzer")
    mcts_config = {
        k: v for k, v in MCTS_CONFIG.items()
        if k not in ('num_simulations', 'exploration_weight', 'use_realizations',
                     'collect_dpo_data', 'dpo_output_dir')
    }
    mcts_agent = MCTSNegotiator(
        model=model,
        domain_config=first_item,
        domain=spec.domain_cls(item_config=first_item),
        num_simulations=args.num_simulations,
        exploration_weight=args.exploration_weight,
        use_realizations=True,      # Required for extract_best_expression
        collect_dpo_data=True,      # Collect seller utterances from MCTS tree
        dpo_output_dir=analyzer_dir,
        **mcts_config,
    )

    print(f"\nConfiguration:")
    print(f"  Simulations/turn: {args.num_simulations}")
    print(f"  Exploration weight: {args.exploration_weight}")
    print(f"  Max rounds: {args.max_rounds}")
    print(f"  Strategy DPO dir: {args.output_dir}/")
    print(f"  Analyzer data dir: {analyzer_dir}/")
    print()

    # Collect DPO data
    total_dpo = 0
    total_turns = 0
    accepted = 0
    all_rewards = []
    all_discounts = []

    for scenario in tqdm(scenarios, desc="Scenarios", disable=args.verbose):
        persona_data = spec.seller_components(scenario)

        # The buyer profile carries the seller persona only so the collection
        # loop can build the seller prompt from it (see run_scenario_with_dpo).
        scenario["buyer"] = {
            **scenario["buyer"],
            "seller_persona": persona_data["persona"],
            "big_five_personality": persona_data["big_five_personality"],
            "decision_making_style": persona_data["decision_making_style"],
        }

        # Labels for the analyzer describe the SELLER — the party it predicts.
        # Passing scenario["buyer"] here (as this script used to) labelled
        # "Seller Target Price" with the buyer's own reservation price, which
        # the buyer already knows. role is stated explicitly so the mistake is
        # visible in the saved files.
        analyzer_labels = {
            "role": "seller",
            "target_price": scenario["seller"].get("target_price"),
            "seller_persona": persona_data["persona"],
            "big_five_personality": persona_data["big_five_personality"],
            "decision_making_style": persona_data["decision_making_style"],
        }

        history, dpo_examples, reward = run_scenario_with_dpo(
            model=model,
            mcts_agent=mcts_agent,
            spec=spec,
            scenario=scenario,
            max_rounds=args.max_rounds,
            verbose=args.verbose,
            output_dir=args.output_dir,
        )

        # How low this seller actually went over the whole episode. This is the
        # analyzer's regression target: at turn k it must forecast the floor the
        # seller reaches by the end, which sits strictly below the current quote
        # about half the time. Taking the minimum within a single MCTS
        # realization instead would coincide with the latest quote 81% of the
        # time, since a realization always ends at the turn being labelled.
        analyzer_labels["observed_seller_floor"] = min_seller_price(history)

        # Save and clear MCTS-internal analyzer data (seller utterances from tree)
        scenario_id = scenario.get("scenario_id", 0)
        mcts_agent.save_dpo_data(
            analyzer_labels,
            prefix=f"scenario{scenario_id}",
        )
        mcts_agent.clear_dpo_buffers()

        num_buyer_turns = len([t for t in history if t.get('speaker') == 'buyer'])
        total_dpo += len(dpo_examples)
        total_turns += num_buyer_turns
        all_rewards.append(reward)

        # Track acceptance and discount
        outcome = "accepted" if reward > -0.5 and reward != -1.0 else "other"
        listing_price = scenario["item"]["listing_price"]
        if reward > 0:
            accepted += 1
            # Find final price
            for turn in reversed(history):
                offer = turn.get('offer', {}) or {}
                if offer.get('price') is not None:
                    discount_pct = (listing_price - offer['price']) / listing_price * 100
                    all_discounts.append(discount_pct)
                    break

    # Summary
    n = len(scenarios)
    mean_reward = sum(all_rewards) / n if n else 0.0
    avg_pairs = total_dpo / n if n else 0.0
    avg_turns = total_turns / n if n else 0.0
    avg_discount = sum(all_discounts) / len(all_discounts) if all_discounts else 0.0

    print()
    print("=" * 70)
    print(f"{spec.name.upper()} DPO COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Scenarios processed:  {n}")
    print(f"Total DPO pairs:      {total_dpo}")
    print(f"Avg pairs/scenario:   {avg_pairs:.1f}")
    print(f"Avg turns/scenario:   {avg_turns:.2f}")
    print(f"Avg MCTS reward:      {mean_reward:.4f}")
    print(f"Scenarios with deals: {accepted} ({accepted/n*100:.1f}%)")
    if all_discounts:
        print(f"Avg discount (deals): {avg_discount:.1f}%")
    print(f"Strategy DPO data:    {args.output_dir}/")
    print(f"Analyzer data:        {analyzer_dir}/")
    print()

    model.cleanup()


if __name__ == "__main__":
    main()
