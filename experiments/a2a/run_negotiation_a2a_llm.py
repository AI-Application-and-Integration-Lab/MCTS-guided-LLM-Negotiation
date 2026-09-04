"""
Run pure-LLM negotiation on the A2A Consumer dataset.

Both buyer and seller are plain LLM calls — no MCTS, no chain-of-thought,
no reflexion.  Baseline for comparison with ProCoT, DMNA, and ICL-AIF.

Buyer acts first.  The seller is a professional (knows cost floor).

Metrics reported:
  - SR  (Success Rate):   fraction of negotiations that reach a deal
  - AT  (Average Turns):  mean buyer turns per negotiation
  - Discount %:           (listing - final) / listing * 100, avg over all
  - Avg Reward:           internal reward signal
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm


from negotiation_llm.models import MockModel, VLLMModel, OpenAIModel
from negotiation_llm.config import MODEL_CONFIG
from negotiation_llm.domains.a2a import A2ANegotiationDomain, load_a2a_scenarios, generate_seller_persona
from negotiation_llm.domains.a2a.prompts import _format_history, parse_a2a_buyer_response
from negotiation_llm.domains.a2a.reward import calculate_a2a_reward_from_terms


# ------------------------------------------------------------------ #
# Buyer prompt                                                         #
# ------------------------------------------------------------------ #

def create_llm_buyer_prompt(
    item_config: Dict[str, Any],
    buyer_target: Optional[float],
    history: List[Dict[str, Any]],
    current_seller_dialogue: Optional[str] = None,
    current_seller_offer: Optional[Dict[str, Any]] = None,
) -> str:
    name = item_config.get("name", "product")
    features = item_config.get("features", "")
    product_type = item_config.get("type", "product")
    listing_price = item_config.get("listing_price", 0)
    target_str = f"${buyer_target:,.0f}" if buyer_target is not None else "as low as possible"

    history_text = _format_history(history)

    seller_context = ""
    if current_seller_dialogue:
        price_info = ""
        if current_seller_offer and current_seller_offer.get("price") is not None:
            price_info = f" [${current_seller_offer['price']:,.0f}]"
        seller_context = f"\nSeller's last message: {current_seller_dialogue}{price_info}"

    return f"""Now enter the role-playing mode. In the following conversation, you will play as a buyer negotiating to purchase a {product_type}.

You are trying to buy: {name}
Product features: {features}
Listed at: ${listing_price:,.0f}
Your target price: {target_str}

Please chat with the Seller using a short and natural sentence. Reply with only one short and succinct sentence.

Conversation Context:
{history_text}{seller_context}

Your Response:"""


# ------------------------------------------------------------------ #
# Single-scenario negotiation                                          #
# ------------------------------------------------------------------ #

def run_llm_negotiation(
    buyer_model,
    seller_model,
    scenario: Dict[str, Any],
    max_rounds: int = 10,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], str, float]:
    buyer_profile = scenario["buyer"]
    item_config = scenario["item"]
    listing_price = item_config["listing_price"]
    buyer_target = buyer_profile.get("target_price")

    domain = A2ANegotiationDomain(
        item_config=item_config,
        seller_persona=generate_seller_persona(scenario),
        seller_target=scenario["seller"].get("target_price"),
    )

    negotiation_history: List[Dict[str, Any]] = []
    turn_records: List[Dict[str, Any]] = []
    negotiation_outcome = "max_rounds_reached"
    scenario_start = time.perf_counter()

    if verbose:
        print(f"\n{'='*70}")
        print(f"A2A LLM SCENARIO {scenario.get('scenario_id', '?')}: "
              f"{item_config['name']} (${listing_price:,.0f})")
        target_str = f"${buyer_target:,.0f}" if buyer_target is not None else "N/A"
        print(f"Buyer target: {target_str}  |  Type: {item_config['type']}")
        print(f"{'='*70}")

    current_seller_offer: Optional[Dict[str, Any]] = None
    current_seller_dialogue: Optional[str] = None

    for round_num in range(max_rounds):
        turn_number = round_num + 1

        if verbose:
            print(f"\n--- Turn {turn_number} ---")

        # ---- Buyer turn ----
        buyer_prompt = create_llm_buyer_prompt(
            item_config=item_config,
            buyer_target=buyer_target,
            history=negotiation_history,
            current_seller_dialogue=current_seller_dialogue,
            current_seller_offer=current_seller_offer,
        )
        _t0 = time.perf_counter()
        buyer_raw = buyer_model.generate(buyer_prompt, max_tokens=128, temperature=0.7)
        turn_time = time.perf_counter() - _t0

        parsed_buyer = parse_a2a_buyer_response(buyer_raw)
        buyer_offer = {
            "price": parsed_buyer.get("price"),
            "action_type": parsed_buyer.get("action_type", "dialogue"),
        }
        buyer_dialogue = parsed_buyer.get("dialogue") or buyer_raw.strip()

        if verbose:
            price_str = f"${buyer_offer['price']:,.0f}" if buyer_offer.get("price") is not None else "N/A"
            print(f"Buyer: {buyer_dialogue}  [{buyer_offer.get('action_type', '?')} @ {price_str}]")

        turn_record = {
            "turn": turn_number,
            "buyer_action": {"offer": buyer_offer, "dialogue": buyer_dialogue},
            "turn_time_sec": round(turn_time, 3),
        }

        negotiation_history.append({
            "speaker": "buyer",
            "offer": buyer_offer,
            "dialogue": buyer_dialogue,
        })

        # Only seller can close — buyer "Agree" signals willingness but seller must confirm
        if buyer_offer.get("action_type") == "accept":
            buyer_offer["action_type"] = "propose"
            parsed_buyer["accepts"] = False

        # ---- Seller turn ----
        seller_prompt = domain.create_opponent_prompt(
            opponent_profile=scenario["seller"],
            history=negotiation_history,
            current_agent_offer=buyer_offer,
            current_agent_dialogue=buyer_dialogue,
        )
        seller_raw = seller_model.generate(seller_prompt)
        parsed_seller = domain.parse_opponent_response(seller_raw)

        seller_offer = domain.build_opponent_offer(parsed_seller)
        seller_dialogue = parsed_seller.get("dialogue", seller_raw)
        seller_action = parsed_seller.get("action_type", "counter")

        if verbose:
            price_str = f"${seller_offer['price']:,.0f}" if seller_offer.get("price") is not None else "N/A"
            print(f"Seller: {seller_dialogue}  [{seller_action} @ {price_str}]")

        turn_record["seller_response"] = {
            "offer": seller_offer,
            "dialogue": seller_dialogue,
            "action_type": seller_action,
        }
        turn_records.append(turn_record)

        negotiation_history.append({
            "speaker": "seller",
            "offer": seller_offer,
            "dialogue": seller_dialogue,
        })

        current_seller_offer = seller_offer
        current_seller_dialogue = seller_dialogue

        if seller_action == "accept" or parsed_seller.get("accepts"):
            negotiation_outcome = "accepted"
            buyer_turns = [h for h in negotiation_history if h.get("speaker") == "buyer"]
            if buyer_turns:
                accepted_price = (buyer_turns[-1].get("offer") or {}).get("price")
                if accepted_price is not None:
                    seller_offer["price"] = accepted_price
            if verbose:
                print("  → Seller accepted buyer's offer!")
            break
        elif seller_action == "reject" or parsed_seller.get("rejects"):
            if verbose:
                print("  → Seller rejected offer (negotiation continues)")

    # ---- Final metrics ----
    final_offer: Dict[str, Any] = {}
    final_price: Optional[float] = None

    if negotiation_outcome == "accepted":
        last_turn = negotiation_history[-1] if negotiation_history else {}
        last_speaker = last_turn.get("speaker")

        if last_speaker == "buyer":
            final_offer = last_turn.get("offer") or {}
            final_price = final_offer.get("price")
        elif last_speaker == "seller":
            buyer_turns = [h for h in negotiation_history if h.get("speaker") == "buyer"]
            if buyer_turns:
                final_offer = buyer_turns[-1].get("offer") or {}
                final_price = final_offer.get("price")

        if final_price is None:
            final_price = listing_price
            final_offer = {"price": final_price, "action_type": "accept"}

        if final_price > listing_price * 1.01:
            if verbose:
                print(f"  [Warning] Final price ${final_price:,.0f} exceeds listing — clamping")
            final_price = listing_price
            final_offer["price"] = final_price

    num_rounds = len([t for t in negotiation_history if t.get("speaker") == "buyer"])
    elapsed_sec = time.perf_counter() - scenario_start

    reward = calculate_a2a_reward_from_terms(
        final_terms=final_offer,
        agent_profile=buyer_profile,
        item_config=item_config,
        outcome=negotiation_outcome,
        num_rounds=num_rounds,
        max_rounds=max_rounds,
    )

    discount_pct = 0.0
    if negotiation_outcome == "accepted" and final_price is not None and listing_price > 0:
        discount_pct = (listing_price - final_price) / listing_price * 100

    if verbose:
        print(f"\n{'='*70}")
        print(f"Outcome: {negotiation_outcome}  |  Turns: {num_rounds}  |  Reward: {reward:.4f}")
        if negotiation_outcome == "accepted":
            print(f"Final: ${final_price:,.0f} / ${listing_price:,.0f}  (discount={discount_pct:.1f}%)")

    return {
        "scenario_id": scenario.get("scenario_id", 0),
        "buyer_profile": buyer_profile,
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
        "timestamp": datetime.now().isoformat(),
    }, negotiation_outcome, reward


# ------------------------------------------------------------------ #
# Results saving                                                       #
# ------------------------------------------------------------------ #

def save_results(results, config, output_file, verbose=False):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    accepted = sum(1 for r in results if r["outcome"] == "accepted")
    rejected = sum(1 for r in results if r["outcome"] == "rejected")
    max_rounds_count = sum(1 for r in results if r["outcome"] == "max_rounds_reached")

    output = {
        "config": config,
        "summary": {
            "total_scenarios": total,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "max_rounds_count": max_rounds_count,
            "success_rate": accepted / total if total else 0.0,
            "avg_reward": sum(r["reward"] for r in results) / total if total else 0.0,
            "avg_rounds": sum(r["num_rounds"] for r in results) / total if total else 0.0,
            "avg_discount_pct": sum(r["discount_pct"] for r in results) / total if total else 0.0,
            "avg_elapsed_sec": round(sum(r.get("elapsed_sec", 0) for r in results) / total, 3) if total else 0.0,
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    if verbose:
        print(f"Results saved to: {output_path}")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Run pure-LLM buyer baseline on A2A Consumer dataset",
    )
    parser.add_argument("--data-file", type=str,
                        default="Agent2Agent-Negotiation-in-Consumer-Setting-Dataset/products.json")
    parser.add_argument("--output-dir", type=str, default="./results/a2a_llm")
    parser.add_argument("--model-name", type=str, default=MODEL_CONFIG["model_name"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--buyer-target-ratio", type=float, default=0.2,
                        help="Buyer target = wholesale + ratio*(retail-wholesale). "
                             "0.0=wholesale, 0.5=midpoint (default: 0.0)")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-vllm", action="store_true",
                        help="Deprecated no-op: vLLM is already the backend unless --use-mock is passed. Kept so existing commands still parse.")
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--use-openai-buyer", action="store_true")
    parser.add_argument("--buyer-model", type=str, default="gpt-4o")
    parser.add_argument("--buyer-api-key", type=str, default=None)
    parser.add_argument("--buyer-base-url", type=str, default=None)
    parser.add_argument("--use-openai-seller", action="store_true")
    parser.add_argument("--seller-model", type=str, default="gpt-4o")
    parser.add_argument("--seller-api-key", type=str, default=None)
    parser.add_argument("--seller-base-url", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print(f"Loading A2A scenarios from: {args.data_file}")
    scenarios = load_a2a_scenarios(
        json_path=args.data_file,
        start_index=args.start_index,
        end_index=args.end_index,
        shuffle=args.shuffle,
        seed=args.seed,
        buyer_target_ratio=args.buyer_target_ratio,
    )
    print(f"Loaded {len(scenarios)} scenarios  (buyer_target_ratio={args.buyer_target_ratio})")
    if not scenarios:
        print("No valid scenarios found.")
        return

    # Build models
    if args.use_mock:
        buyer_model = seller_model = MockModel()
    elif args.use_openai_buyer:
        buyer_model = OpenAIModel(model_name=args.buyer_model,
                                  api_key=args.buyer_api_key, base_url=args.buyer_base_url)
        if args.use_openai_seller:
            seller_model = (buyer_model if args.seller_model == args.buyer_model
                            else OpenAIModel(model_name=args.seller_model,
                                             api_key=args.seller_api_key, base_url=args.seller_base_url))
        else:
            seller_model = buyer_model
    else:
        cfg = MODEL_CONFIG.copy(); cfg["model_name"] = args.model_name
        buyer_model = VLLMModel(**cfg)
        if args.use_openai_seller:
            seller_model = OpenAIModel(model_name=args.seller_model,
                                       api_key=args.seller_api_key, base_url=args.seller_base_url)
        else:
            seller_model = buyer_model

    buyer_label = args.buyer_model if args.use_openai_buyer else args.model_name
    seller_label = (args.seller_model if args.use_openai_seller
                    else (args.buyer_model if args.use_openai_buyer else args.model_name))

    print(f"\nConfiguration:")
    print(f"  Buyer:  {buyer_label}  |  Seller: {seller_label}")
    print(f"  Max rounds: {args.max_rounds}  |  Output: {args.output_dir}/\n")

    all_results = []
    for i, scenario in enumerate(tqdm(scenarios, desc="Scenarios", disable=args.verbose), 1):
        result, outcome, reward = run_llm_negotiation(
            buyer_model=buyer_model,
            seller_model=seller_model,
            scenario=scenario,
            max_rounds=args.max_rounds,
            verbose=args.verbose,
        )
        all_results.append(result)

        if i % 25 == 0:
            n = len(all_results)
            acc = sum(1 for r in all_results if r["outcome"] == "accepted")
            rew = sum(r["reward"] for r in all_results) / n
            disc = sum(r["discount_pct"] for r in all_results) / n
            print(f"\n[{i}/{len(scenarios)}] SR: {acc/n*100:.1f}% | "
                  f"Reward: {rew:.4f} | Discount: {disc:.1f}%")

    total = len(all_results)
    accepted = sum(1 for r in all_results if r["outcome"] == "accepted")
    avg_reward = sum(r["reward"] for r in all_results) / total
    avg_rounds = sum(r["num_rounds"] for r in all_results) / total
    avg_discount = sum(r["discount_pct"] for r in all_results) / total
    total_elapsed = sum(r.get("elapsed_sec", 0) for r in all_results)

    print(f"\n{'='*70}")
    print(f"A2A PURE-LLM RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total:        {total}")
    print(f"SR:           {accepted/total*100:.1f}%  ({accepted}/{total})")
    print(f"AT:           {avg_rounds:.2f}")
    print(f"Avg Discount: {avg_discount:.1f}%")
    print(f"Avg Reward:   {avg_reward:.4f}")
    print(f"Elapsed:      {total_elapsed:.1f}s")
    print(f"{'='*70}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"a2a_llm_results_{timestamp}.json")
    config = {
        "script": "run_negotiation_a2a_llm",
        "data_file": args.data_file,
        "buyer_model": buyer_label,
        "seller_model": seller_label,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "max_rounds": args.max_rounds,
        "buyer_target_ratio": args.buyer_target_ratio,
        "use_mock": args.use_mock,
        "use_vllm": args.use_vllm,
    }
    save_results(all_results, config, output_file, verbose=True)

    models = {id(m): m for m in [buyer_model, seller_model]}
    for m in models.values():
        m.cleanup()

    print("A2A pure-LLM negotiation complete!")


if __name__ == "__main__":
    main()
