"""
Analyzer model training for opponent characteristic prediction.

Trains the analyzer LoRA to predict the seller's target price, next action
and persona from the negotiation context so far.

Example data directory: ./cb_dpo_train/analyzer (produced by collect_dpo.py)
Data format: *_seller_utterances_*.json with negotiation history
"""
try:
    from unsloth import FastLanguageModel
except ImportError:  # keep --help usable without the training extra
    FastLanguageModel = None
import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer,SFTConfig


from negotiation_llm.training.config import (
    TRAINING_MODEL_CONFIG,
    LORA_CONFIG,
    SFT_TRAINING_CONFIG,
    DATA_CONFIG,
)


def detect_domain_from_data(data_dir: str) -> str:
    """Detect domain by file naming convention, then by offer fields."""
    data_dir = Path(data_dir)
    json_files = sorted(data_dir.glob("*_seller_utterances_*.json"))
    if not json_files:
        return "unknown"
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    for utt in data.get("utterances", []):
        for realization in utt.get("realizations", []):
            for turn in realization:
                if "price" in turn.get("offer", {}):
                    return "cb"
    return "unknown"


def _cb_history_to_text(history: List[Dict[str, Any]]) -> str:
    """Format negotiation history for CB domain."""
    if not history:
        return "Negotiation Start\n"
    lines = ["Negotiation History:"]
    for turn in history:
        speaker = turn.get('speaker', 'unknown')
        dialogue = turn.get('dialogue', '')
        offer = turn.get('offer', {}) or {}
        price = offer.get('price')
        label = "Buyer" if speaker == 'buyer' else "Seller"
        if price is not None:
            lines.append(f"  {label}: {dialogue} [${price:.0f}]")
        else:
            lines.append(f"  {label}: {dialogue}")
    return "\n".join(lines) + "\n"


def load_customer_data_from_directory(data_dir: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """
    Load all seller utterance data from directory.

    Args:
        data_dir: Directory containing seller_utterances JSON files
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of seller utterance examples with seller profile
    """
    data_dir = Path(data_dir)
    all_examples = []
    stale_profiles = []

    # Find all seller_utterances JSON files
    json_files = sorted(data_dir.glob("*_seller_utterances_*.json"))

    print(f"Found {len(json_files)} data files in {data_dir}")

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract seller profile and utterances
        seller_profile = data.get("seller_profile", {})
        utterances = data.get("utterances", [])

        # The analyzer predicts the SELLER, so these labels must describe the
        # seller. Data collected before that was fixed saved the buyer profile
        # here, which made "Seller Target Price" the buyer's own target.
        if seller_profile.get("role") == "buyer":
            stale_profiles.append(json_file.name)

        # Attach seller profile to each utterance
        for utterance in utterances:
            utterance["seller_profile"] = seller_profile
            all_examples.append(utterance)

        if max_samples and len(all_examples) >= max_samples:
            all_examples = all_examples[:max_samples]
            break

    if stale_profiles:
        print(
            f"\n  WARNING: {len(stale_profiles)} of {len(json_files)} files carry a "
            f"buyer profile where the seller's is expected (e.g. {stale_profiles[0]}).\n"
            f"  These were collected before the analyzer label fix. Persona fields are\n"
            f"  still the seller's, and the target price now comes from the transcript,\n"
            f"  so training will proceed — but re-collect with the current\n"
            f"  collect_dpo.py for a clean fallback label.\n"
        )

    print(f"Loaded {len(all_examples)} seller utterances with profiles")
    return all_examples


# Removed format_negotiation_history - now using history_to_text from prompt.py


def _extract_realization_parts(utterance: Dict[str, Any]):
    """Extract common realization parts: history, buyer turn, seller turn.

    Returns None if the realization is empty/invalid.
    """
    realizations = utterance.get("realizations", [[]])
    if not realizations or not realizations[0]:
        return None

    realization = realizations[0]
    if len(realization) < 2:
        return None

    buyer_turn = realization[-2]
    seller_turn = realization[-1]

    # Build negotiation history (all turns before the last buyer-seller pair)
    history = realization[:-2] if len(realization) > 2 else []

    return {
        "history": history,
        "buyer_turn": buyer_turn,
        "seller_turn": seller_turn,
    }


def _observed_seller_floor(realization: List[Dict[str, Any]]) -> Optional[float]:
    """
    Lowest price the seller has quoted so far in this transcript.

    This is the label for "Seller Target Price". The obvious alternative — the
    scenario's stated seller reservation price — is *not* the quantity that
    governs behaviour here: create_cb_seller_prompt ignores its seller_target
    argument (it accepts seller_target and never reads it), so the simulated seller has no floor and
    empirically bottoms out around 0.75x its stated target, within +/-15% of it
    only 17% of the time. Training the analyzer on a number the opponent does
    not respect would give MCTS a systematically wrong opponent model.

    A running minimum over the turns seen so far uses no future information, is
    observable at inference, and tightens as the negotiation proceeds — which is
    exactly the estimate rollouts need.

    """
    prices = [
        turn.get("offer", {}).get("price")
        for turn in realization
        if turn.get("speaker") == "seller"
    ]
    prices = [float(p) for p in prices if isinstance(p, (int, float))]
    return min(prices) if prices else None


def format_cb_example(utterance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format a CB domain utterance for analyzer SFT training.

    Task: given buyer offer + seller response + history, predict seller's
    target price and next action (accept/counter/reject + counter price).
    """
    parts = _extract_realization_parts(utterance)
    if parts is None:
        return None

    seller_profile = utterance.get("seller_profile", {})
    buyer_offer = parts["buyer_turn"].get("offer", {}) or {}
    buyer_dialogue = parts["buyer_turn"].get("dialogue", "")
    seller_offer = parts["seller_turn"].get("offer", {}) or {}
    seller_dialogue = parts["seller_turn"].get("dialogue", "")

    buyer_price = buyer_offer.get("price")
    seller_price = seller_offer.get("price")
    seller_action = seller_offer.get("action_type", "counter")

    history_text = _cb_history_to_text(parts["history"])

    buyer_price_str = f"${buyer_price:.0f}" if buyer_price is not None else "N/A"

    prompt = f"""Given the following price negotiation context, predict the seller's target price and next action.

{history_text}
Buyer's Current Offer: {buyer_price_str}
Buyer: {buyer_dialogue}

Seller Response: {seller_dialogue}

Task: Based on the seller's response and the negotiation context, predict the seller's target price and likely next action.

Prediction:"""

    # Completion: ground truth from the transcript.
    # Prefer the seller's own observed floor; fall back to the scenario's stated
    # seller target only before the seller has quoted any price. seller_profile
    # must describe the SELLER — if it carries role="buyer" the collection script
    # is stale and this label is the buyer's own target (see git history).
    # Preference order:
    #   1. the floor the seller reached over the whole episode (a real forecast
    #      target: strictly below the current quote ~48% of the time)
    #   2. the floor within this realization (weaker — a realization ends at the
    #      turn being labelled, so it equals the latest quote ~81% of the time)
    #   3. the scenario's stated seller target, for data collected before (1)
    #      was recorded
    seller_target = seller_profile.get("observed_seller_floor")
    if seller_target is None:
        seller_target = _observed_seller_floor((utterance.get("realizations") or [[]])[0])
    if seller_target is None:
        seller_target = seller_profile.get("target_price")
    target_str = f"${seller_target:.0f}" if seller_target is not None else "Unknown"
    price_str = f"${seller_price:.0f}" if seller_price is not None else "N/A"
    big_five_personality = seller_profile.get("big_five_personality", "")
    decision_making_style = seller_profile.get("decision_making_style", "")

    completion = f"""
- Seller Target Price: {target_str}
- Seller Action: {seller_action}
- Seller Counter Price: {price_str}
- Big Five Personality: {big_five_personality}
- Decision Making Style: {decision_making_style}"""

    return {"text": prompt + completion}


def format_customer_example(utterance: Dict[str, Any], domain: str = "cb") -> Optional[Dict[str, Any]]:
    """Format a seller utterance for analyzer SFT training."""
    return format_cb_example(utterance)


def prepare_datasets(
    data_dir: str,
    train_test_split: float = 0.9,
    max_samples: int = None,
    seed: int = 42,
    domain: str = None,
) -> tuple[Dataset, Dataset]:
    """
    Prepare training and validation datasets.

    Args:
        data_dir: Directory with analyzer data files
        train_test_split: Fraction of data for training
        max_samples: Maximum samples to load
        seed: Random seed for shuffling
        domain: "cb" or "a2a" (auto-detected if None)

    Returns:
        (train_dataset, eval_dataset)
    """
    # Auto-detect domain if not specified
    if domain is None:
        domain = detect_domain_from_data(data_dir)
    print(f"Domain: {domain}")

    raw_utterances = load_customer_data_from_directory(data_dir, max_samples)
    formatted_examples = [format_customer_example(utt, domain=domain) for utt in raw_utterances]

    # Filter out None examples (invalid data)
    formatted_examples = [ex for ex in formatted_examples if ex is not None]

    # Create dataset
    dataset = Dataset.from_list(formatted_examples)

    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    split_idx = int(len(dataset) * train_test_split)

    train_dataset = dataset.select(range(split_idx))
    eval_dataset = dataset.select(range(split_idx, len(dataset)))

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    print(dataset[0])

    return train_dataset, eval_dataset


def load_model_and_tokenizer(model_config: Dict[str, Any], lora_config: Dict[str, Any]):
    """
    Load base model and tokenizer with quantization.

    Args:
        model_config: Model configuration dict

    Returns:
        (model, tokenizer)
    """
    model_name = model_config["base_model"]
    quantization = model_config.get("quantization", None)
    lora_rank = lora_config["r"]
    print(f"Loading model: {model_name}")
    print(f"Quantization: {quantization}")

    # Load tokenizer
    if FastLanguageModel is None:
        raise ImportError(
            "unsloth is required for training. Install the extra:\n"
            "    pip install -e '.[training]'"
        )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,   # Context length - can be longer, but uses more memory
        load_in_4bit = True,     # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = False, # We have full finetuning now!
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank*2, # *2 speeds up training
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = 3407,
    )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train analyzer model for seller response prediction")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./cb_dpo_train/analyzer",
        help="Directory containing seller_utterances JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/analyzer",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=TRAINING_MODEL_CONFIG["base_model"],
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=SFT_TRAINING_CONFIG["num_train_epochs"],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SFT_TRAINING_CONFIG["per_device_train_batch_size"],
        help="Training batch size per device"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=SFT_TRAINING_CONFIG["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)"
    )
    parser.add_argument(
        "--use-unsloth",
        action="store_true",
        help="Use unsloth for faster training"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["cb", "a2a"],
        default=None,
        help="Negotiation domain (auto-detected from data if omitted)"
    )

    args = parser.parse_args()

    # Update configs with CLI args
    model_config = TRAINING_MODEL_CONFIG.copy()
    model_config["base_model"] = args.model_name

    training_config = SFT_TRAINING_CONFIG.copy()
    training_config["output_dir"] = args.output_dir
    training_config["num_train_epochs"] = args.epochs
    training_config["per_device_train_batch_size"] = args.batch_size
    training_config["learning_rate"] = args.learning_rate

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config, LORA_CONFIG)


    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(
        data_dir=args.data_dir,
        train_test_split=DATA_CONFIG["train_test_split"],
        max_samples=args.max_samples,
        seed=DATA_CONFIG["seed"],
        domain=args.domain,
    )


    # Create training arguments
    training_args = SFTConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        warmup_steps=training_config["warmup_steps"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        eval_steps=training_config["eval_steps"],
        bf16=training_config["bf16"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        optim=training_config["optim"],
        remove_unused_columns=training_config["remove_unused_columns"],
        report_to=training_config["report_to"],
    )

    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,
    )

    print("\n" + "=" * 80)
    domain_desc = {
        "cb": "CB seller target price and action prediction",
        "a2a": "A2A seller target price and action prediction",
    }.get(args.domain or "", "opponent characteristic prediction")
    print(f"Starting SFT training for analyzer model ({domain_desc})")
    print("=" * 80)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    print(f"Epochs: {training_config['num_train_epochs']}")
    print(f"Batch size: {training_config['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"Output directory: {training_config['output_dir']}")
    print("=" * 80 + "\n")

    # Train
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(training_config["output_dir"])
    tokenizer.save_pretrained(training_config["output_dir"])

    print("\nTraining complete!")
    print(f"Model saved to: {training_config['output_dir']}")


if __name__ == "__main__":
    main()
