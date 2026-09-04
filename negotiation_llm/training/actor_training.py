"""
Actor model training for negotiation strategy selection.

Trains a model to select optimal negotiation strategies based on MCTS-collected
DPO preference pairs.

Example data directory: ./cb_dpo_train (produced by collect_dpo.py)
Data format: strategy_dpo JSON files with chosen/rejected strategy pairs
"""
try:
    from unsloth import FastLanguageModel
except ImportError:  # keep --help usable without the training extra
    FastLanguageModel = None
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer


from negotiation_llm.training.config import (
    TRAINING_MODEL_CONFIG,
    LORA_CONFIG,
    DPO_TRAINING_CONFIG,
    DATA_CONFIG,
)


def detect_domain(data_dir: Path) -> str:
    """Detect domain from the *_<domain>_dpo_*.json naming convention."""
    found = {d for d in ("cb", "a2a") if list(data_dir.glob(f"*_{d}_dpo_*.json"))}
    if len(found) == 1:
        return found.pop()
    if len(found) > 1:
        print(f"Warning: {sorted(found)} DPO data mixed in one directory; pass --domain")
    return "unknown"


def load_dpo_data_from_directory(data_dir: str, max_samples: int = None) -> tuple[List[Dict[str, Any]], str]:
    """
    Load all DPO training data from directory.

    Args:
        data_dir: Directory containing DPO JSON files
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        Tuple of (examples list, detected domain string)
    """
    data_dir = Path(data_dir)
    all_examples = []

    domain = detect_domain(data_dir)

    json_files = sorted(
        list(data_dir.glob("*_cb_dpo_*.json")) + list(data_dir.glob("*_a2a_dpo_*.json"))
    )

    print(f"Found {len(json_files)} data files in {data_dir} (domain: {domain})")

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        examples = data.get("examples", [])
        all_examples.extend(examples)

        if max_samples and len(all_examples) >= max_samples:
            all_examples = all_examples[:max_samples]
            break

    print(f"Loaded {len(all_examples)} DPO examples")
    return all_examples, domain


def _format_cb_response(data: Dict[str, Any]) -> str:
    """
    Format a CB domain response: " [strategy]dialogue".

    The leading space matters. The prompt ends at "Strategy:", and
    MCTSNegotiator scores the phrase f" [{strategy.value}]" — with a space —
    when it computes strategy priors. Tokenizers attach a leading space to the
    following token, so a completion starting "[" does not share a first token
    with the phrase " [", and DPO would be shifting the probability of a
    sequence the search never scores.
    """
    strategy = data.get("strategy", "unknown")
    dialogue = data.get("dialogue", "")
    return f" [{strategy}]{dialogue}"


def format_dpo_example(example: Dict[str, Any], domain: str = "cb") -> Dict[str, str]:
    """
    Format a DPO example for training.

    Args:
        example: Raw DPO example with prompt, chosen, rejected
        domain: "cb" or "a2a"

    Returns:
        Formatted example with prompt, chosen, rejected strings
    """
    prompt = example["prompt"]
    formatter = _format_cb_response

    chosen_data = example["chosen"]
    rejected_data = example["rejected"]

    return {
        "prompt": prompt,
        "chosen": formatter(chosen_data),
        "rejected": formatter(rejected_data),
        "chosen_reward": chosen_data.get("avg_reward", 0.0),
        "rejected_reward": rejected_data.get("avg_reward", 0.0),
    }


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
        data_dir: Directory with DPO data files
        train_test_split: Fraction of data for training
        max_samples: Maximum samples to load
        seed: Random seed for shuffling
        domain: "cb" or "a2a" (auto-detected if None)

    Returns:
        (train_dataset, eval_dataset)
    """
    # Load raw data
    raw_examples, detected_domain = load_dpo_data_from_directory(data_dir, max_samples)
    domain = domain or detected_domain

    # Format examples
    formatted_examples = [format_dpo_example(ex, domain=domain) for ex in raw_examples]

    # Create dataset
    dataset = Dataset.from_list(formatted_examples)

    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    split_idx = int(len(dataset) * train_test_split)

    train_dataset = dataset.select(range(split_idx))
    eval_dataset = dataset.select(range(split_idx, len(dataset)))

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")

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
        lora_alpha = lora_config["lora_alpha"], # *2 speeds up training
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = 3407,
    )
    return model, tokenizer



def main():
    parser = argparse.ArgumentParser(description="Train actor model for strategy selection")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./cb_dpo_train",
        help="Directory containing strategy_dpo JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/actor",
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
        default=DPO_TRAINING_CONFIG["num_train_epochs"],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DPO_TRAINING_CONFIG["per_device_train_batch_size"],
        help="Training batch size per device"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DPO_TRAINING_CONFIG["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["cb", "a2a"],
        default=None,
        help="Negotiation domain (auto-detected from filenames if omitted)"
    )

    args = parser.parse_args()

    # Update configs with CLI args
    model_config = TRAINING_MODEL_CONFIG.copy()
    model_config["base_model"] = args.model_name

    training_config = DPO_TRAINING_CONFIG.copy()
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

    # Create DPO config
    dpo_config = DPOConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
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
        max_prompt_length=training_config["max_prompt_length"],
        max_length=training_config["max_length"],
        beta=training_config["beta"],
        report_to=training_config["report_to"],
    )

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    domain_label = {"cb": "CB buyer", "a2a": "A2A buyer"}.get(args.domain or "cb", args.domain or "cb")
    print("\n" + "=" * 80)
    print(f"Starting DPO training for actor model ({domain_label} strategy selection)")
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
