"""
Training configuration for negotiation models.
"""

# Model Configuration
TRAINING_MODEL_CONFIG = {
    "base_model": "Qwen/Qwen3-14B",  # Base model for training
    "quantization": "4bit",  # Options: None, "4bit", "8bit"
    "device_map": "auto",
    "max_seq_length": 1024,  # Maximum sequence length for training
}

# LoRA Configuration
LORA_CONFIG = {
    "r": 32,  # LoRA rank
    "lora_alpha": 32,  # LoRA alpha parameter
    "lora_dropout": 0.05,  # Dropout probability for LoRA layers
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Modules to apply LoRA
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# DPO Training Configuration (for actor_training.py)
DPO_TRAINING_CONFIG = {
    "output_dir": "./checkpoints/actor",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,  # Effective batch size = 2 * 8 = 16
    "learning_rate": 5e-6,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 3,
    "eval_steps": 250,
    "bf16": False,  # Use bfloat16 for training (requires Ampere GPU or newer)
    "fp16": True,  # Use fp16 if bf16 is not available
    "gradient_checkpointing": True,  # Enable gradient checkpointing to save memory
    "optim": "adamw_8bit",  # Use 8-bit AdamW optimizer
    "remove_unused_columns": False,
    "max_prompt_length": 1024,  # Maximum length for prompts
    "max_length": 2048,  # Maximum total length
    "beta": 0.1,  # DPO beta parameter (temperature for preference learning)
    "report_to": "none",  # Disable wandb/tensorboard
}

# SFT Training Configuration (for analyzer_training.py)
SFT_TRAINING_CONFIG = {
    "output_dir": "./checkpoints/analyzer",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,  # Effective batch size = 2 * 4 = 8
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 3,
    "eval_steps": 250,
    "bf16": False,
    "fp16": True,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",
    "remove_unused_columns": False,
    "max_prompt_length": 1024,
    "max_length": 2048,
    "report_to": "none",
}


# Data Processing Configuration
DATA_CONFIG = {
    "train_test_split": 0.9,  # 90% training, 10% validation
    "shuffle": True,
    "seed": 42,
    "max_samples": None,  # None = use all data, or set a number for testing
}
