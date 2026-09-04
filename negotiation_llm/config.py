"""
Default configuration for negotiation runs.

Per-domain and per-method overrides come from the CLI; these are the shared
baselines. Anything dataset-specific belongs in the domain package, not here.
"""

# Model Configuration
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-14B",
    "quantization": "4bit",  # Options: None, "4bit", "8bit"
    "device_map": "cuda:0",
    "max_new_tokens": 512,
    "temperature": 1.0,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "gpu_memory_utilization": 0.7,  # Reduced to leave memory for prompt_logprobs
    "tensor_parallel_size": 1,
}

# Negotiation Settings
NEGOTIATION_CONFIG = {
    "max_rounds": 10,  # Maximum negotiation rounds before deadlock
    "convergence_threshold": 0.01,
    "save_negotiations": True,
    "negotiations_dir": "negotiations",
}

# MCTS Configuration
MCTS_CONFIG = {
    "exploration_weight": 1.4,  # PUCT exploration constant (c_puct)
    "num_simulations": 50,  # Number of MCTS simulations per move
    # Cap on strategies expanded per node; None uses every strategy the domain
    # offers. This was 10 while both CB and A2A define 11 dialogue acts, which
    # silently made the 11th (DISAGREE) unreachable by the search — an
    # off-by-one, not a compute budget. Setting it below the strategy count
    # silently drops the tail of the list.
    "max_children": None,
    "heuristic_samples": 3,  # Samples for acceptance heuristic (legacy)
    "use_realizations": True,  # Enable realization caching (GDPZero-style)
    "max_realizations": 1,  # Offer variants cached per strategy
    "use_response_selection": True,  # Select best realization after search
    "collect_dpo_data": False,  # Enable automatic DPO training data collection
    "dpo_output_dir": "dpo_data",  # Directory for DPO training data
}
