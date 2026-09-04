---
base_model: unsloth/qwen3-14b-unsloth-bnb-4bit
library_name: peft
model_name: analyzer
tags:
- base_model:adapter:unsloth/qwen3-14b-unsloth-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
licence: license
pipeline_tag: text-generation
---

# Model Card for analyzer

Predicts the seller's price floor, next action and persona from the transcript so far. Its estimate becomes the opponent model inside subsequent MCTS rollouts. Trained with SFT on seller turns collected from the same rollouts.

A LoRA adapter (r=32) over [unsloth/qwen3-14b-unsloth-bnb-4bit](https://huggingface.co/unsloth/qwen3-14b-unsloth-bnb-4bit), served alongside the other adapter through vLLM multi-LoRA.
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="./checkpoints/analyzer", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.18.0
- TRL: 0.24.0
- Transformers: 4.57.1
- Pytorch: 2.8.0
- Datasets: 4.3.0
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```