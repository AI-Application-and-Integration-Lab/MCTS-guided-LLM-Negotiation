---
base_model: unsloth/qwen3-14b-unsloth-bnb-4bit
library_name: peft
model_name: actor
tags:
- base_model:adapter:unsloth/qwen3-14b-unsloth-bnb-4bit
- dpo
- lora
- transformers
- trl
- unsloth
licence: license
pipeline_tag: text-generation
---

# Model Card for actor

Selects the negotiation dialogue act and writes the buyer's utterance. Trained with DPO on preference pairs harvested from MCTS rollouts, where the chosen strategy is the one with the highest search value at that turn.

A LoRA adapter (r=32) over [unsloth/qwen3-14b-unsloth-bnb-4bit](https://huggingface.co/unsloth/qwen3-14b-unsloth-bnb-4bit), served alongside the other adapter through vLLM multi-LoRA.
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="./checkpoints/actor", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with DPO, a method introduced in [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290).

### Framework versions

- PEFT 0.18.0
- TRL: 0.24.0
- Transformers: 4.57.1
- Pytorch: 2.8.0
- Datasets: 4.3.0
- Tokenizers: 0.22.1

## Citations

Cite DPO as:

```bibtex
@inproceedings{rafailov2023direct,
    title        = {{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}},
    author       = {Rafael Rafailov and Archit Sharma and Eric Mitchell and Christopher D. Manning and Stefano Ermon and Chelsea Finn},
    year         = 2023,
    booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023},
    url          = {http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html},
    editor       = {Alice Oh and Tristan Naumann and Amir Globerson and Kate Saenko and Moritz Hardt and Sergey Levine},
}
```

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