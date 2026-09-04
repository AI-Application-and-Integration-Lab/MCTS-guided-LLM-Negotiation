# MCTS-guided-LLM-Negotiation

This repository contains the implementation of Online Strategic Reasoning for LLM-Based Negotiation via Opponent-Aware Monte Carlo Tree Search.

A buyer agent negotiates the price of an item against an LLM-simulated seller.
Instead of sampling utterances directly, the agent runs MCTS over **dialogue
acts** — propose a counter, use comparatives, ask a question — and an LLM
realizes the selected act into the actual sentence and price. A second LoRA
adapter, the **analyzer**, reads the transcript after every exchange and
predicts the seller's reservation price, next move and persona; that estimate
becomes the opponent model inside subsequent rollouts.

---

## Install

```bash
git clone https://github.com/AI-Application-and-Integration-Lab/MCTS-guided-LLM-Negotiation.git
cd MCTS-guided-LLM-Negotiation
pip install -r requirements.txt
```

## Download the datasets

**CraigslistBargain** — He et al. 2018.

```python
import pandas as pd
from huggingface_hub import hf_hub_download

for split in ("train", "test", "validation"):
    path = hf_hub_download(
        "stanfordnlp/craigslist_bargains",
        f"default/craigslist_bargains-{split}.parquet",
        repo_type="dataset", revision="refs/convert/parquet",
    )
    pd.read_parquet(path).to_csv(f"CB/{split}.csv", index=False)
```

**A2A-NT** — the agent-to-agent negotiation product catalogue from Zhu et al.
2025 ([paper](https://arxiv.org/abs/2506.00073) ·
[code](https://github.com/ShenzheZhu/A2A-NT)).

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "Chouoftears/Agent2Agent-Negotiation-in-Consumer-Setting-Dataset",
    repo_type="dataset", allow_patterns=["*.json"],
    local_dir="Agent2Agent-Negotiation-in-Consumer-Setting-Dataset",
)
```

## Get the adapters

The two LoRA adapters live on the Hugging Face Hub:
[**Jason-Huang/MCTS-guided-LLM-Negotiation**](https://huggingface.co/Jason-Huang/MCTS-guided-LLM-Negotiation).

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "Jason-Huang/MCTS-guided-LLM-Negotiation",
    local_dir="checkpoints",
)
```

That writes `checkpoints/actor` and `checkpoints/analyzer` — the defaults for
`--actor-adapter` and `--analyzer-adapter`, so a real run needs no extra flags.
The runners fail loudly if either path is missing rather than silently falling
back to the base model.

## Run

With a GPU and the adapters downloaded to `checkpoints/`:

```bash
# short smoke run on 2 scenarios
python -m experiments.cb.run_negotiation_cb_mcts \
    --data-file CB/test.csv --end-index 2 --num-simulations 3 --verbose
```

Full runs:

```bash
python -m experiments.cb.run_negotiation_cb_mcts \
    --data-file CB/test.csv --end-index 300 --seed 42

python -m experiments.a2a.run_negotiation_a2a_mcts \
    --data-file Agent2Agent-Negotiation-in-Consumer-Setting-Dataset/products.json \
    --end-index 100 --seed 42 --buyer-target-ratio 0.2
```

Results are written to `results/<domain>_<method>/<domain>_<method>_results_<timestamp>.json`.

Run everything from the repository root. Default data and checkpoint paths are relative to it.


## Training

The actor (DPO) and analyzer (SFT) adapters are trained on data generated based on MCTS rollouts.

```bash
python -m negotiation_llm.data.collect_dpo --domain cb --data-file CB/train.csv --output-dir cb_dpo_train
python -m negotiation_llm.training.actor_training    --data-dir cb_dpo_train --domain cb
python -m negotiation_llm.training.analyzer_training --data-dir cb_dpo_train/analyzer --domain cb
```

Collection works for either domain.

## Layout

```
negotiation_llm/
├── config.py          MODEL_CONFIG, MCTS_CONFIG, NEGOTIATION_CONFIG
├── models.py          BaseModel + HuggingFace / vLLM / OpenAI / Mock backends
├── mcts/              search core — domain-agnostic
│   ├── domain.py      NegotiationDomain ABC (the search contract)
│   ├── node.py        PUCT node
│   └── negotiator.py  MCTSNegotiator
├── domains/           cb, a2a — each a NegotiationDomain + a DomainSpec
├── training/          actor (DPO) and analyzer (SFT)
└── data/              DPO collection from rollouts
experiments/{cb,a2a}/  one runner per method
```

## Data & licences

Code is MIT ([LICENSE](LICENSE)). The datasets are not ours to relicense:

- **CraigslistBargain** — He, Chen, Balakrishnan & Liang, *Decoupling Strategy
  and Generation in Negotiation Dialogues*, EMNLP 2018.
  [arXiv:1808.09637](https://arxiv.org/abs/1808.09637) ·
  [project](https://stanfordnlp.github.io/cocoa/) ·
  [`stanfordnlp/craigslist_bargains`](https://huggingface.co/datasets/stanfordnlp/craigslist_bargains)
- **A2A-NT** — Zhu, Sun, Nian, South, Pentland & Pei, *The Automated but Risky
  Game: Modeling and Benchmarking Agent-to-Agent Negotiations and Transactions
  in Consumer Markets*, 2025.
  [arXiv:2506.00073](https://arxiv.org/abs/2506.00073) ·
  [code](https://github.com/ShenzheZhu/A2A-NT) ·
  [`Chouoftears/Agent2Agent-Negotiation-in-Consumer-Setting-Dataset`](https://huggingface.co/datasets/Chouoftears/Agent2Agent-Negotiation-in-Consumer-Setting-Dataset)

## Citation

```bibtex
@software{mcts_guided_llm_negotiation,
  title  = {MCTS-guided LLM Negotiation: Strategy-level search with a learned opponent model},
  year   = {2026},
  url    = {https://github.com/AI-Application-and-Integration-Lab/MCTS-guided-LLM-Negotiation}
}
```

If you use the evaluation datasets, cite them too:

```bibtex
@misc{zhu2025automatedriskygamemodeling,
      title={The Automated but Risky Game: Modeling and Benchmarking Agent-to-Agent Negotiations and Transactions in Consumer Markets},
      author={Shenzhe Zhu and Jiao Sun and Yi Nian and Tobin South and Alex Pentland and Jiaxin Pei},
      year={2025},
      eprint={2506.00073},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.00073},
}

@inproceedings{he-etal-2018-decoupling,
    title = "Decoupling Strategy and Generation in Negotiation Dialogues",
    author = "He, He  and
      Chen, Derek  and
      Balakrishnan, Anusha  and
      Liang, Percy",
    editor = "Riloff, Ellen  and
      Chiang, David  and
      Hockenmaier, Julia  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1256/",
    doi = "10.18653/v1/D18-1256",
    pages = "2333--2343",
}
```
