# DualNet-AL prototype

Bi-network meta-learning for class-balanced multi-class active learning (Phase II prototype).

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt
```

## Run

```
.venv\Scripts\python dualnet_al.py --smoke     # 1-2 min end-to-end self-check
.venv\Scripts\python dualnet_al.py             # full run: meta-train on CIFAR-100-LT, evaluate on CIFAR-10-LT
.venv\Scripts\streamlit run dashboard.py       # learning curves, per-class F1, label distribution
```

`--qnet results/qnet.pt` reuses a meta-trained Query Network. `--help` lists every option.

## How it works

| Piece | What it is |
|---|---|
| Task Network | 6-layer CNN, retrained from scratch on the labelled pool each round |
| Query Network | MLP over 7 dataset-agnostic features per unlabelled sample: normalised entropy, margin, least confidence, expected class deficit (soft pseudo-label x per-class shortfall), distance to labelled set, budget used, pool balance |
| Meta-training target | within-round rank of `CE(p, y) * (1 + deficit[y])`: how wrong the model is on a sample x how under-represented its true class is |
| Transfer protocol | Query Network meta-trained on CIFAR-100-LT, then frozen and evaluated on CIFAR-10-LT, so no target labels reach the scorer |
| Baselines | random, entropy, Core-Set (k-center greedy) |
| Metrics | accuracy, macro-F1, tail-class F1 (rarest 30% of classes), labelled-pool imbalance ratio, AULC |

## Not built yet

- Learning Loss baseline (needs a loss-prediction head on the Task Network)
- ResNet-18 backbone
- In-batch diversity for the Query Network's top-k picks
- Measured-benefit meta-target (macro-F1 gain from short retrains) instead of the loss x rarity proxy
