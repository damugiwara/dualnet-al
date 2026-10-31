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

## Results

CIFAR-10-LT (imbalance 100, 12,406-image pool), 500 → 2,500 labels in 8 rounds of 250, 3 seeds.
Query Network meta-trained on CIFAR-100-LT, then frozen. Mean ± std at 2,500 labels:

| Strategy | Accuracy | Macro-F1 | Tail-class F1 | Label imbalance (max/min) | AULC |
|---|---|---|---|---|---|
| Random | 48.7 ± 0.6% | 0.442 ± 0.007 | 0.236 ± 0.019 | 146.4 ± 19.2 | 0.428 ± 0.005 |
| Entropy | 57.1 ± 0.5% | 0.536 ± 0.011 | 0.339 ± 0.036 | 51.4 ± 13.7 | 0.485 ± 0.005 |
| Core-Set | 55.2 ± 0.8% | 0.520 ± 0.012 | 0.354 ± 0.038 | 51.0 ± 19.0 | 0.469 ± 0.010 |
| **DualNet-AL** | **58.0 ± 0.5%** | **0.552 ± 0.012** | **0.388 ± 0.043** | **28.5 ± 3.2** | **0.489 ± 0.011** |

- Clear gain: the labelled pool is about half as imbalanced as with the best baseline (28.5 vs ~51).
- Accuracy, macro-F1, tail-class F1 and AULC are best for DualNet-AL, but the margins over entropy are within about 2 std; more seeds are needed to confirm them.
- Full run: 59 min on an RTX 4060 Laptop GPU. Raw data: [`results/cifar10-lt100_20261031-134210.json`](results/cifar10-lt100_20261031-134210.json), log: [`results/full_run.log`](results/full_run.log), trained Query Network: [`results/qnet.pt`](results/qnet.pt).

## Not built yet

- Learning Loss baseline (needs a loss-prediction head on the Task Network)
- ResNet-18 backbone
- In-batch diversity for the Query Network's top-k picks
- Measured-benefit meta-target (macro-F1 gain from short retrains) instead of the loss x rarity proxy
