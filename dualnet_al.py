"""DualNet-AL prototype: bi-network meta-learning for class-balanced multi-class active learning.

Task Network : small CNN, retrained from scratch on the labelled pool every round.
Query Network: MLP meta-learner that scores unlabelled samples from dataset-agnostic features
               (uncertainty, distance to the labelled set, expected class deficit, round context).

Meta-training runs AL episodes on a SOURCE dataset (pool labels known) and regresses the
within-round rank of
    benefit_i = CE(p_i, y_i) * (1 + deficit[y_i])
i.e. "how wrong the Task Network is on it" x "how under-represented its true class is".
Evaluation runs on a different TARGET dataset with the Query Network frozen, so no target
labels leak into the scorer.

    python dualnet_al.py --smoke     # ~1-2 min end-to-end self-check
    python dualnet_al.py             # meta-train on CIFAR-100-LT, evaluate on CIFAR-10-LT
    streamlit run dashboard.py       # view results
"""
import argparse
import io
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent
HF = 'https://huggingface.co/datasets/uoft-cs/{}/resolve/main/{}/{}-00000-of-00001.parquet'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = torch.tensor([0.507, 0.487, 0.441], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.267, 0.256, 0.276], device=DEV).view(1, 3, 1, 1)
STRATEGIES = ['random', 'entropy', 'coreset', 'dualnet']
NF = 7  # Query Network input features


# ------------------------------------------------------------------ data
def fetch(name, split):
    """(x uint8 NHWC, y) from the Hugging Face mirror of CIFAR (the Toronto host is very slow), cached as .npz."""
    cache = ROOT / 'data' / f'{name}_{split}.npz'
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        tmp = cache.with_suffix('.parquet')
        print(f'downloading {name} {split} ...', flush=True)
        urllib.request.urlretrieve(HF.format(name, 'plain_text' if name == 'cifar10' else name, split), tmp)
        df = pd.read_parquet(tmp)
        x = np.stack([np.array(Image.open(io.BytesIO(img['bytes'])).convert('RGB')) for img in df['img']])
        np.savez(cache, x=x, y=df['label' if name == 'cifar10' else 'fine_label'].to_numpy())
        tmp.unlink()
    d = np.load(cache)
    return d['x'], d['y']


def load(name, imbalance, test_size, seed=0):
    """CIFAR-10/100 held on DEV as uint8. imbalance > 1 makes the train pool long-tailed; test stays balanced."""
    x, y = fetch(name, 'train')
    xt, yt = fetch(name, 'test')
    C = int(y.max()) + 1
    rng = np.random.default_rng(seed)
    if imbalance > 1:  # exponential profile, as in CIFAR-LT (Cao et al., 2019)
        n_max = np.bincount(y).max()
        idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:int(n_max * imbalance ** (-c / (C - 1)))]
                              for c in range(C)])
        x, y = x[idx], y[idx]
    t = rng.permutation(len(yt))[:test_size]
    to_dev = lambda a: torch.tensor(a, device=DEV).permute(0, 3, 1, 2).contiguous()
    return dict(name=f'{name}-lt{imbalance:g}' if imbalance > 1 else name, C=C,
                x=to_dev(x), y=torch.tensor(y, device=DEV), xt=to_dev(xt[t]), yt=yt[t])


def norm(x):
    return (x.float() / 255 - MEAN) / STD


def augment(x):
    x = torch.where(torch.rand(len(x), 1, 1, 1, device=x.device) < 0.5, x.flip(3), x)
    i, j = np.random.randint(0, 9, 2)  # ponytail: one crop offset per batch; per-sample crops if accuracy needs it
    return F.pad(x, (4, 4, 4, 4), mode='reflect')[:, :, i:i + 32, j:j + 32]


# ------------------------------------------------------------------ networks
class TaskNet(nn.Module):
    def __init__(self, C):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                         nn.MaxPool2d(2))
        self.body = nn.Sequential(blk(3, 64), blk(64, 128), blk(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(256, C)

    def forward(self, x):
        e = self.body(x)
        return self.head(e), e


class QueryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(NF, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, f):
        return self.mlp(f).squeeze(-1)


def train_task(x, y, C, steps, bs=128):
    net = TaskNet(C).to(DEV)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.05, total_steps=steps)
    net.train()
    for _ in range(steps):
        i = torch.randint(len(x), (bs,), device=DEV)
        loss = F.cross_entropy(net(augment(norm(x[i])))[0], y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return net


@torch.no_grad()
def infer(net, x, bs=2048):
    net.eval()
    out = [net(norm(x[i:i + bs])) for i in range(0, len(x), bs)]
    return torch.cat([o[0].softmax(1) for o in out]), torch.cat([o[1] for o in out])


def evaluate(net, ds):
    pred = infer(net, ds['xt'])[0].argmax(1).cpu().numpy()
    f1 = f1_score(ds['yt'], pred, labels=list(range(ds['C'])), average=None, zero_division=0)
    return float((pred == ds['yt']).mean()), f1


# ------------------------------------------------------------------ query features and strategies
def min_dist(a, b, chunk=4096):
    return torch.cat([torch.cdist(a[i:i + chunk], b).min(1).values for i in range(0, len(a), chunk)])


def deficit(counts):
    """Per-class shortfall against a uniform share: 1 = no labels yet, 0 = at or above its share."""
    C = len(counts)
    return (1 / C - counts / counts.sum()).clamp(min=0) * C


def features(probs, emb, lab, counts, frac):
    """Dataset-agnostic features for every unlabelled sample, all roughly in [0, 1]."""
    C = probs.shape[1]
    p = probs[~lab]
    top2 = p.topk(2, 1).values
    share = counts / counts.sum()
    balance = -(share * share.clamp(min=1e-12).log()).sum() / math.log(C)  # 1 = perfectly balanced pool
    d = min_dist(emb[~lab], emb[lab])
    return torch.cat([torch.stack([
        -(p * p.clamp(min=1e-12).log()).sum(1) / math.log(C),  # normalised entropy
        1 - (top2[:, 0] - top2[:, 1]),                          # margin uncertainty
        1 - top2[:, 0],                                         # least confidence
        p @ deficit(counts),                                    # expected class deficit (soft pseudo-label)
        d / d.mean(),                                           # distance to labelled set, relative
    ], 1), torch.tensor([frac, float(balance)], device=DEV).expand(len(p), 2)], 1)


def kcenter(emb, unl, lab, k):
    """Core-Set greedy k-center (Sener & Savarese, 2018)."""
    eu = emb[unl]
    d = min_dist(eu, emb[lab])
    picks = []
    for _ in range(k):
        i = int(d.argmax())
        picks.append(i)
        d = torch.minimum(d, (eu - eu[i]).norm(dim=1))
    return unl[picks]


@torch.no_grad()
def select(strategy, k, probs, emb, lab, counts, frac, qnet):
    unl = (~lab).nonzero().squeeze(1)
    if strategy == 'random':
        return unl[torch.randperm(len(unl), device=DEV)[:k]]
    if strategy == 'entropy':
        p = probs[unl]
        return unl[(-(p * p.clamp(min=1e-12).log()).sum(1)).topk(k).indices]
    if strategy == 'coreset':
        return kcenter(emb, unl, lab.nonzero().squeeze(1), k)
    # ponytail: top-k of independent scores; add in-batch diversity if picks turn out redundant
    return unl[qnet(features(probs, emb, lab, counts, frac)).topk(k).indices]


# ------------------------------------------------------------------ active-learning loop
def run_al(ds, strategy, seed, args, qnet=None, on_round=None):
    """One pool-based batch AL run. on_round(...) fires before each selection (meta-training hook)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, C = len(ds['y']), ds['C']
    lab = torch.zeros(N, dtype=torch.bool, device=DEV)
    lab[torch.randperm(N, device=DEV)[:args.init]] = True  # same seed -> same initial set for every strategy
    pool_counts = torch.bincount(ds['y'], minlength=C).float()
    tail = (pool_counts <= pool_counts.quantile(0.3)).cpu().numpy()  # rarest 30% of classes
    hist = []
    for r in range(args.rounds + 1):
        counts = torch.bincount(ds['y'][lab], minlength=C).float()
        net = train_task(ds['x'][lab], ds['y'][lab], C, args.steps)
        acc, f1 = evaluate(net, ds)
        h = dict(round=r, n=int(lab.sum()), acc=acc, macro_f1=float(f1.mean()), tail_f1=float(f1[tail].mean()),
                 imbalance=float(counts.max() / counts.min().clamp(min=1)),
                 f1=f1.round(4).tolist(), counts=counts.int().tolist())
        hist.append(h)
        print(f"  {ds['name']:12} {strategy:8} seed={seed} r={r} n={h['n']:5} acc={acc:.3f} "
              f"tailF1={h['tail_f1']:.3f} imb={h['imbalance']:.1f}", flush=True)
        if r == args.rounds:
            break
        probs, emb = infer(net, ds['x'])
        frac = r / args.rounds
        if on_round:
            on_round(probs, emb, lab, counts, frac)
        pick = select(strategy, args.batch, probs, emb, lab, counts, frac, qnet)
        assert len(pick) == args.batch and not lab[pick].any(), 'selection must be k new unlabelled samples'
        lab[pick] = True
    return hist


def meta_train(src, args):
    """Episodic meta-training of the Query Network on a source dataset whose pool labels are known."""
    qnet = QueryNet().to(DEV)
    opt = torch.optim.Adam(qnet.parameters(), 1e-3)
    X, T = [], []

    def on_round(probs, emb, lab, counts, frac):
        y = src['y'][~lab]
        # ponytail: proxy benefit (loss x class rarity); upgrade = measured macro-F1 gain from short retrains
        benefit = -probs[~lab].gather(1, y[:, None]).squeeze(1).clamp(min=1e-12).log() * (1 + deficit(counts)[y])
        X.append(features(probs, emb, lab, counts, frac))
        T.append(benefit.argsort().argsort().float() / (len(benefit) - 1))  # within-round rank in [0, 1]
        fx, ft = torch.cat(X), torch.cat(T)
        qnet.train()
        for _ in range(args.meta_steps):
            i = torch.randint(len(fx), (512,), device=DEV)
            loss = F.mse_loss(qnet(fx[i]), ft[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        qnet.eval()
        print(f'    query-net mse={loss.item():.4f} (constant-guess baseline 0.083)', flush=True)

    for ep in range(args.meta_episodes):
        print(f"meta-episode {ep + 1}/{args.meta_episodes} on {src['name']}")
        run_al(src, 'dualnet', 10_000 + ep, args, qnet, on_round)
    return qnet


def summarise(results):
    out = {}
    for s, seeds in results.items():
        last = lambda m: [h[-1][m] for h in seeds]
        aulc = [np.trapezoid([r['acc'] for r in h], [r['n'] for r in h]) / (h[-1]['n'] - h[0]['n']) for h in seeds]
        out[s] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in dict(
            final_acc=last('acc'), final_macro_f1=last('macro_f1'), final_tail_f1=last('tail_f1'),
            final_imbalance=last('imbalance'), aulc=aulc).items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='cifar10', choices=['cifar10', 'cifar100'])
    ap.add_argument('--imbalance', type=float, default=100, help='max/min class ratio of the pool (1 = balanced)')
    ap.add_argument('--meta-data', default='cifar100', choices=['cifar10', 'cifar100'])
    ap.add_argument('--meta-imbalance', type=float, default=100)
    ap.add_argument('--init', type=int, default=500, help='initial random labelled set')
    ap.add_argument('--batch', type=int, default=250, help='labels acquired per round')
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--steps', type=int, default=1200, help='Task Network SGD steps per round')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--meta-episodes', type=int, default=3)
    ap.add_argument('--meta-steps', type=int, default=300, help='Query Network updates per meta round')
    ap.add_argument('--test-size', type=int, default=10000)
    ap.add_argument('--strategies', default=','.join(STRATEGIES))
    ap.add_argument('--qnet', help='load a meta-trained Query Network (.pt) instead of meta-training')
    ap.add_argument('--smoke', action='store_true', help='tiny end-to-end run as a self-check')
    args = ap.parse_args()
    if args.smoke:
        args.init, args.batch, args.rounds, args.steps, args.seeds = 50, 50, 2, 20, 1
        args.meta_episodes, args.meta_steps, args.test_size = 1, 20, 500
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    strategies = args.strategies.split(',')
    t0 = time.time()

    qnet = None
    if 'dualnet' in strategies:
        qnet = QueryNet().to(DEV)
        if args.qnet:
            qnet.load_state_dict(torch.load(args.qnet, map_location=DEV))
        else:
            if args.meta_data == args.data:
                print('WARNING: meta-training and evaluating on the same dataset leaks labels into the Query Network')
            qnet = meta_train(load(args.meta_data, args.meta_imbalance, args.test_size), args)
            torch.save(qnet.state_dict(), out / 'qnet.pt')
        qnet.eval()

    ds = load(args.data, args.imbalance, args.test_size)
    print(f"target pool {ds['name']}: {len(ds['y'])} images, {ds['C']} classes, device={DEV}")
    results = {s: [run_al(ds, s, seed, args, qnet) for seed in range(args.seeds)] for s in strategies}
    summary = summarise(results)

    name = f"{'smoke_' if args.smoke else ''}{ds['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    (out / f'{name}.json').write_text(json.dumps(dict(config=vars(args), summary=summary, results=results)))
    metrics = list(next(iter(summary.values())))
    print(f"\n{'strategy':10}" + ''.join(f'{m:>20}' for m in metrics))
    for s, d in summary.items():
        print(f'{s:10}' + ''.join(f'{d[m][0]:>13.3f} +-{d[m][1]:.3f}' for m in metrics))
    print(f'\nsaved results/{name}.json  ({(time.time() - t0) / 60:.1f} min)')

    if args.smoke:
        assert all(len(h) == args.rounds + 1 and h[-1]['n'] == args.init + args.batch * args.rounds
                   for hs in results.values() for h in hs)
        print('smoke OK')


if __name__ == '__main__':
    main()
