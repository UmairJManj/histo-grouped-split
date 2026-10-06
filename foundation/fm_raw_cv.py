"""Raw whole-patch frozen-feature CV (laptop protocol that reached 0.963 on seed-42 CV).

Per model: CLS of the whole 512 patch resized to 224 (no Macenko), L2-normalised per model,
models concatenated; StandardScaler + LogisticRegression, C by inner StratifiedKFold(3, rs=0)
on unique training patches; fit with sample_weight = n_copies; val accuracy copy-weighted.
Options: --feat cls|mean|clsmean, --views 1 (identity) or 8 (train on dihedral views, TTA = mean prob).

Usage: python fm_raw_cv.py --models phikon,phikon2 --seeds 42,43,44 [--views 8] [--tag x]
Appends rows to results/fm_raw_cv.csv (one per model-set/seed/fold).
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from fm_heads import copies, folds

HERE = Path(__file__).parent
RAW = HERE / "fm_features_raw"
OUT = HERE / "results" / "fm_raw_cv.csv"
GRID = (0.001, 0.01, 0.1, 1.0)


def load(models, feat):
    xs = []
    for m in models:
        d = np.load(RAW / f"whole_{m}.npz")
        parts = {"cls": [d["cls"]], "mean": [d["mean"]], "clsmean": [d["cls"], d["mean"]]}[feat]
        x = np.concatenate([p / np.linalg.norm(p, axis=-1, keepdims=True) for p in parts], -1)
        xs.append(x / np.linalg.norm(x, axis=-1, keepdims=True))
    return np.concatenate(xs, -1).astype(np.float32), d["group_id"], d["label"]  # (368, 8, D)


def fit(X, y, w, C):
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y, sample_weight=w)


def predict_proba(model, X):
    sc, lr = model
    return lr.predict_proba(sc.transform(X))


def train_views(X, y, w, idx, views):
    Xv = X[idx, :views].reshape(-1, X.shape[-1])
    return Xv, np.repeat(y[idx], views), np.repeat(w[idx], views)


def pick_C(X, y, w, views):
    skf = StratifiedKFold(3, shuffle=True, random_state=0)
    accs = {C: [] for C in GRID}
    for a, b in skf.split(X[:, 0], y):
        for C in GRID:
            m = fit(*train_views(X, y, w, a, views), C)
            accs[C].append((predict_proba(m, X[b, 0]).argmax(1) == y[b]).mean())
    return max(GRID, key=lambda C: np.mean(accs[C]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--feat", default="cls")
    ap.add_argument("--views", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    models = args.models.split(",")
    X, gid, y = load(models, args.feat)
    pos = {int(g): i for i, g in enumerate(gid)}
    ncp = copies()
    w = np.array([ncp[int(g)] for g in gid], dtype=float)
    new = not OUT.exists()
    fh = open(OUT, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(["tag", "models", "feat", "views", "seed", "fold", "C", "acc", "acc_unique", "f1", "acc_tta", "f1_tta"])
    for seed in map(int, args.seeds.split(",")):
        accs = []
        for k, (tr, va) in enumerate(folds(seed), 1):
            t0 = time.time()
            a = np.array([pos[g] for g in tr]); b = np.array([pos[g] for g in va])
            C = pick_C(X[a], y[a], w[a], args.views)
            m = fit(*train_views(X, y, w, a, args.views), C)
            p0 = predict_proba(m, X[b, 0])
            pt = np.mean([predict_proba(m, X[b, v]) for v in range(8)], 0)
            yt, wt = y[b], w[b]
            r0, rt = p0.argmax(1), pt.argmax(1)
            acc = np.average(r0 == yt, weights=wt)
            row = [args.tag, "+".join(models), args.feat, args.views, seed, k, C, acc, (r0 == yt).mean(),
                   f1_score(yt, r0, average="macro", sample_weight=wt), np.average(rt == yt, weights=wt),
                   f1_score(yt, rt, average="macro", sample_weight=wt)]
            wr.writerow(row); fh.flush()
            accs.append(acc)
            print(f"{'+'.join(models)} {args.feat} v{args.views} s{seed} f{k} C={C} acc={acc:.4f} tta={row[10]:.4f} ({time.time()-t0:.0f}s)", flush=True)
        print(f"== {'+'.join(models)} {args.feat} v{args.views} seed {seed}: mean acc {np.mean(accs):.4f}", flush=True)


if __name__ == "__main__":
    main()
