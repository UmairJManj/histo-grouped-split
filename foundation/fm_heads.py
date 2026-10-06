"""Heads on frozen foundation-model features, grouped CV (same folds as the CNN runs).

Training unit: the 315 unique pool patches (one row per MD5 group), augmented with
their 8 dihedral views. Evaluation: the held-out fold's patches, TTA = softmax mean
over the 8 dihedral views, scored copy-weighted (each patch counts n_copies times,
i.e. file-level accuracy as in the CNN runs) and also per unique patch.
Embedding per view = mean over the 6 crops (5 native 224 crops + 1 resize).

Heads per feature set: logistic regression (C by inner 3-fold CV on train patches),
linear SVM (C by inner CV), cosine kNN (k by inner CV), MLP (fixed recipe).
Fusion heads over several foundation models: HistoFormerFusion head (per-branch
projection, attention scorer, softmax-weighted sum, MLP) and an equal-weight
(mean) fusion control, both trained with the same recipe.

Usage:
  python fm_heads.py single --models a,b --seeds 42           # every model x feat x head
  python fm_heads.py fusion --models a,b,c --seeds 42 --feat clsmean
Results: results/fm_cv_<mode>.csv (one row per model/feat/head/seed/fold).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

HERE = Path(__file__).parent
FEAT = HERE / "fm_features"
OUT = HERE / "results"
N_CLASSES = 22


VIEWSETS = ("crops5", "whole", "cat", "all")


def load_bank(name, feat, viewset="all", bank_dir=None):
    """feat: cls|mean|clsmean. viewset over the 6 crops (0-4 native 224 crops, 5 = whole 512->224):
    crops5 = mean of native crops, whole = whole patch only, cat = [crops5, whole], all = mean of 6."""
    d = np.load((bank_dir or FEAT) / f"{name}.npz")
    parts = {"cls": [d["cls"]], "mean": [d["mean"]], "clsmean": [d["cls"], d["mean"]]}[feat]
    x = np.concatenate(parts, axis=-1).astype(np.float32)  # (368, 8, 6, D)
    x = {"crops5": lambda: x[:, :, :5].mean(2), "whole": lambda: x[:, :, 5],
         "cat": lambda: np.concatenate([x[:, :, :5].mean(2), x[:, :, 5]], -1), "all": lambda: x.mean(2)}[viewset]()
    return x, d["group_id"], d["label"]


def copies():
    return {int(r["group_id"]): int(r["n_copies"]) for r in csv.DictReader(open(HERE / "nuinsseg_mapping.csv"))}


def folds(seed):
    s = json.load(open(HERE / "splits.json"))["per_seed"][str(seed)]["cv"]
    return [(sorted({g for _, _, g in f["train"]}), sorted({g for _, _, g in f["val"]})) for f in s]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def inner_pick(make, grid, X, y):
    """choose a hyperparameter by inner 3-fold CV on (unique) training patches, view 0 only."""
    skf = StratifiedKFold(3, shuffle=True, random_state=0)
    best, best_acc = grid[0], -1
    for g in grid:
        accs = []
        for a, b in skf.split(X[:, 0], y):
            m = make(g).fit(X[a].reshape(-1, X.shape[-1]), np.repeat(y[a], X.shape[1]))
            accs.append((m.predict(X[b, 0]) == y[b]).mean())
        if np.mean(accs) > best_acc:
            best, best_acc = g, np.mean(accs)
    return best


def sk_head(kind, Xtr, ytr, Xev):
    """returns probs (n_ev, 8 views, C)"""
    flat = lambda X: X.reshape(-1, X.shape[-1])
    sc = StandardScaler().fit(flat(Xtr))
    Xtr = sc.transform(flat(Xtr)).reshape(Xtr.shape); Xev = sc.transform(flat(Xev)).reshape(Xev.shape)
    if kind == "logreg":
        mk = lambda c: LogisticRegression(C=c, max_iter=3000)
        p = inner_pick(mk, [0.001, 0.01, 0.1, 1.0], Xtr, ytr)
        m = mk(p).fit(flat(Xtr), np.repeat(ytr, 8))
        out = m.predict_proba(flat(Xev))
    elif kind == "linsvm":
        mk = lambda c: LinearSVC(C=c, max_iter=20000)
        p = inner_pick(mk, [0.0001, 0.001, 0.01, 0.1], Xtr, ytr)
        m = mk(p).fit(flat(Xtr), np.repeat(ytr, 8))
        out = softmax(m.decision_function(flat(Xev)))
    else:  # knn
        mk = lambda k: KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
        p = inner_pick(mk, [1, 3, 5, 9], Xtr, ytr)
        m = mk(p).fit(flat(Xtr), np.repeat(ytr, 8))
        out = m.predict_proba(flat(Xev))
    full = np.zeros((out.shape[0], N_CLASSES)); full[:, m.classes_] = out
    return full.reshape(Xev.shape[0], 8, N_CLASSES), p


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, N_CLASSES))

    def forward(self, xs):
        return self.net(xs[0]), None


class FusionHead(nn.Module):
    """HistoFormerFusion head on frozen embeddings; mode='attn' or 'mean' (equal-weight control)."""

    def __init__(self, dims, mode, d=512):
        super().__init__()
        self.mode = mode
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(k), nn.Linear(k, d), nn.LayerNorm(d), nn.GELU(),
                                                 nn.Dropout(0.2)) for k in dims])
        self.score = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.LayerNorm(d // 2), nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(d // 2, N_CLASSES))

    def forward(self, xs):
        z = torch.stack([p(x) for p, x in zip(self.proj, xs)], 1)  # (B, M, d)
        w = torch.softmax(self.score(z).squeeze(-1), 1) if self.mode == "attn" else \
            torch.full(z.shape[:2], 1.0 / z.shape[1], device=z.device)
        return self.head((z * w.unsqueeze(-1)).sum(1)), w


def torch_head(make, Xtrs, ytr, Xevs, seed, epochs=150):
    """Xtrs: list of (n, 8, D_i). Trains on all 8 views; returns probs (n_ev, 8, C) and mean weights."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu = [X.reshape(-1, X.shape[-1]).mean(0) for X in Xtrs]
    sd = [X.reshape(-1, X.shape[-1]).std(0) + 1e-6 for X in Xtrs]
    norm = lambda Xs: [torch.tensor((X - m) / s, dtype=torch.float32) for X, m, s in zip(Xs, mu, sd)]
    tr, ev = norm(Xtrs), norm(Xevs)
    n = tr[0].shape[0]
    xtr = [t.reshape(-1, t.shape[-1]) for t in tr]
    yt = torch.tensor(np.repeat(ytr, 8))
    counts = np.bincount(ytr, minlength=N_CLASSES).astype(np.float32)
    cw = torch.tensor(np.where(counts > 0, counts.sum() / (N_CLASSES * np.maximum(counts, 1)), 0.0), dtype=torch.float32)
    model = make()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            logits, _ = model([x[b] for x in xtr])
            loss = lossf(logits, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        logits, w = model([t.reshape(-1, t.shape[-1]) for t in ev])
    probs = torch.softmax(logits, 1).numpy().reshape(ev[0].shape[0], 8, N_CLASSES)
    wmean = w.mean(0).numpy().tolist() if w is not None else None
    return probs, wmean


def score(probs, y, w):
    p0, ptta = probs[:, 0].argmax(1), probs.mean(1).argmax(1)
    acc = lambda p: float(np.average(p == y, weights=w))
    return {"acc": acc(p0), "acc_tta": acc(ptta),
            "acc_unique": float((p0 == y).mean()), "acc_unique_tta": float((ptta == y).mean()),
            "f1": float(f1_score(np.repeat(y, w), np.repeat(p0, w), average="macro")),
            "f1_tta": float(f1_score(np.repeat(y, w), np.repeat(ptta, w), average="macro"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "fusion"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--feats", default="cls,mean,clsmean")
    ap.add_argument("--heads", default="logreg,linsvm,knn,mlp")
    ap.add_argument("--tag", default="")
    ap.add_argument("--viewsets", default="all")
    ap.add_argument("--bank", default="", help="feature dir (default fm_features; fm_features_raw = no Macenko)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cp = copies()
    rows = []
    out_csv = OUT / f"fm_cv_{args.mode}{args.tag}.csv"
    for seed in map(int, args.seeds.split(",")):
        bank_dir = HERE / args.bank if args.bank else None
        for feat, vs in [(f, v) for f in args.feats.split(",") for v in args.viewsets.split(",")]:
            banks = {m: load_bank(m, feat, vs, bank_dir) for m in args.models.split(",")}
            gid = next(iter(banks.values()))[1]
            lab = next(iter(banks.values()))[2]
            idx = {int(g): i for i, g in enumerate(gid)}
            for k, (trg, evg) in enumerate(folds(seed), 1):
                tri = np.array([idx[g] for g in trg]); evi = np.array([idx[g] for g in evg])
                w = np.array([cp[g] for g in evg])
                ytr, yev = lab[tri], lab[evi]
                if args.mode == "single":
                    for m, (X, _, _) in banks.items():
                        for head in args.heads.split(","):
                            if head == "mlp":
                                probs, wm = torch_head(lambda: MLP(X.shape[-1]), [X[tri]], ytr, [X[evi]], seed)
                                hp = "fixed"
                            else:
                                probs, hp = sk_head(head, X[tri], ytr, X[evi])
                            rows.append({"mode": "single", "models": m, "feat": feat, "viewset": vs, "bank": args.bank or "macenko", "head": head, "seed": seed,
                                         "fold": k, "hparam": hp, **score(probs, yev, w)})
                            print(rows[-1], flush=True)
                else:
                    Xs = [b[0] for b in banks.values()]
                    dims = [X.shape[-1] for X in Xs]
                    for head in ("attn", "mean", "concat_logreg"):
                        if head == "concat_logreg":
                            Xc = np.concatenate(Xs, -1)
                            probs, wm = sk_head("logreg", Xc[tri], ytr, Xc[evi])[0], None
                        else:
                            probs, wm = torch_head(lambda: FusionHead(dims, head), [X[tri] for X in Xs], ytr,
                                                   [X[evi] for X in Xs], seed)
                        rows.append({"mode": "fusion", "models": "+".join(banks), "feat": feat, "viewset": vs,
                                     "bank": args.bank or "macenko", "head": head,
                                     "seed": seed, "fold": k, "hparam": json.dumps(wm), **score(probs, yev, w)})
                        print(rows[-1], flush=True)
            with open(out_csv, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
