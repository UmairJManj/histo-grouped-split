"""Frozen whole-patch raw protocol on grouped CV (seeds 42/43/44), with fusion heads.

Features: fm_features_raw/whole_<model>.npz (raw, no Macenko, whole 512 patch -> 224
bicubic, CLS). Each model block is L2-normalised.
Heads (training rows = the unique pool patches of the fold's train part, sample
weight = n_copies; scoring = copy-weighted file-level accuracy and macro F1 on the
held-out fold):
  concat_lr   blocks concatenated -> StandardScaler -> LogisticRegression, C in
              {0.001, 0.01, 0.1, 1} chosen by inner StratifiedKFold(3, shuffle, rs=0)
              on the training patches (laptop protocol)
  attn        HistoFormerFusion head (per-model projection, attention scorer, softmax-
              weighted sum, MLP), trained on the 8 dihedral views of each training patch
  mean        same head with equal (mean) weights: control
'tta' columns average the softmax over the 8 dihedral views (attn/mean/concat_lr).

Usage: python fm_protocol.py --models phikon,phikon2 --seeds 42,43,44 [--heads concat_lr,attn,mean]
Writes results/fm_protocol_<models>.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import fm_heads as fh

HERE = Path(__file__).parent
RAW = HERE / "fm_features_raw"


def load(models):
    blocks, gid, lab = [], None, None
    for m in models:
        d = np.load(RAW / f"whole_{m}.npz")
        x = d["cls"].astype(np.float32)  # (368, 8, D)
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        blocks.append(x)
        gid, lab = d["group_id"], d["label"]
    return blocks, gid, lab


def concat_lr(Xtr, ytr, wtr, Xev_views):
    """Xtr (n, D) identity view; Xev_views (n_ev, 8, D). returns probs (n_ev, 8, C), C picked."""
    best, best_acc = None, -1
    skf = StratifiedKFold(3, shuffle=True, random_state=0)
    for c in (0.001, 0.01, 0.1, 1.0):
        accs = []
        for a, b in skf.split(Xtr, ytr):
            sc = StandardScaler().fit(Xtr[a])
            m = LogisticRegression(C=c, max_iter=5000).fit(sc.transform(Xtr[a]), ytr[a], sample_weight=wtr[a])
            accs.append(np.average(m.predict(sc.transform(Xtr[b])) == ytr[b], weights=wtr[b]))
        if np.mean(accs) > best_acc:
            best, best_acc = c, np.mean(accs)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=best, max_iter=5000).fit(sc.transform(Xtr), ytr, sample_weight=wtr)
    flat = Xev_views.reshape(-1, Xev_views.shape[-1])
    p = m.predict_proba(sc.transform(flat))
    full = np.zeros((p.shape[0], fh.N_CLASSES)); full[:, m.classes_] = p
    return full.reshape(Xev_views.shape[0], Xev_views.shape[1], fh.N_CLASSES), best


def score(probs, y, w):
    p0, pt = probs[:, 0].argmax(1), probs.mean(1).argmax(1)
    f1 = lambda p: float(f1_score(np.repeat(y, w), np.repeat(p, w), average="macro"))
    return {"acc": float(np.average(p0 == y, weights=w)), "acc_tta": float(np.average(pt == y, weights=w)),
            "f1": f1(p0), "f1_tta": f1(pt), "acc_unique": float((p0 == y).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--heads", default="concat_lr,attn,mean")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    models = args.models.split(",")
    blocks, gid, lab = load(models)
    cp = fh.copies()
    idx = {int(g): i for i, g in enumerate(gid)}
    rows = []
    out = HERE / "results" / f"fm_protocol_{'+'.join(models)}{args.tag}.csv"
    for seed in map(int, args.seeds.split(",")):
        for k, (trg, evg) in enumerate(fh.folds(seed), 1):
            tri = np.array([idx[g] for g in trg]); evi = np.array([idx[g] for g in evg])
            wtr = np.array([cp[g] for g in trg]); wev = np.array([cp[g] for g in evg])
            ytr, yev = lab[tri], lab[evi]
            for head in args.heads.split(","):
                if head == "concat_lr":
                    Xc = np.concatenate(blocks, -1)
                    probs, hp = concat_lr(Xc[tri, 0], ytr, wtr, Xc[evi])
                else:
                    dims = [b.shape[-1] for b in blocks]
                    probs, hp = fh.torch_head(lambda: fh.FusionHead(dims, head), [b[tri] for b in blocks], ytr,
                                              [b[evi] for b in blocks], seed)
                rows.append({"models": "+".join(models), "head": head, "seed": seed, "fold": k,
                             "hparam": json.dumps(hp), **score(probs, yev, wev)})
                print(rows[-1], flush=True)
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.groupby(["head", "seed"])[["acc", "acc_tta", "f1"]].mean().round(4).to_string())
    print(df.groupby("head")[["acc", "acc_tta", "f1"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
