"""Step 3: HistoFormerFusion attention head vs equal-weight fusion on frozen raw whole-patch CLS
embeddings (one branch per foundation model), same grouped folds and seeds as fm_raw_cv.py.
Heads from fm_heads.py (FusionHead attn/mean, trained on 8 dihedral views, 150 epochs).

Usage: python fm_raw_fusion.py --models phikon,phikon2 --seeds 42,43,44
Appends rows to results/fm_raw_fusion.csv.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from fm_heads import FusionHead, copies, folds, score, torch_head

HERE = Path(__file__).parent
OUT = HERE / "results" / "fm_raw_fusion.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--heads", default="attn,mean")
    args = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", 4)))
    models = args.models.split(",")
    Xs = []
    for m in models:
        d = np.load(HERE / "fm_features_raw" / f"whole_{m}.npz")
        Xs.append((d["cls"] / np.linalg.norm(d["cls"], axis=-1, keepdims=True)).astype(np.float32))
    gid, y = d["group_id"], d["label"]
    pos = {int(g): i for i, g in enumerate(gid)}
    cp = copies()
    new = not OUT.exists()
    fh = open(OUT, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(["models", "head", "seed", "fold", "weights", "acc", "acc_tta", "acc_unique", "f1", "f1_tta"])
    for seed in map(int, args.seeds.split(",")):
        for head in args.heads.split(","):
            accs = []
            for k, (tr, va) in enumerate(folds(seed), 1):
                a = np.array([pos[g] for g in tr]); b = np.array([pos[g] for g in va])
                w = np.array([cp[g] for g in va])
                probs, wm = torch_head(lambda: FusionHead([X.shape[-1] for X in Xs], head),
                                       [X[a] for X in Xs], y[a], [X[b] for X in Xs], seed)
                s = score(probs, y[b], w)
                wr.writerow(["+".join(models), head, seed, k, json.dumps(wm), s["acc"], s["acc_tta"],
                             s["acc_unique"], s["f1"], s["f1_tta"]]); fh.flush()
                accs.append(s["acc"])
                print(f"{head} s{seed} f{k} acc={s['acc']:.4f} tta={s['acc_tta']:.4f} w={wm}", flush=True)
            print(f"== {'+'.join(models)} {head} seed {seed}: mean acc {np.mean(accs):.4f}", flush=True)


if __name__ == "__main__":
    main()
