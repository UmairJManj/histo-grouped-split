"""Step 4: score the FROZEN frozen-feature configuration on the fixed grouped test set ONCE.

Fits on all 315 unique pool patches (test groups excluded), predicts the 53 test patches,
expands to the 153 test files (each file = its patch's prediction). Heads:
  concat_lr : fm_raw_cv protocol (L2 per model, concat, scaler + LR, C by inner CV, n_copies weights)
  attn|mean : fm_heads.FusionHead, one model per seed (42, 43, 44) + 3-seed softmax ensemble
Writes results/fm_raw_test_<tag>.json (+ per-class and confusion CSVs, + predictions CSV).

Usage: python fm_raw_test.py --head concat_lr --views 8 --tag final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

import fm_raw_cv as rc
from aggregate import clopper_pearson, group_bootstrap_acc, group_bootstrap_f1
from fm_heads import FusionHead, N_CLASSES, copies, torch_head

HERE = Path(__file__).parent
RES = HERE / "results"


def metrics(t, prob, grp, seed=0):
    p = prob.argmax(1)
    k, n = int((p == t).sum()), len(t)
    lo, hi = clopper_pearson(k, n)
    alo, ahi = group_bootstrap_acc(t, p, grp, seed=seed)
    flo, fhi = group_bootstrap_f1(t, p, grp, seed=seed)
    return {"n_files": n, "n_correct": k, "accuracy": accuracy_score(t, p), "acc_cp95": [lo, hi], "acc_gboot95": [alo, ahi],
            "f1_macro": f1_score(t, p, average="macro"), "f1_gboot95": [flo, fhi], "mcc": matthews_corrcoef(t, p),
            "kappa": cohen_kappa_score(t, p), "roc_auc_macro": roc_auc_score(t, prob, multi_class="ovr", average="macro",
                                                                               labels=list(range(N_CLASSES)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, choices=["concat_lr", "attn", "mean"])
    ap.add_argument("--models", default="phikon,phikon2")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    models = args.models.split(",")
    s = json.load(open(HERE / "splits.json"))
    names = [n.replace("human_tissue_image-", "") for n in s["class_names"]]
    files = pd.DataFrame(s["test"], columns=["path", "label", "group"])
    X, gid, y = rc.load(models, "cls")
    pos = {int(g): i for i, g in enumerate(gid)}
    test_g = sorted(set(files.group))
    pool = np.array([i for i, g in enumerate(gid) if int(g) not in set(test_g)])
    tst = np.array([pos[g] for g in test_g])
    assert len(pool) == 315 and len(tst) == 53
    cp = copies()
    w = np.array([cp[int(g)] for g in gid], dtype=float)
    out = {"config": vars(args), "per_seed": {}}
    if args.head == "concat_lr":
        C = rc.pick_C(X[pool], y[pool], w[pool], args.views)
        m = rc.fit(*rc.train_views(X, y, w, pool, args.views), C)
        probs = {"single": rc.predict_proba(m, X[tst, 0])}
        out["C"] = C
    else:
        dims = [np.load(rc.RAW / f"whole_{m}.npz")["cls"].shape[-1] for m in models]  # phikon 768, phikon2 1024
        assert sum(dims) == X.shape[-1]
        cut = np.cumsum([0] + dims)
        Xn = [X[..., cut[i]:cut[i + 1]] for i in range(len(models))]
        probs = {}
        for seed in (42, 43, 44):
            pr, wm = torch_head(lambda: FusionHead(dims, args.head), [x[pool] for x in Xn], y[pool],
                                [x[tst] for x in Xn], seed)
            probs[str(seed)] = pr[:, 0]
            out.setdefault("attn_weights", {})[str(seed)] = wm
        probs["ensemble"] = np.mean([probs[k] for k in ("42", "43", "44")], 0)
    gi = {g: j for j, g in enumerate(test_g)}
    t, grp = files.label.to_numpy(), files.group.to_numpy()
    for key, pr in probs.items():
        full = np.zeros((len(test_g), N_CLASSES)); full[:] = pr  # classes all present in pool
        fp = full[[gi[g] for g in grp]]
        out["per_seed"][key] = metrics(t, fp, grp)
        print(key, json.dumps({k: v for k, v in out["per_seed"][key].items()}), flush=True)
    key = "ensemble" if "ensemble" in probs else "single"
    fp = probs[key][[gi[g] for g in grp]]
    pred = fp.argmax(1)
    P, R, F, N = precision_recall_fscore_support(t, pred, labels=list(range(N_CLASSES)), zero_division=0)
    pd.DataFrame({"class": names, "precision": P, "recall": R, "f1": F, "support": N}).to_csv(
        RES / f"fm_raw_test_{args.tag}_per_class.csv", index=False)
    pd.DataFrame(confusion_matrix(t, pred, labels=list(range(N_CLASSES))), index=names, columns=names).to_csv(
        RES / f"fm_raw_test_{args.tag}_confusion.csv")
    files.assign(pred=pred, pred_name=[names[i] for i in pred], prob_max=fp.max(1)).to_csv(
        RES / f"fm_raw_test_{args.tag}_predictions.csv", index=False)
    json.dump(out, open(RES / f"fm_raw_test_{args.tag}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    sys.exit(main())
