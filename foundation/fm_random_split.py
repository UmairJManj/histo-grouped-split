"""Secondary result: primary pipeline under a file-level random stratified 80/20 split.

Same features/head as the declared primary (phikon+phikon2 CLS, L2+concat, StandardScaler+LR, 8 views,
C by inner 3-fold CV on training files) but splitting the 1,070 FILES at random (copies of a patch can sit
on both sides), as in the commonly used random-split protocol for this dataset. Seeds 0-9.
Writes results/fm_random_split.csv + prints mean/std and the fraction of test files with a train copy.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from fm_heads import copies
from fm_raw_cv import GRID, fit, load, predict_proba

HERE = Path(__file__).parent
X, gid, y = load(["phikon", "phikon2"], "cls")          # (368, 8, D) per unique patch
ncp = copies()
fg = np.concatenate([[i] * ncp[int(g)] for i, g in enumerate(gid)])  # file -> unique-patch row
fy = y[fg]
rows = []
for seed in range(10):
    tr, te = train_test_split(np.arange(len(fg)), test_size=0.2, stratify=fy, random_state=seed)
    leak = np.isin(fg[te], fg[tr]).mean()
    skf = StratifiedKFold(3, shuffle=True, random_state=0)
    best = max(GRID, key=lambda C: np.mean([
        (predict_proba(fit(X[fg[tr[a]]].reshape(-1, X.shape[-1]), np.repeat(fy[tr[a]], 8), None, C),
                       X[fg[tr[b]], 0]).argmax(1) == fy[tr[b]]).mean()
        for a, b in skf.split(tr, fy[tr])]))
    m = fit(X[fg[tr]].reshape(-1, X.shape[-1]), np.repeat(fy[tr], 8), None, best)
    p = np.mean([predict_proba(m, X[fg[te], v]) for v in range(8)], 0).argmax(1)
    acc, f1 = (p == fy[te]).mean(), f1_score(fy[te], p, average="macro")
    rows.append([seed, len(tr), len(te), best, leak, acc, f1])
    print(f"seed {seed} C={best} test-with-train-copy={leak:.3f} acc={acc:.4f} f1={f1:.4f}", flush=True)
with open(HERE / "results" / "fm_random_split.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["seed", "n_train", "n_test", "C", "frac_test_with_train_copy", "acc", "macro_f1"]); w.writerows(rows)
a = np.array([r[5] for r in rows]); f = np.array([r[6] for r in rows]); l = np.array([r[4] for r in rows])
print(f"== random 80/20 file split, 10 seeds: acc {a.mean():.4f} ± {a.std(ddof=1):.4f}, macro-F1 {f.mean():.4f}, test-with-train-copy {l.mean():.3f}")
