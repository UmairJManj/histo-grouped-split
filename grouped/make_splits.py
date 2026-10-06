"""Build the group-aware split for the HistoFormerFusion retrain.

Group = MD5 hash of the image file, so every byte-identical copy lands in
one partition. Test hold-out, CV folds and the final train/val split are all
StratifiedGroupKFold on (class, group). Test hold-out uses seed 42 for
everyone; CV folds and the final train/val split are re-drawn for seeds 42-44. Writes splits.json and
data_table.csv, and fails if any hash appears in two partitions.
"""
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

DATA = Path(os.environ.get("HISTO_DATA_DIR", "/nesi/nobackup/aut04653/jfd1673/histo_runs/data")) / "Dataset"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUT = Path(__file__).parent
SEED = 42           # test hold-out seed (fixed for every model and seed)
SEEDS = (42, 43, 44)  # CV folds + final train/val are re-drawn per seed
N_TEST_SPLITS = 7   # 1 of 7 group folds held out = ~14% of groups as test
N_FOLDS = 5
N_VAL_SPLITS = 7    # final model: 1 of 7 pool group folds as val


def discover():
    dirs = sorted(d for d in DATA.iterdir() if d.is_dir())
    samples = []
    for i, d in enumerate(dirs):
        samples += [(str(f), i) for f in sorted(d.rglob("*")) if f.suffix.lower() in EXT]
    return samples, [d.name for d in dirs]


def group_fold(idx, labels, groups, n_splits, which=0, seed=SEED):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(idx)), labels[idx], groups[idx]))
    return [(idx[a], idx[b]) for a, b in folds] if which is None else (idx[folds[which][0]], idx[folds[which][1]])


def assert_disjoint(name, groups, *parts):
    sets = [set(groups[p]) for p in parts]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared = sets[i] & sets[j]
            assert not shared, f"{name}: {len(shared)} hashes in partitions {i} and {j}"
    print(f"[ok] {name}: no hash shared across {len(parts)} partitions")


def write_assignment(splits, hashes, paths):
    """Long table: one row per (file, seed) with its role in the test hold-out, CV and final split."""
    md5 = dict(zip(paths, hashes))
    rows = []
    test = {p for p, _, _ in splits["test"]}
    for seed, sp in splits["per_seed"].items():
        fold_of = {p: k + 1 for k, f in enumerate(sp["cv"]) for p, _, _ in f["val"]}
        final_role = {p: "train" for p, _, _ in sp["final"]["train"]} | {p: "val" for p, _, _ in sp["final"]["val"]}
        for p, l, g in splits["test"] + sp["final"]["train"] + sp["final"]["val"]:
            rows.append([Path(p).parent.name + "/" + Path(p).name, splits["class_names"][l], md5[p], g, seed,
                         "test" if p in test else "pool", fold_of.get(p, ""), final_role.get(p, "test")])
    rows.sort(key=lambda r: (int(r[4]), r[0]))
    with open(OUT / "split_assignment.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "md5", "group_id", "seed", "partition", "cv_val_fold", "final_role"])
        w.writerows(rows)


def main():
    samples, names = discover()
    paths = np.array([p for p, _ in samples])
    labels = np.array([l for _, l in samples])
    hashes = np.array([hashlib.md5(open(p, "rb").read()).hexdigest() for p in paths])
    gid = {h: i for i, h in enumerate(sorted(set(hashes)))}
    groups = np.array([gid[h] for h in hashes])
    assert all(len(set(labels[groups == g])) == 1 for g in set(groups)), "a hash spans two classes"
    print(f"{len(paths)} files, {len(gid)} groups, {len(names)} classes")

    all_idx = np.arange(len(paths))
    pool, test = group_fold(all_idx, labels, groups, N_TEST_SPLITS)
    assert_disjoint("pool vs test", groups, pool, test)

    as_list = lambda ix: [[paths[i], int(labels[i]), int(groups[i])] for i in ix]
    splits = {
        "class_names": names, "test_seed": SEED, "seeds": list(SEEDS),
        "group_rule": "MD5 of file bytes; all copies in one partition",
        "test": as_list(test), "per_seed": {},
    }
    sizes = {}
    for seed in SEEDS:
        cv = group_fold(pool, labels, groups, N_FOLDS, which=None, seed=seed)
        for k, (tr, va) in enumerate(cv):
            assert_disjoint(f"seed {seed} cv fold {k + 1}", groups, tr, va, test)
        fin_tr, fin_va = group_fold(pool, labels, groups, N_VAL_SPLITS, seed=seed)
        assert_disjoint(f"seed {seed} final train/val/test", groups, fin_tr, fin_va, test)
        splits["per_seed"][str(seed)] = {
            "final": {"train": as_list(fin_tr), "val": as_list(fin_va)},
            "cv": [{"train": as_list(tr), "val": as_list(va)} for tr, va in cv],
        }
        sizes[str(seed)] = {"cv_fold_sizes": [[len(tr), len(va)] for tr, va in cv],
                            "final_train_val": [len(fin_tr), len(fin_va)]}
    json.dump(splits, open(OUT / "splits.json", "w"))

    with open(OUT / "data_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "pool_files", "pool_groups", "test_files", "test_groups", "total_files", "total_groups"])
        for c, n in enumerate(names):
            pf = pool[labels[pool] == c]; tf = test[labels[test] == c]
            w.writerow([n.replace("human_tissue_image-", ""), len(pf), len(set(groups[pf])), len(tf),
                        len(set(groups[tf])), len(pf) + len(tf), len(set(groups[pf])) + len(set(groups[tf]))])
        w.writerow(["TOTAL", len(pool), len(set(groups[pool])), len(test), len(set(groups[test])),
                    len(paths), len(gid)])

    summary = {
        "files": len(paths), "groups": len(gid),
        "pool_files": len(pool), "pool_groups": len(set(groups[pool])),
        "test_files": len(test), "test_groups": len(set(groups[test])),
        "test_group_fraction": round(len(set(groups[test])) / len(gid), 4),
        "classes_missing_from_test": [names[c] for c in range(len(names)) if c not in set(labels[test])],
        "per_seed_sizes": sizes,
        "leakage_check": "passed: no MD5 hash in two partitions (test/pool, every CV fold and final train/val, all seeds)",
    }
    json.dump(summary, open(OUT / "split_summary.json", "w"), indent=2)
    write_assignment(splits, hashes, paths)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
