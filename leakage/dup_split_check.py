"""Reconstruct the job-7172798 split and count exact-duplicate images across it."""
import hashlib, json, random
from collections import defaultdict
from pathlib import Path

DATA = Path("/nesi/nobackup/aut04653/jfd1673/histo_runs/data/Dataset")
EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
OUT = Path(__file__).parent


def discover():
    dirs = sorted(d for d in DATA.iterdir() if d.is_dir())
    samples = []
    for i, d in enumerate(dirs):
        samples += [(str(f), i) for f in d.rglob("*") if f.suffix.lower() in EXT]
    return samples, [d.name for d in dirs]


def stratified_split(samples, val_frac, test_frac, seed):
    # copied from train_histo.py
    by_class = defaultdict(list)
    for p, l in samples:
        by_class[l].append(p)
    rng = random.Random(seed)
    tr, va, te = [], [], []
    for l, paths in by_class.items():
        paths = paths.copy(); rng.shuffle(paths); n = len(paths)
        nt = max(1, int(round(n * test_frac))) if n >= 3 else 0
        nv = max(1, int(round(n * val_frac))) if n - nt >= 2 else 0
        ntr = n - nv - nt
        tr += [(p, l) for p in paths[:ntr]]
        va += [(p, l) for p in paths[ntr:ntr + nv]]
        te += [(p, l) for p in paths[ntr + nv:ntr + nv + nt]]
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te


samples, names = discover()
tr, va, te = stratified_split(samples, 0.15, 0.15, 42)
print("split", len(tr), len(va), len(te), "first train:", tr[0][0])

h = {p: hashlib.md5(open(p, "rb").read()).hexdigest() for p, _ in samples}
groups = defaultdict(list)
for p, l in samples:
    groups[h[p]].append((p, l))
dups = {k: v for k, v in groups.items() if len(v) > 1}
files_in_dups = sum(len(v) for v in dups.values())
cross_class = sum(1 for v in dups.values() if len({l for _, l in v}) > 1)

pool_hashes = {h[p] for p, _ in tr + va}
test_leak = [(p, l) for p, l in te if h[p] in pool_hashes]
res = {
    "n_images": len(samples), "unique_images": len(groups),
    "dup_groups": len(dups), "files_in_dup_groups": files_in_dups,
    "max_group_size": max(len(v) for v in dups.values()),
    "cross_class_dup_groups": cross_class,
    "split": [len(tr), len(va), len(te)], "first_train_sample": tr[0][0],
    "test_images_with_exact_copy_in_cv_pool": len(test_leak),
    "test_leak_per_class": {names[l]: sum(1 for _, x in test_leak if x == l) for l in range(len(names))},
}
json.dump(res, open(OUT / "dup_split_check.json", "w"), indent=2)
print(json.dumps(res, indent=2))
