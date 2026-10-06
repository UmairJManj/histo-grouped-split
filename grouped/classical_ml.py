"""Classical ML baselines on handcrafted features, on the same grouped split.

Features per image (after the same Macenko normalisation, resized to 224):
  GLCM contrast/correlation/energy/homogeneity at distances 1,2,4 and angles
  0,45,90,135 degrees (grey levels quantised to 64), uniform LBP histogram
  (P=8, R=1 and P=16, R=2), and 32-bin RGB and HSV histograms per channel.
Features are standardised (scaler fit on the training part only).

Classifiers: SVM (RBF), Random Forest, k-NN. Hyperparameters are chosen by
macro F1 on a validation split only:
  final -> tune on per_seed[seed].final.val, refit on final.train, evaluate on test
  foldK -> inner StratifiedGroupKFold(7) on the fold's train part, tune on the
           inner val, refit on the whole fold train, evaluate on the fold val
Outputs mirror train_grouped.py: runs/<clf>/seed<S>/<unit>/result.json + predictions.npz

Usage: python classical_ml.py [--limit N]
"""
import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import histo_core as hc

HERE = Path(__file__).parent
RUNS = HERE / os.environ.get("HISTO_RUNS_DIR", "runs")
REFERENCE = "human_tissue_image-spleen/image_876.png"
GRIDS = {
    "svm": [dict(C=c, gamma=g) for c in (0.1, 1, 10, 100) for g in ("scale", 1e-3, 1e-2)],
    "rf": [dict(n_estimators=500, max_depth=d, max_features=f)
           for d in (None, 20) for f in ("sqrt", 0.1)],
    "knn": [dict(n_neighbors=k, weights=w) for k in (1, 3, 5, 7, 11) for w in ("uniform", "distance")],
}


def make_clf(name, params, seed):
    if name == "svm":
        return SVC(kernel="rbf", probability=True, random_state=seed, **params)
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    return KNeighborsClassifier(**params)


def features(path):
    img = np.array(Image.open(path).convert("RGB"))
    img = hc.apply_stain_normalization(img)
    img = np.array(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
    grey = (rgb2gray(img) * 255).astype(np.uint8)
    q = (grey // 4).astype(np.uint8)  # 64 grey levels
    glcm = graycomatrix(q, distances=[1, 2, 4], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=64, symmetric=True, normed=True)
    f_glcm = np.concatenate([graycoprops(glcm, p).ravel()
                             for p in ("contrast", "correlation", "energy", "homogeneity")])
    f_lbp = []
    for p, r in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(grey, p, r, method="uniform")
        h, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2), density=True)
        f_lbp.append(h)
    hsv = rgb2hsv(img)
    f_col = [np.histogram(img[..., c], bins=32, range=(0, 256), density=True)[0] for c in range(3)]
    f_col += [np.histogram(hsv[..., c], bins=32, range=(0, 1), density=True)[0] for c in range(3)]
    return np.concatenate([f_glcm, *f_lbp, *f_col]).astype(np.float32)


def init_worker(ref):
    hc.init_stain_normalizer(ref)


def extract_all(paths, ref, workers):
    cache = HERE / "classical_features.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if list(d["paths"]) == list(paths):
            return d["X"]
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(ref,)) as ex:
        X = np.stack(list(ex.map(features, paths, chunksize=8)))
    tmp = cache.with_name(f"classical_features.{os.getpid()}.tmp.npz")
    np.savez(tmp, X=X, paths=np.array(paths))
    os.replace(tmp, cache)  # atomic: parallel per-seed jobs may race on the cache
    return X


def tune_fit(name, X, y, tr, va, ev, seed):
    """Pick params by macro F1 on va (scaler fit on tr), refit on tr, predict ev."""
    best, best_f1 = None, -1
    sc = StandardScaler().fit(X[tr])
    for params in GRIDS[name]:
        clf = make_clf(name, params, seed).fit(sc.transform(X[tr]), y[tr])
        f1 = f1_score(y[va], clf.predict(sc.transform(X[va])), average="macro", zero_division=0)
        if f1 > best_f1:
            best, best_f1 = params, f1
    return best, best_f1


def run_unit(name, seed, unit, X, y, g, idx_of, splits, names):
    seed_split = splits["per_seed"][str(seed)]
    rows = lambda part: np.array([idx_of[p] for p, _, _ in part])
    t0 = time.time()
    if unit == "final":
        tr, va, ev = rows(seed_split["final"]["train"]), rows(seed_split["final"]["val"]), rows(splits["test"])
        params, val_f1 = tune_fit(name, X, y, tr, va, ev, seed)
        fit_idx = tr
    else:
        k = int(unit.removeprefix("fold")) - 1
        fold_tr, ev = rows(seed_split["cv"][k]["train"]), rows(seed_split["cv"][k]["val"])
        inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
        a, b = next(inner.split(X[fold_tr], y[fold_tr], g[fold_tr]))
        params, val_f1 = tune_fit(name, X, y, fold_tr[a], fold_tr[b], ev, seed)
        fit_idx = fold_tr
    sc = StandardScaler().fit(X[fit_idx])
    clf = make_clf(name, params, seed).fit(sc.transform(X[fit_idx]), y[fit_idx])
    train_seconds = time.time() - t0
    t1 = time.perf_counter()
    probs = clf.predict_proba(sc.transform(X[ev]))
    infer_ms = (time.perf_counter() - t1) / len(ev) * 1000
    full = np.zeros((len(ev), len(names)))
    full[:, clf.classes_] = probs
    preds = full.argmax(1)
    metrics = hc._metrics_for_json(hc.compute_all_metrics(y[ev], preds, full, len(names), names))

    out = RUNS / name / f"seed{seed}" / unit
    out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "model": name, "unit": unit, "seed": seed,
        "eval_set": "test" if unit == "final" else "cv_val_fold",
        "n_train": int(len(fit_idx)), "n_eval": int(len(ev)), "n_features": int(X.shape[1]),
        "checkpoint_rule": "hyperparameters chosen by macro F1 on the validation split, then refit on train",
        "best_params": params, "val_f1_macro_at_best": float(val_f1),
        "train_seconds": round(train_seconds, 1),
        "inference_ms_per_image_cpu_batched": round(infer_ms, 3),
        "metrics": metrics,
    }, open(out / "result.json", "w"), indent=2)
    np.savez(out / "predictions.npz", probs=full, targets=y[ev], groups=g[ev], attention=np.zeros(0))
    print(f"[done] {name}/seed{seed}/{unit} acc={metrics['accuracy']:.4f} "
          f"f1={metrics['f1_macro']:.4f} params={params}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N units")
    ap.add_argument("--seeds", default="", help="comma-separated subset of seeds (default: all)")
    args = ap.parse_args()
    splits = json.load(open(HERE / "splits.json"))
    names = splits["class_names"]
    hc.CFG.NUM_CLASSES, hc.CFG.CLASS_NAMES = len(names), names

    every = splits["test"] + splits["per_seed"]["42"]["final"]["train"] + splits["per_seed"]["42"]["final"]["val"]
    every = sorted({p: (p, l, gg) for p, l, gg in every}.values())
    paths = [p for p, _, _ in every]
    y = np.array([l for _, l, _ in every])
    g = np.array([gg for _, _, gg in every])
    idx_of = {p: i for i, p in enumerate(paths)}
    ref = next(p for p in paths if p.endswith(REFERENCE))

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    t0 = time.time()
    X = extract_all(paths, ref, workers)
    print(f"features {X.shape} in {time.time() - t0:.0f}s", flush=True)

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else splits["seeds"]
    units = [(n, s, u) for s in seeds for u in ["final"] + [f"fold{k}" for k in range(1, 6)]
             for n in GRIDS]
    for name, seed, unit in units[: args.limit or None]:
        if (RUNS / name / f"seed{seed}" / unit / "result.json").exists() and not args.limit:
            continue
        run_unit(name, seed, unit, X, y, g, idx_of, splits, names)


if __name__ == "__main__":
    main()
