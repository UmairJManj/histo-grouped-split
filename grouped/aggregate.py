"""Collect runs/ into tables, statistics and figures for the HistoFormerFusion paper.

Reads runs/<model>/seed<S>/<unit>/{result.json,predictions.npz,history.json}
and writes into results/:
  cv_per_fold.csv, cv_summary.csv          5-fold grouped CV x 3 seeds
  test_per_seed.csv, test_summary.csv      held-out grouped test, mean/SD over seeds,
                                           Clopper-Pearson CI (acc) + group bootstrap CI (macro F1)
  per_class_<model>.csv, confusion_<model>.csv   best seed of fusion and of the best baseline
  stats_paired.csv                         fusion vs each model: paired t, Wilcoxon, Holm
  mcnemar.csv                              fusion vs best baseline on test predictions
  compute.csv                              time, peak GPU memory, params, checkpoint size, latency
  attention.csv                            mean fusion weights per branch on the test set
  fig_*.png/.pdf (300 DPI)                 bars, confusion matrices, ROC, training curves, attention
  SUMMARY.md
"Best seed" is chosen on validation (lowest best_val_loss for deep models,
highest validation macro F1 for classical ones), never on the test set.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).parent
import os
RUNS = HERE / os.environ.get("HISTO_RUNS_DIR", "runs")
OUT = HERE / os.environ.get("HISTO_RESULTS_DIR", "results")
FUSION = "fusion"
ORDER = ["svm", "rf", "knn", "vgg16", "resnet", "densenet121", "efficientnet_b0", "inception",
         "vit", "meanfusion", "fusion"]
LABEL = {"svm": "SVM (RBF)", "rf": "Random Forest", "knn": "k-NN", "vgg16": "VGG16",
         "resnet": "ResNet50", "densenet121": "DenseNet121", "efficientnet_b0": "EfficientNet-B0",
         "inception": "InceptionResNetV2", "vit": "ViT-B/16", "meanfusion": "Equal-weight fusion",
         "fusion": "HistoFormerFusion"}
CLASSICAL = {"svm", "rf", "knn"}
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "specificity_macro",
           "roc_auc_macro", "mcc", "cohen_kappa"]
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 100})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def short(c):
    return c.replace("human_tissue_image-", "")


def load_runs():
    rows = []
    for f in sorted(RUNS.glob("*/seed*/*/result.json")):
        r = json.load(open(f))
        r["dir"] = f.parent
        rows.append(r)
    return rows


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def group_bootstrap_f1(targets, preds, groups, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug), replace=True)])
        # macro over the classes present in this resample: with 1-2 test groups per class,
        # a resample often omits a class entirely, which would otherwise count as F1 = 0
        vals.append(f1_score(targets[idx], preds[idx], labels=np.unique(targets[idx]),
                             average="macro", zero_division=0))
    return np.percentile(vals, [2.5, 97.5])


def fmt(m, s):
    return f"{m:.4f} ± {s:.4f}"


def main():
    OUT.mkdir(exist_ok=True)
    runs = load_runs()
    names = json.load(open(HERE / "splits.json"))["class_names"]
    models = [m for m in ORDER if any(r["model"] == m for r in runs)]

    # ---------- CV ----------
    cv = pd.DataFrame([{"model": r["model"], "seed": r["seed"], "fold": int(r["unit"][4:]),
                        **{k: r["metrics"][k] for k in METRICS}}
                       for r in runs if r["unit"].startswith("fold")])
    if len(cv):
        cv = cv.sort_values(["model", "seed", "fold"])
        cv.to_csv(OUT / "cv_per_fold.csv", index=False)
        g = cv.groupby("model")[METRICS]
        cv_sum = pd.DataFrame({k: [fmt(g.mean().loc[m, k], g.std(ddof=1).loc[m, k]) for m in g.mean().index]
                               for k in METRICS}, index=g.mean().index)
        cv_sum.insert(0, "n_folds", g.size())
        cv_sum = cv_sum.reindex([m for m in ORDER if m in cv_sum.index])
        cv_sum.index = [LABEL[m] for m in cv_sum.index]
        cv_sum.to_csv(OUT / "cv_summary.csv", index_label="model")
        per_seed = cv.groupby(["model", "seed"])[["accuracy", "f1_macro"]].mean().round(4)
        per_seed.to_csv(OUT / "cv_per_seed_means.csv")

    # ---------- test ----------
    test_rows, preds_of = [], {}
    for r in runs:
        if r["unit"] != "final":
            continue
        p = np.load(r["dir"] / "predictions.npz")
        probs, t, grp = p["probs"], p["targets"], p["groups"]
        pr = probs.argmax(1)
        k, n = int((pr == t).sum()), len(t)
        lo, hi = clopper_pearson(k, n)
        flo, fhi = group_bootstrap_f1(t, pr, grp, seed=r["seed"])
        sel = (-r["val_f1_macro_at_best"]) if r["model"] in CLASSICAL else r["best_val_loss"]
        preds_of[(r["model"], r["seed"])] = dict(probs=probs, t=t, pr=pr, attn=p["attention"], r=r)
        test_rows.append({"model": r["model"], "seed": r["seed"], **{m: r["metrics"][m] for m in METRICS},
                          "correct": k, "n": n, "acc_cp_lo": lo, "acc_cp_hi": hi,
                          "f1_boot_lo": flo, "f1_boot_hi": fhi, "selection_score": sel,
                          "best_epoch": r.get("best_epoch")})
    test = pd.DataFrame(test_rows).sort_values(["model", "seed"])
    test.to_csv(OUT / "test_per_seed.csv", index=False)
    g = test.groupby("model")
    test_sum = pd.DataFrame({k: [fmt(g[k].mean()[m], g[k].std(ddof=1)[m] if g.size()[m] > 1 else 0.0)
                                 for m in g.size().index] for k in METRICS}, index=g.size().index)
    test_sum.insert(0, "n_seeds", g.size())
    best_seed = {m: int(d.sort_values("selection_score").iloc[0]["seed"]) for m, d in g}
    for m in test_sum.index:
        b = test[(test.model == m) & (test.seed == best_seed[m])].iloc[0]
        test_sum.loc[m, "best_seed(val)"] = best_seed[m]
        test_sum.loc[m, "acc_best_seed [95% CP]"] = f"{b.accuracy:.4f} [{b.acc_cp_lo:.4f}, {b.acc_cp_hi:.4f}]"
        test_sum.loc[m, "f1_best_seed [95% group boot]"] = f"{b.f1_macro:.4f} [{b.f1_boot_lo:.4f}, {b.f1_boot_hi:.4f}]"
    test_sum = test_sum.reindex([m for m in ORDER if m in test_sum.index])
    test_sum.index = [LABEL[m] for m in test_sum.index]
    test_sum.to_csv(OUT / "test_summary.csv", index_label="model")

    # best baseline = highest mean CV macro F1 (test mean if CV not yet available), excluding fusion
    cv_complete = len(cv) and set(cv.model) >= set(test.model) and cv.groupby("model").size().min() >= 5
    ref = cv.groupby("model")["f1_macro"].mean() if cv_complete else g["f1_macro"].mean()
    best_rule = "highest mean CV macro F1" if cv_complete else "highest mean test macro F1 (CV incomplete)"
    ref = ref.drop(FUSION, errors="ignore")
    best_base = ref.idxmax()

    # ---------- per-class + confusion for fusion and best baseline ----------
    for m in (FUSION, best_base):
        if (m, best_seed.get(m)) not in preds_of:
            continue
        d = preds_of[(m, best_seed[m])]
        rep = d["r"]["metrics"]["classification_report"]
        pc = pd.DataFrame({short(c): rep[c] for c in names}).T[["precision", "recall", "f1-score", "support"]]
        pc.to_csv(OUT / f"per_class_{m}.csv", index_label="class")
        cm = confusion_matrix(d["t"], d["pr"], labels=range(len(names)))
        pd.DataFrame(cm, index=[short(c) for c in names], columns=[short(c) for c in names]) \
            .to_csv(OUT / f"confusion_{m}.csv")
        plot_confusion(cm, names, f"{LABEL[m]} (seed {best_seed[m]})", f"fig_confusion_{m}")

    # ---------- paired statistics on CV folds ----------
    if len(cv) and FUSION in set(cv.model):
        stat_rows = []
        f = cv[cv.model == FUSION].set_index(["seed", "fold"])
        for metric in ("accuracy", "f1_macro"):
            block = []
            for m in models:
                if m == FUSION or m not in set(cv.model):
                    continue
                o = cv[cv.model == m].set_index(["seed", "fold"])
                common = f.index.intersection(o.index)
                a, b = f.loc[common, metric].values, o.loc[common, metric].values
                t_stat, t_p = stats.ttest_rel(a, b)
                try:
                    w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox")
                except ValueError:
                    w_stat, w_p = np.nan, 1.0
                block.append({"metric": metric, "comparison": f"HistoFormerFusion vs {LABEL[m]}",
                              "n_pairs": len(common), "mean_diff": float(np.mean(a - b)),
                              "t": t_stat, "p_t": t_p, "W": w_stat, "p_wilcoxon": w_p})
            if block:
                bdf = pd.DataFrame(block)
                bdf["p_t_holm"] = multipletests(bdf.p_t.fillna(1), method="holm")[1]
                bdf["p_wilcoxon_holm"] = multipletests(bdf.p_wilcoxon.fillna(1), method="holm")[1]
                stat_rows.append(bdf)
        if stat_rows:
            pd.concat(stat_rows).to_csv(OUT / "stats_paired.csv", index=False)

    # ---------- McNemar on test predictions ----------
    mc = []
    pairs = [(s, s) for s in sorted({s for (m, s) in preds_of if m == FUSION})]
    pairs.append((best_seed.get(FUSION), best_seed.get(best_base)))
    for sf, sb in pairs:
        if (FUSION, sf) not in preds_of or (best_base, sb) not in preds_of:
            continue
        a, b = preds_of[(FUSION, sf)], preds_of[(best_base, sb)]
        ca, cb = a["pr"] == a["t"], b["pr"] == b["t"]
        table = [[int(np.sum(ca & cb)), int(np.sum(ca & ~cb))], [int(np.sum(~ca & cb)), int(np.sum(~ca & ~cb))]]
        res = mcnemar(table, exact=True)
        mc.append({"fusion_seed": sf, "baseline": LABEL[best_base], "baseline_seed": sb,
                   "both_correct": table[0][0], "fusion_only": table[0][1], "baseline_only": table[1][0],
                   "both_wrong": table[1][1], "statistic": res.statistic, "p_exact": res.pvalue,
                   "note": "best seeds (val-selected)" if (sf, sb) == pairs[-1] else "same seed"})
    if mc:
        pd.DataFrame(mc).to_csv(OUT / "mcnemar.csv", index=False)

    # ---------- compute ----------
    comp = []
    for m in models:
        rs = [preds_of[(m, s)]["r"] for (mm, s) in preds_of if mm == m]
        comp.append({"model": LABEL[m],
                     "train_h_final_run(mean)": round(np.mean([r["train_seconds"] for r in rs]) / 3600, 3),
                     "peak_gpu_mem_gb": max((r.get("peak_gpu_mem_gb") or 0) for r in rs) or None,
                     "params_M": round(rs[0]["n_params"] / 1e6, 2) if "n_params" in rs[0] else None,
                     "checkpoint_mb": rs[0].get("checkpoint_mb"),
                     "inference_ms_per_image": np.mean([r.get("inference_ms_per_image_bs1")
                                                        or r.get("inference_ms_per_image_cpu_batched") for r in rs]),
                     "hardware": rs[0].get("gpu", "CPU (16 cores)")})
    pd.DataFrame(comp).to_csv(OUT / "compute.csv", index=False)

    # ---------- attention ----------
    att = [{"seed": s, **preds_of[(FUSION, s)]["r"]["attention_mean"]}
           for (m, s) in sorted(preds_of) if m == FUSION and "attention_mean" in preds_of[(FUSION, s)]["r"]]
    if att:
        pd.DataFrame(att).to_csv(OUT / "attention.csv", index=False)
        plot_attention(preds_of[(FUSION, best_seed[FUSION])]["attn"], best_seed[FUSION])

    # ---------- figures ----------
    plot_bars(cv, test, models)
    if (FUSION, best_seed.get(FUSION)) in preds_of:
        d = preds_of[(FUSION, best_seed[FUSION])]
        plot_roc(d["probs"], d["t"], names, best_seed[FUSION])
        plot_curves(json.load(open(d["r"]["dir"] / "history.json")), best_seed[FUSION])

    global BEST_RULE
    BEST_RULE = best_rule
    write_summary(cv, test_sum, best_base, best_seed, models)
    print(f"wrote {OUT}")


def plot_confusion(cm, names, title, fname):
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.2, 6.4), layout="constrained")
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    n = len(names)
    for i in range(n):
        for j in range(n):
            if norm[i, j] >= 0.05:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(n), [short(c) for c in names], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n), [short(c) for c in names], fontsize=7)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Row-normalised proportion")
    save(fig, fname)


def plot_bars(cv, test, models):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained", sharey=True)
    x = np.arange(len(models))
    cv_title = (f"Grouped 5-fold CV ({cv.seed.nunique()} seed{'s' * (cv.seed.nunique() > 1)})"
                if len(cv) else "Grouped CV (pending)")
    test_title = f"Held-out grouped test ({test.seed.nunique()} seed{'s' * (test.seed.nunique() > 1)})"
    for ax, (df, title) in zip(axes, ((cv, cv_title), (test, test_title))):
        if not len(df):
            ax.set_visible(False); continue
        for i, (metric, lab) in enumerate((("accuracy", "Accuracy"), ("f1_macro", "Macro F1"))):
            g = df.groupby("model")[metric]
            mean = [g.mean().get(m, np.nan) for m in models]
            sd = [g.std(ddof=1).get(m, np.nan) for m in models]
            ax.bar(x + (i - 0.5) * 0.38, mean, 0.38, yerr=sd, capsize=2, color=OKABE[4 if i == 0 else 0],
                   label=lab, error_kw={"lw": 0.8})
        ax.set_xticks(x, [LABEL[m] for m in models], rotation=55, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Score (mean ± SD)")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, "fig_bars_acc_f1")


def plot_roc(probs, t, names, seed):
    aucs = []
    for c in range(len(names)):
        if (t == c).any():
            aucs.append((roc_auc_score(t == c, probs[:, c]), c))
    worst = sorted(aucs)[:3]
    fig, ax = plt.subplots(figsize=(3.6, 3.4), layout="constrained")
    grid = np.linspace(0, 1, 501)
    tprs = []
    for _, c in aucs:
        fpr, tpr, _ = roc_curve(t == c, probs[:, c])
        tprs.append(np.interp(grid, fpr, tpr))
    macro_auc = np.mean([a for a, _ in aucs])
    ax.plot(grid, np.mean(tprs, 0), color=OKABE[7], lw=1.8, label=f"Macro average (AUC {macro_auc:.3f})")
    for k, (a, c) in enumerate(worst):
        fpr, tpr, _ = roc_curve(t == c, probs[:, c])
        ax.plot(fpr, tpr, color=OKABE[[5, 4, 2][k]], lw=1.2, label=f"{short(names[c])} (AUC {a:.3f})")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"HistoFormerFusion ROC, test (seed {seed})", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "fig_roc_fusion")


def plot_curves(h, seed):
    ep = np.arange(1, len(h["train_loss"]) + 1)
    best = int(np.argmin(h["val_loss"])) + 1
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), layout="constrained")
    for ax, key, lab in ((axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")):
        ax.plot(ep, h[f"train_{key}"], color=OKABE[4], label="Train")
        ax.plot(ep, h[f"val_{key}"], color=OKABE[5], label="Validation")
        ax.axvline(best, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel("Epoch"); ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=7)
    axes[2].plot(ep, h["lr"], color=OKABE[2])
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning rate (head group)")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle(f"HistoFormerFusion final model (seed {seed}); dotted line = selected epoch {best}", fontsize=9)
    save(fig, "fig_training_curves_fusion")


def plot_attention(w, seed):
    fig, ax = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
    labs = ["ResNet50", "InceptionResNetV2", "ViT-B/16"]
    parts = ax.violinplot([w[:, i] for i in range(3)], showmeans=True, showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(OKABE[[0, 1, 2][i]]); b.set_alpha(0.8)
    parts["cmeans"].set_color("black")
    for i in range(3):
        ax.text(i + 1, w[:, i].mean() + 0.04, f"{w[:, i].mean():.3f}", ha="center", fontsize=7)
    ax.set_xticks([1, 2, 3], labs, fontsize=7)
    ax.set_ylabel("Fusion weight"); ax.set_ylim(0, 1)
    ax.set_title(f"Per-image fusion weights, test (seed {seed})", fontsize=9)
    save(fig, "fig_attention_weights")


def write_summary(cv, test_sum, best_base, best_seed, models):
    lines = ["# HistoFormerFusion grouped retrain: summary", "",
             "All numbers come from runs/ via aggregate.py. The split is grouped by MD5 hash (no image copy crosses partitions).",
             f"Best baseline ({BEST_RULE}, excluding HistoFormerFusion): **{LABEL[best_base]}**.",
             "Best seed per model is chosen on validation, never on the test set.", "",
             "## Held-out test (mean ± SD over seeds)", "", test_sum.to_markdown(), ""]
    if (OUT / "cv_summary.csv").exists():
        lines += ["## 5-fold grouped CV (5 folds x 3 seeds)", "",
                  pd.read_csv(OUT / "cv_summary.csv").to_markdown(index=False), ""]
    for f in ("stats_paired.csv", "mcnemar.csv", "compute.csv", "attention.csv"):
        if (OUT / f).exists():
            lines += [f"## {f}", "", pd.read_csv(OUT / f).to_markdown(index=False, floatfmt=".4g"), ""]
    lines += ["## Data", "", pd.read_csv(HERE / "data_table.csv").to_markdown(index=False), ""]
    (OUT / "SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
