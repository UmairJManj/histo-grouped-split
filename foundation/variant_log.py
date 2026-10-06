"""Log of every improvement variant tried (validation only; test is never read here).

Scans runs/ and <variant>/runs/ for result.json (+ val_tta.json backfills) and
writes results/variant_log.csv and prints it, sorted by variant then model/seed.
No seed ensemble here: each seed redraws its own final train/val split, so the
validation sets differ and a seed ensemble can only be scored on the fixed test set.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def main():
    rows = []
    for f in sorted(HERE.glob("**/runs/*/seed*/final/result.json")):
        if "smoke" in str(f):
            continue
        r = json.load(open(f))
        if not (f.parent / "history.json").exists():  # classical models have no training history
            continue
        extra = json.load(open(f.parent / "val_tta.json")) if (f.parent / "val_tta.json").exists() else {}
        get = lambda k: r.get(k, extra.get(k))
        hist = json.load(open(f.parent / "history.json"))
        variant = r.get("variant", "base")
        rows.append({"variant": variant, "model": r["model"], "seed": r["seed"],
                     "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                     "patience": r.get("early_stop_patience"), "best_val_loss": round(r["best_val_loss"], 4),
                     "val_acc": round(get("val_acc_at_best") or hist["val_acc"][r["best_epoch"] - 1], 4),
                     "val_acc_tta": get("val_acc_tta"), "val_f1": get("val_f1_macro"),
                     "val_f1_tta": get("val_f1_macro_tta"),
                     "val_acc_tta4": get("val_acc_tta4"), "val_f1_tta4": get("val_f1_macro_tta4"), "train_min": round(r["train_seconds"] / 60, 1),
                     "peak_gb": r.get("peak_gpu_mem_gb"), "gpu": r.get("gpu")})
    df = pd.DataFrame(rows).sort_values(["variant", "model", "seed"])
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "variant_log.csv", index=False)
    print(df.to_string(index=False))
    ab = out / "abandoned_variants.csv"
    if ab.exists():
        print("\nabandoned (logged, not completed):\n" + pd.read_csv(ab).to_string(index=False))

if __name__ == "__main__":
    main()
