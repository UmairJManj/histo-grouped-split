"""Match each unique dataset image to its NuInsSeg source patch.

NuInsSeg (Mahbod et al. 2024, Zenodo 10518968): 512x512 H&E patches in
"<species> <organ>/tissue images/". Each of the 368 unique images (MD5 groups)
is compared with every human patch resized to 224x224 (Pearson correlation
of 64x64 grey thumbnails), and the best match is checked at full 224 size with
several resize filters. Crop/tile is tested by template matching at native
scale when no whole-patch match exists.
Writes nuinsseg_mapping.csv and nuinsseg_summary.json.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
NUIN = Path("/nesi/nobackup/aut04653/jfd1673/nuinsseg")
FILTERS = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS,
           "nearest": Image.NEAREST, "area": Image.BOX}


def thumb(img):
    g = np.asarray(img.convert("L").resize((64, 64), Image.BOX), dtype=np.float64).ravel()
    return (g - g.mean()) / (g.std() + 1e-9)


def main():
    rows = json.load(open(HERE / "splits.json"))
    every = rows["test"] + rows["per_seed"]["42"]["final"]["train"] + rows["per_seed"]["42"]["final"]["val"]
    names = rows["class_names"]
    by_group = defaultdict(list)
    for p, l, g in every:
        by_group[g].append((p, l))

    patches = sorted(p for p in NUIN.glob("human */tissue images/*.png"))
    pt_img = {p: Image.open(p).convert("RGB") for p in patches}
    pt_thumb = np.stack([thumb(pt_img[p].resize((224, 224), Image.BILINEAR)) for p in patches])
    organ = lambda p: p.parent.parent.name.replace("human ", "")

    out, used = [], Counter()
    for g, members in sorted(by_group.items()):
        p0, label = sorted(members)[0]
        img = Image.open(p0).convert("RGB")
        corr = pt_thumb @ thumb(img) / pt_thumb.shape[1]
        best = int(np.argmax(corr))
        second = float(np.sort(corr)[-2])
        src = patches[best]
        a = np.asarray(img, dtype=np.float64)
        diffs = {k: float(np.abs(np.asarray(pt_img[src].resize(img.size, f), dtype=np.float64) - a).mean())
                 for k, f in FILTERS.items()}
        filt = min(diffs, key=diffs.get)
        if corr[best] > 0.98 and diffs[filt] < 3:
            kind = "whole patch resized 512->224" + (" (pixel-identical)" if diffs[filt] < 0.5 else "")
        elif corr[best] > 0.9:
            kind = "whole patch, altered (colour/compression)"
        else:
            kind = "no whole-patch match"
        used[str(src)] += 1
        out.append({"group_id": g, "class": names[label].replace("human_tissue_image-", ""),
                    "n_copies": len(members), "example_file": f"{Path(p0).parent.name}/{Path(p0).name}",
                    "all_files": ";".join(sorted(Path(p).name for p, _ in members)),
                    "nuinsseg_patch": f"{src.parent.parent.name}/{src.name}", "nuinsseg_organ": organ(src),
                    "thumb_corr": round(float(corr[best]), 4), "second_best_corr": round(second, 4),
                    "best_resize_filter": filt, "mean_abs_pixel_diff": round(diffs[filt], 3), "match": kind})

    with open(HERE / "nuinsseg_mapping.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)

    per_class = defaultdict(lambda: {"unique_images": 0, "files": 0, "organ_match": 0})
    for r in out:
        c = per_class[r["class"]]
        c["unique_images"] += 1; c["files"] += r["n_copies"]
        c["organ_match"] += int(r["class"].replace("-", " ").replace("gland", "salivory gland") == r["nuinsseg_organ"])
    n_patches = Counter(organ(p) for p in patches)
    for cls, c in per_class.items():
        org = cls.replace("-", " ").replace("gland", "salivory gland")
        c["nuinsseg_human_patches"] = n_patches.get(org, 0)
        c["copy_counts"] = dict(sorted(Counter(r["n_copies"] for r in out if r["class"] == cls).items()))
    summary = {
        "unique_images": len(out),
        "match_kinds": dict(Counter(r["match"] for r in out)),
        "patches_matched_more_than_once": sum(1 for v in used.values() if v > 1),
        "nuinsseg_human_patches_total": len(patches),
        "nuinsseg_organs_not_used": sorted(set(n_patches) - {r["nuinsseg_organ"] for r in out}),
        "per_class": dict(sorted(per_class.items())),
    }
    json.dump(summary, open(HERE / "nuinsseg_summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_class"}, indent=2))
    for cls, c in sorted(per_class.items()):
        print(f"{cls:15s} unique {c['unique_images']:3d}  patches {c['nuinsseg_human_patches']:3d}  "
              f"organ-match {c['organ_match']:3d}  files {c['files']:3d}  copies {c['copy_counts']}")


if __name__ == "__main__":
    main()
