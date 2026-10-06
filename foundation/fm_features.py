"""Frozen foundation-model feature bank for the 368 unique NuInsSeg patches.

For every unique patch (MD5 group) the ORIGINAL 512x512 NuInsSeg image is
Macenko-normalised (fixed reference, see REFERENCE_PATCH), then for each of the
8 dihedral transforms we take 6 views: 5 native-scale 224 crops (4 corners and
centre) plus one 512->224 resize. Each model returns the CLS token and the mean
patch token per view.

Output per model: fm_features/<name>.npz with
  cls  (368, 8, 6, D) float16, mean (368, 8, 6, D) float16,
  group_id (368,), label (368,), patch (368,)
Copies of a patch share its features; map files -> group_id via splits.json.

Usage: python fm_features.py [--models a,b,...] [--limit N]
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import histo_core as hc

HERE = Path(__file__).parent
OUT = HERE / "fm_features"
NUIN = Path("/nesi/nobackup/aut04653/jfd1673/nuinsseg")
REFERENCE_PATCH = "human bladder/tissue images/human_bladder_01.png"  # Macenko target (seed-42 final-train image)
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
HOPT = ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))

# name: (kind, source, (mean, std)); for timm models the checkpoint's pretrained_cfg mean/std wins
MODELS = {
    "phikon": ("hf", "owkin/phikon", IMAGENET),
    "phikon2": ("hf", "owkin/phikon-v2", IMAGENET),
    "hoptimus0": ("timm", "hf-hub:1aurent/vit_giant_patch14_reg4_224.h-optimus-v0", HOPT),
    "midnight": ("hf", "kaiko-ai/midnight", HALF),
    "kaiko_vitl14": ("timm", "hf-hub:1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms", HALF),
    "lunit_vits8": ("timm", "hf-hub:1aurent/vit_small_patch8_224.lunit_dino",
                    ((0.70322989, 0.53606487, 0.66096631), (0.21716536, 0.26081574, 0.20723464))),
    "lunit_r50bt": ("timm", "hf-hub:1aurent/resnet50.lunit_bt",
                    ((0.70322989, 0.53606487, 0.66096631), (0.21716536, 0.26081574, 0.20723464))),
    "ctranspath": ("timm", "hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath", IMAGENET),
    # ImageNet controls
    "in_vitb16": ("timm", "vit_base_patch16_224.augreg_in21k_ft_in1k", HALF),
    "in_resnet50": ("timm", "resnet50.tv2_in1k", IMAGENET),
    "in_irv2": ("timm", "inception_resnet_v2.tf_in1k", HALF),
}


class ConvStem(torch.nn.Module):
    """CTransPath patch embedding (Wang et al. 2022): conv stem instead of a linear patchify."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, **kwargs):
        super().__init__()
        img_size = img_size[0] if isinstance(img_size, (list, tuple)) else img_size
        patch_size = patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size
        assert patch_size == 4 and embed_dim % 8 == 0
        layers, cin, cout = [], in_chans, embed_dim // 8
        for _ in range(2):
            layers += [torch.nn.Conv2d(cin, cout, 3, 2, 1, bias=False), torch.nn.BatchNorm2d(cout), torch.nn.ReLU(True)]
            cin, cout = cout, cout * 2
        layers.append(torch.nn.Conv2d(cin, embed_dim, 1))
        self.proj = torch.nn.Sequential(*layers)
        self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()
        self.img_size, self.patch_size = (img_size, img_size), (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))  # NHWC, as timm's swin expects


class Encoder(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        kind, src, (mean, std) = MODELS[name]
        self.kind = kind
        if kind == "hf":
            from transformers import AutoModel
            self.m = AutoModel.from_pretrained(src, trust_remote_code=True)
        else:
            import timm
            kw = {"img_size": 224} if ("vit_" in src or "h-optimus" in src) else {}  # DINOv2 ports default to 518
            if "CTransPath" in src:
                kw = {"embed_layer": ConvStem}
            self.m = timm.create_model(src, pretrained=True, num_classes=0, **kw)
            cfg = self.m.pretrained_cfg  # use each checkpoint's own normalisation
            mean, std = tuple(cfg.get("mean", mean)), tuple(cfg.get("std", std))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x01):  # x01: (B,3,224,224) float in [0,1]
        x = (x01 - self.mean) / self.std
        if self.kind == "hf":
            t = self.m(pixel_values=x).last_hidden_state
            n_prefix = 1 + int(getattr(self.m.config, "num_register_tokens", 0) or 0)
        else:
            t = self.m.forward_features(x)
            if t.ndim == 4:  # CNN feature map (B,C,H,W) or swin (B,H,W,C)
                if t.shape[1] < t.shape[-1]:
                    t = t.permute(0, 3, 1, 2)
                v = t.mean((2, 3))
                return v, v
            n_prefix = getattr(self.m, "num_prefix_tokens", 1)
        return t[:, 0], t[:, n_prefix:].mean(1)


def views(img):
    """img: (3,512,512) float [0,1] -> (8, 6, 3, 224, 224): dihedral x (4 corners, centre, resize)."""
    out = []
    for f in (False, True):
        base = torch.flip(img, dims=(2,)) if f else img
        for k in range(4):
            r = torch.rot90(base, k, dims=(1, 2))
            H = r.shape[-1]
            c, s = (H - 224) // 2, 224
            crops = [r[:, :s, :s], r[:, :s, -s:], r[:, -s:, :s], r[:, -s:, -s:], r[:, c:c + s, c:c + s],
                     F.interpolate(r[None], size=(224, 224), mode="bilinear", antialias=True,
                                   align_corners=False)[0]]
            out.append(torch.stack(crops))
    return torch.stack(out)


def extract(enc, imgs, cls_all, mean_all, batch, dev):
    for img in imgs:
        v = views(img).reshape(-1, 3, 224, 224)  # 48 views
        cs, ms = [], []
        for i in range(0, len(v), batch):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                c, m = enc(v[i:i + batch].to(dev))
            cs.append(c.float().cpu()); ms.append(m.float().cpu())
        cls_all.append(torch.cat(cs).reshape(8, 6, -1)); mean_all.append(torch.cat(ms).reshape(8, 6, -1))


def load_patches(limit, raw=False):
    rows = list(csv.DictReader(open(HERE / "nuinsseg_mapping.csv")))[: limit or None]
    names = json.load(open(HERE / "splits.json"))["class_names"]
    short = {n.replace("human_tissue_image-", ""): i for i, n in enumerate(names)}
    hc.CFG.LOAD_SIZE = 512
    if not raw:
        hc.init_stain_normalizer(str(NUIN / REFERENCE_PATCH))
    imgs, meta = [], []
    for r in rows:
        p = NUIN / r["nuinsseg_patch"].replace("/", "/tissue images/", 1)
        a = np.asarray(Image.open(p).convert("RGB"))
        a = a if raw else hc.apply_stain_normalization(a)
        imgs.append(torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0)
        meta.append((int(r["group_id"]), short[r["class"]], r["nuinsseg_patch"]))
    return imgs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--raw", action="store_true", help="no Macenko normalisation; writes fm_features_raw/")
    args = ap.parse_args()
    global OUT
    if args.raw:
        OUT = HERE / "fm_features_raw"
    OUT.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imgs, meta = load_patches(args.limit, args.raw)
    print(f"{len(imgs)} patches loaded and stain-normalised", flush=True)
    for name in args.models.split(","):
        dst = OUT / f"{name}.npz"
        if dst.exists() and not args.limit:
            print(f"skip {name} (exists)"); continue
        lock = OUT / f".{name}.lock"
        if not args.limit:
            try:
                lock.mkdir()  # one job per model when several jobs share the work
            except FileExistsError:
                print(f"skip {name} (locked by another job)"); continue
        t0 = time.time()
        try:
            enc = Encoder(name).to(dev).eval()
        except Exception as e:
            print(f"FAILED to load {name}: {e!r}"[:400], flush=True)
            continue
        cls_all, mean_all = [], []
        try:
            extract(enc, imgs, cls_all, mean_all, args.batch, dev)
        except Exception as e:
            print(f"FAILED to extract {name}: {e!r}"[:400], flush=True)
            continue
        cls_all = torch.stack(cls_all).numpy().astype(np.float16)
        mean_all = torch.stack(mean_all).numpy().astype(np.float16)
        if args.limit:
            dst = OUT / f"smoke_{name}.npz"
        np.savez(dst, cls=cls_all, mean=mean_all, group_id=np.array([m[0] for m in meta]),
                 label=np.array([m[1] for m in meta]), patch=np.array([m[2] for m in meta]))
        print(f"{name}: cls {cls_all.shape} in {time.time() - t0:.0f}s, finite={np.isfinite(cls_all).all()}", flush=True)
        del enc
        torch.cuda.empty_cache() if dev == "cuda" else None


if __name__ == "__main__":
    main()
