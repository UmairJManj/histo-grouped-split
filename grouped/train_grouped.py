"""Group-aware retrain of HistoFormerFusion and its baselines.

One call trains one (model, unit) pair on the split in splits.json:
  unit = final  -> train on final.train, select on final.val, evaluate on test
  unit = foldK  -> train on cv[K-1].train, evaluate on cv[K-1].val
Folds and the final train/val split come from splits.json["per_seed"][seed];
the test set is the same for every seed.

The training methodology is train_histo.py unchanged (imported as histo_core):
global CFG (batch 16, bf16 AMP, AdamW 1e-4 head / 1e-5 backbones, focal loss
gamma 2 + class weights + label smoothing 0.1, weighted sampler, 4 frozen
epochs, 40 epochs, patience 20, best-val-loss checkpoint restored).

Variants (env HISTO_VARIANT, underscore-separated tokens; results in <variant>/runs/):
  hires    same files, groups, multiplicity and split, but each file is read from
           its ORIGINAL 512x512 NuInsSeg patch (nuinsseg_mapping.csv); cached at 512,
           Macenko on the 512 image, RandomResizedCrop from 512; ResNet50 384,
           InceptionResNetV2 384, ViT-B/16 384 (vit_base_patch16_384); other CNNs 384.
  phikon   ViT branch = owkin/phikon (ViT-B/16, H&E-pretrained, CLS token, 768-d) at 224
  phikon2  ViT branch = owkin/phikon-v2 (ViT-L/16, CLS token, 1024-d) at 224
  pNN      early-stopping patience NN
  eNN      max epochs NN (cosine schedule spans NN epochs)
  aug      RandomResizedCrop scale 0.5-1.0 and ColorJitter 0.3/0.3/0.3/0.1 (else 0.75-1.0, 0.2/0.2/0.2/0.05)
  rot90    training also applies a random 90-degree rotation (k=0..3), giving the full dihedral
           group with the existing flips (H&E has no canonical orientation)
  bdNN     branch dropout: during training each fusion branch weight is zeroed with
           probability NN/100 per image (at least one branch kept), weights renormalised
Every run also stores validation and test predictions with 8-view dihedral TTA.

Usage: python train_grouped.py --model <MODELS> --unit final|fold1..fold5 --seed 42|43|44
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

import histo_core as hc

HERE = Path(__file__).parent
# extra CNN baselines: timm tag, input size (all ImageNet mean/std, torchvision weights where available)
EXTRA_CNNS = {
    "vgg16": ("vgg16.tv_in1k", 224),
    "densenet121": ("densenet121.tv_in1k", 224),
    "efficientnet_b0": ("efficientnet_b0.ra_in1k", 224),
}
MODELS = ("fusion", "meanfusion", "vit", "resnet", "inception", *EXTRA_CNNS)
IMAGENET = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
REFERENCE = "human_tissue_image-spleen/image_876.png"  # Macenko target of job 7172798


VARIANT = os.environ.get("HISTO_VARIANT", "base")
TOKENS = set(VARIANT.split("_"))
IS_HIRES = "hires" in TOKENS
VIT_BACKEND = "phikon2" if "phikon2" in TOKENS else "phikon" if "phikon" in TOKENS else "timm"
PATIENCE = next((int(t[1:]) for t in TOKENS if t[:1] == "p" and t[1:].isdigit()), None)
MAX_EPOCHS = next((int(t[1:]) for t in TOKENS if t[:1] == "e" and t[1:].isdigit()), None)
BRANCH_DROP = next((int(t[2:]) / 100 for t in TOKENS if t[:2] == "bd" and t[2:].isdigit()), 0.0)
STRONG_AUG = "aug" in TOKENS
ROT90 = "rot90" in TOKENS
PHIKON = {"phikon": ("owkin/phikon", 768), "phikon2": ("owkin/phikon-v2", 1024)}
NUINSSEG = Path("/nesi/nobackup/aut04653/jfd1673/nuinsseg")
HIRES = {"LOAD_SIZE": 512, "RESNET_SIZE": 384, "INCEPTION_SIZE": 384, "VIT_SIZE": 384,
         "VIT_NAME": "vit_base_patch16_384.augreg_in21k_ft_in1k"}
HIRES_EXTRA_SIZE = 384


def hires_path_map():
    """dataset file (class_dir/image_n.png) -> original 512x512 NuInsSeg patch."""
    import csv
    out = {}
    for r in csv.DictReader(open(HERE / "nuinsseg_mapping.csv")):
        folder = r["example_file"].split("/")[0]
        for f in r["all_files"].split(";"):
            out[f"{folder}/{f}"] = str(NUINSSEG / r["nuinsseg_patch"].replace("/", "/tissue images/", 1))
    return out


class PhikonBackbone(nn.Module):
    """HuggingFace Phikon / Phikon-v2 encoder; CLS token as the feature (ImageNet mean/std)."""

    def __init__(self, which, target_size=224):
        super().__init__()
        from transformers import AutoModel
        name, self.feature_dim = PHIKON[which]
        self.model = AutoModel.from_pretrained(name)
        self.target_size = target_size
        self.register_buffer("mean", torch.tensor(IMAGENET["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET["std"]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x_uint8):
        x = x_uint8.float() / 255.0
        if x.shape[-1] != self.target_size:
            x = torch.nn.functional.interpolate(x, size=(self.target_size, self.target_size),
                                                mode="bilinear", align_corners=False, antialias=True)
        return self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, 0]

    def set_frozen(self, frozen):
        for p in self.model.parameters():
            p.requires_grad = not frozen


def strong_train_transform():
    T, C = hc.T, hc.CFG
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(C.LOAD_SIZE, scale=(0.5, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        T.ToDtype(torch.uint8, scale=True),
    ])


class RandomRot90:
    """Rotate a CHW tensor by a random multiple of 90 degrees."""

    def __init__(self, base):
        self.base = base

    def __call__(self, img):
        x = self.base(img)
        return torch.rot90(x, int(torch.randint(4, (1,))), dims=(-2, -1))


def add_branch_dropout(model, p):
    """Wrap model.fusion.forward: in training, drop branch weights with prob p, renormalise."""
    inner = model.fusion.forward

    def forward(features_list):
        fused, weights = inner(features_list)
        if not model.training or p <= 0:
            return fused, weights
        projected = torch.stack([proj(f) for proj, f in zip(model.fusion.projections, features_list)], dim=1)
        keep = (torch.rand_like(weights) >= p).float()
        keep[keep.sum(1) == 0, torch.randint(weights.shape[1], (1,)).item()] = 1.0
        w = weights * keep
        w = w / w.sum(1, keepdim=True).clamp_min(1e-8)
        return (projected * w.unsqueeze(-1)).sum(1), w

    model.fusion.forward = forward
    return model


def apply_variant():
    if MAX_EPOCHS:
        hc.CFG.EPOCHS = MAX_EPOCHS
    if STRONG_AUG:
        hc.build_train_transform = strong_train_transform
    if ROT90:
        base_builder = hc.build_train_transform
        hc.build_train_transform = lambda: RandomRot90(base_builder())
    if IS_HIRES:
        for k, v in HIRES.items():
            setattr(hc.CFG, k, v)
    if VIT_BACKEND != "timm":
        hc.CFG.VIT_SIZE = 224
        hc.CFG.VIT_NAME = PHIKON[VIT_BACKEND][0]
        hc.CFG.FEATURE_DIM_VIT = PHIKON[VIT_BACKEND][1]
        hc.build_vit_backbone = lambda: PhikonBackbone(VIT_BACKEND, hc.CFG.VIT_SIZE)


def train_resumable(model, train_loader, val_loader, class_weights, run_name, epochs=None, log_every=20):
    """hc.train_one_run with identical maths, plus a per-epoch resume state
    (<ckpt dir>/<run>_resume.pt) so a run survives job requeues/timeouts."""
    C = hc.CFG
    epochs = epochs or C.EPOCHS
    device = C.DEVICE
    model.to(device)
    loss_fn = hc.FocalLoss(class_weights=class_weights if C.USE_CLASS_WEIGHTS else None,
                           gamma=C.FOCAL_GAMMA, label_smoothing=C.LABEL_SMOOTHING).to(device)
    optimizer, scheduler = hc.build_optimizer_and_scheduler(
        model, epochs, steps_per_epoch=len(train_loader), warmup_epochs=C.WARMUP_EPOCHS)
    amp = hc.AMPContext(C.USE_AMP, device, C.AMP_DTYPE)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss, best_state, patience_counter, start = float("inf"), None, 0, 0
    ckpt_path = os.path.join(C.OUTPUT_DIR, f"{run_name}_best.pt")
    resume_path = os.path.join(C.OUTPUT_DIR, f"{run_name}_resume.pt")
    if os.path.exists(resume_path):
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"]); amp.scaler.load_state_dict(st["scaler"])
        history, best_val_loss, patience_counter, start = st["history"], st["best_val_loss"], st["patience"], st["epoch"]
        best_state = torch.load(ckpt_path, map_location="cpu") if os.path.exists(ckpt_path) else None
        torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state_all(st["rng_cuda"]) if st.get("rng_cuda") else None
        print(f"  -> resumed {run_name} at epoch {start + 1} (best val_loss {best_val_loss:.4f})")
    stopped = st.get("stopped", False) if start else False
    for epoch in range(start, epochs):
        if stopped:
            break
        model.set_backbones_frozen(epoch < C.FREEZE_BACKBONES_EPOCHS)
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        t0 = time.time()
        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                logits = model(imgs)
                loss = loss_fn(logits, labels)
            amp.backward_and_step(loss, optimizer, model, C.GRAD_CLIP_NORM)
            scheduler.step()
            preds = logits.detach().argmax(dim=1)
            running_loss += loss.item() * imgs.size(0)
            running_correct += (preds == labels).sum().item()
            running_n += imgs.size(0)
            if (step + 1) % log_every == 0:
                print(f"  [{run_name}] epoch {epoch+1}/{epochs} step {step+1}/{len(train_loader)} "
                      f"loss={running_loss/running_n:.4f} acc={running_correct/running_n:.4f}")
        val_loss, val_acc, _, _ = hc.evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(running_loss / max(1, running_n))
        history["train_acc"].append(running_correct / max(1, running_n))
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[-1]["lr"])
        frozen_note = " [backbones frozen]" if epoch < C.FREEZE_BACKBONES_EPOCHS else ""
        print(f"[{run_name}] Epoch {epoch+1}/{epochs}{frozen_note} ({time.time()-t0:.1f}s) "
              f"train_loss={history['train_loss'][-1]:.4f} train_acc={history['train_acc'][-1]:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val_loss - C.MIN_DELTA:
            best_val_loss, patience_counter = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            print(f"  -> new best val_loss={val_loss:.4f}, checkpoint saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= C.EARLY_STOP_PATIENCE:
                print(f"  -> early stopping triggered at epoch {epoch+1} "
                      f"(no improvement for {C.EARLY_STOP_PATIENCE} epochs)")
                stopped = True
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": amp.scaler.state_dict(),
                    "history": history, "best_val_loss": best_val_loss, "patience": patience_counter,
                    "epoch": epoch + 1, "stopped": stopped, "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   resume_path + ".tmp")
        os.replace(resume_path + ".tmp", resume_path)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_tta(model, loader, views=8):
    """Softmax averaged over test-time views: 8 = dihedral (4 rotations x horizontal flip);
    4 = flips only (identity, horizontal, vertical, both), matching the training flips."""
    model.eval()
    probs, targets = [], []
    for imgs, labels in loader:
        x = imgs.to(hc.CFG.DEVICE, non_blocking=True)
        acc = 0
        if views == 8:
            batch = [torch.rot90(torch.flip(x, dims=(3,)) if f else x, k, dims=(2, 3))
                     for f in (False, True) for k in range(4)]
        else:
            batch = [x, torch.flip(x, dims=(3,)), torch.flip(x, dims=(2,)), torch.flip(x, dims=(2, 3))]
        for v in batch:
            acc = acc + torch.softmax(model(v).float(), dim=1)
        probs.append((acc / len(batch)).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def projection(dim):
    # same per-branch block as hc.AttentionFusion
    return nn.Sequential(nn.Linear(dim, hc.CFG.FUSION_DIM), nn.LayerNorm(hc.CFG.FUSION_DIM),
                         nn.GELU(), nn.Dropout(0.2))


def classifier(num_classes):
    # same head as hc.TriBackboneEnsemble
    d = hc.CFG.FUSION_DIM
    return nn.Sequential(nn.Linear(d, d // 2), nn.LayerNorm(d // 2), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(d // 2, num_classes))


class SingleBackbone(nn.Module):
    """One backbone -> same projection -> same classifier head."""

    def __init__(self, which, num_classes):
        super().__init__()
        builders = {"resnet": hc.build_resnet_backbone, "inception": hc.build_inception_backbone,
                    "vit": hc.build_vit_backbone}
        if which in builders:
            self.backbone = builders[which]()
        else:
            tag, size = EXTRA_CNNS[which]
            size = HIRES_EXTRA_SIZE if IS_HIRES else size
            self.backbone = hc.BackboneWrapper(tag, size, feature_dim=None, **IMAGENET)
            with torch.no_grad():  # VGG16's pooled pre-logits are 4096-d, not num_features (512)
                dummy = torch.zeros(1, 3, size, size, dtype=torch.uint8)
                self.backbone.feature_dim = self.backbone.eval()(dummy).shape[1]
            self.backbone.train()
        self.proj = projection(self.backbone.feature_dim)
        self.classifier = classifier(num_classes)

    def forward(self, x_uint8, return_attention=False):
        logits = self.classifier(self.proj(self.backbone(x_uint8)))
        return (logits, None) if return_attention else logits

    def set_backbones_frozen(self, frozen):
        self.backbone.set_frozen(frozen)

    def get_param_groups(self, base_lr, backbone_lr_mult):
        head = list(self.proj.parameters()) + list(self.classifier.parameters())
        return [{"params": list(self.backbone.parameters()), "lr": base_lr * backbone_lr_mult},
                {"params": head, "lr": base_lr}]


class MeanFusion(nn.Module):
    """Same projections as AttentionFusion, attention replaced by a fixed mean."""

    def __init__(self, feature_dims, fusion_dim):
        super().__init__()
        self.projections = nn.ModuleList([projection(d) for d in feature_dims])

    def forward(self, features_list):
        stacked = torch.stack([p(f) for p, f in zip(self.projections, features_list)], dim=1)
        weights = torch.full(stacked.shape[:2], 1.0 / stacked.shape[1], device=stacked.device)
        return stacked.mean(dim=1), weights


def build_model(name, num_classes):
    if name == "fusion":
        return hc.TriBackboneEnsemble(num_classes)
    if name == "meanfusion":
        model = hc.TriBackboneEnsemble(num_classes)
        dims = [hc.CFG.FEATURE_DIM_RESNET, hc.CFG.FEATURE_DIM_INCEPTION, hc.CFG.FEATURE_DIM_VIT]
        model.fusion = MeanFusion(dims, hc.CFG.FUSION_DIM)
        return model
    return SingleBackbone(name, num_classes)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, targets, attn = [], [], []
    for imgs, labels in loader:
        logits, w = model(imgs.to(hc.CFG.DEVICE, non_blocking=True), return_attention=True)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        targets.append(labels.numpy())
        if w is not None:
            attn.append(w.float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(targets), (np.concatenate(attn) if attn else None)


@torch.no_grad()
def inference_ms_per_image(model, path, n_warmup=10, n_runs=100):
    """Batch-1 forward latency incl. per-backbone resize, excluding disk I/O and stain norm."""
    model.eval()
    x = hc.build_eval_transform()(Image.open(path).convert("RGB")).unsqueeze(0).to(hc.CFG.DEVICE)
    for _ in range(n_warmup):
        model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return round((time.perf_counter() - t0) / n_runs * 1000, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="override, smoke tests only")
    ap.add_argument("--limit", type=int, default=0, help="cap samples per partition, smoke tests only")
    args = ap.parse_args()

    splits = json.load(open(HERE / "splits.json"))
    names = splits["class_names"]
    hc.CFG.NUM_CLASSES, hc.CFG.CLASS_NAMES = len(names), names
    seed_split = splits["per_seed"][str(args.seed)]
    if args.unit == "final":
        tr, ev = seed_split["final"]["train"], seed_split["final"]["val"]
        test = splits["test"]
    else:
        k = int(args.unit.removeprefix("fold")) - 1
        tr, ev = seed_split["cv"][k]["train"], seed_split["cv"][k]["val"]
        test = ev
    apply_variant()
    if IS_HIRES:
        src = hires_path_map()
        key = lambda p: "/".join(Path(p).parts[-2:])
        as_samples = lambda rows: [(src[key(p)], l) for p, l, _ in rows][: args.limit or None]
    else:
        as_samples = lambda rows: [(p, l) for p, l, _ in rows][: args.limit or None]
    tr_s, ev_s, te_s = as_samples(tr), as_samples(ev), as_samples(test)
    test_groups = np.array([g for _, _, g in test][: args.limit or None])

    runs_root = HERE / VARIANT / "runs" if VARIANT != "base" else HERE / "runs"
    out_dir = HERE / os.environ["HISTO_RUNS_DIR"] if "HISTO_RUNS_DIR" in os.environ else runs_root
    out_dir = out_dir / args.model / f"seed{args.seed}" / args.unit
    out_dir.mkdir(parents=True, exist_ok=True)
    hc.CFG.OUTPUT_DIR = os.environ.get("HISTO_CKPT_DIR", str(out_dir))  # checkpoints go to nobackup
    os.makedirs(hc.CFG.OUTPUT_DIR, exist_ok=True)

    ref_file = hires_path_map()[REFERENCE] if IS_HIRES else REFERENCE
    ref = next((p for p, _ in tr_s if p.endswith(ref_file)), tr_s[0][0])
    hc.init_stain_normalizer(ref)
    hc.CFG.EARLY_STOP_PATIENCE = int(os.environ.get("HISTO_PATIENCE", hc.CFG.EARLY_STOP_PATIENCE))
    if PATIENCE is not None:
        hc.CFG.EARLY_STOP_PATIENCE = PATIENCE
    hc.CFG.SEED = args.seed
    hc.set_seed(args.seed)

    train_loader, val_loader, test_loader = hc.build_dataloaders(tr_s, ev_s, te_s)
    model = build_model(args.model, len(names))
    if BRANCH_DROP and args.model in ("fusion", "meanfusion"):
        add_branch_dropout(model, BRANCH_DROP)
    n_params = sum(p.numel() for p in model.parameters())
    class_weights = hc.compute_class_weights(tr_s, len(names), hc.CFG.DEVICE)

    run_name = f"{args.model}_seed{args.seed}_{args.unit}"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train = train_resumable if VARIANT != "base" else hc.train_one_run   # base runs: unchanged code path
    model, history = train(model, train_loader, val_loader, class_weights,
                           run_name=run_name, epochs=args.epochs)
    train_seconds = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    ckpt = Path(hc.CFG.OUTPUT_DIR) / f"{run_name}_best.pt"
    ckpt_mb = ckpt.stat().st_size / 1024 ** 2 if ckpt.exists() else None
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    probs, targets, attn = predict(model, test_loader)
    preds = probs.argmax(1)
    infer_ms = inference_ms_per_image(model, te_s[0][0]) if args.unit == "final" else None
    metrics = hc._metrics_for_json(hc.compute_all_metrics(targets, preds, probs, len(names), names))
    probs_tta, _ = predict_tta(model, test_loader)
    metrics_tta = hc._metrics_for_json(hc.compute_all_metrics(targets, probs_tta.argmax(1), probs_tta,
                                                              len(names), names))
    # validation predictions (model selection material: variants, TTA, ensembling)
    val_probs, val_t, _ = predict(model, val_loader)
    val_probs_tta, _ = predict_tta(model, val_loader)
    val_probs_tta4, _ = predict_tta(model, val_loader, views=4)
    probs_tta4, _ = predict_tta(model, test_loader, views=4)

    result = {
        "model": args.model, "unit": args.unit, "seed": args.seed, "variant": VARIANT,
        "input_sizes": {k: getattr(hc.CFG, k) for k in ("LOAD_SIZE", "RESNET_SIZE", "INCEPTION_SIZE", "VIT_SIZE")},
        "eval_set": "test" if args.unit == "final" else "cv_val_fold",
        "n_train": len(tr_s), "n_val": len(ev_s), "n_eval": len(te_s),
        "n_params": n_params,
        "checkpoint_rule": "lowest validation loss (min_delta 1e-4), restored before evaluation",
        "early_stop_patience": hc.CFG.EARLY_STOP_PATIENCE, "max_epochs": hc.CFG.EPOCHS,
        "best_epoch": best_epoch, "epochs_run": len(history["val_loss"]),
        "best_val_loss": float(min(history["val_loss"])),
        "val_acc_at_best": float((val_probs.argmax(1) == val_t).mean()),
        "val_acc_tta": float((val_probs_tta.argmax(1) == val_t).mean()),
        "val_f1_macro": float(hc.f1_score(val_t, val_probs.argmax(1), average="macro", zero_division=0)),
        "val_f1_macro_tta": float(hc.f1_score(val_t, val_probs_tta.argmax(1), average="macro", zero_division=0)),
        "val_acc_tta4": float((val_probs_tta4.argmax(1) == val_t).mean()),
        "val_f1_macro_tta4": float(hc.f1_score(val_t, val_probs_tta4.argmax(1), average="macro", zero_division=0)),
        "vit_backend": VIT_BACKEND, "strong_aug": STRONG_AUG, "rot90": ROT90, "branch_dropout": BRANCH_DROP,
        "train_seconds": round(train_seconds, 1),
        "peak_gpu_mem_gb": round(peak_gb, 2),
        "checkpoint_mb": round(ckpt_mb, 1) if ckpt_mb else None,
        "inference_ms_per_image_bs1": infer_ms,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "stain_reference": ref,
        "metrics": metrics,
        "metrics_tta": metrics_tta,
    }
    if attn is not None:
        result["attention_mean"] = dict(zip(["resnet50", "inception_resnet_v2", "vit_b16"],
                                            attn.mean(0).round(4).tolist()))
    json.dump(result, open(out_dir / "result.json", "w"), indent=2)
    json.dump(history, open(out_dir / "history.json", "w"), indent=2)
    np.savez(out_dir / "predictions.npz", probs=probs, targets=targets, groups=test_groups,
             attention=attn if attn is not None else np.zeros(0), probs_tta=probs_tta, probs_tta4=probs_tta4,
             val_probs=val_probs, val_probs_tta=val_probs_tta, val_probs_tta4=val_probs_tta4, val_targets=val_t)
    if VARIANT == "base":
        print(f"[done] {args.model}/seed{args.seed}/{args.unit} acc={metrics['accuracy']:.4f} "
              f"f1_macro={metrics['f1_macro']:.4f} best_epoch={best_epoch} ({train_seconds/60:.1f} min)")
    else:  # screening variants: never surface test metrics before the configuration is frozen
        print(f"[done] {VARIANT}/{args.model}/seed{args.seed}/{args.unit} VAL acc={result['val_acc_at_best']:.4f} "
              f"tta4={result['val_acc_tta4']:.4f} tta8={result['val_acc_tta']:.4f} best_epoch={best_epoch} "
              f"({train_seconds/60:.1f} min)")


if __name__ == "__main__":
    main()
