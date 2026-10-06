"""Auto-generated from tissue_ensemble_fixed_(1).ipynb for NeSI SLURM.
Colab cells (pip/upload/zip/debug) removed; paths & epochs from env vars:
  HISTO_DATA_DIR, HISTO_OUTPUT_DIR, HISTO_EPOCHS
"""


import os
import json
import math
import random
import warnings
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import timm
from PIL import Image

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. CONFIG -- edit this block for your run, nothing else needs touching
# ==============================================================================

class CFG:
    # --- paths ---
    DATA_DIR = os.environ.get("HISTO_DATA_DIR", "dataset")            # contains human_tissue_image-<class>/ subfolders
    OUTPUT_DIR = os.environ.get("HISTO_OUTPUT_DIR", "outputs")          # checkpoints, plots, logs all go here

    # --- classes are auto-discovered from folder names, but listed here for reference ---
    # human_tissue_image-bladder, -brain, -cardia, -cerebellum, -epiglottis, -gland,
    # -jejunum, -kidney, -liver, -lung, -melanoma, -muscle, -oesophagus, -pancreas,
    # -peritoneum, -pylorus, -rectum, -spleen, -testis, -tongue, -tonsile, -umbilical-cord

    # --- split ---
    VAL_FRACTION = 0.15
    TEST_FRACTION = 0.15
    SPLIT_SEED = 42

    # --- cross-validation ---
    USE_KFOLD = True
    N_FOLDS = 5                     # set USE_KFOLD=False to do a single train/val/test run

    # --- image loading ---
    LOAD_SIZE = 320                 # images are loaded/cached at this size; backbones resize down from here
    STAIN_NORMALIZE = True          # Macenko H&E stain normalization (set False if not histology / torchstain unavailable)

    # --- backbones (verified timm tags, June 2026) ---
    RESNET_NAME = "resnet50.tv2_in1k"
    INCEPTION_NAME = "inception_resnet_v2.tf_in1k"
    VIT_NAME = "vit_base_patch16_224.augreg_in21k_ft_in1k"

    RESNET_SIZE = 176
    INCEPTION_SIZE = 299
    VIT_SIZE = 224

    FEATURE_DIM_RESNET = 2048
    FEATURE_DIM_INCEPTION = 1536
    FEATURE_DIM_VIT = 768

    FUSION_DIM = 768                # shared projection dim before attention fusion

    # --- training ---
    # NOTE on BATCH_SIZE: this ensemble carries ~167M params (ResNet50 ~25M +
    # InceptionResNetV2 ~56M + ViT-B/16 ~86M) all resident and trained
    # simultaneously, each running its own forward/backward at its own native
    # resolution (176/299/224). On a Colab T4 (16GB, no bf16), BATCH_SIZE=16
    # commonly OOMs once the backbones unfreeze (activations for all three
    # branches + AMP buffers + optimizer state for 167M params). Use 8 on T4.
    # L4/A100/V100 (24GB+, bf16-capable) can comfortably run 16, and you can
    # likely push higher there. CFG.DEVICE auto-detects bf16 support below;
    # it does NOT auto-detect total VRAM, so set BATCH_SIZE for your GPU.
    BATCH_SIZE = 16                 # set to 8 on T4; 16 is safe on L4/A100/V100
    NUM_WORKERS = 4
    EPOCHS = int(os.environ.get("HISTO_EPOCHS", "40"))
    WARMUP_EPOCHS = 3
    BASE_LR = 1e-4
    BACKBONE_LR_MULT = 0.1          # backbones fine-tune slower than the fusion head
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1           # blended with focal loss below
    GRAD_CLIP_NORM = 1.0
    EARLY_STOP_PATIENCE = 20
    MIN_DELTA = 1e-4

    # --- progressive unfreezing ---
    # 🐛 FIX: this was set to 10, equal to EPOCHS (also 10), so backbones were
    # frozen for the ENTIRE run and never actually fine-tuned -- confirmed by
    # the training logs, which printed "[backbones frozen]" on every single
    # epoch. Set well below EPOCHS so there's real fine-tuning time left.
    FREEZE_BACKBONES_EPOCHS = 4      # train fusion head only for this many epochs first

    # --- focal loss ---
    FOCAL_GAMMA = 2.0
    USE_CLASS_WEIGHTS = True

    # --- mixed precision ---
    USE_AMP = True
    AMP_DTYPE = torch.bfloat16      # bf16 is stable on A100/L4/TPU-adjacent and modern Colab GPUs;
                                     # falls back to fp16 automatically on older GPUs (see setup code)

    # --- reproducibility ---
    SEED = 42

    # --- device ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- misc ---
    NUM_CLASSES = None              # filled in automatically after dataset discovery
    CLASS_NAMES = None              # filled in automatically


os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
print(f"Device: {CFG.DEVICE}")
if CFG.DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    # bf16 needs Ampere+ (compute capability >= 8.0); fall back to fp16 on older GPUs (e.g. T4 = 7.5)
    major, _ = torch.cuda.get_device_capability(0)
    if major < 8:
        CFG.AMP_DTYPE = torch.float16
        print("GPU is pre-Ampere -> using fp16 AMP instead of bf16")

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Total VRAM: {total_vram_gb:.1f} GB")
    if total_vram_gb < 20 and CFG.BATCH_SIZE > 8:
        print(
            f"WARNING: detected ~{total_vram_gb:.0f}GB VRAM (likely a T4) with "
            f"BATCH_SIZE={CFG.BATCH_SIZE}. This tri-backbone ensemble (~167M params: "
            f"ResNet50 + InceptionResNetV2 + ViT-B/16, all training simultaneously at "
            f"three different resolutions) is likely to OOM at this batch size once "
            f"backbones unfreeze. Strongly recommend setting CFG.BATCH_SIZE = 8 on T4. "
            f"Reserve BATCH_SIZE=16 for L4/A100/V100."
        )


# ==============================================================================
# 2. REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic-ish but doesn't kill performance on cuDNN
    torch.backends.cudnn.benchmark = True


set_seed(CFG.SEED)


# ==============================================================================
# 3. DATASET DISCOVERY + STRATIFIED SPLIT
# ==============================================================================
# Works directly on the ImageFolder-style layout you already have:
#   dataset/human_tissue_image-<tissue>/*.jpg|png|...
# This is the SAME layout Keras's image_dataset_from_directory, PyTorch's
# ImageFolder, FastAI's ImageDataLoaders.from_folder, and torchvision's
# ImageFolder all expect natively -- no restructuring needed.

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_dataset(data_dir: str):
    """Scan data_dir for class subfolders and build a (filepath, label_idx) list."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"'{data_dir}' not found. Expected structure:\n"
            f"  {data_dir}/human_tissue_image-bladder/*.jpg\n"
            f"  {data_dir}/human_tissue_image-brain/*.jpg\n  ..."
        )

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise FileNotFoundError(f"No class subfolders found inside '{data_dir}'.")

    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = []
    for d in class_dirs:
        label = class_to_idx[d.name]
        files = [f for f in d.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS]
        for f in files:
            samples.append((str(f), label))

    if len(samples) == 0:
        raise FileNotFoundError(
            f"Found {len(class_dirs)} class folders but 0 images with extensions "
            f"{IMG_EXTENSIONS}. Check your file types."
        )

    print(f"Discovered {len(class_names)} classes, {len(samples)} total images.")
    counts = Counter(label for _, label in samples)
    for name, idx in class_to_idx.items():
        print(f"  {name:45s} {counts.get(idx, 0):6d} images")

    return samples, class_names, class_to_idx


def stratified_split(samples, val_frac: float, test_frac: float, seed: int):
    """Per-class stratified split into train/val/test, so rare tissue classes
    still get representation in every split."""
    by_class = defaultdict(list)
    for path, label in samples:
        by_class[label].append(path)

    rng = random.Random(seed)
    train_set, val_set, test_set = [], [], []

    for label, paths in by_class.items():
        paths = paths.copy()
        rng.shuffle(paths)
        n = len(paths)
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_frac))) if n - n_test >= 2 else 0
        n_train = n - n_val - n_test

        if n_train <= 0:
            # extremely small class: guarantee at least 1 train sample
            n_train = 1
            n_val = min(n_val, max(0, n - n_train - n_test))

        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:n_train + n_val + n_test]

        train_set += [(p, label) for p in train_paths]
        val_set += [(p, label) for p in val_paths]
        test_set += [(p, label) for p in test_paths]

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)
    return train_set, val_set, test_set


# ==============================================================================
# 4. STAIN NORMALIZATION (Macenko) -- optional, for H&E-style histology images
# ==============================================================================
# Uses torchstain (actively maintained, pip install torchstain). Wrapped
# defensively: Macenko's matrix decomposition can fail on near-flat / low
# stain-contrast tiles, so on any failure we silently fall back to the
# original image rather than crashing a training run over one bad tile.

_stain_normalizer = None


def init_stain_normalizer(reference_image_path: str = None):
    """Fit a Macenko normalizer to a reference image so all images are mapped
    to a consistent stain appearance. If no reference is given, uses the
    first image found in the dataset as the reference target."""
    global _stain_normalizer
    if not CFG.STAIN_NORMALIZE:
        return None
    try:
        import torchstain
    except ImportError:
        print("torchstain not installed (`pip install torchstain`) -- "
              "disabling stain normalization.")
        CFG.STAIN_NORMALIZE = False
        return None

    if reference_image_path is None:
        return None

    target = np.array(Image.open(reference_image_path).convert("RGB").resize((CFG.LOAD_SIZE, CFG.LOAD_SIZE)))
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
    try:
        normalizer.fit(target)
    except Exception as e:
        print(f"Stain normalizer failed to fit reference image ({e}); disabling.")
        CFG.STAIN_NORMALIZE = False
        return None

    _stain_normalizer = normalizer
    print(f"Stain normalizer fitted on reference: {reference_image_path}")
    return normalizer


def apply_stain_normalization(img_np: np.ndarray) -> np.ndarray:
    """img_np: HxWx3 uint8 RGB. Returns HxWx3 uint8 RGB, normalized if possible,
    original image untouched on any failure."""
    if _stain_normalizer is None or not CFG.STAIN_NORMALIZE:
        return img_np
    try:
        result, _, _ = _stain_normalizer.normalize(I=img_np)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    except Exception:
        return img_np  # fall back silently -- one bad tile shouldn't crash training


# ==============================================================================
# 5. AUGMENTATION
# ==============================================================================
# Built on torchvision.transforms.v2 (the current, maintained API). Applied at
# CFG.LOAD_SIZE; each backbone branch then resizes down to its own native
# resolution inside the model forward pass.

from torchvision.transforms import v2 as T

def build_train_transform():
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(CFG.LOAD_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        T.ToDtype(torch.uint8, scale=True),
    ])


def build_eval_transform():
    return T.Compose([
        T.ToImage(),
        T.Resize((CFG.LOAD_SIZE, CFG.LOAD_SIZE), antialias=True),
        T.ToDtype(torch.uint8, scale=True),
    ])


# ==============================================================================
# 6. DATASET + DATALOADER
# ==============================================================================
# Returns one image tensor at CFG.LOAD_SIZE (uint8, HWC->CHW via ToImage).
# Per-backbone resize + normalization happens inside the model, NOT here,
# because each backbone needs a different resolution and normalization.

class TissueDataset(Dataset):
    def __init__(self, samples, transform, stain_normalize=False):
        self.samples = samples
        self.transform = transform
        self.stain_normalize = stain_normalize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)

        if self.stain_normalize:
            img_np = apply_stain_normalization(img_np)

        img = Image.fromarray(img_np)
        img_tensor = self.transform(img)  # uint8 CHW tensor at LOAD_SIZE
        return img_tensor, label


def make_weighted_sampler(samples):
    """Inverse-frequency sampler so rare tissue classes are seen as often as
    common ones during training (complements, doesn't replace, class weights
    in the loss)."""
    labels = [label for _, label in samples]
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders(train_samples, val_samples, test_samples):
    train_ds = TissueDataset(train_samples, build_train_transform(), CFG.STAIN_NORMALIZE)
    val_ds = TissueDataset(val_samples, build_eval_transform(), CFG.STAIN_NORMALIZE)
    test_ds = TissueDataset(test_samples, build_eval_transform(), CFG.STAIN_NORMALIZE)

    train_sampler = make_weighted_sampler(train_samples)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, sampler=train_sampler,
        num_workers=CFG.NUM_WORKERS, pin_memory=(CFG.DEVICE == "cuda"),
        drop_last=True, persistent_workers=(CFG.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=(CFG.DEVICE == "cuda"),
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=(CFG.DEVICE == "cuda"),
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )
    return train_loader, val_loader, test_loader


# ==============================================================================
# 7. BACKBONE WRAPPERS
# ==============================================================================
# Each wrapper owns its own resize + normalization, because the three
# backbones disagree on both (see header note). Input to every wrapper is the
# SAME uint8 CHW batch at CFG.LOAD_SIZE coming out of the DataLoader; each
# wrapper independently produces the resolution/normalization it needs.

class BackboneWrapper(nn.Module):
    def __init__(self, timm_name, target_size, mean, std, feature_dim):
        super().__init__()
        self.model = timm.create_model(timm_name, pretrained=True, num_classes=0, global_pool="avg")
        self.target_size = target_size
        self.feature_dim = feature_dim
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.frozen = False

    def preprocess(self, x_uint8):
        # x_uint8: (B, 3, H, W) uint8 in [0,255]
        x = x_uint8.float() / 255.0
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                               mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return x

    def forward(self, x_uint8):
        x = self.preprocess(x_uint8)
        return self.model(x)  # (B, feature_dim)

    def set_frozen(self, frozen: bool):
        self.frozen = frozen
        for p in self.model.parameters():
            p.requires_grad = not frozen
        # BatchNorm/dropout behave correctly even when frozen, since we still
        # call .train()/.eval() on the whole ensemble; freezing only stops
        # gradient updates to backbone weights, it doesn't touch running stats
        # mode -- that's controlled separately if you want it (see note below).


def build_resnet_backbone():
    return BackboneWrapper(
        CFG.RESNET_NAME, CFG.RESNET_SIZE,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        feature_dim=CFG.FEATURE_DIM_RESNET,
    )


def build_inception_backbone():
    return BackboneWrapper(
        CFG.INCEPTION_NAME, CFG.INCEPTION_SIZE,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        feature_dim=CFG.FEATURE_DIM_INCEPTION,
    )


def build_vit_backbone():
    return BackboneWrapper(
        CFG.VIT_NAME, CFG.VIT_SIZE,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        feature_dim=CFG.FEATURE_DIM_VIT,
    )


# ==============================================================================
# 8. ATTENTION FUSION
# ==============================================================================
# Each backbone's feature vector is projected to a shared FUSION_DIM space,
# then a small attention network learns a per-sample, per-backbone scalar
# weight (softmax over the 3 backbones) before summing -- i.e. the model
# learns, per image, how much to trust ResNet vs InceptionResNetV2 vs ViT,
# rather than fixing that weighting a priori.

class AttentionFusion(nn.Module):
    def __init__(self, feature_dims, fusion_dim):
        super().__init__()
        self.n_branches = len(feature_dims)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            for dim in feature_dims
        ])
        # attention score network: takes each projected feature, outputs 1 scalar logit
        self.attention_scorer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.Tanh(),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(self, features_list):
        # features_list: list of (B, dim_i) tensors, one per backbone
        projected = [proj(feat) for proj, feat in zip(self.projections, features_list)]
        stacked = torch.stack(projected, dim=1)  # (B, n_branches, fusion_dim)

        scores = self.attention_scorer(stacked).squeeze(-1)     # (B, n_branches)
        weights = torch.softmax(scores, dim=1)                  # (B, n_branches)

        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)    # (B, fusion_dim)
        return fused, weights  # return weights too, for interpretability/plots


# ==============================================================================
# 9. FULL ENSEMBLE MODEL
# ==============================================================================

class TriBackboneEnsemble(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = build_resnet_backbone()
        self.inception = build_inception_backbone()
        self.vit = build_vit_backbone()

        feature_dims = [CFG.FEATURE_DIM_RESNET, CFG.FEATURE_DIM_INCEPTION, CFG.FEATURE_DIM_VIT]
        self.fusion = AttentionFusion(feature_dims, CFG.FUSION_DIM)

        self.classifier = nn.Sequential(
            nn.Linear(CFG.FUSION_DIM, CFG.FUSION_DIM // 2),
            nn.LayerNorm(CFG.FUSION_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(CFG.FUSION_DIM // 2, num_classes),
        )

    def forward(self, x_uint8, return_attention=False):
        f_resnet = self.resnet(x_uint8)
        f_inception = self.inception(x_uint8)
        f_vit = self.vit(x_uint8)

        fused, attn_weights = self.fusion([f_resnet, f_inception, f_vit])
        logits = self.classifier(fused)

        if return_attention:
            return logits, attn_weights
        return logits

    def set_backbones_frozen(self, frozen: bool):
        self.resnet.set_frozen(frozen)
        self.inception.set_frozen(frozen)
        self.vit.set_frozen(frozen)

    def get_param_groups(self, base_lr, backbone_lr_mult):
        """Separate param groups: backbones fine-tune at a lower LR than the
        freshly-initialized fusion + classifier head."""
        backbone_params = (
            list(self.resnet.parameters())
            + list(self.inception.parameters())
            + list(self.vit.parameters())
        )
        head_params = list(self.fusion.parameters()) + list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
            {"params": head_params, "lr": base_lr},
        ]


# ==============================================================================
# 10. FOCAL LOSS (multi-class, with class weights + label smoothing)
# ==============================================================================
# Standard multi-class focal loss (Lin et al. 2017), generalized with
# per-class alpha weights (computed from training-set class frequency) and
# blended with label smoothing for calibration. Pure PyTorch -- no TF Addons
# equivalent needed; this is the same loss in spirit.

class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weights", class_weights if class_weights is not None else None,
                              persistent=False)

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # label smoothing applied to the one-hot target distribution
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.label_smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        pt = (probs * true_dist).sum(dim=1).clamp(min=1e-7)   # "probability of true class" (smoothed)
        focal_term = (1 - pt) ** self.gamma

        ce = -(true_dist * log_probs).sum(dim=1)               # smoothed cross-entropy

        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]
            loss = focal_term * ce * sample_weights
        else:
            loss = focal_term * ce

        return loss.mean()


def compute_class_weights(train_samples, num_classes, device):
    """Inverse-frequency class weights, normalized so mean weight == 1.0,
    used as the alpha term in focal loss (sampler above handles the
    data-balancing side; this handles the loss-balancing side)."""
    counts = Counter(label for _, label in train_samples)
    freqs = np.array([counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    weights = 1.0 / freqs
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ==============================================================================
# 11. OPTIMIZER / SCHEDULER / AMP SETUP
# ==============================================================================

def build_optimizer_and_scheduler(model, total_epochs, steps_per_epoch, warmup_epochs):
    param_groups = model.get_param_groups(CFG.BASE_LR, CFG.BACKBONE_LR_MULT)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=CFG.WEIGHT_DECAY)

    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay to 0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


class AMPContext:
    """Wraps autocast + GradScaler so the training loop doesn't need to branch
    on CPU vs CUDA vs bf16 vs fp16 -- it just calls .step(loss, optimizer)."""
    def __init__(self, enabled, device, dtype):
        self.enabled = enabled and device == "cuda"
        self.device = device
        self.dtype = dtype
        # GradScaler is only meaningful for fp16; bf16 has enough dynamic range
        # to skip loss scaling entirely (this is the standard modern guidance).
        self.use_scaler = self.enabled and dtype == torch.float16
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.use_scaler)

    def autocast(self):
        if self.enabled:
            return torch.autocast(device_type=self.device, dtype=self.dtype)
        return torch.autocast(device_type="cpu", enabled=False)

    def backward_and_step(self, loss, optimizer, model, clip_norm):
        if self.use_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()


# ==============================================================================
# 12. TRAINING LOOP
# ==============================================================================

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds, all_targets = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * imgs.size(0)
        total_correct += (preds == labels).sum().item()
        total_n += imgs.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    avg_loss = total_loss / max(1, total_n)
    acc = total_correct / max(1, total_n)
    all_preds = torch.cat(all_preds).numpy() if all_preds else np.array([])
    all_targets = torch.cat(all_targets).numpy() if all_targets else np.array([])
    return avg_loss, acc, all_preds, all_targets


def train_one_run(model, train_loader, val_loader, class_weights, run_name="run",
                   epochs=None, log_every=20):
    epochs = epochs or CFG.EPOCHS
    device = CFG.DEVICE
    model.to(device)

    loss_fn = FocalLoss(
        class_weights=class_weights if CFG.USE_CLASS_WEIGHTS else None,
        gamma=CFG.FOCAL_GAMMA,
        label_smoothing=CFG.LABEL_SMOOTHING,
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, epochs, steps_per_epoch=len(train_loader), warmup_epochs=CFG.WARMUP_EPOCHS
    )
    amp = AMPContext(CFG.USE_AMP, device, CFG.AMP_DTYPE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    ckpt_path = os.path.join(CFG.OUTPUT_DIR, f"{run_name}_best.pt")

    for epoch in range(epochs):
        if epoch < CFG.FREEZE_BACKBONES_EPOCHS:
            model.set_backbones_frozen(True)
        else:
            model.set_backbones_frozen(False)

        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        t0 = time.time()

        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                logits = model(imgs)
                loss = loss_fn(logits, labels)

            amp.backward_and_step(loss, optimizer, model, CFG.GRAD_CLIP_NORM)
            scheduler.step()

            preds = logits.detach().argmax(dim=1)
            running_loss += loss.item() * imgs.size(0)
            running_correct += (preds == labels).sum().item()
            running_n += imgs.size(0)

            if (step + 1) % log_every == 0:
                print(f"  [{run_name}] epoch {epoch+1}/{epochs} step {step+1}/{len(train_loader)} "
                      f"loss={running_loss/running_n:.4f} acc={running_correct/running_n:.4f}")

        train_loss = running_loss / max(1, running_n)
        train_acc = running_correct / max(1, running_n)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[-1]["lr"])

        frozen_note = " [backbones frozen]" if epoch < CFG.FREEZE_BACKBONES_EPOCHS else ""
        print(f"[{run_name}] Epoch {epoch+1}/{epochs}{frozen_note} "
              f"({elapsed:.1f}s) train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - CFG.MIN_DELTA:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            patience_counter = 0
            print(f"  -> new best val_loss={val_loss:.4f}, checkpoint saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= CFG.EARLY_STOP_PATIENCE:
                print(f"  -> early stopping triggered at epoch {epoch+1} "
                      f"(no improvement for {CFG.EARLY_STOP_PATIENCE} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_path = os.path.join(CFG.OUTPUT_DIR, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history


# ==============================================================================
# 13a. FULL METRICS SUITE (shared by k-fold CV and final test-set evaluation)
# ==============================================================================
# Every metric requested for the publication-ready report: accuracy, macro
# and weighted precision/recall/F1, balanced accuracy, per-class + macro
# specificity, MCC, Cohen's kappa, macro ROC-AUC (one-vs-rest), confusion
# matrix, and the full sklearn classification report. Computed once here so
# k-fold CV and the final held-out test evaluation report identical metrics.

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


def _specificity_per_class(cm: np.ndarray) -> np.ndarray:
    """One-vs-rest specificity per class from a confusion matrix:
    TN / (TN + FP) for each class, with rows=true, cols=predicted."""
    total = cm.sum()
    spec = np.zeros(cm.shape[0], dtype=np.float64)
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        spec[c] = tn / denom if denom > 0 else 0.0
    return spec


def compute_all_metrics(targets: np.ndarray, preds: np.ndarray, probs: np.ndarray,
                         num_classes: int, class_names=None) -> dict:
    """targets: (N,) int labels. preds: (N,) int predicted labels.
    probs: (N, num_classes) softmax probabilities (needed for ROC-AUC).
    Returns a flat dict of every metric in the publication spec, plus the
    raw confusion matrix and classification report (as nested objects)."""
    targets = np.asarray(targets)
    preds = np.asarray(preds)

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    spec_per_class = _specificity_per_class(cm)

    # macro ROC-AUC (one-vs-rest) needs every class present in targets;
    # guard against folds/splits where a rare class has zero val samples
    try:
        roc_auc_macro = roc_auc_score(
            targets, probs, multi_class="ovr", average="macro",
            labels=list(range(num_classes)),
        )
    except ValueError:
        roc_auc_macro = float("nan")

    target_names = class_names if class_names is not None else [str(i) for i in range(num_classes)]
    report = classification_report(
        targets, preds, labels=list(range(num_classes)), target_names=target_names,
        zero_division=0, output_dict=True,
    )

    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "precision_macro": precision_score(targets, preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(targets, preds, average="weighted", zero_division=0),
        "recall_macro": recall_score(targets, preds, average="macro", zero_division=0),
        "recall_weighted": recall_score(targets, preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(targets, preds, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "specificity_macro": float(np.mean(spec_per_class)),
        "specificity_per_class": spec_per_class.tolist(),
        "mcc": matthews_corrcoef(targets, preds),
        "cohen_kappa": cohen_kappa_score(targets, preds),
        "roc_auc_macro": roc_auc_macro,
        "confusion_matrix": cm,                # kept as np.ndarray for plotting; strip before json.dump
        "classification_report": report,
    }
    return metrics


def _metrics_for_json(metrics: dict) -> dict:
    """Returns a copy of a compute_all_metrics() dict safe for json.dump
    (confusion_matrix -> list, everything else already JSON-safe)."""
    out = dict(metrics)
    if isinstance(out.get("confusion_matrix"), np.ndarray):
        out["confusion_matrix"] = out["confusion_matrix"].tolist()
    return out


# ==============================================================================
# 13. K-FOLD CROSS-VALIDATION DRIVER
# ==============================================================================
# Stratified k-fold over the TRAIN+VAL pool only (the held-out test set from
# stratified_split is never touched during CV -- it's reserved for final,
# unbiased reporting). Each fold trains a fresh model from scratch and is
# evaluated on its own held-out fold; final metrics are the mean +/- std
# across folds, which is what you report as your model's true generalization
# estimate rather than a single lucky split.

def make_kfold_splits(samples, n_folds, seed):
    """Stratified k-fold: each class's samples are distributed round-robin
    across folds so every fold has proportional class representation."""
    by_class = defaultdict(list)
    for path, label in samples:
        by_class[label].append(path)

    rng = random.Random(seed)
    folds = [[] for _ in range(n_folds)]

    for label, paths in by_class.items():
        paths = paths.copy()
        rng.shuffle(paths)
        for i, p in enumerate(paths):
            folds[i % n_folds].append((p, label))

    return folds


def run_kfold_cv(train_val_samples, num_classes, n_folds=None, class_names=None):
    n_folds = n_folds or CFG.N_FOLDS
    folds = make_kfold_splits(train_val_samples, n_folds, CFG.SPLIT_SEED)

    fold_results = []
    for fold_idx in range(n_folds):
        print(f"\n{'='*70}\nFOLD {fold_idx + 1}/{n_folds}\n{'='*70}")

        val_fold = folds[fold_idx]
        train_fold = [s for i, f in enumerate(folds) if i != fold_idx for s in f]

        train_loader, val_loader, _ = build_dataloaders(train_fold, val_fold, val_fold)
        # third loader (test) is unused here; val_fold doubles as both val+holdout
        # within this fold since CV folds ARE the holdout mechanism

        model = TriBackboneEnsemble(num_classes=num_classes)
        class_weights = compute_class_weights(train_fold, num_classes, CFG.DEVICE)

        model, history = train_one_run(
            model, train_loader, val_loader, class_weights,
            run_name=f"fold{fold_idx + 1}",
        )

        loss_fn = FocalLoss(class_weights=class_weights, gamma=CFG.FOCAL_GAMMA,
                             label_smoothing=CFG.LABEL_SMOOTHING).to(CFG.DEVICE)
        val_loss, val_acc, preds, targets = evaluate(model, val_loader, loss_fn, CFG.DEVICE)

        # full probability-based metrics need softmax probs, not just argmax preds
        probs, _, _ = get_predictions_with_probs(model, val_loader, CFG.DEVICE)
        fold_metrics = compute_all_metrics(targets, preds, probs, num_classes, class_names=class_names)
        fold_metrics["fold"] = fold_idx + 1
        fold_metrics["val_loss"] = val_loss

        fold_results.append(fold_metrics)
        print(
            f"Fold {fold_idx + 1} result: "
            f"acc={fold_metrics['accuracy']:.4f} "
            f"balanced_acc={fold_metrics['balanced_accuracy']:.4f} "
            f"f1_macro={fold_metrics['f1_macro']:.4f} "
            f"f1_weighted={fold_metrics['f1_weighted']:.4f} "
            f"mcc={fold_metrics['mcc']:.4f} "
            f"kappa={fold_metrics['cohen_kappa']:.4f} "
            f"roc_auc_macro={fold_metrics['roc_auc_macro']:.4f}"
        )

    accs = [r["accuracy"] for r in fold_results]
    bal_accs = [r["balanced_accuracy"] for r in fold_results]
    f1_macros = [r["f1_macro"] for r in fold_results]
    f1_weighteds = [r["f1_weighted"] for r in fold_results]
    mccs = [r["mcc"] for r in fold_results]
    kappas = [r["cohen_kappa"] for r in fold_results]
    roc_aucs = [r["roc_auc_macro"] for r in fold_results]

    print(f"\n{'='*70}\nCROSS-VALIDATION SUMMARY ({n_folds} folds)\n{'='*70}")
    print(f"Accuracy:          {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"Balanced Accuracy: {np.mean(bal_accs):.4f} +/- {np.std(bal_accs):.4f}")
    print(f"F1 Macro:          {np.mean(f1_macros):.4f} +/- {np.std(f1_macros):.4f}")
    print(f"F1 Weighted:       {np.mean(f1_weighteds):.4f} +/- {np.std(f1_weighteds):.4f}")
    print(f"MCC:               {np.mean(mccs):.4f} +/- {np.std(mccs):.4f}")
    print(f"Cohen Kappa:       {np.mean(kappas):.4f} +/- {np.std(kappas):.4f}")
    print(f"ROC-AUC Macro:     {np.mean(roc_aucs):.4f} +/- {np.std(roc_aucs):.4f}")

    with open(os.path.join(CFG.OUTPUT_DIR, "cv_results.json"), "w") as f:
        json.dump([_metrics_for_json(r) for r in fold_results], f, indent=2)

    return fold_results


# ==============================================================================
# 14. EVALUATION
# ==============================================================================

@torch.no_grad()
def get_predictions_with_probs(model, loader, device):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(probs.argmax(dim=1).cpu().numpy())
        all_targets.append(labels.numpy())
    return (np.concatenate(all_probs), np.concatenate(all_preds), np.concatenate(all_targets))


def full_evaluation_report(model, test_loader, class_names, device, output_dir):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    num_classes = len(class_names)
    probs, preds, targets = get_predictions_with_probs(model, test_loader, device)

    metrics = compute_all_metrics(targets, preds, probs, num_classes, class_names=class_names)

    print(f"\n{'='*70}\nTEST SET EVALUATION\n{'='*70}")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
    print(f"Precision (macro):   {metrics['precision_macro']:.4f}")
    print(f"Precision (weighted):{metrics['precision_weighted']:.4f}")
    print(f"Recall (macro):      {metrics['recall_macro']:.4f}")
    print(f"Recall (weighted):   {metrics['recall_weighted']:.4f}")
    print(f"F1 (macro):          {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):       {metrics['f1_weighted']:.4f}")
    print(f"Specificity (macro): {metrics['specificity_macro']:.4f}")
    print(f"MCC:                 {metrics['mcc']:.4f}")
    print(f"Cohen Kappa:         {metrics['cohen_kappa']:.4f}")
    print(f"ROC-AUC (macro):     {metrics['roc_auc_macro']:.4f}\n")

    print(classification_report(targets, preds, labels=list(range(num_classes)),
                                 target_names=class_names, zero_division=0))

    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(metrics["classification_report"], f, indent=2)

    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(_metrics_for_json(metrics), f, indent=2)

    cm = metrics["confusion_matrix"]
    np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)

    # per-class ROC/PR (one-vs-rest), useful for multi-class imbalanced problems
    # and for the ROC plot; roc_auc_macro above is the single summary number,
    # this is the per-class curve data needed to actually draw the plot.
    targets_onehot = np.eye(num_classes)[targets]
    roc_data, pr_data = {}, {}
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(targets_onehot[:, c], probs[:, c])
        roc_data[class_names[c]] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
        prec, rec, _ = precision_recall_curve(targets_onehot[:, c], probs[:, c])
        pr_data[class_names[c]] = {"precision": prec.tolist(), "recall": rec.tolist()}

    return {
        **metrics,
        "macro_f1": metrics["f1_macro"],         # back-compat aliases used elsewhere in this file
        "weighted_f1": metrics["f1_weighted"],
        "report": metrics["classification_report"],
        "roc_data": roc_data, "pr_data": pr_data,
        "probs": probs, "preds": preds, "targets": targets,
    }


# ==============================================================================
# 15. PLOTS
# ==============================================================================

import matplotlib
matplotlib.use("Agg")  # headless-safe for Colab/Kaggle batch runs
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()

    axes[2].plot(history["lr"])
    axes[2].set_title("Learning Rate (head group)"); axes[2].set_xlabel("epoch")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_confusion_matrix(cm, class_names, output_path, normalize=True):
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig_size = max(10, len(class_names) * 0.5)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, square=True,
                cbar_kws={"shrink": 0.8})
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (row-normalized)" if normalize else ""))
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_roc_curves(roc_data, output_path):
    plt.figure(figsize=(9, 8))
    for class_name, data in roc_data.items():
        plt.plot(data["fpr"], data["tpr"], label=f"{class_name} (AUC={data['auc']:.3f})", linewidth=1.5)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Per-Class ROC Curves (One-vs-Rest)")
    plt.legend(fontsize=7, loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved ROC curves to {output_path}")


@torch.no_grad()
def plot_attention_weight_distribution(model, loader, device, output_path):
    """Shows how much the model relies on each backbone (ResNet/Inception/ViT)
    on average, and the per-sample spread -- useful for sanity-checking that
    the fusion is actually doing something (e.g. not collapsing to one branch)."""
    model.eval()
    all_weights = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        _, attn = model(imgs, return_attention=True)
        all_weights.append(attn.cpu().numpy())
    all_weights = np.concatenate(all_weights, axis=0)  # (N, 3)

    branch_names = ["ResNet50", "InceptionResNetV2", "ViT-B/16"]
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=all_weights, palette="Set2")
    plt.xticks(range(3), branch_names)
    plt.ylabel("Attention weight")
    plt.title("Learned Attention Fusion Weights per Backbone (test set)")
    means = all_weights.mean(axis=0)
    for i, m in enumerate(means):
        plt.text(i, m + 0.03, f"mean={m:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved attention weight distribution to {output_path}")
    return all_weights


# ==============================================================================
# 16. INFERENCE HELPER (single image)
# ==============================================================================

@torch.no_grad()
def predict_single_image(model, image_path, class_names, device, return_attention=True):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    if CFG.STAIN_NORMALIZE:
        img_np = apply_stain_normalization(img_np)
    img = Image.fromarray(img_np)

    transform = build_eval_transform()
    img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, LOAD_SIZE, LOAD_SIZE) uint8

    if return_attention:
        logits, attn = model(img_tensor, return_attention=True)
    else:
        logits = model(img_tensor)
        attn = None

    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())

    result = {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }
    if attn is not None:
        attn_np = attn.cpu().numpy()[0]
        result["backbone_attention"] = {
            "resnet50": float(attn_np[0]),
            "inception_resnet_v2": float(attn_np[1]),
            "vit_b16": float(attn_np[2]),
        }
    return result


def load_model_for_inference(checkpoint_path, num_classes, device):
    model = TriBackboneEnsemble(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ==============================================================================
# 17. MAIN ENTRYPOINT (ROBUST + FIXED FOR COLAB DATASETS)
# ==============================================================================

def main():
    # ── fallback: define CFG here if not already in global scope ────────────
    try:
        CFG
    except NameError:
        import os
        import torch as _torch
        class CFG:
            DATA_DIR        = os.environ.get("HISTO_DATA_DIR", "dataset")     # env-driven on NeSI; falls back to ./dataset
            OUTPUT_DIR      = os.environ.get("HISTO_OUTPUT_DIR", "outputs")   # env-driven on NeSI; falls back to ./outputs
            VAL_FRACTION    = 0.15
            TEST_FRACTION   = 0.15
            SPLIT_SEED      = 42
            USE_KFOLD       = True
            N_FOLDS         = 5
            LOAD_SIZE       = 320
            STAIN_NORMALIZE = True
            RESNET_NAME     = "resnet50.tv2_in1k"
            INCEPTION_NAME  = "inception_resnet_v2.tf_in1k"
            VIT_NAME        = "vit_base_patch16_224.augreg_in21k_ft_in1k"
            RESNET_SIZE     = 176
            INCEPTION_SIZE  = 299
            VIT_SIZE        = 224
            FEATURE_DIM_RESNET    = 2048
            FEATURE_DIM_INCEPTION = 1536
            FEATURE_DIM_VIT       = 768
            FUSION_DIM      = 768
            BATCH_SIZE      = 8
            NUM_WORKERS     = 4
            EPOCHS          = int(os.environ.get("HISTO_EPOCHS", "40"))
            WARMUP_EPOCHS   = 3
            BASE_LR         = 1e-4
            BACKBONE_LR_MULT       = 0.1
            WEIGHT_DECAY           = 1e-4
            LABEL_SMOOTHING        = 0.1
            GRAD_CLIP_NORM         = 1.0
            EARLY_STOP_PATIENCE    = 20
            MIN_DELTA              = 1e-4
            FREEZE_BACKBONES_EPOCHS = 4
            FOCAL_GAMMA     = 2.0
            USE_CLASS_WEIGHTS = True
            USE_AMP         = True
            AMP_DTYPE       = _torch.float16
            SEED            = 42
            DEVICE          = "cuda" if _torch.cuda.is_available() else "cpu"
            NUM_CLASSES     = None
            CLASS_NAMES     = None
        import builtins
        builtins.CFG = CFG
    # ────────────────────────────────────────────────────────────────────────

    # --------------------------------------------------------------------------
    # AUTO-FIX: resolve correct dataset root
    # Handles these common Colab layouts:
    #   /content/dataset/human_tissue_image-*/   ← direct
    #   /content/dataset/Dataset/human_tissue_image-*/  ← nested
    # --------------------------------------------------------------------------
    import os

    def resolve_dataset_root(path):
        """Walk down single-subfolder wrappers until we find class folders."""
        path = os.path.abspath(path)
        while True:
            if not os.path.isdir(path):
                break
            subdirs = [d for d in os.listdir(path)
                       if os.path.isdir(os.path.join(path, d))]
            # If there's exactly one subfolder and it contains no images itself,
            # descend into it (handles arbitrary nesting depth)
            if len(subdirs) == 1:
                candidate = os.path.join(path, subdirs[0])
                # Check if candidate has image files directly or has class subfolders
                has_images = any(
                    f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))
                    for f in os.listdir(candidate)
                    if os.path.isfile(os.path.join(candidate, f))
                )
                if not has_images:
                    path = candidate
                    continue
            break
        return path

    CFG.DATA_DIR = resolve_dataset_root(CFG.DATA_DIR)
    print(f"\n📁 Using dataset root: {CFG.DATA_DIR}")

    # Sanity check before going further
    if not os.path.isdir(CFG.DATA_DIR):
        raise FileNotFoundError(
            f"❌ Dataset directory not found: {CFG.DATA_DIR}\n"
            f"👉 Make sure you uploaded and extracted your dataset zip correctly.\n"
            f"👉 Expected something like: /content/dataset/Dataset/human_tissue_image-brain/"
        )

    # --------------------------------------------------------------------------
    # 1. discover data
    # --------------------------------------------------------------------------
    samples, class_names, class_to_idx = discover_dataset(CFG.DATA_DIR)

    if class_names is None or len(class_names) < 2:
        raise ValueError(
            f"❌ Invalid dataset: only {len(class_names) if class_names else 0} class(es) found.\n"
            f"👉 Check folder structure inside: {CFG.DATA_DIR}\n"
            f"👉 Expected: <DATA_DIR>/class_name/*.jpg"
        )

    CFG.NUM_CLASSES = len(class_names)
    CFG.CLASS_NAMES = class_names

    print("\n✅ Classes found:", class_names)
    print("📊 Number of classes:", CFG.NUM_CLASSES)
    print("🖼️ Total samples:", len(samples))

    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(CFG.OUTPUT_DIR, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # --------------------------------------------------------------------------
    # 2. split dataset
    # --------------------------------------------------------------------------
    train_samples, val_samples, test_samples = stratified_split(
        samples,
        CFG.VAL_FRACTION,
        CFG.TEST_FRACTION,
        CFG.SPLIT_SEED
    )

    if len(train_samples) == 0 or len(test_samples) == 0:
        raise ValueError(
            "❌ Train/Test split failed. Check stratified_split logic or dataset balance."
        )

    train_val_pool = train_samples + val_samples

    print(f"\n📊 Dataset Split:")
    print(f"Train+Val pool: {len(train_val_pool)}")
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # --------------------------------------------------------------------------
    # 3. stain normalization
    # --------------------------------------------------------------------------
    if CFG.STAIN_NORMALIZE and len(train_samples) > 0:
        init_stain_normalizer(train_samples[0][0])

    # --------------------------------------------------------------------------
    # 4. cross-validation
    # --------------------------------------------------------------------------
    fold_results = None

    if CFG.USE_KFOLD:
        print(f"\n🔁 Running {CFG.N_FOLDS}-Fold Cross Validation...")

        fold_results = run_kfold_cv(
            train_val_pool,
            CFG.NUM_CLASSES,
            CFG.N_FOLDS,
            class_names=class_names
        )

    # --------------------------------------------------------------------------
    # 5. FINAL TRAINING
    # --------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("🚀 FINAL MODEL TRAINING (train + val combined)")
    print(f"{'='*70}")

    final_train_pool, final_val_slice, _ = stratified_split(
        train_val_pool,
        val_frac=CFG.VAL_FRACTION,
        test_frac=0.0,
        seed=CFG.SPLIT_SEED,
    )

    final_train_loader, final_val_loader, test_loader = build_dataloaders(
        final_train_pool,
        final_val_slice,
        test_samples
    )

    final_model = TriBackboneEnsemble(num_classes=CFG.NUM_CLASSES)

    final_class_weights = compute_class_weights(
        final_train_pool,
        CFG.NUM_CLASSES,
        CFG.DEVICE
    )

    final_model, final_history = train_one_run(
        final_model,
        final_train_loader,
        final_val_loader,
        final_class_weights,
        run_name="final_model",
    )

    # --------------------------------------------------------------------------
    # 6. evaluation + visualization
    # --------------------------------------------------------------------------
    plot_training_curves(
        final_history,
        os.path.join(CFG.OUTPUT_DIR, "training_curves.png")
    )

    eval_results = full_evaluation_report(
        final_model,
        test_loader,
        class_names,
        CFG.DEVICE,
        CFG.OUTPUT_DIR
    )

    plot_confusion_matrix(
        eval_results["confusion_matrix"],
        class_names,
        os.path.join(CFG.OUTPUT_DIR, "confusion_matrix.png")
    )

    plot_roc_curves(
        eval_results["roc_data"],
        os.path.join(CFG.OUTPUT_DIR, "roc_curves.png")
    )

    plot_attention_weight_distribution(
        final_model,
        test_loader,
        CFG.DEVICE,
        os.path.join(CFG.OUTPUT_DIR, "attention_weights.png")
    )

    print(f"\n✅ All outputs saved to: {CFG.OUTPUT_DIR}/")
    print("🎯 Training Complete")

    return final_model, eval_results


if __name__ == "__main__":
    main()
