# Duplicate-aware evaluation: code

Code for "Duplicate-Aware Evaluation of Frozen Foundation Model Features and Fine-Tuned Networks for 22-Class Human Tissue Classification". This package holds no data and no model checkpoints.

## Layout

| Path | What it is |
|---|---|
| `original/train_histo.py` | Training script of the first run (NeSI job 7172798): ResNet50 + InceptionResNetV2 + ViT-B/16 with attention fusion. Image-level random split. **Do not use its results**: see `leakage/`. |
| `original/run_histo.sl` | Slurm script of that run. |
| `leakage/dup_split_check.py` | Rebuilds the original split and counts byte-identical (MD5) copies across it. |
| `grouped/make_splits.py` | Group-aware split. Group = MD5 of the file, so every copy stays in one partition. Hold-out test is 1 of 7 StratifiedGroupKFold folds (seed 42). 5-fold StratifiedGroupKFold CV and the final train/val split are drawn for seeds 42, 43 and 44. Fails if any hash crosses partitions. |
| `grouped/split_assignment.csv` | File, class, MD5, group, seed, partition, CV fold and final role for every image. |
| `grouped/histo_core.py` | Byte-identical copy of `original/train_histo.py`, imported as a library so the training methodology is unchanged. |
| `grouped/train_grouped.py` | Trains one (model, seed, unit) pair: HistoFormerFusion, equal-weight fusion, ViT-B/16, ResNet50, InceptionResNetV2, VGG16, DenseNet121 or EfficientNet-B0. |
| `grouped/classical_ml.py` | SVM / Random Forest / k-NN on GLCM, LBP and RGB/HSV histogram features, on the same split. |
| `grouped/aggregate.py` | Tables, paired statistics (t-test, Wilcoxon, Holm, McNemar), compute table, figures. |
| `grouped/lane.sl`, `grouped/lane_debug.sl`, `grouped/classical.sl`, `grouped/smoke_cpu.sl`, `grouped/common_env.sh`, `grouped/tasks*.txt`, `grouped/status.sh` | Slurm work-queue scripts used on NeSI. Each lane claims (model, seed, unit) tasks from `tasks.txt` with mkdir locks. Paths are NeSI-specific; edit `common_env.sh`. |
| `foundation/` | Frozen foundation model track: feature extraction (`fm_features.py`), grouped cross-validation of the logistic regression models (`fm_raw_cv.py`), fusion heads (`fm_raw_fusion.py`, `fm_heads.py`), held-out test scoring (`fm_raw_test.py`), the random file-level split (`fm_random_split.py`) and the timestamped analysis declarations (`DECLARATIONS.md`). |
| `dataset_origin/` | `match_nuinsseg.py` maps each unique image to its NuInsSeg source patch: all 368 are whole human patches, bicubic-resized from 512 to 224 and pixel-identical. Output in `nuinsseg_mapping.csv`. |

## Data

The images are NuInsSeg human tissue patches (Mahbod et al., 2024, Sci Data, doi:10.1038/s41597-024-03117-2; Zenodo 10518968), resized to 224x224. Download NuInsSeg from Zenodo; it is not redistributed here.

## Data layout

`$HISTO_DATA_DIR/Dataset/human_tissue_image-<class>/image_<n>.png`: 22 class folders, 1,070 PNG files, 368 unique images by MD5.

## Commands

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
export HISTO_DATA_DIR=/path/to/data   # contains Dataset/
cd grouped
python make_splits.py                  # splits.json, split_summary.json, data_table.csv, split_assignment.csv
python train_grouped.py --model fusion --seed 42 --unit final   # one unit; models/units listed in tasks.txt
python classical_ml.py                 # all classical units
python aggregate.py                    # results/
```

On Slurm: `sbatch smoke_cpu.sl`, then submit any number of `lane.sl` jobs (any GPU type), plus `sbatch classical.sl`; finally `python aggregate.py`. `tasks.txt` holds the 24 finals (8 models x 3 seeds) and the 40 seed-42 CV folds; `tasks_tier3.txt` holds the seed 43/44 CV.

## Training methodology (all deep models)

AdamW, 1e-4 for the head and 1e-5 for the backbones, weight decay 1e-4, 3 warmup epochs then cosine decay. Batch 16, bf16 autocast, 40 epochs, backbones frozen for the first 4. Focal loss (gamma 2) with class weights and label smoothing 0.1, plus an inverse-frequency weighted sampler. The best-validation-loss checkpoint is restored before evaluation. Early-stopping patience is 10 in the grouped retrain (`HISTO_PATIENCE`) and was 20 in the original run. Macenko stain normalisation is fitted on the first training image of each split (the first bladder image in that split's training set, e.g. `bladder/image_0.png`; it is the same for every model on a given split and is logged in the output). The original run used `spleen/image_876.png`, but that image falls in the grouped test set, so it is not used. Inputs are cached at 320 px and resized per backbone: ResNet50 176, InceptionResNetV2 299, others 224.
