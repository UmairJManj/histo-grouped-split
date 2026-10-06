# Pre-registered analysis declarations (HistoFormerFusion grouped retrain)

## 2026-10-01 18:19 NZDT: declared BEFORE any test-set scoring of the improvement variants
1. Headline model: the 3-seed (42, 43, 44) softmax ensemble with 8-view dihedral TTA
   (4 rotations x horizontal flip, softmax averaged), evaluated once on the fixed grouped
   test set (153 files / 53 MD5 groups). Single-seed test results are also reported as mean and SD.
2. Baselines get the identical treatment: 3-seed ensemble + TTA, plus single-seed mean and SD,
   all under the same frozen variant (resolution, backbone weights, recipe).
3. Selection: variants are screened on single-seed (42) validation only. Before freezing, the
   chosen configuration is compared with the 224 px base on seed-42 grouped 5-fold CV. If CV
   disagrees with validation, CV decides.
4. Every variant tried is logged in results/variant_log.csv (losers included), and the paper
   states the number of variants tried.
5. The test set is scored once per final configuration, after freezing. Test predictions written
   to disk by screening runs are not read before then.

## 2026-10-01 18:54 NZDT: AMENDMENT (before any test result was read)
Point 1 is amended. The TTA mode (none / 4-view flips / 8-view dihedral) is chosen on validation for
the frozen configuration and applied identically to all models, baselines included.
Reason: on validation, 8-view TTA hurt the 224 px models (equal-weight fusion 0.893 to 0.840,
ViT 0.878 to 0.847; HistoFormerFusion unchanged at 0.878). Training had no 90-degree rotations.

## 2026-10-02 12:39 NZDT: FROZEN-FEATURE TRACK, declared BEFORE any test-set scoring of it
Selection on mean grouped 5-fold CV over seeds 42/43/44 (15 folds), pool only; test never read.
Candidates (results/fm_raw_cv.csv, results/fm_raw_fusion.csv), mean CV accuracy:
  phikon+phikon2 concat-LR, 8-view dihedral training 0.9647 | same, identity view only 0.9629 |
  +lunit_vits8 0.9626 | +hoptimus0 0.9582 | +kaiko 0.9560 | +midnight 0.9426 | cls+mean tokens 0.9418 |
  equal-weight fusion head 0.9589 | HistoFormerFusion attention head 0.9574 |
  phikon2 alone 0.9531 (v1) / 0.9506 (v8) | phikon alone 0.9501 (v1) / 0.9476 (v8).
1. PRIMARY (headline): phikon + phikon-v2, raw whole 512 patch -> 224 bicubic (no Macenko), CLS token,
   L2 per model, concatenated, StandardScaler + LogisticRegression (C in {0.001,0.01,0.1,1} by inner
   StratifiedKFold(3, rs=0) on pool), trained on the 8 dihedral views of the 315 unique pool patches,
   sample_weight = n_copies, NO TTA (identity view; CV TTA 0.9615 < 0.9647). Deterministic: one model.
2. SECONDARY (paper method): HistoFormerFusion attention head on the same frozen embeddings, seeds 42/43/44,
   reported per seed and as the 3-seed softmax ensemble (identity view). Control: equal-weight fusion, same.
3. Baselines: phikon alone and phikon-v2 alone under the primary protocol.
4. Test (153 files / 53 patches) is scored once per configuration above: accuracy, macro F1, MCC, kappa,
   macro AUC, per-class P/R/F1, confusion, Clopper-Pearson CI and group-bootstrap CI (1,000 patch resamples).
   Number of frozen-feature variants tried before this declaration: 15 (plus 64 earlier Macenko-bank combos).
