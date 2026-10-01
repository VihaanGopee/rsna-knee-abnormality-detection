# RSNA Knee Abnormality Detection — Project Scope

**Competition:** [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
**Deadline:** October 22, 2026 11:59 PM UTC (~22 days)
**Metric:** Macro-averaged AUC ROC over 12 labels
**Prizes:** $77,000 (Main: 1st $9K … 6th–10th $5K each; Efficiency track ~$18K)

---

## 1. The Problem, Precisely

Per knee MRI **study**, predict 12 abnormality confidences:
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.

**The defining constraint:** 4,407 train studies, but only **58 have gold labels**. The other 4,349 have DICOM images + multilingual radiology reports (~9–12 languages). Reports are **not available at test time**.

This is a **weak-supervision problem**, not a standard supervised one. The entire competition is won or lost on:
1. How good your training labels are (extracted from reports), and
2. How well your models learn from noisy labels without overfitting 58 gold studies.

**Public leaderboard state (Sep 2026):** top ~0.957, dense wall at 0.943–0.945 (~81 teams ≥0.945), top-10 within 0.006. The public top is dominated by **one heavily-forked community ensemble** — every member is a mounted public checkpoint; the reference 0.936 "trains nothing." Analysts warn of public-LB overfitting and expect a private shakeup.

**Our thesis:** the forked ensemble caps around 0.94x on robust validation. The gap to 0.957+ comes from (a) better labels, (b) actually training custom models, (c) ensemble diversity from disagreement — all things we control.

---

## 2. Design Philosophy (the Soheil standard)

Justin studied Soheil Ayati's 2nd-place Biohub writeup and asked for that same intelligence. Mapped to this competition:

| Soheil's principle | Our implementation |
|---|---|
| Multiple detector banks, not one model | Diverse model *families* (DINOv2, DINOv3, CoAtNet arms), each with folds — diversity from input representation + pretraining regime, not backbone count |
| Synthetic data for hard cases | Rare-label augmentation (MCL 16%, Lateral OA 19% prevalence); targeted augmentation for under-represented findings |
| Never threshold weak evidence early | **Abstain-masking**: report silent on a finding → mask the loss, don't teach a negative. Soft graded labels (0–1), never hard 0/1 |
| Learned scorers, not heuristics | Fine-tuned LLM labeler (nobody public has done this); learned series-fusion; teacher-student refinement |
| Recovery stage revisiting discards | Noisy-Student second pass; test-set pseudo-labeling as the endgame move |
| 5-fold discipline per bank | Scanner-grouped folds (random K-fold inflates AUC by **0.087** — measured) |

**Non-negotiable rules:**
- The 58 gold studies are a **regression guard, never a tuner**. They've been reused so often they can't be an independent test set.
- **Report-label CV ranks models; it never estimates the leaderboard.** A metadata-only model scored 0.669 on report-label CV vs 0.531 on LB — a 0.138 gap between report-truth and scored-truth.
- **Generate our own labels.** Three of four surveyed public label sets are contaminated (copy gold-58 verbatim into "weak" labels).
- No board move under 0.003 is signal (reseed floor ±0.003).

---

## 3. Phase 0 — Data Pipeline & Cache (the foundation everything stands on)

**Measured facts about the data:**
- ~819k DICOM files (~710 GB extrapolated), 4,407 studies / 24,371 series
- Median 30 slices/series (range 20–45), median 5 series/study (range 3–14)
- Intensity max spans 690…8,736 across series (**12.7× range**) → per-series windowing is mandatory
- Headers are de-identified: **no site label exists**
- Axial fluid-sensitive series exists for **100% of studies** — the guaranteed fallback

**Preprocessing recipe (consensus, verified):**
1. pydicom (+pylibjpeg); apply RescaleSlope/Intercept; invert MONOCHROME1
2. **Sort slices by IPP·(IOP_row × IOP_col), NEVER by filename** — filename order has Spearman ρ = −0.012 vs true order; silently destroys 2.5D triplets while loss still falls
3. Per-series 1st–99th percentile clip → [0,1] (never a global window)
4. Resample to fixed mm/px; 130mm anatomical crop; mirror R knees to L; quantize uint8 into sharded cache
5. Series selection: **one fluid-sensitive series per plane** (sagittal/coronal/axial), same function at train and inference
6. Host's `Fluid_Sensitive`/`Fat_Suppression` flags are degenerate in train — recover contrast from ScanningSequence/SeriesDescription/TR/TE. `Anatomical_Plane` is 100% trustworthy.

**Fold grouping key (scanner fingerprint):** Manufacturer + Model + SoftwareVersions + MagneticFieldStrength + ImagingFrequency rounded to 2 decimals → 178 usable groups. (Raw precision = 8,618 near-unique = grouped-KFold in disguise.)

**Report-sharing gotcha:** 49 report texts shared across 183 studies — fold grouping must include report-sharing, not just scanner.

**CSV gotcha:** `train.csv` is **latin-1, not UTF-8** — pandas default silently mangles accented characters in Spanish/French reports. Multi-line reports: 58,556 physical lines for 4,407 rows — use a real CSV parser.

**Cache engineering:**
- Preprocess once → uint8 cache shards → Kaggle Dataset → mount in train/infer kernels
- **Cache version string must encode every byte-affecting knob** — stale versions silently mix incompatible caches
- **Train/infer pixel path must be byte-verified equal**, not just error-free
- File access is ~free (0.059 s/study); pixel decoding is the constraint

**Deliverable:** `notebooks/phase0-preprocess.ipynb` → versioned uint8 cache dataset on Kaggle.

---

## 4. Phase 1 — Label Extraction (the highest-leverage phase)

**Measured labeler quality vs the 58 gold:**

| Extractor | Agreement | Source |
|---|---|---|
| Qwen3-14B-AWQ + vLLM, closed vocab | **0.881** | Public notebook (LB 0.883) |
| gpt-5.4-mini, graded 0–1 | 0.869 mean AUC | tranbadat2607 (2.4M tokens, full corpus) |
| Rule-based baseline | 0.814 | tranbadat2607 |
| Hand-built multilingual lexicon | 0.769 | existentialistlogarithmic |

Per-target extremes: ACL ~0.97–0.99 (easiest), **Synovitis ~0.68** (hardest).

**The fundamental ceiling (most important finding):** report labels ≠ image labels. Only **13 of 27 true Synovitis cases are even mentioned in reports** — a perfect extractor caps at 0.8076 there. Label effort should target *frequently-mentioned but poorly-extracted* findings, not completeness-capped ones.

**Our labeler design (nobody public has done this):**
1. **Closed-vocabulary LLM classification → deterministic probability mapping → graded soft labels (0–1).** Preserves LLM uncertainty; trains better under BCE. (+0.112 label quality measured for this pattern over lexicon baselines.)
2. **Fine-tune the labeler** (CheX-LLM style: the chest X-ray literature went CheXpert 0.8864 → CheXbert 0.9047 → CheX-LLM **0.9115**, with largest gains on negated/uncertain mentions). Fine-tune on the 58 gold + high-confidence LLM labels.
3. **Validate the extractor on the 58 gold first** — the universal rule among serious teams.
4. **Per-language calibration** of output probabilities.
5. **Labeler ensemble** (2–3 diverse LLMs) + confidence-weighted training downstream.
6. **Abstain-masking downstream**: silence on a finding → mask loss, never teach negative.

**Deliverable:** `notebooks/phase1-label-extraction.ipynb` → our own clean label table (soft, 4,407 × 12) as a Kaggle Dataset. **Never use public label tables for validation** (contaminated).

---

## 5. Phase 2 — Baseline Models (prove the pipeline before the moats)

**Architecture consensus (what works):**
- **2.5D self-supervised ViTs**, not 3D CNNs. Workhorse: **DINOv2 ViT-S/14** (22M, fully fine-tuned) + attention pooling over slices
- Blend arms: **DINOv3 ViT-S/16**, **RadImageNet ResNet-50** (HF `Lab-Rasool/RadImageNet`)
- **CoAtNet** (conv-attention hybrid) holds best measured solo: **0.932**
- 224→336px gives **+0.017**; 384px is measured dead (worse, 3× cost)
- Fusion: attention-pool slices → per-series embedding → study-level (mean-within-plane+concat, or cross-attention over all series)
- **Diversity comes from input representation + pretraining regime, not backbone count** (3 backbones, same input = +0.001 measured)

**Training recipe:**
- **Loss: ASL (asymmetric loss)**, γ−=4/γ+=0–1 + probability margin — the community default for imbalanced multi-label; the margin rejects likely-mislabeled hard negatives (exactly the report-label noise). Don't stack with focal/pos_weight. Label smoothing 0.05–0.1 on weak heads.
- **Two-stage schedule:** Stage 1 on all 4,407 with weak labels + heavy aug → Stage 2 fine-tune on 58 gold with low LR. Gold override + **8× upweight** (measured working).
- **Augmentation:** horizontal flip is the landmine — 5/12 labels are laterality-sensitive. Either never flip, or flip with **Medial↔Lateral label swaps** (train aug and TTA both). Rotation/gamma/scale jitter shared across whole series. No mixup/cutmix reported.
- **TTA is measured dead** (10/12 variants correlate >0.98 with parent). Skip it.
- **float64 precision end-to-end** in the submission path (rounding creates AUC tie-penalties).

**Validation discipline:**
- Scanner-grouped folds (Section 3). **No fold decision without the grouping.**
- Decide everything on scanner-grouped OOF. Fixed fusion weights — **no LB-probed blend weights** (that's the shakeup trap: fork fingerprints at 0.936/0.937 + documented LB-probed weights = classic shakeup setup).

**Deliverables:**
- `notebooks/phase2a-din
...[truncated 8236 chars]