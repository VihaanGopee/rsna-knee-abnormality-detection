# RSNA Knee Abnormality Detection — Project Scope

**Competition:** [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
**Deadline:** October 22, 2026 11:59 PM UTC (~22 days)
**Metric:** Macro-averaged AUC ROC over 12 labels
**Prizes:** $77,000 (Main: 1st $9K … 6th–10th $5K each; Efficiency track ~$18K)

---

## 0. Stress-Test Revisions (2026-09-30)

Justin challenged this plan before execution. Three flaws were found and fixed:

1. **Compute fiction.** Custom SSL pretraining (Phase 3) does not fit Kaggle's 30h/week quota.
   It is now **Tier 3 — conditional** on ablation evidence + sufficient Colab units, not a committed phase.
2. **Phase 1 oversell.** Fine-tuning the LLM labeler on 58 gold examples is experimental
   (tiny sample, partly circular). The committed win is simpler: our own **clean, uncontaminated
   labels** via the best measured method (Qwen3-14B closed-vocab, 0.881) + abstain-masking.
   Labeler fine-tuning is a stretch experiment, not a certain moat.
3. **Unmeasured throughput.** Phase sizes were guessed without training-speed numbers.
   **Every phase now starts with a timing probe** (1 epoch on a subset → real iters/sec → sized plan).
   See `EXECUTION_PROTOCOL.md` — no GPU run starts without passing pre-flight.

**Compute budget (confirmed 2026-09-30):** ~90 GPU-hours Kaggle (2×T4, 30h/week × ~3 weeks)
+ **200 Google Colab units**. Exchange rates: T4 ~2 units/hr (≈100h), A100 ~14/hr (≈14h).
Budget: label extraction ~20u, probes/ablations ~20u, main training on Kaggle quota + Colab
overflow, **A100 reserve ~10–14h held for the single highest-leverage run**. Full SSL pretraining
(700+ units) stays shelved unless ablations prove features are the bottleneck.

**Score calibration (honest ranges, not false precision):**
- Base case 0.94–0.945 (top ~50–100, robust to shakeup)
- Good execution 0.945–0.95 (prize fringe)
- Everything lands 0.95+ (top-10 contention, $5K+)
- Depends most on: (1) label quality, (2) ensemble diversity, (3) surviving the shakeup.
  A robust 0.945 can outrank a fragile 0.955 when the private board lands.

## 0b. Verification Revisions — Round 2 (2026-09-30)

Four verification tracks cross-checked the plan against primary sources. What changed:

**Licenses — DINOv3: GO.** Custom "DINOv3 License" (Meta), commercial use permitted,
attribution "Built with DINOv3" required. Weights are HF-gated: download once with a token,
republish as a Kaggle Dataset (internet is OFF at scoring — no on-the-fly timm download).
**Surprise: RadImageNet is the real license risk** (no stated license per competitor audit) —
demoted from blend arm to **Tier-2 non-load-bearing only**. Fallback ranking if ever needed:
DINOv2 (Apache-2.0) → SigLIP/SigLIP2 (Apache-2.0) → ConvNeXt (Apache-2.0).

**Leaderboard — bar moved to 0.960** (Sep 28; was 0.957). 4,682 teams. Public LB = 30% of test,
private = 70%. Shakeup fingerprints confirmed (492 teams @ 0.936, 163 @ 0.937; top public
author warns of overfit). Score targets stand, but 0.95+ is now more clearly top-10 fringe.

**Efficiency track — real, and worth a week-3 side bet.** Official formula verified:
one extra inference hour ≈ 0.05 AUC. A 0.90 model finishing in 30 min beats a 0.94 8-hour
ensemble. Prizes $7K/$6K/$5K. **Scott Willis is 3rd (0.958) with a small ResNet/EfficientNet
@224 and ~5-min scoring** — small models compete; architecture size is not destiny.
Plan: dedicated small/fast notebook in week 3, separate from the main ensemble. No rule
conflict; final 2-selection split decided at the end.

**Label effort reallocated (per-target ceilings, measured):**
- **Synovitis is the ONLY completeness-capped target** (13/27 mentioned; ceiling 0.8076).
  Route closed — accept it, don't burn effort.
- **Effort goes to:** PF OA, Lateral OA, Medial OA, Fracture, Lateral Meniscus —
  frequently mentioned but poorly extracted; LLM closed-vocab beats lexicons by +0.112 here,
  the largest single available number.
- **Already near-complete:** ACL (0.97–0.99), Effusion, Medial Meniscus, Baker's, Contusion.
- **"Two graded findings can carry the entire target"** — Synovitis→0.90 = +0.14,
  PF OA→0.93 = +0.10 of the +0.204 needed for +0.017 macro. Separation lives in the graded four.

**Phase 1 concretized:**
- Torres's exact prompt/tokens are unrecoverable (nobody reproduced 0.881), but the
  **transferable invention is verified**: closed vocabulary → deterministic Python map
  using measured positive rates. The number lives in the calibration table we build ourselves.
- tranbadat2607's complete 771-line implementation (direct grading, 0.869) gives us the
  working skeleton to adapt.
- **Exit gate: ≥0.86 macro-agreement vs gold-58.** Hybrid calibration (LLM-corpus rates as
  prior, gold-58 as check, shrink noisy cells toward corpus rate).
- **Starts with a 100-report timing probe** — no public throughput data exists.
  Qwen3-14B-AWQ + vLLM on Colab T4, ~3–8h estimated, inside the 20u budget.

**Two tensions flagged (resolve by our own ablation, not by faith):**
- (a) Forum consensus says masking noisy cells is dead — but that's masking on *low model
  confidence*, a different operation from our abstain-masking on *report silence*.
  Keep as a **gated experiment**, not an assumption.
- (b) Forum says nothing above 288px beats 0.940; our briefs measured 224→336px at +0.017.
  **Add 288-vs-336 to the probe ablation.** Don't assume.

**Grayscale → DINOv2: settled.** Replicate to 3 channels + ImageNet mean/std. Universal
standard; zero public evidence for anything exotic.

**New closed routes (don't re-run):** Synovitis reader improvements, DINOv2-as-second-family
in one setup (−0.035/−0.148), 288px geometry assumed optimal, per-finding blend weights,
self-distillation on a leaking rig.

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

## Tiering (what's committed vs stretch)

- **Tier 1 — committed:** Phase 0 → Phase 1 (clean labels) → Phase 2 (one strong family)
  → Phase 4 (ensemble + survival). **Bank a robust submission by end of week 2.**
- **Tier 2 — if compute allows:** teacher-student refinement, second model family, LibAUC head fine-tune.
- **Tier 3 — conditional:** custom SSL pretraining on the corpus. Only if ablations prove
  features are the bottleneck AND the unit budget supports it. Currently shelved.

---

## 3. Phase 0 — Data Pipeline & Cache (Tier 1)

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
   - **Laterality note (verified 2026-09-30):** the R→L mirror is *canonicalization*, not
     augmentation — the anatomical medial meniscus lands on the canonical medial side, so
     **no Medial/Lateral label swap is needed**. The swap requirement applies only to
     *random* horizontal-flip augmentation, which stays **OFF** everywhere because the
     cache is already canonicalized. Do not "fix" this by adding a swap.
   - Laterality source tag and fallback (if tag missing) to be confirmed in the DICOM audit.
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

## 4. Phase 1 — Label Extraction (Tier 1, the highest-leverage phase)

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

## 5. Phase 2 — Baseline Models (Tier 1 — one strong family, trained for real)

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
- `notebooks/phase2-timing-probe.ipynb` — 1 epoch on a 10% subset → measured iters/sec,
  memory footprint, loss curve sanity. **Sizes everything downstream. Nothing long runs before this.**
- `notebooks/phase2a-dinov2.ipynb` — DINOv2 ViT-S/14, 336px, ASL, two-stage, scanner-grouped folds.
- **Go/no-go gate:** single family must clear **0.93 on scanner-grouped OOF**. Below that,
  stop — the problem is labels or pipeline, not model capacity. Do not ensemble a broken base.

---

## 6. Phase 3 — Custom Moats (Tier 2 committed experiments, Tier 3 conditional)

**Tier 2 (committed if Tier 1 gates pass):**
1. **Teacher-student label refinement** — train a teacher on our labels, mix teacher predictions
   50/50 with LLM labels (quantile-matched), retrain. **+0.009 LB measured** (TianK003).
2. **Noisy Student iteration** — soft pseudo-labels, RandAugment + dropout/stochastic depth noise,
   equal-or-larger student, iterate.
3. **LibAUC head fine-tuning** — AUCMLoss + PESG as a head-only pass on 2×T4.
   Unexplored in public logs; cheap to try, kill fast if flat.

**Tier 3 (conditional — shelved unless ablations prove features are the bottleneck):**
- **Self-supervised pretraining on the competition corpus** (MAE on ~819k slices).
  The architectural moat nobody public has. Requires 700+ Colab units — does not fit the
  200-unit budget. Revisit only with fresh evidence + fresh budget.

---

## 7. Phase 4 — Ensemble + Submission Engineering (Tier 1)

- **Rank-mean blending** of *disagreeing* families (blend gain comes from disagreement,
  ρ=0.542 across families vs 0.905–0.986 within family). Fixed weights — no LB probing.
- **"Adding members is the only operation that has ever moved this board"** —
  member count +0.002 measured twice, blend-weight tuning +0.000. Spend effort on members, not weights.
- **Survival engineering:** decode DICOMs once, share across members; incremental
  `submission.csv` every 25 studies; runtime projection; staged degradation
  (drop weakest members first if behind schedule); failed study → 0.5, never crash.
- **Never P100.** `machine_shape: NvidiaTeslaT4` (wrong names silently fall back to P100).

**Deliverable:** `notebooks/rsna-knee-final-v1.ipynb` — the canonical submission notebook.

---

## 8. Timeline (22 days)

| Week | Focus | Milestone |
|---|---|---|
| Week 1 (Sep 30–Oct 6) | Phase 0 + Phase 1 | Cache built; clean label table generated; extractor validated vs gold-58 |
| Week 2 (Oct 7–13) | Phase 2 | Timing probe → DINOv2 trained → **0.93 OOF gate** → robust submission banked |
| Week 3 (Oct 14–21) | Tier 2 + Phase 4 | Teacher-student, second family, ensemble; final submission by Oct 21 |
| Oct 22 | Buffer | Final submission deadline 11:59 PM UTC |

Entry/team-merger deadline: **October 15** — accept rules before then.

---

## 9. Decision Gates (the plan stops here if these fail)

1. **Phase 1:** extractor agreement vs gold-58. If our labeler < 0.85, stop and fix labels —
   no model recovers from bad targets.
2. **Phase 2 probe:** measured throughput. If 1 epoch projects beyond the unit budget,
   cut resolution or family count before committing.
3. **Phase 2 OOF:** single family ≥ 0.93 on scanner-grouped OOF. Below → labels/pipeline bug, not capacity.
4. **Phase 4:** every ensemble member must *disagree* with existing members (ρ < 0.9).
   Agreeing members add nothing — skip them.

---

## 10. Docs & Discipline

- `EXECUTION_PROTOCOL.md` — pre-flight checklist, kill gates, gotcha list, clearance rule.
  **No GPU run starts without Muse's clearance.**
- `docs/run-log.md` — every GPU run logged: date, notebook version, config, probe numbers,
  projected vs actual runtime, metric, artifacts.
- `docs/` — the four research briefs backing every claim in this scope.