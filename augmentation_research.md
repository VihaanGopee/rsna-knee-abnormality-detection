# MRI-Specific Augmentation Research — RSNA Knee Abnormality Detection

Researched 2026-10-08. Sources: public competition repos (incl. a 0.910-LB
ensemble writeup), MONAI docs, knee-MRI deep-learning papers, and the
"I Can't Believe TTA Is Not Better" medical-imaging TTA study.

## Key findings

1. **A 0.910 public-LB team runs a 24-member TTA ensemble** with "jittered
   multi-window TTA" and two physical-scale crop configs. Heavy TTA is a real
   lever at the top of the leaderboard — but see caution #2.
2. **Naive TTA can HURT medical-image accuracy** (up to -31pp in a systematic
   study across MedMNIST + 4 architectures). Mechanism: distribution shift
   between augmented and training inputs, amplified by batch-norm statistics
   mismatch. Ablations show: **intensity-only TTA preserves far more accuracy
   than geometric TTA**; always include the unaugmented original; aggregation
   method (mean/vote/confidence-weight) matters little. TTA must be validated
   on local val — never deployed blind.
3. **Bias field is THE canonical MRI augmentation.** B1 inhomogeneity creates
   smooth low-frequency multiplicative intensity variation across every MRI
   scan. MONAI's `RandBiasField` simulates it directly. No ImageNet pipeline has
   an equivalent.
4. **Knee anatomical constraints** (from knee-MRI papers):
   - Horizontal flip = SAFE (left/right knees are mirror-symmetric; several
     papers explicitly normalize L/R orientation by mirroring one side).
   - Rotation must be small (±10–15°). Knee MRI is acquired in standardized
     planes; large rotations are anatomically unrealistic.
   - NO vertical flip (femur-above / tibia-below orientation is meaningful).
   - NO aggressive RandomResizedCrop (menisci, collateral ligaments and
     peripheral pathology live at image edges; cropping = spatial dropout of
     the disease). One medical-imaging group bans it outright for this reason.
   - No CutMix/MixUp: with 12 independent labels tied to different anatomical
     regions, mixing images corrupts label semantics.
5. Typical knee-MRI paper augmentations: flip, small rotation, translation,
   brightness/contrast jitter, Gaussian noise. Modest but standard.

## Recommended TRAIN pipeline (per-sample, 2.5D)

Applied consistently across all slices of a plane for geometric transforms
(preserves through-plane consistency); intensity transforms may vary per slice.
Implementable in Albumentations; bias field via MONAI or the numpy fallback below.

| # | Transform | Params | p | Rationale |
|---|-----------|--------|---|-----------|
| 1 | HorizontalFlip | — | 0.5 | L/R knee symmetry; anatomically valid |
| 2 | Rotate | limit=12, border_mode=reflect | 0.5 | Small rotations only; reflect padding avoids black corners |
| 3 | ShiftScaleRotate | shift_limit=0.05, scale_limit=0.05, rotate_limit=0 | 0.4 | Subtle translation/zoom (scanner positioning variance) |
| 4 | RandomBrightnessContrast | brightness_limit=0.15, contrast_limit=0.15 | 0.5 | Scanner gain / sequence contrast variance |
| 5 | GaussNoise | var_limit=(10, 50) on 0–255 scale ≈ std 0.02–0.05 normalized | 0.3 | Thermal noise (Gaussian approx of Rician is standard practice) |
| 6 | BiasField (MRI-specific) | degree 0.3–0.5, order 3 | 0.3 | Simulates B1 inhomogeneity — the single most MRI-characteristic augmentation |
| 7 | ElasticTransform | alpha=20, sigma=5, alpha_affine=5 | 0.2 | Subtle soft-tissue deformation; keep small |
| 8 | CoarseDropout | 1 hole, 24–48 px, fill=mean | 0.15 | Forces global context; keep probability low so small pathology (fracture lines) is rarely erased |

Do NOT add: VerticalFlip, Rotate >15°, aggressive RandomResizedCrop / crop-and-resize,
CutMix, MixUp.

**Bias-field numpy fallback** (if not pulling in MONAI): generate a smooth field
by upsampling a small random grid (e.g. 4×4 uniform in [1-degree, 1+degree])
with cubic interpolation to image size, then multiply the image. Equivalent in
spirit to MONAI `RandBiasField(prob=0.3, degree=0.4)`.

**Expected gain (train augs):** +0.005 to +0.015 val macro AUC over a basic
flip/rotate/brightness baseline. The bias-field + noise terms are the novel
part; geometric terms are table stakes. This will NOT by itself close a 0.07
gap — it stacks with labels + ensemble.

## Recommended TTA pipeline (inference)

Per the TTA caution study: **intensity-only, always include the original, mean
aggregation, validate on local val before trusting.**

Views per test study (logits averaged):
1. Original (unaugmented) — anchors the prediction
2. Horizontal flip
3. Brightness +0.10
4. Brightness −0.10
5. Contrast (gamma) 1.10
6. Contrast (gamma) 0.90
7. Intensity window variant A (narrow window: clip to [p5, p95] then rescale)
8. Intensity window variant B (wide window: clip to [p1, p99] then rescale)

The windowing views mirror the 0.910 team's "multi-window TTA" and simulate
how radiologists re-window scans. All 8 are intensity-only → minimal BN-shift
risk. Geometric TTA (rotations) is deliberately excluded per the caution study.

Cost: 8× inference. At ~15 s/study single-view, that's ~2 min/study → for the
small test set this is affordable in a single submission notebook run, but
measure first.

**Expected gain (TTA):** +0.005 to +0.012 LB AUC *if* it validates locally.
There is a real chance it is flat or slightly negative — the study found
degradation in most model/dataset combos. Decision rule: run TTA on the local
58-gold + val split; ship it only if macro AUC improves ≥ +0.003, else drop it.

## Stacking order for the parent agent

1. Add train augs (#5 GaussNoise, #6 BiasField are the deltas) → next training run.
2. Build TTA into the submission notebook behind a flag → validate on gold/val,
   keep iff ≥ +0.003.
3. These stack with the severity-label win and the planned multi-seed ensemble;
   combined realistic LB delta from this workstream: +0.01 to +0.025.

## Implementation notes for the existing codebase

- The current v2.3 notebook presumably has a basic Albumentations pipeline;
  add transforms 5–8 to it. Keep everything else (loss, MIL, EMA) unchanged
  so the ablation is clean.
- For 2.5D slice stacks: use Albumentations `additional_targets` or apply the
  Compose once per plane with a fixed seed across its 8 slices for geometric
  ops. Simplest correct approach: replay mode (`A.ReplayCompose`) — compose
  once on slice 0, replay on slices 1–7 for geometric determinism, allow fresh
  intensity sampling per slice.
- Submission notebook: implement TTA as a loop over the 8 views inside the
  existing per-study inference, accumulating logits then averaging. No
  architectural change needed.
