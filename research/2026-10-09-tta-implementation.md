# TTA Implementation Research — 2026-10-09

**Deliverables:**
- `~/workspace/rsna-knee/tta_inference.py` — drop-in TTA inference (paste into submission notebook)
- `~/workspace/rsna-knee/tta_validate.py` — gold-58 validation + per-view ablation (paste as cell after model load)

## 1. What TTA transforms are safe for knee MRI?

**Safe (used):**
| View | Transform | Why safe |
|---|---|---|
| identity | none | anchor; always included |
| hflip | horizontal flip | L/R knees mirror-symmetric; finding presence is flip-invariant. Phase-0 audit: no R→L canonicalization exists, so the model already sees both lateralities |
| bright_up/down | ±0.10 intensity shift | scanner gain variation; within training augmentation range (gain 0.85–1.15 was in train augs) |
| gamma_090/110 | x^0.9 / x^1.1 | contrast variants; training used gamma 0.7–1.4, so these are in-distribution |
| window_narrow | clip [0.15,0.85]→[0,1] | mimics tighter DICOM window/level; mirrors the 0.910 team's multi-window TTA |
| window_soft | [0,1]→[0.05,0.95] | gentle contrast compression; catches overconfident edge responses |

**Unsafe (excluded) and why:**
- **Vertical flip** — femur-above/tibia-below orientation is anatomically meaningful; flipping it creates images that never occur clinically.
- **Rotation** — acquisition planes are standardized (±12° max in training). TTA rotation risks testing the model on geometry it never learned; the medical-TTA caution literature shows geometric TTA is where degradation happens.
- **CutMix/MixUp-style** — corrupts 12-label multi-label semantics tied to distinct anatomical regions.
- **Aggressive crop** — menisci/ligaments live at image edges; cropping can erase the pathology.

**Key safety rule implemented:** transforms apply ONLY to present planes (`plane_mask == 1`). Zero-filled missing planes are never brightened/shifted — that would inject fake signal.

## 2. How to implement efficiently?

All 8 views stack on the **batch dimension** → **one forward pass per study**:
- Per plane: (8, S, 3, H, W) = 8 views × 8 slices × 3 ch × 288 × 288 fp32 ≈ 63 MB
- 3 planes ≈ 190 MB — comfortable on a T4 (16 GB)
- `plane_mask` tiled to (8, 3); the MIL fusion handles it identically per view
- Single batched forward amortizes kernel-launch overhead vs 8 separate calls

Estimated cost: ~8× backbone FLOPs per study, but test set is only ~1,300 studies and DICOM decode (not GPU) dominates the current pipeline — real wall-clock multiplier is ~3–4×, not 8×.

`model.eval()` is enforced inside the predict function: BN running stats are frozen, so TTA views cannot corrupt normalization. (The reported "TTA degrades accuracy up to 31pp" failure mode involves aggressive geometric TTA + BN mismatch; intensity-only + frozen BN avoids it.)

## 3. How to aggregate?

**Default: mean of sigmoid probabilities** across the 8 views. Rationale:
- The metric is macro AUC (a ranking metric); mean preserves calibrated probability scale for downstream rank-averaging in ensembles.
- Standard across Kaggle medical-imaging TTA; the 0.910 team's recipe uses mean aggregation.
- Median available via `TTA_AGG = "median"` (more robust to a single bad view, slightly less calibrated).

Do NOT do: max (overconfident), majority vote (meaningless for probabilities), learned weights (overfits to val).

## 4. Validation protocol (implemented in tta_validate.py)

Runs on the 58 expert-labeled gold studies (train studies → preprocessed from `train_series/`):

1. **Baseline**: identity view only → macro AUC
2. **Full TTA**: all 8 views → macro AUC
3. **Ablation**: identity + each single transform (2 views) → per-view delta vs baseline

**Ship rule** (from prior research consensus):
- Ship TTA iff `full_TTA_macro − baseline_macro ≥ +0.003`
- Drop any individual view with ablation delta `< −0.001` before shipping
- Per-finding AUC table printed for inspection (a view may help meniscus but hurt fracture)

Degenerate findings (<2 positives or <2 negatives in the 58) are skipped in the macro rather than crashing.

## Integration into the submission notebook

1. Paste `tta_inference.py` contents into a new cell after the existing INFERENCE cell (or replace `predict_studies` call).
2. Replace the call `results, failed = predict_studies(test_uids, test_dir)` with `results, failed = predict_studies_tta(test_uids, test_dir)`.
3. `TTA_ENABLED = False` reproduces single-view behavior bit-identically (verified: views list collapses to `[identity]`).
4. For RadImageNet checkpoints, set `RADIMAGENET_NORM = True` (or rely on the notebook global) — TTA transforms run in [0,1] space, then map to [-1,1].
5. Run `tta_validate.py` as a cell on Kaggle (needs `gold_58.csv` attached + `train_series/`); follow the printed verdict.

## Expected gain

Prior estimates: **+0.005 to +0.012 LB** if it validates; real chance of flat/negative (hence the mandatory gold check). Zero training GPU cost — pure inference.

## Test log

- `py_compile` clean on both files (2026-10-09)
- All 8 numpy transforms unit-tested: shape/dtype/range preserved, hflip involution holds, windows nontrivial, gamma monotonic
- Missing-plane gating verified by code inspection (mask==1 check before every transform)
- No torch available locally; torch-dependent paths (batching, forward) follow the existing submission notebook's proven patterns exactly
