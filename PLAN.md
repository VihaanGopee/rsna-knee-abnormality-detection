# RSNA Knee Abnormality Detection — Winning Plan
**Date:** 2026-10-06 | **Deadline:** Oct 22, 2026 11:59 PM UTC | **Budget:** 45 GPU-hours/week (T4×2) — ~90h over 16 days

## BASELINE RESULT (2026-10-06): 0.804 LB

The CoAtNet baseline scored **0.804** on the leaderboard — well below the 0.88-0.93 expected range.
Gap to first (0.964): **0.16**. This changes the strategy fundamentally.

**What 0.804 tells us:**
- CoAtNet + weak labels (0.8473) is not competitive at this level
- Incremental improvements (+0.06-0.08) would reach ~0.86-0.88 — still far from 0.964
- The architecture AND labels both need to change, not just get tuned

**Revised strategy (2026-10-06 evening):**
- **Primary:** EfficientNet-B3 @ 288px (the proven competitor recipe) + public teacher labels (0.8927 vs gold)
- **CoAtNet:** demoted to ensemble diversity member only, not the lead
- **Labels:** public teacher table (0.8927) replaces our 0.8473 labels as the base — this is now the #1 lever
- **Target:** 0.90+ via better labels + proven architecture, then ensemble to 0.93-0.95 — ~90h over 16 days
**Goal:** Maximize macro AUC. Target 0.92–0.95 (top 100). Stretch: 0.960+ (top 10).

---

## 1. Current State

**Trained model (v12, completed 2026-10-06 ~03:40 UTC):**
- Architecture: CoAtNet-2.5D (`coatnet_0_rw_224`) + 12 gated-attention MIL heads
- Stage 1: 8 epochs on 4,407 weak-labeled studies → val macro 0.7917, gold-macro 0.8395
- Stage 2: 5 epochs on 43 train-fold gold → val macro 0.7918, gold-macro 0.8463
- Checkpoints: `/kaggle/working/phase2_ckpt/stage1_best.pt`, `stage2_best.pt`
- Config: 224px, 8 slices/plane, single GPU (DataParallel crashes CoAtNet), batch 2×8 accum

**Critical caveat:** Gold-macro measured on 15 held-out studies → SE ±0.09. True LB could be 0.76–0.94. **No reliable signal until baseline submission.**

---

## 2. Competition Landscape (2026-10-06)

| Metric | Value |
|---|---|
| Top LB | 0.964 |
| Top-10 cutoff | ~0.960 |
| Teams ≥ 0.950 | 123 |
| Public plateau | 0.943 (984 teams) |
| Total teams | 5,293 |
| Prize (1st) | $9,000 |

**Entry deadline: Oct 15** — must accept competition rules or cannot compete. **VERIFY IMMEDIATELY.**

---

## 3. Research Synthesis (8 agents)

### 3.1 Architecture
- **Stay 2.5D.** Literature unambiguous for small medical data.
- **EfficientNet-B3 @ 288px** is the proven winner (TianK003, 0.944). NOT ConvNeXt (measured no gain). NOT bigger backbones (measured null twice).
- CoAtNet-0 stays as ensemble diversity member (different family → decorrelated errors).

### 3.2 Labels (the 10x insight)
**Report-to-image gap:** Weak labels are accurate *report* labels, but reports systematically differ from images. For synovitis, model scores 0.779 — *below* its 0.790 supervision. No architecture fixes this.

**Original plan (isotonic calibration) was KILLED by red team:** n=58 is 10× below the 500+ minimum for isotonic regression. It would overfit and make things worse.

**Revised approach:**
- Use **Platt scaling** (2-param logistic) instead of isotonic, OR skip calibration for findings with <15 positives
- **Teacher-label mix:** 0.5 LLM + 0.5 quantile-matched public teacher table (biggest known lever)
- ~~**Verify Raptor dataset** (`rsna-knee-teacher-tables`)~~ — DEAD 2026-10-06: Raptor table is private/inaccessible. Replacement verified public: `stevenleehans/rsna-knee-llm-report-labels` (`llm_labels_v4_blend.csv`, 0.8927 vs gold, two independent measurements — beats our 0.8473)
- Fallback: pure LLM labels if the public table is ever unavailable

### 3.3 Training Strategy
- **3-fold ensemble** (not 5): captures 80% of gain at 60% cost
- **Rank-average** (per-label percentile ranks), NOT arithmetic mean — confirmed by ISIC winner + live competitor
- **SWA** (last 3 epochs): +0.005–0.015, nearly free
- **TTA**: +0.003–0.006, free at inference
- **30 epochs** (not 8): winners train longer
- **OOF noise cleaning**: flag weak labels where |LLM − model| > 0.4, retrain (RSNA 2024 2nd-place trick)
- **Do NOT add inter-class static weights** — ASL authors explicitly advise against (interacts with focusing params)

### 3.4 Unconventional (high-EV additions)
- **Pseudo-labeling:** Ensemble → predict test → high-confidence pseudo-labels → retrain. +0.005–0.015, ~7h. The "private team" move.
- **Spatial re-extraction:** Re-run LLM for synovitis surrogates/severity (not binary). 0 GPU hours (API only, parallel).
- **Skip:** Multimodal CLIP, true 3D, SSL pretraining (unless budget remains), diffusion augmentation

---

## 4. Red Team Findings (all addressed)

| # | Risk | Severity | Fix |
|---|---|---|---|
| 1 | Isotonic on n=58 overfits | PLAN-KILLER | Use Platt scaling or skip for rare findings |
| 2 | Raptor table private (verified inaccessible 2026-10-06) | PLAN-KILLER → RESOLVED | Swapped to public `stevenleehans/rsna-knee-llm-report-labels` (`llm_labels_v4_blend.csv`, 0.8927 vs gold); verify gold-row independence before GPU spend; fallback to pure LLM |
| 3 | 0.846 is mirage (±0.09) | HIGH | Baseline submission as hard decision gate with pre-committed thresholds |
| 4 | EfficientNet recipe = single data point | MEDIUM | Keep CoAtNet as floor; architecture matters less than labels+ensemble |
| 5 | 42h budget, 3h buffer fragile | MEDIUM | Checkpoint every epoch; prioritize: drop pseudo-labeling first, then 3rd fold |

**Blind spots fixed:**
- Inference pipeline broken (DataParallel stub crashed) → must build clean single-GPU inference
- Oct 15 deadline unverified → verify immediately
- Test distribution shift → check test vs train finding prevalence

---

## 5. The Plan

### Phase 0 — IMMEDIATE (Day 1, 0 GPU-h)
- [ ] Verify Oct 15 entry deadline (accept rules if not done)
- [x] ~~Verify `rsna-knee-teacher-tables`~~ — DEAD (private). Attach `stevenleehans/rsna-knee-llm-report-labels` instead
- [ ] Build clean inference notebook (single-GPU, no DataParallel)
- [ ] **Submit current CoAtNet model → get baseline LB**

**Decision gate (pre-committed):**
- LB ≥ 0.92: Proceed with full plan
- LB 0.88–0.92: Proceed, drop pseudo-labeling
- LB < 0.88: STOP, reassess architecture/labels

### Phase A — Labels (Days 1–3, 0 GPU-h, parallel with Phase 0)
- [ ] Download `stevenleehans/rsna-knee-llm-report-labels` (`llm_labels_v4_blend.csv`); verify format (StudyInstanceUID + 12 cols, [0,1]) and that the 58 gold rows are NOT verbatim 0/1 (verbatim copies poison cross-validation)
- [ ] Build LLM blend (mean of hans_v4, pilkwang, sol56 where available)
- [ ] Quantile-match teacher table to LLM distribution (exact function in §7)
- [ ] Mix: 0.5 LLM + 0.5 teacher (gold rows stay hard 0/1)
- [ ] Platt scaling per finding on 58 gold (skip if <15 positives)
- [ ] Start synovitis surrogate re-extraction via API (parallel)

### Phase B — Core Training (Days 3–10, ~28 GPU-h)
- [ ] Train 2× EfficientNet-B3 @ 288px:
  - 30 epochs, SWA last 3, LR 3e-4 (backbone) / 1e-3 (head)
  - Frozen BN, heavy augmentation, drop-path 0.1
  - AdamW, weight decay 0.02, warmup 10% + cosine decay
  - AMP enabled, grad clip 1.0, EMA 0.998
  - Batch 2, accum 2 (effective 4)
  - Different seeds
  - (~7h each = 14h)
- [ ] Train 1× ConvNeXt-Tiny @ 224px (architecture hedge):
  - Same recipe as EfficientNet-B3 (30 epochs, SWA, etc.)
  - Tests whether the "ConvNeXt no gain" finding was wrong
  - Cross-family diversity > same-family repeats
  - (~7h)
- [ ] Use trained CoAtNet (v12, 0.846 gold-macro) as 4th member (no retrain needed)
  - Total: 4 members, 3 families (EfficientNet, ConvNeXt, CoAtNet)

### Phase C — Ensemble & Push (Days 10–14, ~10 GPU-h)
- [ ] Rank-average ensemble (per-label percentile ranks)
- [ ] Measure per-member contribution; drop below-mean members from over-represented families
- [ ] TTA: 8 views (rotations ±10°, brightness jitter) at inference
- [ ] If budget remains: pseudo-labeling round (~7h)

### Phase D — Final (Days 15–16, buffer)
- [ ] Select 2 final submissions by gold CV, NEVER by public LB alone
- [ ] Submit before Oct 22, 11:59 PM UTC

**Total: ~42 GPU-h** (3h buffer)

**Architecture hedge rationale (2026-10-06):** Betting 21h on EfficientNet-B3 based on one competitor (rank 337) is a single point of failure. 2× EfficientNet + 1× ConvNeXt + 1× CoAtNet gives cross-family diversity with the same GPU cost. If EfficientNet-B3 underperforms, the ensemble survives.

---

## 6. Implementation Specs

### 6.1 EfficientNet-B3 Config
```python
backbone = "efficientnet_b3"  # timm, ra2_in1k weights
img_size = 288                 # resize from 336px cache via F.interpolate
drop_path_rate = 0.1
num_classes = 0

epochs = 30
lr_backbone = 3e-4
lr_head = 1e-3
weight_decay = 0.02            # not on biases/norms/1-D
optimizer = "AdamW"
lr_schedule = "linear warmup 10% + cosine decay to 0"
freeze_bn = True               # encoder BN in eval mode (see §6.2)
ema_decay = 0.998
swa_last = 3
grad_clip = 1.0
amp = True
batch_studies = 2
grad_accum = 2
```

### 6.2 Frozen BN
```python
# At start of every epoch, AFTER model.train():
if freeze_bn:
    for mod in model.enc.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()
# BN uses pretrained running stats; affine params still train
```

### 6.3 Quantile Matching
```python
def quantile_match(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    out = np.full(pred.shape, np.nan)
    m = np.isfinite(pred)
    ref = np.asarray(ref, dtype=float)
    ref = ref[np.isfinite(ref)]
    if m.sum() == 0 or len(ref) == 0:
        return out
    r = pd.Series(pred[m]).rank(method="average").to_numpy()
    q = (r - 0.5) / m.sum()
    out[m] = np.quantile(ref, np.clip(q, 0.0, 1.0))
    return out

# Mix:
target = 0.5 * LLM_blend + 0.5 * quantile_match(teacher_v4_blend, LLM_blend)
```

### 6.4 SWA (manual weight averaging)
```python
# During training: maintain EMA (decay 0.998), snapshot to CPU at each epoch end
# Keep last 3 in ring buffer. After final epoch:
def average_state_dicts(sds):
    out = {}
    for k, v in sds[-1].items():
        if v.dtype.is_floating_point:
            out[k] = torch.stack([sd[k].float() for sd in sds]).mean(0).to(v.dtype)
        else:
            out[k] = v.clone()  # e.g., BN num_batches_tracked
    return out
```

### 6.5 Rank-Average Ensemble
```python
def rank_mean(frames):
    """frames: list of DataFrames with StudyInstanceUID + 12 label cols"""
    base = frames[0][["StudyInstanceUID"]].copy()
    for lab in LABELS:
        acc = np.zeros(len(base))
        for f in frames:
            acc += f[lab].rank(pct=True).to_numpy()
        base[lab] = acc / len(frames)
    return base
# Rank PER LABEL independently. Average ranks, not probabilities.
```

### 6.6 Heavy Augmentation (p=0.9, float [0,1])
| Transform | Range |
|---|---|
| Rotation | ±15° |
| Zoom | 0.90–1.15 |
| Shift | ±8% |
| Gamma | 0.7–1.4 |
| Contrast | 0.8–1.25 |
| Gain | 0.85–1.15 |
| Cutout | p=0.3, 20–50% side rectangle → 0 |

**No flips** (medial ≠ lateral).

### 6.7 Teacher Labels (public)
- **CORRECTION (2026-10-06):** The Raptor teacher table is private/dead. Actual table: public `stevenleehans/rsna-knee-llm-report-labels`, file `llm_labels_v4_blend.csv` (4,407 rows, StudyInstanceUID + 12 label columns [0,1], excludes 58 gold rows). Verified 0.8927 vs gold by two independent teams. Old entry below kept for reference.
- Dataset: `rsna-knee-teacher-tables`
- File: `raptor_teacher.csv`
- Path: `/kaggle/input/rsna-knee-teacher-tables/raptor_teacher.csv`
- Format: StudyInstanceUID + 12 label columns, [0,1]
- Excludes 58 gold rows
- **Verify accessibility before GPU spend**

### 6.8 Severity-Grade Labels (2026-10-07, Justin's committed program)

The 4 weakest teacher findings are all severity-graded (Synovitis 0.76, PF OA 0.83) — severity extraction is the competitive moat, not done publicly by any team. Decision path:
- 12-finding all-in-one prompt probed at 0.8319 macro vs teacher 0.8927 → REJECTED (attention split: the same 4 severity findings dropped to 72-86%).
- Focused 4-finding prompt (synovitis, pfoa, medial_oa, lateral_oa) hit **100% binary accuracy (58/58)** on the 58 gold studies. KEEP teacher labels for the 8 non-severity findings; grade only the 4 severity findings.

Execution (his side): Justin pastes prompts into the Antigravity Gemini 3.8 Flash agent chat — one **new chat per batch**, 9 batches of ~500 rows. Naming: `graded_batch_N_M.json` = Python slice `rows[N:M]` (indices N through M-1, no overlap), e.g. batch 1 = `graded_batch_0_500.json` (rows 0–499, done 2026-10-07 ~19:34 PDT, 41 min; gold in-slice studies rows 29/63/109/445/470 matched 100%). ~6h total at this pace. Each file is a flat array of `{"StudyInstanceUID", "synovitis", "pfoa", "medial_oa", "lateral_oa"}`.

Grading rules: not mentioned / normal / intact → "none"; "trace" / low-grade → "mild". Hoffa edema, superolateral fat pad edema, fat pad impingement, hoffitis → synovitis "mild". Synovitis moderate = moderate synovitis / chronic reactive synovitis / synovial hypertrophy with moderate effusion / lipoma arborescens; severe = PVNS / massive destructive synovitis. OA (pfoa, medial_oa, lateral_oa): mild = grade 1–2 / low-grade chondrosis / early osteophytosis / superficial fissuring; moderate = grade 3 / high-grade partial-thickness defect >50% / deep fissuring; severe = grade 4 / full-thickness cartilage denudation / bone-on-bone / eburnation.

Merge (assistant side): `merge_severity_teacher.py` — tested, one command:
```
python3 merge_severity_teacher.py --teacher llm_labels_v4_blend.csv \
    --batches 'graded_batch_*.json' --outdir ./severity_merge_v1
```
Semantics: the 4 severity findings become binary targets (none→0, mild/moderate/severe→1 — mapping validated at 100% on gold); the other 8 findings keep teacher values untouched, including 0.5 "not addressed" cells. The notebook's `2*|p-0.5|` confidence mask then assigns weight 1.0 to every replaced cell (script asserts this), so former ignored 0.5 cells become real supervision. Fails LOUD on: invalid grades, duplicate UIDs across batches, batch UIDs missing from the teacher table, batch-file count ≠ 9 (catches a forgotten batch or a stray probe JSON matching the glob; override with `--expect-batches N`). Outputs (Kaggle notebook path, `notebooks/phase2-merge-severity.ipynb` v1.2 — Justin runs this, no CLI): `/kaggle/working/merged_labels_v1.csv` (training drop-in, 13 cols, [0,1], zero NaNs). The script path additionally emits `severity_ordinal_v1.csv` (uid + 4 ordinal columns 0–3, -1 = ungraded, reserved for future severity-auxiliary-head run) and `merge_report.json` (distributions, replaced-0.5 count, disagreement rate vs confident teacher cells).

Next training run: download `/kaggle/working/merged_labels_v1.csv` from the merge notebook and upload it to Kaggle as a new dataset, attach it, set `TEACHER_CSV_NAME = 'merged_labels_v1.csv'` in `phase2-effnet-b3-v2.2` — no other notebook changes needed (gold override and 0.5-mask logic untouched). Note: the notebook's verbatim-gold warning may tick up — informational only, since severity binaries now match gold on graded studies.

---

## 7. Expected Outcomes (REVISED post-0.804 baseline)

**Baseline: 0.804** (CoAtNet + 0.8473 labels). Gap to 0.964: 0.16.

**Revised forecast with EfficientNet-B3 + 0.8927 teacher labels:**
- Better labels (0.8473 → 0.8927): +0.03-0.05 → 0.834-0.854
- EfficientNet-B3 @ 288px, 30 epochs + SWA (vs CoAtNet 8 epochs): +0.02-0.04 → 0.854-0.894
- 3-member ensemble (2× EffNet-B3 + 1× ConvNeXt): +0.015-0.025 → 0.869-0.919
- TTA: +0.003-0.006 → 0.872-0.925
- 2nd-gen pseudo-labeling: +0.005-0.015 → 0.877-0.940

**Realistic target: 0.88-0.92** (top 100-300)
**Stretch target: 0.93-0.95** (top 50) — requires everything to hit upper estimates
**First place (0.964):** Out of reach. The 0.804 baseline reveals a 0.16 gap that 90h cannot close.

## 7a. Original Expected Outcomes (pre-baseline, for reference)

| Step | Expected LB | Cumulative GPU-h |
|---|---|---|
| Baseline (current CoAtNet) | 0.88–0.93 (unknown!) | 2h |
| + Calibrated labels | +0.005–0.015 | 0h (CPU) |
| + 3× EfficientNet-B3 | 0.90–0.935 | 23h |
| + CoAtNet diversity | +0.003–0.005 | 30h |
| + Rank-average + TTA | +0.005–0.008 | 30h |
| + Pseudo-labeling | +0.005–0.015 | 37h |

**Realistic target: 0.92–0.95** (top 100–300)
**Stretch: 0.960+** (top 10, needs everything + luck)
**First (0.964):** In play with 90h over 2 weeks + private-label iteration. The gap is execution, not resources.

---

## 8. What NOT To Do (measured dead ends)

- ❌ Larger backbones (measured null twice)
- ❌ ConvNeXt training (no gain)
- ❌ LLM relabeling with same prompt (0.847 is ceiling for binary mention)
- ❌ Per-finding static loss weights (ASL authors advise against)
- ❌ Isotonic calibration on n=58 (overfits; use Platt or skip)
- ❌ Self-distillation / same-family students (closed)
- ❌ Architecture search (field converged; labels + ensemble win)
- ❌ Tuning on public LB (noise ±0.004; top-10 span is 0.004)

---

## 9. Research Agents (8 total)

1. Architecture → ConvNeXt-Tiny (overruled by competition intel)
2. Training strategy → 3-fold, rank-average, SWA, TTA
3. Labels → U-SelfTrained, all-58-gold final, per-finding audit
4. Competition intel → EfficientNet-B3 @ 288px, public teacher-table mix, 0.964 top
5. Unconventional → Pseudo-labeling, spatial re-extraction, skip SSL
6. Label deep-dive → Report-to-image gap, Platt calibration (not isotonic)
7. Red team → Killed isotonic, flagged private-Raptor dependency (→ swapped to public table 2026-10-06), budget risks
8. Implementation specs → Exact hyperparameters, code-ready functions

---

*If the winner's writeup doesn't match this plan, we know exactly where we diverged.*

---

## 10. Bug Saga (v4–v12) — Do Not Repeat

| Version | Bug | Cost | Lesson |
|---|---|---|---|
| v4 | Scanner-proxy fold collapse (all UIDs same prefix → fold 0) | 15 min GPU | Test with real UID formats, not synthetic |
| v5 | Rotation dtype: np.cos/sin → float64, grid_sample refused | 19 min GPU | Force float32 at every numpy→torch boundary |
| v6 | coatnet_0_224 has no pretrained weights (silent random init) + 256px vs 224px native | 15 min GPU | Verify weights exist; match backbone native size |
| v7 | Augment brightness: rng.rand() float64 → DoubleTensor crash | 15 min GPU | Same as v5 — audit ALL rng.* calls |
| v8 | .view() on DataParallel tensors → CUDA misaligned address | 15 min GPU | Use .reshape(), never .view() under DP |
| v9 | Sweep found last .view() (plane_mask) | 0 (proactive) | Systematic pattern sweep works |
| v10 | DataParallel splits batch=2 → 1/GPU, CoAtNet kernels crash | 15 min GPU | Probe on single GPU; DP untested for full run |
| v11 | OOM: 96 images/batch > 14.56GB T4 | 15 min GPU | N_SLICES 16→8 (48 images) |
| v12 | Full run crashed (DataParallel, same as v10) | 10 min GPU | Disabled DP entirely; batch 2×8 accum |

**Total wasted:** ~2 hours GPU. **Root cause:** Testing components in isolation, not the integrated pipeline.

## 11. Kaggle Setup Checklist

**Datasets to attach:**
- [ ] `phase0runA`, `phase0runB`, `phase0runC` (preprocessed MRI shards)
- [ ] `rsna-knee-phase1-weak-labels-v1` (weak_labels_v1.csv)
- [ ] Competition data `rsna-knee-abnormality-detection` (train.csv, test/)
- [ ] `stevenleehans/rsna-knee-llm-report-labels` (llm_labels_v4_blend.csv) — public, verified 2026-10-06
- [ ] Model checkpoints (stage2_best.pt as dataset for submission)

**Notebook settings:**
- [ ] Accelerator: GPU T4 ×2 (training) / T4 ×1 (inference)
- [ ] Internet: ON (for timm pretrained weights)
- [ ] Expected runtime: Training ~4h, Inference ~2-6h

**Before each run:**
- [ ] Check `VERSION:` string at top of log matches expected
- [ ] Delete stale notebook, download fresh from GitHub
- [ ] Verify all 5 datasets attached

## 12. File Inventory (GitHub: VihaanGopee/rsna-knee-abnormality-detection)

| File | Purpose |
|---|---|
| `notebooks/phase2-train.ipynb` | Training (v12, single-GPU) |
| `notebooks/phase2-submit.ipynb` | Inference/submission (v1) |
| `notebooks/phase2-effnet-b3-v2.2` | EfficientNet-B3 training (canonical, fp32 loss, no DP) |
| `merge_severity_teacher.py` | Merge 9 severity batch JSONs into teacher table (binary severity targets) |
| `severity_batch_verify.py` | Quality-check a graded batch JSON vs the 58 gold studies |
| `validate_severity.py` | Validate the focused 4-finding severity prompt vs gold |
| `phase1_severity_antigravity.py` | Antigravity-style severity grading script (assistant-side) |
| `platt_calibration.py` | Per-finding bias correction |
| `PLAN.md` | This document |
| `phase1_*.py` | Phase 1 label extraction (archived) |

## 13. Key Links

- GitHub: https://github.com/VihaanGopee/rsna-knee-abnormality-detection
- Competition: (RSNA Knee Abnormality Detection on Kaggle)
- TianK003 repo: https://github.com/tiank003/rsna-kneemri-kaggle-competition
- Deadline: Oct 22, 2026 11:59 PM UTC
- Entry deadline: Oct 15, 2026 (VERIFY)
