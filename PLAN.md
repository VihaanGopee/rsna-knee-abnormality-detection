# RSNA Knee Abnormality Detection — Winning Plan
**Date:** 2026-10-06 | **Deadline:** Oct 22, 2026 11:59 PM UTC | **Budget:** 45 GPU-hours (T4×2)
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
- **Raptor label mix:** 0.5 LLM + 0.5 quantile-matched Raptor (biggest known lever)
- **Verify Raptor dataset** (`rsna-knee-teacher-tables`) is accessible before committing GPU hours
- Fallback: pure LLM labels if Raptor unavailable

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
| 2 | Raptor labels undefined | PLAN-KILLER | Verify dataset access BEFORE GPU spend; fallback to pure LLM |
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
- [ ] Verify `rsna-knee-teacher-tables` dataset accessible
- [ ] Build clean inference notebook (single-GPU, no DataParallel)
- [ ] **Submit current CoAtNet model → get baseline LB**

**Decision gate (pre-committed):**
- LB ≥ 0.92: Proceed with full plan
- LB 0.88–0.92: Proceed, drop pseudo-labeling
- LB < 0.88: STOP, reassess architecture/labels

### Phase A — Labels (Days 1–3, 0 GPU-h, parallel with Phase 0)
- [ ] Download Raptor labels, verify format and independence from gold
- [ ] Build LLM blend (mean of hans_v4, pilkwang, sol56 where available)
- [ ] Quantile-match Raptor to LLM distribution (exact function in §7)
- [ ] Mix: 0.5 LLM + 0.5 Raptor (gold rows stay hard 0/1)
- [ ] Platt scaling per finding on 58 gold (skip if <15 positives)
- [ ] Start synovitis surrogate re-extraction via API (parallel)

### Phase B — Core Training (Days 3–10, ~28 GPU-h)
- [ ] Train 3× EfficientNet-B3 @ 288px:
  - 30 epochs, SWA last 3, LR 3e-4 (backbone) / 1e-3 (head)
  - Frozen BN, heavy augmentation, drop-path 0.1
  - AdamW, weight decay 0.02, warmup 10% + cosine decay
  - AMP enabled, grad clip 1.0, EMA 0.998
  - Batch 2, accum 2 (effective 4)
  - Different seeds
  - (~7h each = 21h)
- [ ] Train 1× CoAtNet full-fit (all 4,407 + all 58 gold, v12 recipe extended to 30 epochs + SWA) (~7h)

### Phase C — Ensemble & Push (Days 10–14, ~10 GPU-h)
- [ ] Rank-average ensemble (per-label percentile ranks)
- [ ] Measure per-member contribution; drop below-mean members from over-represented families
- [ ] TTA: 8 views (rotations ±10°, brightness jitter) at inference
- [ ] If budget remains: pseudo-labeling round (~7h)

### Phase D — Final (Days 15–16, buffer)
- [ ] Select 2 final submissions by gold CV, NEVER by public LB alone
- [ ] Submit before Oct 22, 11:59 PM UTC

**Total: ~42 GPU-h** (3h buffer for retries/crashes)

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
target = 0.5 * LLM_blend + 0.5 * quantile_match(Raptor, LLM_blend)
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

### 6.7 Raptor Labels
- Dataset: `rsna-knee-teacher-tables`
- File: `raptor_teacher.csv`
- Path: `/kaggle/input/rsna-knee-teacher-tables/raptor_teacher.csv`
- Format: StudyInstanceUID + 12 label columns, [0,1]
- Excludes 58 gold rows
- **Verify accessibility before GPU spend**

---

## 7. Expected Outcomes

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
**First (0.964):** Out of reach with 45h — leaders have 140h+ and private labels

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
4. Competition intel → EfficientNet-B3 @ 288px, Raptor mix, 0.964 top
5. Unconventional → Pseudo-labeling, spatial re-extraction, skip SSL
6. Label deep-dive → Report-to-image gap, Platt calibration (not isotonic)
7. Red team → Killed isotonic, flagged Raptor dependency, budget risks
8. Implementation specs → Exact hyperparameters, code-ready functions

---

*If the winner's writeup doesn't match this plan, we know exactly where we diverged.*
