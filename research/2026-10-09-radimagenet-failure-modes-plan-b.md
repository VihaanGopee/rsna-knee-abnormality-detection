# RadImageNet Failure Modes & Plan B — Decision Tree

**Date:** 2026-10-09
**Context:** About to bet ~7 GPU-hours on RadImageNet ResNet-50 (v2.4). v2.3 (ImageNet B3, severity labels) finished: best EMA 0.8231, best val 0.8179, best gold 0.8487.

---

## 1. Known Failure Modes of RadImageNet Transfer

### Documented negative results (peer-reviewed)

| Study | Task | Result |
|---|---|---|
| Dental radiographs (Springer, 2025) | Panoramic + cephalometric classification | RadImageNet did **NOT** beat ImageNet; CLAHE aug failed to help either |
| Suganuma et al. | PET/CT segmentation | **ImageNet beat RadImageNet** |
| Nehary et al. | Lung ultrasound frame classification | **ImageNet beat RadImageNet, specifically for ResNet-50 and DenseNet-121** |
| Remzan et al. | Brain tumor classification | RadImageNet did **not consistently** beat ImageNet across architectures/metrics |
| Thermography (review) | Out-of-distribution modality | RadImageNet **underperformed** |
| Dovile et al. | Reproduction attempt | Comparable performance; **could not fully reproduce** Mei et al.'s AUC claims |

**Key pattern:** Failures cluster on (a) modalities/anatomies far from RadImageNet's core distribution, and (b) ResNet-50 specifically (Nehary). We are using ResNet-50.

### The large-dataset compression problem (from Mei et al. themselves)

Their own Figure 4 (bigger datasets):
- COVID-19: +6.1% → still decent
- Pneumonia CXR: **+1.9%**
- SARS-CoV-2 CT: **+1.7%**
- Hemorrhage CT: **+0.9%**

The +4.5–4.8% (ACL/meniscus MRI) numbers are from **small datasets**. We have 4,407 studies — medium-large. Expect the gain to compress toward the +1–2% range, i.e., **+0.008 to +0.017 absolute**, not +0.03–0.05.

### Self-supervised on knee MRI: ImageNet wins at scale

Critical study (arXiv 2403.16499, sagittal T2W FS knee MRI, Stanford data):

| Pretraining | 80 subjects | 800 subjects (100%) |
|---|---|---|
| None | 0.433 | 0.562 |
| **ImageNet** | **0.682** | **0.824** |
| SimCLR | 0.698 | 0.800 |
| BYOL | 0.577 | 0.753 |
| MAE | 0.548 | 0.624 |
| Relative-location SSL | 0.740 | 0.815 |

**At full data (800 subjects), ImageNet (0.824) beat every SSL method.** SSL only helped in the low-data regime (<200 subjects). We have 4,407 studies — firmly in the regime where ImageNet wins. This is a warning against the DINOv2 path, and a caution that domain pretraining advantages shrink with data scale.

### Implementation failure modes (ours to prevent)

1. **Normalization mismatch.** RadImageNet trained on `(x-127.5)*2/255` → [-1,1]. If the v2.4 loader silently feeds [0,1], features are garbage and the run fails for a stupid reason. The PROBE must verify input range.
2. **Wrong LR destroys features.** Backbone LR 1e-4 is the paper's value. If someone bumps it to 3e-4 "to match v2.3," the domain-adapted features get catastrophically overwritten in early epochs.
3. **Weight-file format.** The .pt uses `nn.Sequential` indexing. `strict=True` must pass — if it silently falls back to `strict=False` with missing keys, we're training a randomly-initialized ResNet-50 (would still learn, but no RadImageNet benefit, wasting 7 hours).
4. **Frozen-BN mismatch.** If BatchNorm layers were frozen during RadImageNet training but we leave them in train mode (or vice versa), statistics shift and early epochs are unstable.

---

## 2. Top 3 Alternative Pretraining Strategies (if RadImageNet fails)

### Alternative 1: MRNet supervised pretraining (STRONGEST)

- **What:** 1,370 Stanford knee MRI exams, 3-plane (sagittal/coronal/axial), labels: abnormal / ACL tear / meniscal tear (3-radiologist majority vote).
- **Why it beats RadImageNet for us:** Knee-specific (not general radiology), supervised (not self-supervised), 3-plane (matches our 2.5D), 3 of 12 findings directly overlap.
- **Cost:** RUA registration (~1 day) + ~3 GPU-hrs pretraining + fine-tune on severity labels.
- **Expected:** +0.010 to +0.020 over ImageNet baseline. Lower ceiling than RadImageNet's published numbers, but far less downside risk — it's the same anatomy, same modality.
- **When to use:** If RadImageNet aborts, start MRNet paperwork immediately (parallel with seed-2 training).

### Alternative 2: ImageNet EfficientNet-B3, seed 2 (SAFEST)

- **What:** Exact v2.3 recipe, different random seed.
- **Why:** Proven 0.8231 EMA. Seed diversity is the most reliable ensemble gain in the literature. Zero implementation risk.
- **Cost:** ~8 GPU-hrs.
- **Expected:** Solo ~0.82 (same as v2.3); ensemble with v2.3 → +0.004 to +0.008.
- **When to use:** Always (regardless of RadImageNet outcome). This is the backbone of the ensemble plan.

### Alternative 3: ConvNeXt-Tiny, ImageNet (DIVERSITY)

- **What:** Different architecture family (large-kernel convs, LayerNorm), ImageNet pretrained, same severity labels, same MIL head.
- **Why:** Cross-architecture disagreement > cross-seed disagreement (Krogh–Vedelsby). If it solos within 0.01 of B3, it adds +0.002 to +0.006 to the ensemble.
- **Cost:** ~8 GPU-hrs.
- **Expected:** Solo 0.81–0.83; ensemble contribution +0.002–0.006.
- **When to use:** After seed 2, if budget remains.

### Explicitly NOT recommended

- **DINOv2/DINOv3 as primary:** No knee MRI evidence; the 2403.16499 study shows SSL loses to ImageNet at our data scale on knee MRI specifically. Only as a 4th ensemble member for diversity, never as the main bet.
- **Self-supervised pretraining on our own 4,407:** We don't have a held-out unlabeled pool (train IS the data). Contrastive pretraining on the same images we'd fine-tune on is circular and the knee-MRI evidence says ImageNet wins at this scale anyway.
- **Freezing the backbone:** With 4,407 studies we have enough data to fine-tune everything. Freezing wastes the adaptation.

---

## 3. Minimum Viable Test: Is RadImageNet Working?

### Reference trajectory (v2.3, ImageNet B3, severity labels)

| Epoch | Val macro | EMA macro | Gold macro |
|---|---|---|---|
| 5 | 0.7737 | 0.7714 | 0.7381 |
| 8 | 0.7957 | 0.8006 | 0.8188 |
| 10 | 0.8049 | 0.8129 | 0.7980 |
| 12 | **0.8179** (best val) | 0.8205 | 0.8310 |
| 13 | 0.8125 | **0.8231** (best EMA) | 0.8277 |

### Decision thresholds for v2.4 (RadImageNet R50)

RadImageNet uses backbone LR 1e-4 (vs 3e-4 for v2.3), so allow a slightly slower start — but the domain head-start should compensate. Thresholds are on **EMA macro** (the stable metric):

**Epoch 5 checkpoint (~1.5 GPU-hrs spent):**
- **EMA ≥ 0.78 → GREEN.** Ahead of v2.3's 0.7714 at ep5. Domain pretraining is helping. Continue.
- **EMA 0.74–0.78 → YELLOW.** Roughly matching v2.3's trajectory (0.7714). Not failing, but no evidence of gain yet. Continue to ep8, but flag for close watch.
- **EMA < 0.74 → RED.** Behind v2.3 by >0.03 at the same point. Either the weights didn't load correctly, normalization is wrong, or RadImageNet genuinely doesn't transfer here. **STOP and diagnose** (check: strict=True load, input range [-1,1], LR values in log) before spending another hour.

**Epoch 8 checkpoint (~2.5 GPU-hrs spent):**
- **EMA ≥ 0.81 → GREEN.** Ahead of v2.3's 0.8006. Working. Continue to completion.
- **EMA 0.78–0.81 → YELLOW.** Matching v2.3. The run isn't failing, but the expected +0.015 isn't materializing. Decision: continue ONLY if gold-macro is also ≥ v2.3's trajectory (0.8188 at ep8). If gold is lower too, the run is a lateral move — abort and reallocate.
- **EMA < 0.78 → RED. ABORT.** Two checkpoints behind. RadImageNet is not beating ImageNet here. Cut losses at ~2.5 GPU-hrs.

**Epoch 10 checkpoint (~3 GPU-hrs spent):**
- **EMA ≥ 0.82 → GREEN.** On track to beat v2.3's 0.8231 peak. Continue.
- **EMA < 0.80 → RED. ABORT.** v2.3 was at 0.8129 here. If we're below 0.80 at ep10, we will not beat 0.8231. Do not spend the remaining 4 hours hoping.

### Gold-macro guardrail (all checkpoints)

If **gold-macro drops >0.03 below v2.3's trajectory** at the same epoch while val/EMA look fine, the model is overfitting to weak labels faster than v2.3 did — a pretraining-independent failure. Note it, but don't abort solely on gold noise (58 studies, ±0.02 swings are normal; see v2.3 ep9→ep10: 0.8487→0.7980).

### The 5-minute pre-flight (before PROBE=False)

1. `strict=True` passed on weight load (check log for "RadImageNet weights loaded OK (strict=True)").
2. Feature dim printed as 2048.
3. PROBE loss decreases across probe steps (not NaN, not flat).
4. First real epoch completes without NaN.

If any of these fail, it's an implementation bug, not a RadImageNet verdict — fix and re-probe, don't abandon.

---

## 4. If We Abort: Best Use of Remaining GPU Budget

### Budget math (45 hrs/week)

| Spent | Hours |
|---|---|
| v2.3 full run | ~8 |
| RadImageNet aborted at ep5 | ~1.5 |
| RadImageNet aborted at ep8 | ~2.5 |
| **Remaining (abort at ep5)** | **~35.5** |
| **Remaining (abort at ep8)** | **~34.5** |

### Reallocation plan (priority order)

**1. EffNet-B3 seed 2 — 8 hrs — HIGHEST PRIORITY, start immediately**
Same v2.3 recipe, different seed, severity-merged labels. Proven 0.8231 EMA solo. Ensemble with v2.3: +0.004 to +0.008. Zero implementation risk. This is the single most reliable gain left on the table.

**2. MRNet RUA paperwork — 0 GPU-hrs, start in parallel**
Register at stanfordmlgroup.github.io/competitions/mrnet/. ~1 day turnaround. If approved, pretraining (~3 hrs) can run while seed 2 trains.

**3. CORAL ordinal head probe — ~2 GPU-hrs**
Fine-tune from v2.3's best checkpoint (ep12): add 4 CORAL heads (none/mild/moderate/severe) on the severity findings, λ=0.3–0.5, 5 epochs. Our moat — no team does severity. If gold improves, it's real signal.

**4. ConvNeXt-Tiny — 8 hrs**
Cross-architecture diversity for the ensemble. Only if solo validates within 0.01 of B3.

**5. TTA in submission notebook — 0 GPU-hrs training**
8-view intensity-only TTA (original, h-flip, brightness ±0.10, contrast gamma 0.90/1.10, 2 window variants). Validate on gold before trusting. +0.005 to +0.012 if it holds.

**6. Label-dependency probe (PCD loss) — ~2 GPU-hrs**
Pairwise Correlation Difference auxiliary loss aligning predicted vs empirical label-correlation matrix. Cheap test of the "12 findings aren't independent" hypothesis.

### What NOT to do with the freed budget

- Don't launch a second RadImageNet variant (different LR, frozen stages) — if the paradigm doesn't transfer, hyperparameter tweaks won't save it.
- Don't start DINOv2 — the knee-MRI evidence says ImageNet wins at our data scale.
- Don't do a 3rd B3 seed — diminishing returns (+0.001–0.003), spend it on ConvNeXt instead.

---

## 5. Decision Tree (one page)

```
PROBE=True
  ├── FAIL (NaN / strict=False / dim≠2048 / loss flat)
  │     → Implementation bug. Fix, re-probe. Do NOT judge RadImageNet yet.
  └── PASS ("PROBE: ALL GREEN")
        └── PROBE=False, launch 25-epoch run
              ├── EPOCH 5 (~1.5 GPU-hrs)
              │     ├── EMA ≥ 0.78 → GREEN: continue
              │     ├── EMA 0.74–0.78 → YELLOW: continue to ep8, watch closely
              │     └── EMA < 0.74 → RED: stop, diagnose loader/norm/LR
              │           ├── Bug found → fix, restart (1 probe, not a verdict)
              │           └── No bug → ABORT RadImageNet → go to PLAN B
              ├── EPOCH 8 (~2.5 GPU-hrs)
              │     ├── EMA ≥ 0.81 → GREEN: continue to completion
              │     ├── EMA 0.78–0.81 → YELLOW: continue only if gold ≥ 0.82
              │     └── EMA < 0.78 → RED: ABORT → PLAN B
              ├── EPOCH 10 (~3 GPU-hrs)
              │     ├── EMA ≥ 0.82 → GREEN: continue (on track to beat 0.8231)
              │     └── EMA < 0.80 → RED: ABORT → PLAN B
              └── EPOCH 25 (done, ~7 GPU-hrs)
                    ├── Best EMA > 0.8231 → SUCCESS: ensemble with v2.3
                    ├── Best EMA 0.81–0.8231 → MARGINAL: ensemble anyway
                    │     (diversity gain may still add +0.005–0.01)
                    └── Best EMA < 0.81 → FAILURE: keep v2.3 as solo,
                          ensemble from seed-2 instead

PLAN B (on abort):
  1. Launch B3 seed 2 immediately (8 hrs) — highest reliability
  2. Start MRNet RUA paperwork (parallel, 0 GPU)
  3. CORAL probe from v2.3 ep12 ckpt (2 hrs)
  4. ConvNeXt-Tiny (8 hrs)
  5. TTA + rank-average ensemble (0 GPU training)
  Expected PLAN B outcome: ~0.90–0.92 LB (vs ~0.89 now)
```

---

## Sources

- RadImageNet negative results (dental, PET/CT, lung US, brain tumor): https://www.springermedizin.de/radimagenet-and-imagenet-as-datasets-for-transfer-learning-in-th/27383006
- RadImageNet review (transferability limits, thermography): https://link.springer.com/article/10.1007/s00521-026-12435-y
- RadImageNet original (Figure 4, gain compression on large datasets): https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/
- SSL vs ImageNet on knee MRI at scale (ImageNet wins at 800 subjects): https://arxiv.org/html/2403.16499v1
- SB-SSL knee MRI (low-data regime only): https://arxiv.org/abs/2208.13923v1
