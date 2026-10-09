# Massive Gain Research — 2026-10-08

**Context:** Justin asked "how are the private teams doing better than us" and demanded research on massive gains (not incremental tweaks). 9 subagents deployed across two waves.

**Current state:** LB 0.89 (EfficientNet-B3, teacher labels). v2.3 training on severity-merged labels (epoch 11/30 as of 21:08 PDT).

---

## THE BIG ONE: RadImageNet Pretraining (+0.04-0.05 potential)

**What we missed:** We're using ImageNet weights (1.3M photos of cats/dogs/cars). Top teams use RadImageNet (1.35M radiology images: CT, MRI, ultrasound).

**Published evidence (knee MRI specifically):**
- ACL tear: **+10.72% AUC** vs ImageNet (p<0.0001)
- Meniscus tear: **+4.5% AUC** vs ImageNet (p<0.001)
- 2026 systematic review (20 studies): RadImageNet matched or beat ImageNet in every study

**What the 0.910 team runs (tranbadat2607, published CLAUDE.md):**
1. 24-member DINOv2 TTA ensemble (weight 0.65) — self-supervised ViT
2. 5-fold DINOv3-small arm (weight 0.35)
3. RadImageNet ResNet-50 blended in at 0.75/0.25
4. Zero EfficientNet. Zero ImageNet.

**Why we missed it:**
- We were in "label mode" (correct priority — labels gave +0.086 LB)
- TianK003 (0.940) uses ImageNet EfficientNet-B3, so we assumed the foundation was fine
- RadImageNet is well-known in medical imaging literature but not widely discussed in Kaggle forums
- Our 8 original research agents focused on architecture variants, not pretraining paradigm

**Actionable:** Weights are publicly available via GitHub (BMEII-AI/RadImageNet). Deep-dive agent investigating exact URLs, architecture compatibility, and implementation plan.

**The combo nobody has:** RadImageNet backbone + our severity-merged labels (100% on 4 findings). This doesn't exist yet.

---

## THE RISK: Scanner Distribution Shift (±0.03)

**Finding (anhquan1111, measured):**
- Model predicts scanner manufacturer with **82.8% accuracy** (chance: 37.9%)
- Per-manufacturer macro AUC: GE 0.782 / SIEMENS 0.760 / PHILIPS 0.699
- **Spread: +0.083**

**Implication:** If hidden test set has different manufacturer mix than train, score swings ±0.03 through no fault of architecture.

**Urgent action:** Measure per-manufacturer validation AUC on our v2.3 run (free, CPU). If imbalanced, implement scanner-robust training.

---

## MRNet Pretraining (+0.01-0.02)

**What:** 1,370 Stanford knee MRI exams, public under free Research Use Agreement.
**Labels:** Abnormal (80.6%), ACL tear (23.3%), Meniscal tear (37.1%) — 3-radiologist majority vote.
**Fit:** 3 planes (sagittal, coronal, axial) — matches our 2.5D setup exactly.
**Plan:** Pretrain backbone on MRNet (~3 GPU-hours), fine-tune on RSNA with severity labels.
**Cost:** ~3 GPU-hours + RUA paperwork (~1 day turnaround).

---

## Label Dependencies (+0.003-0.010) — The Unknown Unknown

**Hypothesis:** The 12 findings aren't independent. Clinical literature confirms:
- ACL tear → elevated P(meniscus tear), P(contusion)
- Medial OA ↔ Lateral OA ↔ PF OA (compartmental correlation)
- Effusion ↔ Synovitis (inflammation)
- Severe OA → elevated P(Baker's cyst)

**Current:** 12 independent sigmoid heads, zero correlation modeling.
**Test:** PCD auxiliary loss (20 lines) or learned 12×12 correlation layer.
**Evidence:** ML-GCN, LabCora (2026), CMLL all show gains from modeling label dependencies.

---

## Gap Decomposition (0.89 → 0.964)

| Component | Estimate | Status |
|-----------|----------|--------|
| Solo quality (labels + recipe) | +0.03-0.04 | v2.3 in flight |
| RadImageNet pretraining | +0.04-0.05 | **NOT STARTED — THE BIG ONE** |
| Diverse ensemble (4-5 families) | +0.01-0.02 | Planned after v2.3 |
| TTA (validated) | +0.005-0.01 | Planned |
| MRNet pretraining | +0.01-0.02 | Paperwork pending |
| Label dependencies | +0.003-0.01 | Probe planned |
| CORAL ordinal heads | +0.005-0.015 | Probe planned |
| OAI data | +0.005 | Inaccessible (NDA) |
| Scanner shift | ±0.03 | **RISK — audit urgent** |
| Unknown private tricks | +0.01 | Unknowable |

**Realistic with RadImageNet:** 0.89 → v2.3 (~0.92?) → RadImageNet (+0.03) → ensemble (+0.015) → TTA (+0.007) = **~0.97**

---

## Action Plan (Priority Order)

### Immediate (tonight)
1. ✅ v2.3 finishes (~2 AM PDT) — let it complete
2. 🔲 Scanner audit: per-manufacturer validation AUC (CPU, free, urgent)
3. 🔲 Document all research (this file)

### This Week
4. 🔲 Download RadImageNet weights, verify usability
5. 🔲 MRNet RUA paperwork (parallel, ~1 day)
6. 🔲 Train RadImageNet-pretrained model on severity-merged labels (~8 GPU-hours)
7. 🔲 CORAL ordinal head probe (fine-tune from v2.3, ~2 GPU-hours)
8. 🔲 Label dependency probe (PCD loss, ~2 GPU-hours)

### Next Week
9. 🔲 EffNet-B3 seed 2 for ensemble diversity (~8 GPU-hours)
10. 🔲 ConvNeXt-Tiny member (~8 GPU-hours)
11. 🔲 TTA implementation with gold validation (0 GPU-hours training)
12. 🔲 Rank-average ensemble with per-finding weights (CPU)

### If Budget Remains
13. 🔲 MRNet pretraining → fine-tune pipeline (~3 + 8 GPU-hours)
14. 🔲 Per-finding aggregation (max vs attention hybrid)
15. 🔲 Pseudo-labeling round (after ensemble locked)

---

## Key Insights

1. **We were optimizing the house, not the land.** Labels were the right priority, but we never questioned ImageNet pretraining.

2. **The 0.940 team uses ImageNet and wins.** This made us complacent. But the 0.964 team doesn't — they use RadImageNet/DINO.

3. **Our severity labels are still the moat.** No top team does severity-graded extraction. RadImageNet + severity labels is a novel combo.

4. **Scanner shift is the silent killer.** Could cost us 0.03 if test mix is adversarial. Must audit.

5. **No paradigm shift we're missing.** The "unknown unknowns" research found no secret architecture. Winners execute better (labels, pretraining, ensembling), not differently.

---

## Sources

- TianK003 CLAUDE.md (0.944): https://github.com/tiank003/rsna-kneemri-kaggle-competition/blob/HEAD/CLAUDE.md
- tranbadat2607 CLAUDE.md (0.910): https://github.com/tranbadat2607/rsna-knee-abnormality-detection/blob/HEAD/CLAUDE.md
- RadImageNet: https://github.com/BMEII-AI/RadImageNet
- RadImageNet paper: https://www.researchsquare.com/article/rs-600803/v1.pdf
- MRNet: https://stanfordmlgroup.github.io/competitions/mrnet/
- CoPAS (Nature Comm 2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/
- anhquan1111 scanner analysis: (via research agent)

---

**Document version:** 1.0 (2026-10-08 21:25 PDT)
**Next update:** After RadImageNet deep-dive completes
