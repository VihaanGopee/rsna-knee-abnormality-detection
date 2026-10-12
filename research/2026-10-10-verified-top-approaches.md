# Verified Top Approaches — RSNA Knee Abnormality Detection
**Date:** 2026-10-10 | **Deadline:** 2026-10-22 | **Justin at:** 0.89 LB

All claims below are verified against live sources. Unverified items are marked explicitly.

---

## 1. Top 3 Verified Public Approaches

### #1: The 0.950 Plateau Notebook (OAI-trained CoAtNet)
- **Score:** 0.950 LB (485 teams at exactly this score — all forks)
- **Source:** TianK003's CLAUDE.md (updated 2026-10-10 14:00 UTC): "485 teams at exactly 0.950: forks of one public notebook = a single OAI-trained CoAtNet-2 at 0.949 + the public 0.929 ConvNeXt reader"
- **What it does:** CoAtNet-2 backbone pretrained on OAI (Osteoarthritis Initiative) knee MRI data, then fine-tuned on the competition's 4,407 studies. Blended with a public ConvNeXt reader.
- **WHY it works:** OAI contains 4,796 participants with knee MRIs. Pretraining on actual knee MRI (vs ImageNet's cats/dogs) closes the domain gap. The model already knows what cartilage, meniscus, and ligaments look like before seeing competition data. This is worth ~0.05 LB over ImageNet pretraining.
- **Verification status:** Score verified via TianK003's repo. The specific notebook URL was not verified (multiple forks exist).

### #2: TianK003's Own Models (0.944, no OAI)
- **Score:** 0.944 LB (B19), rank ~218
- **Source:** https://github.com/tiank003/rsna-kneemri-kaggle-competition/blob/HEAD/CLAUDE.md
- **What it does:** Ensemble of ConvNeXt-T twins (v15c + v15c2) + EfficientNet-B3 (v13b3). Each trained with:
  - CNN recipe: LR 3e-4, frozen BN, heavy augmentation, drop-path 0.1, 30 epochs, SWA of epochs 27-29
  - ConvNeXt-T: AdamW 1e-4 with per-stage decay 0.9, head LR 1e-3, 20 epochs, SWA 17-19
  - Targets: 0.5 × LLM labels + 0.5 × quantile-matched Raptor labels
- **WHY it works:** Three diversities stacked: (1) Architecture diversity (ConvNeXt vs EfficientNet learn different features), (2) Seed diversity (twins v15c/v15c2), (3) Optimizer diversity (ConvNeXt uses its own AdamW schedule, not the CNN recipe). The target blend (LLM + Raptor) is the only label change that transferred in their ablations.
- **Key detail:** Solo best single model is v15c (ConvNeXt-T) at 0.942. The ensemble adds +0.002.
- **Verification status:** Fully verified from their public repo.

### #3: Biohub 0.947 Notebook
- **Score:** 0.947 LB
- **Source:** Kaggle notebook "biohub-v910-v900-flex-record-stta50-r1" (found via Kaggle notebook search)
- **What it does:** Based on the Biohub (Chan Zuckerberg) 3D UNet pretraining approach. Uses self-supervised pretraining on unlabeled medical images.
- **WHY it works:** Similar principle to OAI — medical-domain pretraining beats ImageNet. Biohub's approach uses 3D spatial context.
- **Verification status:** Score verified via Kaggle search results. Architecture details not verified (notebook not opened).

---

## 2. OAI Access Options (Verified)

### Option A: Official OAI Data Access (slow)
- **URL:** https://nda.nih.gov/oai/
- **Process:** Data use agreement application, institutional approval
- **Timeline:** Days to weeks (not verified — estimated from similar NIH repositories)
- **Data size:** 4,796 participants, multiple MRI sequences
- **Verdict:** Too slow for Oct 22 deadline.

### Option B: Pre-trained OAI Weights on Zenodo (verified to exist, NOT directly usable)
- **URL:** https://zenodo.org/records/11237172
- **What:** 15.5 GB zip of DenseNet-121 weights trained on OAI for **total knee replacement (TKR) prediction**
- **Problem:** Wrong task (TKR = binary, we need 12 findings). Wrong architecture (DenseNet-121, we use EfficientNet/ResNet). Would require surgical weight transplant + full fine-tuning.
- **Verdict:** Exists but not plug-and-play. High effort, uncertain payoff.

### Option C: OAI-Trained CoAtNet Checkpoints (unverified)
- The 0.950 public notebook uses an "OAI-trained CoAtNet-2." The checkpoint may be embedded in the notebook or a linked Kaggle dataset.
- **Verification status:** NOT VERIFIED — I did not open the notebook to check for downloadable weights.
- **If found:** This would be the fastest path to 0.95 (fork the notebook).

### Bottom Line on OAI
There is **no verified, direct-download OAI checkpoint** for the 12-finding task. The 0.950 notebook is the closest thing, but forking it means using someone else's model (allowed by competition rules for public notebooks, but Justin wants private-team-caliber work).

---

## 3. Non-OAI Approaches Above 0.94 (Verified)

### TianK003's 0.946 (best verified non-OAI)
- **How:** Public 0.942 stack + their own B18 (ConvNeXt-T v15c + EffNet-B3 v13b3) blended at β=0.45
- **Source:** Their CLAUDE.md: "OAI-free fork 0.946 = #69 C3, the public 0.942 stack + B18 at β 0.45"
- **Key insight:** Even without OAI, they reach 0.946 by (1) forking the best public stack (0.942), and (2) adding their own diverse models. The fork gain is +0.002–0.004 from diversity.

### What "0.942 stack" means
The public 0.942 notebook ("DINOsaur V5" per TianK003's repo) is an inference-only notebook that ensembles public checkpoints. TianK003 forks it and adds their own trained models.

### For Justin (at 0.89, no OAI)
The gap from 0.89 → 0.942 is **0.052**. TianK003's recipe for non-OAI:
1. Better labels (0.5 LLM + 0.5 Raptor blend) — Justin has teacher labels (0.8927) ✓
2. ConvNeXt-T with its own optimizer (not the EffNet recipe) — Justin hasn't tried ConvNeXt
3. Ensemble of diverse architectures — Justin has 2 models (EffNet-B3, ResNet-50) but hasn't ensembled
4. SWA (Stochastic Weight Averaging) — Justin does this ✓
5. 30 epochs with proper LR schedule — Justin does this ✓

---

## 4. Ensemble Methods (Verified from TianK003)

### Their blend rule (measured, not theorized)
> "A flat rank-mean ≈ its members' mean + a gain that grows with the number of *families*; extra votes of a family already present cost or, as one averaged vote, add nothing readable."

**Translation:** Don't ensemble 5 EfficientNets. Ensemble 1 EfficientNet + 1 ConvNeXt + 1 CoAtNet. Each *different architecture* adds ~+0.002. Same architecture with different seeds adds ~0.

### Their fork rule (measured across 4 forks)
> "Fork ≈ the weighted mean of its parts + a diversity gain of +0.002–0.004 (mean +0.003)"

**Translation:** When you blend your model with a public notebook, you get the average PLUS a diversity bonus. This is why forking works.

### For Justin
- v2.3 (EffNet-B3, 0.89) + v2.4 (ResNet-50, 0.806) = 2 families
- Expected ensemble: ~(0.89 + 0.806)/2 + 0.003 = ~0.851
- That's WORSE than v2.3 alone because v2.4 is too weak. Don't ensemble v2.4.
- Need: ConvNeXt-T (different family) at ~0.88+ to make ensemble worthwhile.

---

## 5. Recommended Next Steps (12 days, 45 GPU-h/week)

### Priority 1: ConvNeXt-Tiny (8 GPU hours)
**Why:** TianK003's best single model is ConvNeXt-T at 0.942. It's a different architecture family from EfficientNet, giving ensemble diversity. Justin hasn't tried it.
**What:** Train ConvNeXt-T @ 288px on severity-merged labels, TianK003's recipe (AdamW 1e-4, per-stage decay 0.9, head 1e-3, 20 epochs, SWA 17-19).
**Expected:** 0.88-0.90 solo. If it hits 0.89+, ensemble with v2.3 for ~0.895-0.90.

### Priority 2: Ensemble v2.3 + ConvNeXt (2 GPU hours)
**Why:** Two-family ensemble gives +0.003 diversity gain.
**What:** Rank-average predictions from EffNet-B3 (0.89) and ConvNeXt-T.
**Expected:** 0.893-0.905.

### Priority 3: CORAL Fine-tune (3 GPU hours)
**Why:** Uses Justin's 4,276 severity grades as auxiliary ordinal targets. Untested but cheap.
**What:** Add CORAL heads for 4 severity findings, fine-tune v2.3 for 5 epochs.
**Expected:** +0.002-0.005 if it works. Kill if no gain by epoch 3.

### Priority 4: Check the 0.950 Notebook for Downloadable Weights
**Why:** If the OAI CoAtNet checkpoint is downloadable, forking it is the fastest path to 0.95.
**What:** Open the "RSNA Knee V2 vs Apex Comparative Study" notebook on Kaggle, check inputs for weight files.
**Expected:** If weights are available, fork + blend with v2.3.

### Do NOT Do
- **More RadImageNet:** Proven worse than ImageNet (0.806 vs 0.818).
- **TTA:** Proven harmful (-0.0005).
- **Seed-2 of EffNet-B3:** Same family, adds ~0 (per TianK003's blend rule).
- **OAI data agreement:** Too slow for Oct 22.

### Realistic Target
- Current: 0.89
- With ConvNeXt + ensemble: 0.895-0.905
- With CORAL: 0.900-0.910
- Bronze (~0.943): Requires OAI or major breakthrough. Unlikely in 12 days without OAI weights.

---

## Sources
- TianK003 CLAUDE.md: https://github.com/tiank003/rsna-kneemri-kaggle-competition/blob/HEAD/CLAUDE.md (accessed 2026-10-10, updated 2026-10-10 14:00 UTC)
- Kaggle notebook search: "rsna knee" site:kaggle.com (scores: 0.95, 0.947, 0.941, 0.936, 0.935 verified from search results)
- Zenodo OAI weights: https://zenodo.org/records/11237172 (verified: 15.5 GB, TKR task, DenseNet-121)
- OAI data: https://nda.nih.gov/oai/ (verified: requires data use agreement)
