# Verification Brief: DINOv3 License + Efficiency Track Rules

**Date:** 2026-09-30
**Purpose:** Go/no-go on DINOv3 as a model family; official efficiency-track rules and strategy.

---

## TASK 1 — DINOv3 LICENSE: ✅ GO

### 1.1 The license, authoritatively

The Hugging Face model card for the exact checkpoint we plan to use
(`facebook/dinov3-vits16-pretrain-lvd1689m`) states:

- **License: "DINOv3 License"** — a custom Meta license. **Not** CC-BY-NC, **not** Apache-2.0.

This resolves a conflict in third-party sources: one repo (`www06sev/mscoffe`)
claims CC-BY-NC-4.0, but the authoritative HF card plus the majority of recent
third-party license audits agree on the custom DINOv3 License. The CC-BY-NC claim
is treated as outdated/wrong.

**What the DINOv3 License permits** (corroborated across 4 independent third-party
license audits: alfaomegagrafx, qtmesheditor, anylearning-oss, alayalab/worldsculpt):

- ✅ **Commercial use permitted** (this is the key point — it is NOT non-commercial)
- ⚠️ With restrictions: export controls, **no military/weapons end-use**, no ITAR-controlled activities
- ⚠️ **Redistribution must be under the same DINOv3 License**
- ⚠️ **Attribution required**: "Built with DINOv3" wherever the weights are documented
- ⚠️ Download is **gated**: must accept Meta's terms on Hugging Face or Meta's portal

None of the restrictions touch our use case (knee MRI research competition).

### 1.2 Does Kaggle's ruleset allow it?

The competition's Code Requirements (quoted from the official competition page):

> "Freely & publicly available external data is allowed, **including pre-trained models**"

DINOv3 weights are publicly available (free terms acceptance — the same bar as
every other gated HF model). **Precedent in this exact competition:** multiple
teams already use DINOv3 — `mattiaangeli/knee-mri-fold-weights` runs
`vit_small_patch16_dinov3.lvd1689m` via timm, and tranbadat's 0.910-public-LB
ensemble includes a 5-fold DINOv3-small arm. No objections recorded.

**Winner obligations compatibility:** winners must open-source their solution and
publish weights. The DINOv3 License's redistribution-under-same-license term is
compatible with this — we comply by carrying the license + "Built with DINOv3"
attribution. No conflict.

### 1.3 The "is a Kaggle prize competition commercial use?" question

**Moot.** Since the DINOv3 License permits commercial use outright, it doesn't
matter whether prize-money competition counts as commercial. We're covered either way.

### 1.4 Practical handling for our pipeline

- **Gated download problem:** the Kaggle scoring notebook runs with internet OFF,
  so timm cannot download weights on the fly. **Action:** download once (Colab/local
  with HF token after accepting terms) → upload as a Kaggle Dataset/Model → mount
  in the notebook. Load via `timm.create_model(..., pretrained=False)` +
  `load_state_dict`, or pre-seed the torch hub cache. This is the established pattern.
- **Attribution:** add "Built with DINOv3" to our repo README, notebook headers, and
  any writeup, per the license.

### 1.5 Fallback ranking (if DINOv3 ever became problematic)

| Rank | Backbone | License | Note |
|---|---|---|---|
| 1 | **DINOv2 ViT-S/14** | Apache-2.0 | Already our workhorse; zero license risk |
| 2 | **SigLIP / SigLIP2** | Apache-2.0 | Google; strong vision-language pretraining |
| 3 | **ConvNeXt (timm, ImageNet-22k)** | Apache-2.0 | License-clean per competitor audit; first-choice CNN member |
| 4 | EVA-02 | Unclear (verify before use) | Do not use without a license read |

### 1.6 ⚠️ FLAG: RadImageNet is the actual license risk, not DINOv3

Our plan includes RadImageNet ResNet-50 as a blend arm. Competitor audit (TianK003,
checked 2026-08-28): **"RadImageNet weights carry no stated licence"** (code MIT,
paper CC BY 4.0, data "by request"). Recommendation from that audit: **treat as
restrictive** until radimagenet.com's Terms are read. 

**Implication:** for a competition whose winner obligations require open-sourcing
weights, an unclear license is a real risk. **Recommendation:** keep RadImageNet as
a Tier-2 blend arm only; do not make it load-bearing for our submission. DINOv2 +
DINOv3 + CoAtNet give us diversity without it.

---

## TASK 2 — EFFICIENCY TRACK RULES

### 2.1 Official formula

From the official competition page (via community mirror of the rules text):

$$\text{Efficiency} = \frac{\text{AUC}_{\text{submission}}}{\text{Benchmark} - \max\text{AUC}} + \frac{\text{RuntimeSeconds}}{32400}$$

- **Minimized.** Lower is better.
- $\text{AUC}_{\text{submission}}$ = our score on the main metric (macro AUC)
- $\text{Benchmark}$ = `sample_submission.csv` score (~0.5, all-0.5 predictions)
- $\max\text{AUC}$ = best private-LB score across all teams
- $\text{RuntimeSeconds}$ = our notebook's scoring runtime; 32,400 s = the 9-hour cap

**Note on sourcing:** two community transcriptions of the official formula disagree
in notation (one omits the submission-AUC term, likely a copy error). The version
above is the self-consistent one — independently verified by working the math:
with Benchmark≈0.5 and maxAUC≈0.95, **one extra hour of runtime costs ≈0.05 AUC**.
A formula without the submission's AUC would contradict the stated goal of
"evaluating both runtime and predictive performance."

**There is a public efficiency leaderboard notebook**, updated daily during the
competition (ranks only, not scores). Final scoring is on private data.

### 2.2 Eligibility (official)

A submission is eligible for the Efficiency Prize iff:

1. It is **among the team's selected submissions** for the Leaderboard Prize
   (or auto-selected per the My Submissions tab rules), **AND**
2. It ranks on the **private LB above the `sample_submission.csv` benchmark**.

**A single submission may be eligible for — and win — both tracks.**

### 2.3 Prizes

- 1st Efficiency: **$7,000**
- 2nd Efficiency: **$6,000**
- 3rd Efficiency: **$5,000**
- Total: $18,000 of the $77,000 pool.

Note: 1st efficiency ($7K) equals 2nd place on the main track ($7K).

### 2.4 What wins it

The math brutally favors speed: **1 hour of inference ≈ 0.05 AUC points.**
Worked example (Benchmark 0.5, maxAUC 0.95):

| Submission | AUC | Runtime | Efficiency (lower wins) |
|---|---|---|---|
| Accurate but slow | 0.94 | 2.0 h | −1.867 |
| Weaker but fast | 0.90 | 0.5 h | −1.944 ← wins |

A single small backbone finishing in 30–60 minutes at ~0.85–0.90 AUC beats a
0.94-AUC 8-hour ensemble. Our main-track ensemble (10–15 members, ~8h) would be
**terrible** for this track. The winner is a **small, fast, decent** model —
not a big accurate one.

### 2.5 Is it worth our effort? Does it conflict with the main track?

**No rule conflict.** Separate scoring, same submission pool; one entry can win both.

**The real tradeoff is the 2 final selections.** Kaggle lets a team select 2
submissions for prize judging. Contending in both tracks likely means spending
one selection on the main-track best and one on the efficiency entry —
giving up a second main-track shot.

**Recommendation: pursue as a low-cost side bet.**

- *For:* $7K first prize is real money; marginal cost is low — our Tier-1 pipeline
  already produces fast single models (a single DINOv2 at 336px runs ~2 s/study,
  ≈45 min on 1,300 studies). The efficiency entry is a *different, smaller*
  notebook, not a distraction from main-track training.
- *Against:* it must not siphon focus in weeks 1–2. Build it in week 3 from the
  best fast single model we already have.
- *Decision point:* final selection split (2 main vs 1 main + 1 efficiency) gets
  made at the end, based on where we stand. No commitment needed now.

**What NOT to do:** do not try to make one notebook serve both tracks. The
objectives are opposed (accuracy wants big ensembles; efficiency wants tiny
models). Two notebooks, two strategies.

---

## BOTTOM LINE FOR THE PARENT

1. **DINOv3: GO.** Custom Meta license, commercial use permitted, competition
   rules explicitly allow pretrained models, in-competition precedent exists,
   winner open-source obligations are compatible. Action items: gated download →
   Kaggle dataset mount; "Built with DINOv3" attribution in docs.
2. **RadImageNet: FLAG.** No stated license — treat as restrictive, keep as
   non-load-bearing Tier-2 arm only.
3. **Efficiency track: real, official, $18K.** Formula strongly rewards speed
   (1h ≈ 0.05 AUC). Pursue as a week-3 side bet with a dedicated small/fast
   notebook; decide the final selection split at the end.

### Sources
- DINOv3 license: HF model card `facebook/dinov3-vits16-pretrain-lvd1689m`
  (license field: "DINOv3 License"); third-party audits: alfaomegagrafx/3daigc-api
  MODEL_LICENSES.md, fernandotonon/qtmesheditor THIRD_PARTY_LICENSES.md,
  nrl-ai/anylearning-oss model_license_policy.md, alayalab/worldsculpt
  THIRD_PARTY_LICENSES.md, condadosai/modern-yolonas apache-weights-plan.md
- Competition rules: official code requirements via ahmadenan/rsna-knee
  COMPETITION.md ("Freely & publicly available external data is allowed,
  including pre-trained models"); efficiency rules via same; formula cross-checked
  against existentialistlogarithmic/knee-abnormality FINDINGS.md §2.13 and README
  tradeoff analysis (1h ≈ 0.0502 AUC, mathematically consistent)
- DINOv3 in-competition use: tranbadat2607/rsna-knee-abnormality-detection
  CLAUDE.md (mattiaangeli DINOv3 arm, 0.910 ensemble)
- RadImageNet license flag: TianK003/RSNA-KneeMRI-kaggle-competition CLAUDE.md
