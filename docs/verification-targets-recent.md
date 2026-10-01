# Verification Brief: Per-Target Ceilings + Recent Developments

**Date:** 2026-09-30
**Purpose:** Extend the report-mention ceiling analysis to all 12 targets; sweep for developments since ~Sep 16, 2026.
**Tag convention:** `VERIFIED` = measured/read in a primary source named below. `INFERRED` = analyst conclusion, marked as such.

---

## Topic 1 — Per-target report-mention ceilings

### 1a. Mention completeness vs the 58 gold (VERIFIED)

Source: existentialistlogarithmic/knee-abnormality `docs/FINDINGS.md` §6 (label_ceiling_probe.py,
run on the 58 gold studies; crude case-folded substring matcher, **no negation/hedging/severity/
laterality handling** — a floor, not a forecast; table read 2026-09-30, page updated ~Sep 22, 2026).

| Target | Positives (gold) | Mentions (crude) | Sens (crude) | Bal. acc (crude) |
|---|---|---:|---:|---:|
| Baker's | 12 | 18 | 0.67 | 0.725 |
| MCL | 9 | 27 | 0.78 | 0.685 |
| Contusion | 19 | 25 | 0.63 | 0.649 |
| Lateral Meniscus | 23 | 35 | 0.78 | 0.648 |
| Synovitis | 27 | 15 | 0.41 | 0.639 |
| Medial OA | 15 | 13 | 0.40 | 0.619 |
| Fracture | 18 | 17 | 0.44 | 0.610 |
| Medial Meniscus | 26 | 34 | 0.69 | 0.596 |
| ACL | 24 | 37 | 0.75 | 0.596 |
| Lateral OA | 11 | 11 | 0.27 | 0.551 |
| Effusion | 35 | 40 | 0.66 | 0.459 |
| PF OA | 21 | 21 | 0.29 | 0.440 |

Macro balanced accuracy of the crude matcher: **0.601** (VERIFIED, n=58).

Refined analysis (same author, dedicated Synovitis writeup, ~Sep 1, 2026, VERIFIED):
with a 45-term multilingual Synovitis lexicon (more terms than Effusion's 19, ACL's 27,
Fracture's 15), only **13 of 27 true Synovitis cases are mentioned at all**, vs
Effusion 35/35, ACL 24/24, Medial Meniscus 26/26 near-complete. Synovitis total mentions
(21) fall below its positives (27) — the only finding written about less often than it is
present. Corpus-wide: Synovitis in 23.6% of reports vs Effusion 90.4%. Vocabulary gap was
checked and ruled out — "the corpus talking, not the lexicon."

Perfect-reader ceiling (VERIFIED, computed not guessed):
with 13 mentioned positives, 14 silent positives, 8 mentioned negatives, 23 silent negatives,
a perfect reader ranks mentioned correctly and cannot separate the 37 silent studies:
AUC = [13·31 + 14·8 + 0.5·14·23] / (27·31) = 676/837 = **0.8076**.
Incumbent reader scores 0.790 → headroom **+0.0176**. Route closed (E059); ~15 GPU-h not spent.

### 1b. Categorization: where label effort pays vs is wasted (INFERRED from the above + LLM benchmarks)

| Category | Targets | Why |
|---|---|---|
| **Completeness-capped — label effort wasted** | Synovitis | Ceiling 0.8076 computed; +0.0176 headroom max. Closed. |
| **Frequently mentioned, poorly extracted — label effort pays** | PF OA, Lateral OA, Medial OA, Fracture, Lateral Meniscus | Severity-thresholded or negation-scope problems. LLM closed-vocab readers reach 0.881 vs lexicon 0.769 (**+0.112 gap** — the largest single available number, per ROADMAP.md). AUC tracks abstain rate monotonically: Synovitis 72% abstain → 0.580; Fracture 62% → 0.759. |
| **Near-complete, extraction-easy** | ACL (0.97–0.99), Effusion (35/35), Medial Meniscus (26/26), Baker's, Contusion | Acute present-or-absent findings; LLM and lexicon agree. |
| **Rare-positive problem (not a mention problem)** | MCL (9/58 positives) | Crude sens 0.78 — mentioned fine — but so rare that always-negative beats raw accuracy (liamd-schneider: 81.0% vs 84.5% majority baseline; F1 0.52). Think in F1, not accuracy. |

Corroborating per-target LLM ranking (VERIFIED): tranbadat2607's CLAUDE.md (updated ~Sep 17, 2026)
— gpt-5.4-mini 0.869 mean AUC on gold, **same per-target ranking: Synovitis weakest ~0.68,
ACL strongest ~0.97–0.99**. liamd-schneider README: Qwen2.5-3B 75.9% acc / 0.64 F1, MCL the
accuracy-baseline exception.

### 1c. Hardest targets for image models even with good labels

- **Meniscus tears are the hardest classic finding** (VERIFIED, literature): MRNet (Bien et al.,
  PLOS Medicine 2018) — ACL 0.965 (95% CI 0.938–0.993), meniscus 0.847 (0.780–0.914),
  abnormality 0.937. Replicated pattern in follow-ups (Azcona et al.: ACL 0.9557, meniscus 0.9081).
- **Severity-graded findings resist both readers and models** (VERIFIED, existentialistlogarithmic
  proxy-OOF analysis): weakest four on report-label OOF are Synovitis 0.7575, PF OA 0.8288,
  Lateral OA 0.8685, Lateral Meniscus 0.8807 — "every one of them only exists after a severity
  threshold is applied." Quote: "a report parse and an image-derived severity-thresholded truth
  agree on whether there is an ACL tear and diverge on whether OA is moderate enough to count."
- **Secondary/rare findings degrade most under domain shift** (VERIFIED, Lancet eClinicalMedicine
  2026, PIIS2589-5370(25)00467-5, 14,962 scans, 5 centres): internal test primary 0.898 /
  secondary 0.815 → external test I primary 0.852 / secondary **0.744**. MCL accuracy comparable
  to senior radiologists (87.1% vs 85.7%) but the secondary group collapses cross-site.
  Architecture note: attention-guided coarse-to-fine + dedicated meniscus localization module.
- **MRNet cross-hospital drop** (VERIFIED): ACL 0.96 → 0.824 with no retraining (Croatia,
  Siemens/T1 vs Stanford/T2); recovered to 0.911 only after retraining. Direct evidence that
  single-site numbers overstate multi-site (16-site competition) performance.

### 1d. Which targets drive the macro-AUC spread between teams (VERIFIED + INFERRED)

- Macro-AUC is the mean of 12 per-finding AUCs → **+0.017 macro = +0.204 summed**
  (existentialistlogarithmic, report-label proxy analysis, ~Sep 10, 2026, VERIFIED).
- "Synovitis to 0.90 is +0.14 and PF OA to 0.93 is +0.10, so **two graded findings can
  carry the entire target**." (VERIFIED quote)
- INFERRED: the acute present/absent findings (ACL, Effusion, Fracture, Contusion, Baker's)
  are near saturation for every serious team — they contribute level, not spread. Team
  separation lives in the graded/severity-thresholded findings: Synovitis, PF OA, Lateral OA,
  Lateral Meniscus. These four are simultaneously the weakest-extracted, the hardest-imaged,
  and the highest-leverage — label + model effort should concentrate here, not on ACL.

---

## Topic 2 — Recent developments sweep (since ~Sep 16, 2026)

### 2a. Leaderboard / field shape (VERIFIED, dated)

- Top **0.957** (~Sep 15); 0.954 (Sep 8); 0.952 (Aug 28). Top-10 inside a 0.006 band.
  (existentialistlogarithmic RESEARCH_BRIEF.md, written Sep 21, 2026)
- Field: ~3,800 teams (Sep 21). 12 teams ≥ 0.950, 48 ≥ 0.945, 141 ≥ 0.940 (Sep 8).
- **Structural read (PATH.md, Sep 8):** fork plateaus at exactly 0.936 (461 teams), 0.939
  (165), 0.937 (129). "The best fork lands at 0.939. 141 teams are above it on unpublished
  work." — "Copying every public asset reaches 0.939. The remaining 0.011 is unpublished
  by construction."
- TianK003 (CLAUDE.md, read Sep 22): at **0.942**; their own blend alone 0.913; best single
  model 0.877 on LB. Confirms the public-notebook author warning: the top shared ensemble is
  "likely overfit to the public leaderboard" — shakeup expected.
- msmile-shiny/cvproject_rsna-knee-detection (updated ~Sep 25): active 2.5D pipeline
  (DINOv2 v5 + RadImageNet diversity member, percentile-rank fusion), 0.920 super-ensemble
  package verified submission-ready. Notable as independent recent work, not a fork.
- tranbadat2607 (updated ~Sep 17): CoAtNet MIL program **stopped — Kaggle GPU quota ran out**;
  E8 LB score pending; "nothing above 0.94 demonstrated yet." Residual-gated CoAtNet +0.007
  on gold-58; public v5-reverse arm is a no-op; 0.939→0.941 gains are LB-probed per-target
  weights. Five public label tables copy gold-58 verbatim (barun2104, rayanbabur, tasmeemreza,
  zaidaliiq1000, yunusgmsoy) — best clean teacher: flight0234 hybrid (0.899).

### 2b. New public notebooks since Sep 16

- **No fundamentally new architecture found** in this sweep. The field is consolidating
  around the shared DINOv2/CoAtNet/RadImageNet ensemble, not inventing. (INFERRED from
  absence across searches; not proof of absence.)
- arjun-bhattarai/rsna-knee-abnormality-detection (updated ~Sep 28): beginner-grade repo,
  initial submission 0.487 (random-init backbone — internet-off gotcha). Notable only as a
  cautionary example, not an approach.

### 2c. New pretrained weight / label datasets

- **No new weight datasets found published since Sep 16** in this sweep (INFERRED from
  absence; the known set remains: dreaddevelopment Raptor CoAtNet checkpoints v5/v10/v8,
  pilkwang DINOv2 checkpoints — measured dead — stevenleehans/rsna-knee-llm-report-labels,
  flight0234 hybrid labels).
- existentialistlogarithmic closed "borrowing public weights" as STALE on Sep 8 (E088):
  public systems moved 0.917 → 0.936–0.939 while the closure sat — "a closure resting on a
  moving external number needs a date and a re-read."

### 2d. Forum / organizer developments

- **No organizer announcements, rule changes, or data fixes found since Sep 16** via public
  web search (Kaggle forum pages are JS-rendered; not directly readable here). **Gap flagged
  for the parent agent:** worth a live-forum check for data-version changes (mount paths
  already changed once mid-competition) and any deadline/rule updates.
- Efficiency LB: `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` notebook is the
  published efficiency leaderboard; its leader is also top-5 on accuracy — "efficiency is
  not being bought with score" (TianK003, Sep 22).

### 2e. New papers (2026) relevant to architecture choices

1. **Lancet eClinicalMedicine 2026** (PIIS2589-5370(25)00467-5; crawled Sep 30, 2026):
   multi-task DLS for **9 knee abnormalities**, 14,962 scans, 5 centres, attention-guided
   **coarse-to-fine** with dedicated meniscus localization. Primary 0.898 → secondary 0.815
   internal; secondary collapses to 0.744 external. **Architectural relevance: HIGH** —
   coarse-to-fine attention + abnormality-specific localization is the closest published
   analogue to per-finding pooling for our 12 targets.
2. **Cureus 2026, Velitsikakis et al.** (review): per-pathology AI performance table —
   ACL AUC up to 0.965, meniscus AUC ≈ 0.90, OA X-ray accuracy 95.7–98.9%. Confirms the
   ACL-easy/meniscus-hard ordering.
3. **BMC Medical Imaging, Sep 2026**: Swin-Unet knee cartilage segmentation (OAMRI dataset,
   47 patients). Segmentation-focused; low direct relevance to our classification task.

### 2f. Closed routes worth knowing (so we don't re-run them)

From existentialistlogarithmic PATH.md (Sep 8–22, all measured, VERIFIED as their results):
TTA null (−0.0006); self-distillation closed (offline rig was leaking, scored −0.013);
Synovitis reader closed (0.8076 ceiling); DINOv2-as-second-family −0.035/−0.148 on two folds
(setup-specific); more epochs converged (peak 18–21); 288px geometry failed on board;
more seeds closed; fold+full-fit mixing purely linear; per-finding blend weights never
separated; Mattia Angeli's cnx-m448 excluded on license (`other`).

---

## Bottom line for the parent agent

1. **Label effort priority (all 12 targets):** Synovitis = closed (0.8076 ceiling, +0.0176 headroom).
   Spend label effort on PF OA, Lateral OA, Medial OA, Fracture, Lateral Meniscus — the
   severity/negation-gap targets where LLM readers beat lexicons by +0.112. ACL/Effusion/
   Medial Meniscus/Baker's/Contusion are near-complete already. MCL is a rarity problem, not
   a mention problem.
2. **Model effort priority:** the graded four (Synovitis, PF OA, Lateral OA, Lateral Meniscus)
   carry the macro spread (+0.14 + +0.10 of the +0.204 available). Meniscus tears are the
   hardest classic imaging target (0.847 vs 0.965 ACL, literature). Consider per-finding
   pooling rules (focal=max-like, diffuse=mean-like) and the Lancet coarse-to-fine pattern.
3. **Field update:** top 0.957 (Sep 15), fork ceiling 0.939, 141 teams on unpublished work.
   No new architecture or weight dataset since Sep 16 found. The 0.011 above the fork line
   is unpublished by construction — consistent with our custom-pipeline thesis.
4. **Gaps:** live Kaggle forum check still needed (organizer announcements, data-version
   changes); live leaderboard top-10 unverified.
