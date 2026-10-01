# Leaderboard Verification Brief — RSNA Knee Abnormality Detection

**Compiled:** 2026-09-30 (UTC). Every number below carries its measurement date.
Kaggle's leaderboard/discussion pages are JS-rendered and unfetchable via text tools;
all LB figures come from dated third-party snapshots (participant repos that read the
Kaggle API). Nothing here is invented.

## 1. Current participation (Kaggle overview page, search-indexed <1 hour ago)

| Metric | Value |
|---|---|
| Entrants | 23,136 |
| Participants | 5,291 |
| Teams | 4,682 |
| Submissions | 72,347 |
| Prizes | $77,000 (confirmed) |
| Time left | 22 days (closes Oct 22, 2026) |

Team-count trajectory: 1,866 (08-18) → 2,559 (08-28) → 2,957 (09-03) → 3,263 (09-07)
→ 3,723 (09-14) → 3,769 (09-15) → ~3,800 (09-21) → **4,682 (09-30)**.
~900 teams joined in the last 9 days.

## 2. Top of leaderboard over time (all dated)

| Date | Top score | Source |
|---|---|---|
| 2026-08-18 | 0.951 | existentialistlogarithmic FINDINGS.md §8 (API read, top 200 of 1,866 teams) |
| 2026-08-28 | 0.952 (ranks 2–9: 0.946–0.949) | TianK003 CLAUDE.md |
| 2026-09-03 | 0.952 (ranks 2–10: 0.947–0.950, API-verified) | TianK003 CLAUDE.md |
| 2026-09-07 | 0.954 | existentialistlogarithmic STATUS.md |
| 2026-09-15 | 0.957 | existentialistlogarithmic PATH.md |
| 2026-09-28 | **0.960** (3rd: Scott Willis 0.958; 20th: 0.955) | TianK003 research.md §2.7.2 (mined discussion thread 735304) |

Movement is steady (~0.001–0.003/week), no sudden jumps. The top-10 band has been
~0.006 wide throughout. **The bar moved: top is 0.960 as of 09-28, not 0.957.**

## 3. Top-15 team names — PARTIAL (could not fully verify)

Verified names at the top (2026-09-28 snapshot):
- **Top: 0.960** (team name not captured in sources)
- **3rd: Scott Willis — 0.958.** Small ResNet/EfficientNet @224, own Gemma-4 labels
  (~0.89 gold agreement), single fold 0.949, ~5-minute scoring — **efficiency leaderboard #1**.
- **20th: 0.955** (team name not captured)

Other named competitors (single-model scores self-reported in thread 735304, not LB ranks):
- CoolinLai: five-fold ResNet @224 — 0.954
- Archit Konde: single-fold 2.5D CoAtNet @224 — 0.950 (OOF gold 0.930)
- NguyenThanhNhan: Qwen 3.5 2B (LoRA + vision encoder) @384 — 0.950 single fold
- Raymond Yuen: full-train CoAtNet @288 — 0.949 ("still label"; multi-source teachers → pseudo-labels; API cost < $5)
- Prateek Grover: CoAtNet-2 @352 — 0.940 single / 0.942 4 folds / 0.945 in public stack
- Tucker Arrants: ResNet/EffNet @224 5-fold — 0.943 ("student consistently outperforms teacher")
- tennogh: 288px — 0.942 ("OOF pseudo-labels pretty well correlated with LB")

**Gap:** a full top-15 table with team names + submission dates was not retrievable
without live leaderboard access. The parent agent should delegate a live-browser read
if exact names are needed.

## 4. Shakeup signals

- **Fork fingerprints (verified ~09-07):** 492 teams at exactly 0.936, 163 at exactly
  0.937 — forks of two public notebooks. 655 teams ≥0.010 ahead of an independent
  0.926 by clicking Copy & Edit.
- **Author warning:** the top public notebooks' own author warns the ensemble is
  "likely overfit to the public leaderboard" after a fork-and-republish race chasing
  0.001–0.003 movements (TianK003 CLAUDE.md).
- **Public LB = ~30% of test data** (stated on the leaderboard page itself, verified
  2026-09-30); final standings use the other 70%.
- **Density cliff:** 837 teams ≥0.938, 706 ≥0.940, 537 ≥0.941, but only 81 ≥0.945
  (2026-09-15). Cliff at ~0.943–0.945.
- **LB-probed weights documented:** the 0.939→0.941 gains came from per-target blend
  weights tuned against the public LB — the classic shakeup-vulnerable pattern.

## 5. Discussion forum (last 2 weeks) — via mined thread, not live access

The Kaggle discussion page is JS-rendered (fetch failed). The substantive recent
discussion is **thread 735304, "Best single-model score"** — 105 comments,
2026-08-14 → 09-28, mined 2026-09-28 by TianK003's 8-agent workflow. Key content:

**Converged recipe (Archit Konde, 09-17..21):**
1. Check whether your OOF predictions beat the extracted labels on the 58 gold —
   if yes, labels are the bottleneck.
2. Mix OOF predictions INTO the labels (50/50 or heavier on model); only works with
   properly out-of-fold predictions ("in-fold ones just parrot the labels back").
3. Keep labels soft — "rounding the labels to 0/1 also hurt"; check calibration on the 58.

**Consensus negatives (do NOT spend GPU on):**
- Masking noisy / low-confidence cells (Tom, Archit)
- Replacing labels outright; in-fold pseudo-labels
- Better extraction or prompt fixes (Tucker, tennogh)
- Tuning extraction on the 58 gold reports
- Higher resolution as the lever — **nothing above 288px beats 0.940 in the thread**
- Architecture sweeps (7 ablations within 0.008 at 0.924)

**Notable individual points:**
- Scott Willis: "spent a lot of time trying to figure out how to work around the low
  quality labels" (method never stated); efficiency #1 at ~5 min/scoring run.
- Raymond Yuen: multi-source teachers → pseudo-labels, "different teachers seem to
  help different targets", total API cost < $5.
- Tucker Arrants: "student consistently outperforms teacher"; "correct them
  [labels], no change in LB".
- tennogh: "OOF pseudo-labels have been pretty well correlated with LB".
- Chris Deotte: pseudo-labelled single models match ensemble CV — reporting them as
  "single" hides that.

**Tensions with our plan (flagged, not resolved):**
- (a) Thread says masking noisy/low-confidence cells is dead; our plan uses
  abstain-masking (mask when report is *silent*, not when confidence is low).
  These are different operations but adjacent — treat silence-masking as an
  experiment with a kill gate, not a certainty.
- (b) Thread says nothing above 288px beats 0.940; our briefs measured 224→336px
  at +0.017. Different contexts (single-model thread vs ensemble measurements);
  resolve with our own timing probe + ablation, not by trusting either.

## 6. Official competition updates / dataset changes

- **No official announcements, dataset version changes, or rule changes found** in
  searches conducted 2026-09-30.
- Known infra change (from earlier data-pipeline research): Kaggle mount paths
  changed mid-competition — datasets now mount nested
  (`/kaggle/input/datasets/<owner>/<name>`); kernels must search at depth ≥4.
- Efficiency Prize has a separate published leaderboard:
  `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` (Kaggle notebook, readable
  via `kaggle kernels output`). Efficiency top ≈ 0.948 vs accuracy top 0.952–0.960;
  ranking is non-monotone in accuracy. Scott Willis (0.958 accuracy, 3rd) leads it.

## 7. What could NOT be verified

1. Full top-15 table with team names and submission dates — needs live leaderboard.
2. Exact current top-10 scores (latest verified: 09-28 top 0.960).
3. Any discussion threads newer than 09-28 — forum is JS-walled.
4. Whether the 0.960 top entry is a fork or custom work.

## 8. Implications for the plan (for parent synthesis)

- The bar is 0.960, not 0.957. Our "0.95+ stretch" framing still holds, but the
  top moved +0.003 in 13 days.
- Scott Willis at 0.958 (3rd) with a *small* ResNet/EfficientNet @224 and 5-minute
  scoring is the most interesting data point: it suggests label quality + efficiency
  discipline can reach the top-3 without massive ensembles.
- The thread's converged recipe (OOF-mix into soft labels) independently validates
  our Tier 2 teacher-student plan.
- The two tension points (masking, resolution) should become explicit ablations
  with kill gates rather than assumed wins.
