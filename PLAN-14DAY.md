# RSNA Knee — 14-Day Execution Schedule

**Created:** 2026-10-09 (Fri) ~07:30 PDT
**Deadline:** Thu 2026-10-22 04:59 PM PDT (final submission)
**Team merger deadline:** Thu 2026-10-15 04:59 PM PDT (non-event — Justin is solo)
**GPU budget:** 45 hrs/week × 2 = ~90 hrs total. Zero spend. Justin runs Kaggle manually.
**Scoring latency:** ~2h15m per submission. **5 submissions/day max.**
**Current state:** v2.3 done (best EMA 0.8231 ep13, best gold 0.8487 ep9, old LB 0.89).
v2.4 (RadImageNet ResNet-50) notebook ready, needs PROBE first.

---

## GPU Budget Plan

| Item | GPU-hrs | Week |
|---|---|---|
| v2.4 PROBE + full 25-epoch run | ~7.5 | 1 |
| EffNet-B3 seed 2 (severity labels, 30 ep) | ~8 | 1 |
| CORAL ordinal-head probe (fine-tune, 5 ep) | ~2 | 1 |
| ConvNeXt-Tiny (severity labels, 30 ep) | ~8 | 1 |
| Label-dependency probe (PCD loss, 5 ep) | ~2 | 1 |
| **Week 1 subtotal** | **~27.5** | (buffer: ~17.5) |
| Optional: MRNet pretrain | ~3 | 2 |
| Optional: 3rd seed / extra member | ~8 | 2 |
| Optional: pseudo-labeling round | ~8 | 2 |
| Rerun/failure buffer | ~26 | 2 |
| **Week 2 subtotal (planned)** | **~0–19** | (buffer: ~26–45) |

**Rule:** Never schedule more than 40 hrs in a week. Keep ≥5 hrs emergency buffer each week.

---

## Day-by-Day Calendar

### FRI OCT 9 (Day 1) — Launch RadImageNet
**GPU:** ~7.5 | **Cumulative:** 7.5

- **AM:** Upload `phase2-rad50-v1.ipynb` to Kaggle (delete stale copies first).
  Attach: `phase0runA/B/C` + `severity-merged-labels-v1` + competition data + **`ipythonx/notop-wg-radimagenet`**.
  Internet ON. Set `PROBE=True`. Run. Confirm `PROBE: ALL GREEN` + VERSION line
  `phase2-rad50-v1 (2026-10-09: RadImageNet ResNet-50 backbone, 2048-dim, [-1,1] norm; severity-merged labels v1)`.
- **Midday:** If PROBE green → `PROBE=False`, launch full 25-epoch run (~6.5 hrs).
- **PM (parallel, CPU):** Run `scanner-audit.ipynb` (needs val predictions; build inference snippet if missing).
- **Evening:** Check v2.4 trajectory (~ep12–15). Loss falling? Val climbing?
- **Justin actions:** 2 notebook launches (probe + full). ~15 min hands-on each.
- **Failure branch:** If PROBE fails on weight loading → try Google Drive direct download in-notebook (fallback documented in `radimagenet_research.md` §9). If still blocked by EOD → pivot to seed-2-first plan on Sat.

### SAT OCT 10 (Day 2) — v2.4 Verdict → Launch Seed 2
**GPU:** ~8 | **Cumulative:** ~15.5

- **AM:** v2.4 finished overnight (~2 AM). Record: best val macro, best EMA, best gold, best epoch.
- **★ DECISION POINT 1 — RadImageNet go/no-go:**
  - **GO** if best EMA ≥ 0.84 (beats v2.3's 0.8231) → RadImageNet path confirmed.
  - **MARGINAL** if 0.82–0.84 → still useful as ensemble member (diversity), proceed.
  - **NO-GO** if < 0.82 or crash → abandon RadImageNet; ensemble = ImageNet members only.
- **AM:** Launch **EffNet-B3 seed 2** (same v2.3 recipe, different seed, severity labels, 30 epochs, ~8 hrs).
  Either path needs this — it's the highest-conviction ensemble member.
- **PM (parallel, CPU):** Build TTA into submission notebook (8-view: original + h-flip + brightness± + contrast×2 + window×2). Behind a flag. Do NOT trust blindly — validate on gold later.
- **Justin actions:** 1 training launch + TTA notebook editing.
- **Evening:** Seed 2 at ~ep12. Sanity-check trajectory vs v2.3's.

### SUN OCT 11 (Day 3) — CORAL Probe, then ConvNeXt
**GPU:** ~10 | **Cumulative:** ~25.5

- **AM:** Seed 2 done overnight. Quick check (should roughly match v2.3: EMA ~0.82).
- **AM:** Launch **CORAL ordinal-head probe** — fine-tune from v2.3 `effnet_best.pt`, 5 epochs, ~2 hrs.
  (Notebook to be built: adds 3-threshold CORAL heads on 4 severity findings, λ=0.4.)
  - If gold improves vs v2.3 baseline → schedule full CORAL run in Week 2.
  - If flat/worse → drop it, no further spend.
- **Midday:** Launch **ConvNeXt-Tiny** (severity labels, 288px, own optimizer AdamW 1e-4, 30 epochs, ~8 hrs).
- **Justin actions:** 2 launches (probe + full). Monitor both.
- **Note:** Kaggle free tier = 1 GPU session at a time; these run sequentially (2 + 8 = 10 hrs, fits in the day).

### MON OCT 12 (Day 4) — Label-Dependency Probe + Ensemble Prep
**GPU:** ~2 | **Cumulative:** ~27.5

- **AM:** ConvNeXt done overnight. Check solo val (must be within ~0.01 of B3 or it's dropped from blend).
- **AM:** Launch **label-dependency probe** — PCD auxiliary loss fine-tune from v2.3, 5 epochs, ~2 hrs.
  (Notebook to be built: aligns predicted 12×12 correlation matrix with empirical label correlations.)
  - If val improves → candidate for Week 2 full run.
  - If flat → drop.
- **PM (CPU):** Ensemble blending code — per-finding rank averaging across members
  (v2.3 + seed2 + v2.4-Rad50 + ConvNeXt-T). Greedy forward selection on validation, never on gold-58.
- **Justin actions:** 1 probe launch + CPU blending work.

### TUE OCT 13 (Day 5) — First Leaderboard Submission (calibration)
**GPU:** 0 | **Cumulative:** ~27.5

- **AM:** Build submission from **best single solo** (v2.4 if GO, else v2.3/seed2 best). Run submission notebook, download CSV.
- **Submit #1** to competition (1 of 5 daily). Note time; score back in ~2h15m.
- **PM:** Score arrives.
- **★ DECISION POINT 2 — LB calibration:**
  - If LB ≥ 0.91 → on track for bronze via ensemble. Proceed as planned.
  - If LB 0.89–0.91 → val→LB translation is weak; lean harder on ensemble + TTA.
  - If LB < 0.89 → something is wrong (overfit to val?). Diagnose before spending more GPU.
- **Justin actions:** submission notebook run + Kaggle submission. ~30 min hands-on.

### WED OCT 14 (Day 6) — Fill Ensemble Gaps
**GPU:** ~0–8 | **Cumulative:** ~27.5–35.5

- Based on Decision Point 2:
  - **If on track:** light day. CPU work — TTA validation on gold-58 (keep transforms with ≥+0.003 gain, drop the rest). Prep pseudo-labeling inputs.
  - **If behind:** launch gap-filler member (~8 hrs):
    - Priority 1: whichever of (v2.4-Rad50 / ConvNeXt) is missing or undertrained.
    - Priority 2: 3rd EffNet-B3 seed (diminishing but cheap insurance).
- **Justin actions:** conditional launch.

### THU OCT 15 (Day 7) — Merger Deadline + Buffer
**GPU:** 0 (buffer) | **Cumulative:** ~27.5–35.5

- **04:59 PM PDT:** team merger deadline passes (no action — solo).
- **Buffer day.** No planned GPU. Absorb any failed/crashed runs from Days 1–6.
- **★ DECISION POINT 3 — Lock ensemble composition.** Decide the final member list:
  - Core: best B3 (v2.3 or seed2) + v2.4-Rad50 (if GO/MARGINAL) + ConvNeXt-T (if within 0.01)
  - Optional: CORAL variant (if probe won), 3rd seed.
- **End of Week 1.** Should have used ~28–36 of 45 hrs. Buffer intact.

### FRI OCT 16 (Day 8) — Week 2 GPU Renews + Ensemble LB
**GPU:** 0 | **Cumulative (W2):** 0

- **AM:** Build ensemble submission (rank-average, per-finding weights from validation).
  Include TTA **only** for transforms that validated on gold-58.
- **Submit #2** (ensemble). Score by PM.
- **★ DECISION POINT 4 — Ensemble verdict:**
  - If ensemble LB ≥ 0.93 → bronze path live. Week 2 = polish (TTA tuning, calibration).
  - If 0.91–0.93 → need one more member or pseudo-labeling.
  - If < 0.91 → diagnose; consider emergency options (see below).
- **Justin actions:** ensemble blending (CPU) + submission.

### SAT OCT 17 (Day 9) — Conditional Member
**GPU:** 0–8 | **Cumulative (W2):** 0–8

- **If Decision Point 4 says "need more":** launch best available member (~8 hrs):
  - Option A: MRNet-pretrained backbone → fine-tune on severity labels (needs MRNet weights; RUA paperwork should be done by now — start it Oct 9 in parallel).
  - Option B: 3rd EffNet-B3 seed.
  - Option C: full CORAL run (if probe won on Oct 11).
- **Else:** rest day / TTA refinement (CPU).
- **Submit #3** (improved ensemble) in PM if ready.

### SUN OCT 18 (Day 10) — TTA Validation + Calibration
**GPU:** 0 | **Cumulative (W2):** 0–8

- **CPU day.** Finalize TTA set on gold-58. Fit per-finding Platt scaling on validation
  (`platt_calibration.py` exists — runs on Kaggle, 0 GPU).
- No GPU planned. Buffer for any overrunning Week-2 training.

### MON OCT 19 (Day 11) — Pseudo-Labeling Decision
**GPU:** 0–8 | **Cumulative (W2):** 0–16

- **★ DECISION POINT 5 — Pseudo-labeling go/no-go** (needs ≥2-model ensemble teacher + calibrated probs):
  - GO if: ensemble LB ≥ 0.93, ≥16 GPU-hrs remain in Week 2, and TTA inference on test (~1 hr) succeeds.
  - Pipeline: ensemble+TTA → test probs → per-finding 0.95-precision thresholds → retrain seed (~7 hrs).
  - NO-GO if: behind schedule, low GPU, or ensemble already ≥0.945 (diminishing returns).
- **Justin actions:** conditional launch.

### TUE OCT 20 (Day 12) — Final Ensemble Submission
**GPU:** 0 | **Cumulative (W2):** ≤16

- **AM:** Assemble FINAL candidate: best ensemble + validated TTA + Platt calibration.
- **Submit #4.** Score by PM (~2h15m).
- **★ DECISION POINT 6 — Final model selection.** This score decides what ships Oct 21.
- **Justin actions:** blending + submission. ~1 hr hands-on.

### WED OCT 21 (Day 13) — FINAL SUBMISSION DAY
**GPU:** 0

- **By 2 PM PDT:** submit FINAL `submission.csv` (best of all runs).
- Score back by ~4:15 PM. Confirm it's sane (not 0.5s, not NaN, 12 columns × N rows).
- **Up to 4 more submissions allowed today** if the first looks broken — use sparingly.
- **Justin actions:** final submission + verification. The most important 30 minutes of the project.

### THU OCT 22 (Day 14) — DEADLINE (buffer only)
**GPU:** 0

- **04:59 PM PDT:** hard deadline.
- **Plan:** NOTHING scheduled. This day exists for emergencies only
  (e.g., Oct 21 submission corrupted → resubmit by 2 PM PDT for scoring buffer).
- If all went well, the work is done and Justin watches the private LB.

---

## Submission Plan (max 5/day)

| # | Date | Content | Purpose |
|---|---|---|---|
| 1 | Tue Oct 13 | Best single solo | LB calibration (val→LB translation) |
| 2 | Fri Oct 16 | 2–4 member ensemble | Ensemble gain measurement |
| 3 | Sat Oct 17 | Ensemble + validated TTA | TTA gain measurement |
| 4 | Tue Oct 20 | Final candidate ensemble | Pre-final check |
| 5 | Wed Oct 21 | **FINAL** | Ships |
| 6 | Thu Oct 22 | Emergency backup | Only if #5 broken |

Total planned: ~5–6 submissions. Well under limits.

---

## Decision Points Summary

| # | When | Question | Outcomes |
|---|---|---|---|
| 1 | Sat Oct 10 AM | Did RadImageNet beat v2.3? (EMA ≥ 0.84) | GO / MARGINAL (ensemble member) / NO-GO (abandon) |
| 2 | Tue Oct 13 PM | Does LB match val? (LB ≥ 0.91) | On track / push ensemble harder / diagnose |
| 3 | Thu Oct 15 | Lock ensemble member list | Final roster for Week 2 |
| 4 | Fri Oct 16 PM | Ensemble LB ≥ 0.93? | Polish / add member / emergency |
| 5 | Mon Oct 19 | Pseudo-labeling worth 8 hrs? | GO (if ≥0.93 + budget) / skip |
| 6 | Tue Oct 20 PM | Which model ships? | Final selection |

---

## Failure Buffers & Contingencies

- **v2.4 PROBE fails (Oct 9):** Fallback to Drive download in-notebook (~30 min fix). If still blocked by EOD → skip RadImageNet, run seed 2 first, revisit RadImageNet Oct 14.
- **Training NaN/crash:** All notebooks have kill-gates + `effnet_last.pt` resume. Lose ~1 hr max per incident. Week 1 buffer (~17 hrs) absorbs 2–3 incidents.
- **Kaggle GPU queue:** If T4×2 queued >30 min, try P100 or run overnight. All multi-hour runs are overnight-friendly.
- **RadImageNet total failure:** Fallback path = v2.3 + seed2 + ConvNeXt-T + TTA → ~0.91–0.92 LB. Below bronze target but no wasted Week 1 (members still useful).
- **LB translation poor (val 0.85 → LB 0.89):** Trust gold-58 over val for model selection; prefer members with best gold-macro.
- **Justin unavailable a day:** Schedule has 2+ slack days (Oct 14, Oct 18). Critical path needs him only on: Oct 9, 10, 11, 13, 16, 20, 21.

## Emergency Options (if LB < 0.91 by Oct 16)

1. **DINOv2 ViT-S member** — `timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)`. One-liner, no manual weights. ~8 hrs. Diversity play.
2. **5-fold CV ensemble** — train 5 folds of B3 (~35 hrs, too expensive). SKIP unless Week 2 is empty.
3. **Aggressive TTA** — 24-view like the 0.910 team (inference-only, 0 training GPU).
4. **Test-set mix check** — if scanner audit (Oct 9) showed manufacturer skew, apply per-manufacturer calibration.

---

## What Justin Must Do (hands-on checklist)

- [ ] **Oct 9 AM:** Upload v2.4, attach 5 datasets, PROBE run (~15 min)
- [ ] **Oct 9 midday:** Launch v2.4 full run (~10 min)
- [ ] **Oct 9 PM:** Run scanner audit notebook CPU (~10 min)
- [ ] **Oct 10 AM:** Read v2.4 results, launch seed 2 (~15 min)
- [ ] **Oct 11 AM:** Launch CORAL probe (~10 min), then ConvNeXt (~10 min)
- [ ] **Oct 12 AM:** Launch label-dep probe (~10 min)
- [ ] **Oct 13 AM:** Submission notebook → download CSV → Kaggle submit (~30 min)
- [ ] **Oct 16 AM:** Ensemble blend + submission (~45 min)
- [ ] **Oct 20 AM:** Final candidate submission (~30 min)
- [ ] **Oct 21 by 2 PM:** FINAL submission + verify (~30 min)

**Total hands-on:** ~4 hours spread over 14 days. Everything else is GPU running unattended.

---

## Target Trajectory

| Milestone | Date | Target LB | Basis |
|---|---|---|---|
| v2.4 solo | Oct 10 | 0.90–0.92 | RadImageNet +0.015–0.025 over 0.89 |
| + seed 2 blend | Oct 13 | 0.91–0.93 | Ensemble +0.004–0.008 |
| + ConvNeXt + TTA | Oct 16 | 0.93–0.945 | Diversity +0.004, TTA +0.007 |
| + pseudo-label (opt) | Oct 20 | 0.935–0.95 | +0.005 |
| **Final** | **Oct 21** | **0.93–0.95** | Bronze zone (top 10%) |

**Honest caveat:** 0.97 (Justin's floor) needs RadImageNet to hit optimistic (+0.035), ensemble to stack perfectly, AND private-team private-data advantages to not matter. The plan maximizes the probability; it does not guarantee it.
