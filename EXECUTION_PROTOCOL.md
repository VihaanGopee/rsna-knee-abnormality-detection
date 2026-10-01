# Execution Protocol — No Slip-Ups

**Rule zero:** No GPU run starts without passing pre-flight. No exceptions.
A 10-minute probe is cheaper than a 5-hour mistake.

## 1. Pre-flight checklist (every training run, no exceptions)

- [ ] Mount listing printed at startup; all datasets attached and found
- [ ] Cache version string matches expected (every byte-affecting knob encoded)
- [ ] Full config dumped to log — every hyperparameter visible, no hidden defaults
- [ ] Smoke test passes: 1 batch forward + backward on a tiny subset
- [ ] Timing probe done: 10-minute run → measured iters/sec → projected total runtime
- [ ] Projected runtime fits session limit with 20% margin (Colab: plan against a 12h conservative cap)
- [ ] Checkpointing configured: saves to Drive every N epochs; resume path tested
- [ ] Kill gates defined (Section 2)
- [ ] Output artifacts defined: exact files, exact paths
- [ ] **Assistant clearance: Muse has reviewed the notebook against this list**

## 2. Kill gates (stop the run — don't watch it die slowly)

| Signal | Action |
|---|---|
| Loss NaN / Inf | Kill immediately, report |
| Loss not decreasing after 15% of planned steps | Kill, diagnose — don't burn the other 85% |
| Throughput < 70% of probe measurement | Pause, investigate (throttling? dataloader bottleneck?) |
| GPU memory creeping upward (leak) | Kill before the hour-4 OOM |
| Val metric flat while train loss falls | Overfitting — kill and adjust, don't finish the run |

## 3. Notebook self-validation (first cell block, runs in <60 seconds)

1. Print mount listing + attachment state
2. Print full config (every knob)
3. Data sanity: tensor shapes, dtypes, label distribution, one decoded batch as stats
4. **Fail fast with a clear error message** — never silently continue on wrong data

## 4. Competition-specific gotcha list (verify before every run)

- [ ] `train.csv` read with **latin-1** encoding (not UTF-8 default)
- [ ] Slices sorted by **IPP·(IOP_row × IOP_col)**, never by filename
- [ ] Horizontal flip only with **Medial↔Lateral label swaps** (or flip disabled entirely)
- [ ] Kaggle GPU: `machine_shape: NvidiaTeslaT4` — **never P100** (dies at first convolution)
- [ ] Cache version string is current; no stale shards
- [ ] Train/infer pixel path **byte-verified equal**, not just error-free
- [ ] `test.csv` treated as a 3-row stub — study list discovered at runtime, never hardcoded
- [ ] Mount paths searched at **depth ≥ 4**; layout printed at startup
- [ ] `submission.csv` written **incrementally every 25 studies** + runtime projection + staged degradation
- [ ] **float64 end-to-end** in the submission path (rounding creates AUC tie-penalties)
- [ ] Folds are **scanner-grouped** (never random) for any validation claim
- [ ] Gold-58 used as **regression guard only** — never for tuning or model selection

## 5. Resume protocol (Colab disconnects happen)

- Every run longer than 1 hour checkpoints to Drive
- Resume command is **tested before** the long run starts, not after it dies
- If a run dies: resume from checkpoint, **never from zero**
- Log records exactly which epoch/step was resumed from

## 6. The clearance rule

**Justin does not start a GPU run until Muse has cleared the notebook against this protocol.**
If it isn't cleared, it doesn't run. This is the mechanism that makes "no oops" real —
the checklist isn't self-serve, it's a gate, and Muse is the gatekeeper.

## 7. Run log (every run gets an entry)

Each GPU run records: date, notebook version (git hash), config summary, probe
throughput, projected vs actual runtime, kill-gate status, final metric, and
artifact locations. The log lives in `docs/run-log.md`. No run is "just a quick test" —
untracked runs are how mistakes hide.

## 8. Version gate (Justin's design — refined)

Every notebook carries an embedded version ID. Justin confirms it with a variable
(no `input()` — the notebook must run unattended after one manual edit):

```python
# === STEP 0: VERSION CONFIRMATION (edit this, then Run All) ===
# Muse's message says the current version is: "phase2a-dinov2-v3"
# Copy it EXACTLY below. Notebook refuses to run on mismatch.
CONFIRM_VERSION = "TYPE-VERSION-HERE"
# ==============================================================

NOTEBOOK_VERSION = "phase2a-dinov2-v3"  # embedded by Muse — do not edit
NOTEBOOK_CHANGELOG = "v3: fixed lateral flip labels"  # one-line human changelog

if CONFIRM_VERSION != NOTEBOOK_VERSION:
    raise SystemExit(
        f"VERSION MISMATCH — not cleared to run.\n"
        f"Embedded: {NOTEBOOK_VERSION} / You typed: {CONFIRM_VERSION}\n"
        f"Stop. Check Muse's message for the correct version ID."
    )
print(f"Version verified: {NOTEBOOK_VERSION} — {NOTEBOOK_CHANGELOG}")
```

Rules:
- The default `"TYPE-VERSION-HERE"` never matches — a fresh upload cannot run without the edit.
- The version ID comes from **Muse's clearance message**, never from inside the notebook.
  (Reading the embedded ID out of the wrong file and typing it in defeats the gate —
  copy from the message.)
- The changelog line tells Justin *what changed*, not just the version string.
- Works on the offline scoring notebook too (no internet needed).
