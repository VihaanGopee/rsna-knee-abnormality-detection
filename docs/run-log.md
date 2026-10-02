# Run log

Every GPU/CPU run goes here: what was run, the version ID, key output, and the decision.

## 2026-10-01 — Phase 0 v1 (Kaggle, CPU-only)

- Notebook: `notebooks/phase0-preprocess.ipynb`, `CONFIRM_VERSION = "phase0-preprocess-v1"`, cache `_u8_v2`
- Audit (50 studies / 273 series): IPP/IOP 100%, header failures 0. **ImageLaterality
  0.00%** — R→L canonicalization skips everything (never guessed).
  **Anatomical_Plane 0.00%** — not a DICOM tag; plane derived geometrically from IOP.
  MONOCHROME2 100% (no inversion needed). RescaleSlope/Intercept 52.38% (defaults
  1.0/0.0 elsewhere). pylibjpeg MISSING but all DICOMs decoded fine (uncompressed).
- Smoke: 5/5 studies, PNGs written, visually pending Justin's check.
- Probe: 17/20 ok, **3 failed** with "no fluid-sensitive series in any plane"
  (de-identified `dummyseriesdesc` + missing TE/TR). Per-study 2.02s →
  **projected full build 2.47h** (under the 6h gate).
- **Gate fired correctly**: `SystemExit: FATAL: 3/20 probe studies failed.` —
  the full build did NOT start. No compute wasted.
- Decision: fix series selection (FS-preferred + best-available fallback), bump to v2.

## 2026-10-01 — Phase 0 v2 (Kaggle, CPU-only)

- Notebook: `notebooks/phase0-preprocess.ipynb`, `CONFIRM_VERSION = "phase0-preprocess-v2"`, cache `_u8_v3`
- Changes vs v1: `select_series` falls back to best-available series per plane when no
  FS series is detectable (deterministic: most slices, then UID); per-plane
  `selection` (`fs`/`fallback`) recorded in manifest; verification cell reports
  fallback count. Laterality policy set to `none_tag_absent` (tag absent 100%).
- Expected: probe 20/20, projected build ~2.5h, then full build + verification.
- Justin to confirm smoke PNGs look anatomically correct (knee centered, 3 planes).

## 2026-10-01 — Phase 0 v2 run ABORTED: /kaggle/working 20 GB cap

- v2 build was progressing cleanly (350/4407, 0 failures, ETA 1.9h) when the
  assistant caught the storage math: shards run ~1.07–1.27 GB per 100 studies →
  full cache ~52 GB, but `/kaggle/working` persists only 20 GB per notebook output
  (verified from multiple recent sources). The run would have died around study
  ~1,700. Justin told to kill it immediately — no partial output is salvageable.
- Fix: v3 (`phase0-preprocess-v3`, cache `_u8_v4`) splits the build into 3 ranges
  of 1,469 studies (Run A: 0–1469, Run B: 1469–2938, Run C: 2938–4407), each its
  own `/kaggle/working` output → its own Kaggle Dataset (~17 GB uncompressed,
  ~8 GB deflate-compressed each). Shards use `np.savez_compressed`; a disk guard
  stops before a shard write under 3 GB free; the probe projects uncompressed
  output and hard-stops if >19 GB. Decoded bytes are identical across ranges —
  training attaches all three datasets.
- Justin can run 2 of the 3 ranges concurrently (independent OUT_DIRs, no shared
  state). Each range ~40 min. After all three: create the 3 datasets from the
  notebook outputs (names printed by the notebook's dataset-metadata cell).

## 2026-10-01 — Phase 0 v3 run B gated (gate was too conservative)

- Run B ([1469, 2938)) passed audit/smoke/probe (20/20, 0 failures, 0.66h
  projected) but the v3 size gate tripped: projected 21.7 GB *uncompressed* > 19.
  The gate measured the wrong thing — shards are deflate-compressed on disk, so
  real output is roughly half the uncompressed projection.
- Fix: v4 (`phase0-preprocess-v4`, cache `_u8_v4` unchanged — decoded bytes
  identical) compresses each probe study's volumes with `np.savez_compressed`
  into a BytesIO and projects *measured compressed* bytes. The 19 GB gate now
  decides on what actually lands on disk.

## 2026-10-01 — Phase 0 v3 run A crashed on first shard write (my bug, my miss)

- Run A ([0, 1469)) passed audit/smoke/probe (20/20, 0.59h projected, 17.7 GB
  uncompressed — under the old gate) and built 50 studies, then died in
  `flush_shard()`: it called `(OUT_DIR/name).stat()` BEFORE `os.replace(tmp, ...)`,
  so the file didn't exist at the final path yet -> FileNotFoundError.
- I had identified this exact defect from the patch transcript before v3 shipped
  and failed to fix it. Owned.
- Fix: v5 (`phase0-preprocess-v5`, cache `_u8_v4` unchanged — shard contents
  identical, only write ordering changed) renames first, then stats. Also cleans
  stale `*.tmp` files at startup.
- Validation this time was real: extracted the actual `flush_shard` from the
  notebook and executed it in a temp dir — first-ever shard write, reload, 9/9
  sha256 checksums match, manifest updated, size accounting correct, empty-flush
  no-op, tmp cleanup. (The sandbox run also proved the 3 GB disk guard fires
  correctly when disk is actually low.)
- Nothing was lost: run A died before writing any shard, so v5 restarts clean.

## 2026-10-01 — Phase 0 v5 run B COMPLETE (1469 studies, 0 failures)

- Range [1469, 2938): audit 285 series (0 header failures), smoke 5/5, probe
  20/20 (0.60h projected; 16.5 GB compressed projected vs 21.7 uncompressed).
- Build: 15 shards, 12.02 GB total, wall 0.50h, failures=0.
- Verify: (a) determinism 15/15 PASS, (b) checksum integrity 4407/4407 arrays
  PASS, (c) 144,350 slices; planes sagittal/coronal/axial x 1469 each.
- 1128/4407 plane-images (25.6%) are fallback (non-FS) selections — recorded in
  manifest for Phase 2.
- Actual 12.02 GB < probe's 16.5 GB projection: 100-study shards compress
  better than per-study probe archives (conservative gate, as designed).

## 2026-10-01 — Phase 0 COMPLETE: all 3 ranges built (4407 studies, 0 failures)

- Run A [0,1469) v5: 1469 studies, 0 failures, 12.30 GB, determinism PASS,
  checksums 4406/4406 PASS. One study missing sagittal plane (1468/1469/1469).
  1067 fallback plane-images.
- Run B [1469,2938) v5: 1469 studies, 0 failures, 12.02 GB, determinism 15/15,
  checksums 4407/4407 PASS. All planes complete. 1128 fallbacks.
- Run C [2938,4407) v5: 1469 studies, 0 failures, 12.11 GB, determinism PASS,
  checksums 4405/4405 PASS. One study axial-only (1468/1468/1469). 1037 fallbacks.
- Total: 4407/4407 studies, 0 failed, ~36.4 GB across 3 outputs, all under the
  20 GB/run cap. Two studies have partial planes (by design — kept with usable
  planes, recorded in manifests). Phase-1 loader MUST tolerate missing planes.
- Next: 3 Kaggle Datasets (one per range output), then Phase 1 (weak labels).

## 2026-10-01 — Phase 1 label-extraction notebook built (phase1-labels-v1)

- Design: closed-vocabulary LLM descriptor extraction (12 targets) + deterministic
  isotonic calibration on the 58 gold studies. No public label tables (contamination).
- Engine: vLLM + Qwen3-14B-AWQ primary; transformers + bnb-4bit Qwen3-14B fallback.
  Same 14B weights either way; thinking mode disabled; latin-1 CSV read; reports
  deduped by text hash (~4,273 unique); one repair retry; failures -> not_mentioned.
- `src/phase1/label_core.py`: 37 local unit tests green (prompt, JSON parse incl.
  think-tag stripping, aliases, PAVA isotonic, AUC vs brute force).
- Full e2e with mock engine on 200-study fake CSV: dedupe mapping, retry path,
  resume, CSV/manifest/calibration writes all verified. E2e caught 2 real bugs:
  gold_df snapshot taken before _hash column existed (KeyError); retry leaving
  None records on double parse failure.
- Gates (probe): macro AUC >= 0.86 vs gold, parse-fail < 15%, projected full < 8h.
- Next: Justin runs MODE="probe" on Kaggle GPU. Only after PASS: MODE="full",
  then a Kaggle Dataset `rsna-knee-phase1-weak-labels-v1`.

## 2026-10-01 — Phase 1 v6 probe FAILED (quantization not applied); v7 built
- v6 probe: 14B 4-bit OOM'd at 14.3GB (should be ~7GB). The OOM fallback to 8B
  ALSO OOM'd. Root cause: transformers 5.0.0's new core_model_loading.py
  pipeline does NOT apply BitsAndBytes 4-bit quantization during load — both
  models materialized in full precision. Not a device_map problem.
- v7 (`phase1-labels-v7`, commit 9d1579bfda06): DROPS quantization entirely.
  Qwen3-8B in FP16 (~16GB) with device_map="auto" + max_memory 13GiB/GPU.
  No vLLM, no bitsandbytes. Simple and guaranteed to fit on 2xT4 (29GB).
- Quality risk: 8B vs 14B. If probe AUC < 0.86, 14B FP16 (28GB, tight) or
  other options will be evaluated. 8B is the working baseline first.

## 2026-10-01 — Phase 1 v7 probe ran (model works, all gates failed); v8 built
- v7 probe (Qwen3-8B FP16): model loaded (16.4GB), 100 reports in 1050s.
  Results: macro AUC 0.8011 (gate 0.86), parse fail 17% (gate <15%),
  10.5s/report -> 12.47h projected (gate <8h). ALL THREE GATES FAILED.
- Per-label AUC: ACL 0.937, MM 0.913, Baker's 0.895 (good); Synovitis 0.625,
  Fracture 0.674, Contusion 0.733, Effusion 0.755, PF OA 0.759 (weak).
- v8 (`phase1-labels-v8`, commit c494d3066fd0): batch 4->8, tokens 3000->2500,
  stronger JSON-only instruction, multilingual cues + clearer definitions for
  the 5 weakest labels. Targets all three gates.

## 2026-10-01 — Phase 1 v8 probe: prompt tweaks changed nothing; v9 tries 14B-AWQ
- v8 probe: AUC 0.7874 (v7: 0.8011 — WORSE), parse 17% (unchanged), 10.46s/report
  (batch 8 gave ZERO speedup over batch 4). Prompt engineering is exhausted;
  8B has hit its capability ceiling. The 0.86 gate needs a bigger model.
- v9 (`phase1-labels-v9`, commit df5cdd089ad2): tries Qwen/Qwen3-14B-AWQ first
  (pre-quantized 4-bit ~7GB, uses AWQ loader not the broken bnb path). Falls back
  to 8B FP16 automatically if AWQ fails. Log will show which model ran.

## 2026-10-02 — Research complete; v10 built (one-and-done plan)
Four parallel research tracks completed overnight:

1. **35B JSON failure: SOLVED.** Root cause: Qwen3.5-35B-A3B routes its entire
   response to the `reasoning_content` API field when thinking is enabled,
   leaving `message.content` empty. The script read only `content` -> 100%
   parse fail. Fix: use Ollama `/api/chat` with `"think": false`. Not a model
   capability issue. (But Mac is still too slow for full run: 10-12h even fixed.)

2. **Platform strategy:** #1 pick is `unsloth/Qwen3-14B-unsloth-bnb-4bit` on
   Kaggle (pre-quantized 4-bit, bypasses the transformers v5 bnb bug, ~7GB,
   ~4h for full run, $0). #2 is GPT-4o-mini API (~$2, <1h, 0.86-0.90 AUC).
   The v5 bnb bug is a confirmed upstream regression (issue #43032).

3. **Pipeline failure modes (P0):**
   - The 0.86 AUC gate is statistically meaningless with n=58 (95% CI +-0.04;
     0.86 is 3.6 sigma from measured 0.79). Replaced with 0.80 sanity threshold.
   - `extract_json` took FIRST JSON not last (draft-then-correct -> uses draft).
   - Non-English output terms silently became `not_mentioned`.
   - Braces in prose before JSON killed the whole report.
   - `not_mentioned` prior from gold may be inflated for full population.
   All parser bugs fixed in v10.

4. **Phase 2 readiness:** Do NOT train on 0.79 labels (teacher bounds student).
   CoAtNet > DINOv2 (measured 0.91-0.92 vs 0.84). No horizontal flips
   (laterality). Scanner-grouped folds needed (random K-fold inflates +0.087).
   8x gold weight, abstain masking, ASL loss. One model first, then ensemble.

v10 (`phase1-labels-v10`, commit ef0367c602da):
- unsloth/Qwen3-14B-unsloth-bnb-4bit (pre-quantized, ~7GB, one T4)
- All 3 P0 parser bugs fixed
- Multilingual term aliases added
- AUC gate: 0.80 sanity threshold (honest about n=58 limits)
- MAX_NEW_TOKENS 200 (was 320), BATCH_SIZE 4 (conservative for 7GB model)

## 2026-10-02 — v10 probe failed (missing bnb install); v11 built
- v10 probe: `unsloth/Qwen3-14B-unsloth-bnb-4bit` failed at load with
  `ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes`.
  The pre-quantized weights still need bnb installed for the 4-bit linear ops
  (not for runtime quantization — that path is bypassed). My mistake: v10
  removed the bnb install entirely.
- v11 (`phase1-labels-v11`, commit 59843462450d): adds `pip install bitsandbytes`
  back. Everything else identical to v10.

## 2026-10-02 — v12 probe OOM'd (single-GPU pin); v13 built
- v12 probe: 14B loaded fine (10.9GB) but OOM'd during extraction:
  `device_map={"": 0}` pinned everything to GPU 0, ignoring the second T4.
- v13 (`phase1-labels-v13`): `device_map="auto"` so accelerate splits the
  model across both T4s, leaving room for KV cache + activations.
