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
