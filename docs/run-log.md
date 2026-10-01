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
