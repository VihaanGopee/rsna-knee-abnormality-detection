# Data Pipeline & DICOM Preprocessing Brief
## RSNA Knee Abnormality Detection (Kaggle 2026)

**Compiled:** 2026-09-30 from public competitor repos, verified EDA logs, and community sources.
**Tagging:** `VERIFIED` = measured/read directly from a competition file, Kaggle API, or a competitor's measured log (source named). `INFERRED` = plausible but not directly measured. Treat INFERRED as hypotheses, not design inputs.

---

## 1. What the DICOM data actually is

| Fact | Detail | Source |
|---|---|---|
| Scale | ~819k DICOM files ≈ ~819k slices; mean file size **683 KiB**; extrapolated total ~0.69 TiB / ~710 GB (above the brief's 570 GB) | `VERIFIED` — existentialistlogarithmic FINDINGS §3.1 (partial listing 165k files / 107.5 GiB) |
| Hierarchy | `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`; filenames are SOP Instance UIDs | `VERIFIED` — multiple repos |
| Studies | 4,407 train studies; 24,371 series; mean 5.53, median 5, min 3, max 14 series/study | `VERIFIED` — train_series.csv |
| Planes (series) | Sagittal 9,864 / Coronal 8,609 / Axial 5,898 | `VERIFIED` — train_series.csv |
| Slices/series | Typically 20–45, **median 30**, long tail to a few hundred | `VERIFIED` — host data-description |
| Test | ~1,300 studies hidden (~215k slices); visible `test.csv` is a 3-row stub, replaced at scoring | `VERIFIED` — data-description |
| Resolutions | **Native median 0.312 mm/px** in-plane | `VERIFIED` — existentialistlogarithmic §10 (192px @ 0.60 mm/px "discarding detail", 288px @ 0.40 mm/px) |
| Intensity range | Max intensity spans **690 … 8,736** across sample series — a **12.7× range** | `VERIFIED` — TianK003 traps.md §6 |
| Photometric | Mostly `MONOCHROME2`; `MONOCHROME1` exists in the wild (invert it) | `VERIFIED`/`INFERRED` — TianK003 notes sample was all MONOCHROME2; hidden test spans 16–19 sites |
| Rescale | Apply `RescaleSlope`/`RescaleIntercept` (their sample had trivial values, but hidden test may not) | `VERIFIED` recommendation |
| De-identification | `InstitutionName`, `StationName`, `DeviceSerialNumber`, all dates, `PatientAge/Size/Weight`, `ProtocolName` are **gone** | `VERIFIED` — 7-file header audit |
| Scanner survivors | `Manufacturer`, `ManufacturerModelName`, `SoftwareVersions`, `MagneticFieldStrength`, `ImagingFrequency`, `TransmitCoilName`, `SeriesDescription`, `Laterality`, `PatientID` (pseudonymised), `PixelSpacing`, `SliceThickness`, `ImageOrientationPatient`, `EchoTime`/`RepetitionTime`/`EchoTrainLength`/`ScanningSequence`/`SequenceVariant`, `PatientSex` | `VERIFIED` — header audit |
| Transfer syntax | Not directly measured in any source I found. Competitors' offline wheelhouses include **pylibjpeg** → some files use JPEG-compressed transfer syntaxes | `INFERRED` — aakashkavuru101 AGENTS.md wheel list (monai, pydicom, pylibjpeg, SimpleITK, timm) |
| Vendors | SIEMENS, Siemens Healthineers, GE MEDICAL SYSTEMS, Philips Healthcare, TOSHIBA seen in a 7-file sample | `VERIFIED` |

### 1a. The scanner fingerprint (no site label exists)
- No site/institution column in any CSV; DICOM headers de-identified → **true site label does not exist**.
- Proxy fingerprint = `Manufacturer` + `ManufacturerModelName` + `SoftwareVersions` (rounded) + `MagneticFieldStrength` + **`ImagingFrequency` rounded to 2 decimals** + `TransmitCoilName`.
- `ImagingFrequency` (Larmor freq, MHz) separates nominally identical 1.5T magnets at the 4th decimal (63.881601 vs 63.870660 vs 63.685261 vs 63.648174 = four different magnets). **Round to 2 decimals**: raw precision gives 8,618 fingerprints/4,410 studies (5,633 singletons — grouped K-fold in disguise); rounding to 2 decimals gives **178 usable groups** (median 8 studies, mean 24.8, largest 246, 42 singletons).
- Random K-fold inflates macro AUC by **+0.087** on metadata alone (gradient-boosted model, no pixels); community reports ~0.05; one team measured +0.136. **Scanner-grouped folds are mandatory.**
- `VERIFIED` — existentialistlogarithmic §§5.2, 9 (measured on Kaggle).

### 1b. Laterality
- `Laterality` tag present in 6/7 sampled files (values `L`, `R`, one empty — the empty case is real and must be handled). `SeriesDescription` sometimes encodes it (`LT_t2_tse_fs_cor_obl_ACL`).
- Best practice: **mirror right knees to a canonical left orientation** in preprocessing (not random flips). TianK003's cache encodes this (`lat20` in version string).
- `VERIFIED` (tag presence) / `INFERRED` (exact TianK003 semantics).

---

## 2. Canonical DICOM → tensor preprocessing (what works)

```python
# The consensus recipe, assembled from measured sources
import pydicom, numpy as np

def dicom_to_array(path):
    ds = pydicom.dcmread(path)                       # needs pylibjpeg for JPEG transfer syntaxes
    x = ds.pixel_array.astype(np.float32)
    x = x * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        x = x.max() - x                              # inverted: high value = black
    return x, ds

def slice_key(ds):
    # NEVER sort by filename (SOP UID): Spearman rho vs true order = -0.012 (measured on 12 series)
    iop = ds.ImageOrientationPatient                  # two 3-vectors: row dir, col dir
    ipp = ds.ImagePositionPatient
    row = np.array(iop[:3]); col = np.array(iop[3:])
    normal = np.cross(row, col)
    return float(np.dot(np.array(ipp), normal))       # fallback: InstanceNumber, then filename
```

Step-by-step, with parameters:

1. **Decode** with pydicom (+ pylibjpeg installed). Apply `RescaleSlope`/`RescaleIntercept`. Invert `MONOCHROME1`. — `VERIFIED` recommendation (liamd-schneider reference-notes; TianK003 traps §6)
2. **Order slices** by projecting `ImagePositionPatient` onto the slice normal (`cross` of the two `ImageOrientationPatient` vectors). Fallback `InstanceNumber` → filename. **Sorting by filename is a silent killer**: measured ρ = −0.012, |ρ|>0.99 in 0/12 series; destroys the 2.5D-triplet premise (three "adjacent" channels become three random slices) while loss still falls. — `VERIFIED` (TianK003 traps §1)
3. **Intensity windowing — per series, never global.** Max intensity spans 690…8,736 (12.7×); a global window crushes some series to black and saturates others. Standard: **clip each series (or each 2.5D triplet jointly) at its own 1st–99th percentile**, then scale to [0,1]. — `VERIFIED` (TianK003 traps §6; liamd-schneider reference-notes)
4. **Physical-scale resampling.** Resample to a fixed mm/px using `PixelSpacing` (native median 0.312 mm/px). Tested operating points: 192px @ 0.60 mm/px, 288px @ 0.40 mm/px, 336px, 384px. Note the 288-vs-192 result: a clean comparison showed 192px winning on the board (0.725 vs 0.688) — resolution interacts with batch size and label noise; don't assume finer is better. — `VERIFIED` (existentialistlogarithmic §11; TianK003 c02=336px)
5. **Crop.** TianK003 uses a **130 mm anatomical crop** (knee-centered) in their cache pipeline. — `VERIFIED` (TianK003 traps §6f, cache version strings)
6. **Quantize to uint8** for the cache. Cache version string must encode **every** byte-affecting knob (px, slice count, per-plane band, normalization percentiles, crop, laterality) — a stale version string silently mixes incompatible caches. — `VERIFIED` (TianK003 traps §23)
7. **Laterality normalization**: mirror R knees to L orientation once, in the cache. — `INFERRED` from `lat20` token

### 2a. Slice sampling per series
- Median 30 slices/series; models sample a fixed count: 16–64 slices seen in the wild (liamd-schneider: center-crop depth to 64; TianK003 c02 per-plane bands `18-12-12-14-8-8`; existentialist: 3 planes × 20 slices).
- 2.5D input construction: per-slice features → pooling (mean / max / gated attention). Hyperparameter search on a small backbone picked **plain max-pooling** over gated attention (fancy didn't win). — `VERIFIED` (liamd-schneider reference-notes)
- Per-slice CLS + patch-mean, plane and laterality embeddings, mean-within-plane pooling — the "Torres" recipe, each argued from a specific failure. — reported in existentialistlogarithmic COMPETITIVE_ANALYSIS (third-party account of a top solution; treat the attribution as `INFERRED`, the technique list as real)

---

## 3. Series selection per study

**The cheapest reliable rule** (`VERIFIED` — existentialistlogarithmic §10, measured on all 24,371 series):

| plane / contrast | studies with it |
|---|---|
| **Axial, fluid-sensitive** | **4,407 — 100%** |
| Sagittal, T1w-ish | 4,266 (96.8%) |
| Coronal, fluid-sensitive | 4,248 (96.4%) |
| Sagittal, fluid-sensitive | 4,150 (94.2%) |
| Coronal, T1w-ish | 3,406 (77.3%) |
| Axial, T1w-ish | 857 (19.4%) |

- Take **one fluid-sensitive series per plane** (sagittal + coronal + axial), degrade gracefully for the 9.4% of studies missing one. **Axial fluid-sensitive is the guaranteed fallback** — it exists for every study.
- All three fluid-sensitive planes co-occur in 90.6% of studies.
- Fluid-sensitive slices per study: median 99, mean 110, p90 159, max 561 (59% of all slices). One-series-per-plane cuts this substantially.
- **Same selection function at train and inference** — training/serving drift here is silent. — `VERIFIED` (liamd-schneider reference-notes)

### 3a. Don't trust the host's contrast flags
- `Fluid_Sensitive` / `Fat_Suppression` are **degenerate in train**: only (1,1) [14,010 rows] and (0,0) [10,361] ever occur — never a mixed pair. Two physically independent properties collapsed into one bit.
- The host warns they are "often correlated, as observed in the training set, [but] **not necessarily equivalent** for every case" — the test set may diverge.
- **Recover both from headers**: `ScanningSequence`, `SeriesDescription`, `SequenceName`, `ScanOptions`, TR/TE. Catch gradient echo from `ScanningSequence` first (short TR by design breaks the TR/TE rule).
- `Anatomical_Plane` **is trustworthy**: 100% agreement with IOP-derived plane on all 24,371 series. Don't recompute it.
- `VERIFIED` — TianK003 traps §5; existentialistlogarithmic §3.10

### 3b. Label → series-type routing (private-team territory)
- Clinical prior: Sagittal best for ACL/menisci/cartilage; Coronal best for collateral ligaments (MCL) and compartment OA; Axial best for patellofemoral joint (PF OA) and Baker's cyst.
- One competitor's pipeline has an explicit `routing.py` — label → series-type routing logic. — `VERIFIED` (repo exists: limammohamedlimam/rsna-knee `src/data/routing.py`); effectiveness `INFERRED`
- Note a counter-measurement: per-plane attribution "looked like +0.082 across 5 of 12 findings; naming the plane from anatomy *before* looking scored −0.0201 against pooling" — selection bias trap; measured on frozen features. — `VERIFIED` (existentialistlogarithmic COMMENTS_FOR_MARTIJN §3)

---

## 4. Data loading at scale (570 GB in a 9h notebook)

### 4a. The budget
- Hidden test ≈ 1,300 studies ≈ 215,000 slices → **~24 s/study** end-to-end including DICOM reading.
- Measured: directory traversal + one header read/series = **0.059 s/study** (77 s for the whole hidden test, 0.2% of cap) — file *access* is free; **pixel decoding is the constraint**. — `VERIFIED` (existentialistlogarithmic §3.19)
- Achieved inference speeds: existentialistlogarithmic Phase 3 = **2.2 s/study** (0.8h of 9h); TianK003 two-member solo ≈ 28–30 min for the hidden test.

### 4b. The cache pattern (how serious teams handle it)
TianK003's `src/cache_pipeline.py`:
- **Preprocess once**: DICOM → uint8 volumes (130 mm crop, per-series 1/99 normalization, laterality) → cache shards as a **Kaggle Dataset**; training/inference kernels mount the cache and `np.load`.
- Cache line in log: `c02_p336_b18-12-12-14-8-8_band2-98_crop130_lat20` (px=336, per-plane slice bands, 2–98 percentile band, 130mm crop, laterality).
- existentalistlogarithmic: v1 192px cache (4 shards), v2 288px cache (8 shards), both verified complete — 4,407 cached studies, 58 gold present.
- TianK003 inference optimization (P-41): **16-thread header scan + 8 decode workers**, byte-identical to single-thread.
- **Critical trap**: one flag must not select *which preprocessing* AND *whether a file is present* — when the infer kernel mounted no cache, the loader silently flipped to an older decode branch (no crop, no mirroring, no per-series norm). Fix: gate preprocessing choice on `mode`, and **verify train/infer pixel equality byte-for-byte**, not by absence of errors. — `VERIFIED` (TianK003 traps §§6d, 6f)

### 4c. Mount-path landmines
- Competition data: `/kaggle/input/competitions/<slug>`; datasets: `/kaggle/input/datasets/<owner>/<name>`; kernel outputs: `/kaggle/input/notebooks/<owner>/<slug>/`. **Never hard-code `/kaggle/input/<name>`.**
- Kaggle changed layout mid-competition (2026-08-29): new slugs mount type-prefixed (deeper). **Search at depth ≥ 4 with fallback**; print the input layout at every kernel startup.
- **NEVER select the P100 accelerator**: Kaggle's PyTorch ships no Pascal CUDA kernels → dies at first convolution (`torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`). Set `"machine_shape": "NvidiaTeslaT4"`. A wrong accelerator name (e.g. `--accelerator GpuT4x2`) is **silently ignored** and falls back to P100. — `VERIFIED` the expensive way (existentialistlogarithmic §12; TianK003 hard constraint 1)
- `VERIFIED` — existentialistlogarithmic §§3.18, 12; TianK003 traps §6f

### 4d. Write submission.csv incrementally
- Write a valid `submission.csv` every 25 studies; project total runtime every 25 studies and degrade in stages (fewer eval slices 32→16, then fewer ensemble folds) if projected over budget. Failed-to-decode studies get neutral 0.5, never a crash. — `VERIFIED` (liamd-schneider reference-notes)

---

## 5. Augmentation (MRI-specific rules)

### 5a. The laterality/flip rule — the most important one
5 of 12 labels are laterality-sensitive (Medial vs Lateral Meniscus, Medial vs Lateral OA; MCL is medial-only anatomy):
- **Option A (conservative):** never horizontal-flip. Rotation/gamma/scale jitter shared across a whole series. — liamd-schneider reference-notes
- **Option B (laterality-aware flip):** horizontal flip IS used, but **Medial Meniscus ↔ Lateral Meniscus and Medial OA ↔ Lateral OA are swapped on flip**, in both augmentation and TTA. Implemented in limammohamedlimam `src/data/augment.py` ("incl. laterality-aware flip"); mandated by aakashkavuru101 AGENTS.md ("Must swap … on flip, in both augmentation and TTA").
- Anatomy subtlety: for **sagittal** slices (side view) a horizontal flip swaps anterior↔posterior, not medial↔lateral; for **coronal/axial** it swaps medial↔lateral. A plane-aware implementation flips only where the label swap is valid.
- The cleanest design (TianK003): **normalize laterality once in the cache** (mirror all R knees to L), then flips are unnecessary.

### 5b. What else is used
- Rotation, gamma, scale jitter — **shared across a whole series** (same transform for all slices; per-slice jitter breaks 2.5D consistency). — liamd-schneider
- "Heavy aug" + drop-path 0.1 in TianK003's P-60 student recipe.
- **TTA is measured dead**: 10 of 12 TTA variants correlate >0.98 with their parent — the attention-pooling model (~60 windows) is nearly invariant to these perturbations. — `VERIFIED` (existentialistlogarithmic COMMENTS_FOR_MARTIJN §3)
- **No mixup/cutmix reported** by any source (label semantics don't permit it cleanly).
- Corrupt-file handling: any study that fails to decode → neutral 0.5, log it. — liamd-schneider

---

## 6. Data gotchas (CSVs and labels)

1. **Multiline reports**: `train.csv` is **58,556 physical lines for 4,407 rows** — reports contain embedded newlines. Must use a real CSV parser (pandas handles it by default). — `VERIFIED` (multiple sources)
2. **Empty label cells are NOT zeros**: 4,349 studies have blank label columns → treat as missing, **mask in loss**. — `VERIFIED` (aakashkavuru101 AGENTS.md)
3. **Exactly 58 fully-labeled studies** — no partially-labeled middle ground. — `VERIFIED`
4. **Shared report texts**: 4,273 distinct reports; **49 texts shared by >1 study, covering 183 studies (largest group 37)**. Studies sharing a report share a target vector → folds must group on report text too, or the model memorizes answers across the split. — `VERIFIED` (TianK003 traps §2)
5. **PatientID pseudonyms may span studies** — if one patient has multiple studies, folds must group on patient too. Status `UNVERIFIED` (flagged for checking in existentialistlogarithmic §5.3).
6. **Degenerate contrast flags** (§3a) — don't collapse `Fluid_Sensitive`/`Fat_Suppression` into one bit; keep both, recover from headers.
7. **Five public label tables copy the gold labels verbatim** — must not be used for gold-58 validation (leak). — `VERIFIED` (tranbadat2607 README)
8. **Submission precision**: don't round probabilities (e.g. `np.round(p, 4)`) — ties cost AUC. Keep full float precision. — reported (berattcelikk log; standard AUC math)
9. **Kernels-only submission**: `is_kernels_submissions_only=True` — `kaggle competitions submit` CLI returns HTTP 400 by design; submissions only via notebook output. — `VERIFIED` (existentialistlogarithmic §2.16)

---

## 7. What private teams likely do differently (inference from evidence)

The public top (~0.957, Sep 2026) is dominated by heavily-forked community ensembles of the **same ingredients**: dreaddevelopment's "Raptor" CoAtNet checkpoints (v5/v10/v8, residual-gated e4/e6/e8), DINOv2 public checkpoints (20-member OOF = 0.840 gold-58), public report-label tables, rank-mean blends, LB-probed per-target weights (0.939–0.941). The private edge, per the competitive analysis of the "Torres" 0.94 stack and the measured dead-ends:

1. **Better label extraction, not more labels.** The largest single measured number in anyone's log is **+0.112 of label quality** from replacing a lexicon labeler with an open-weights LLM reader (closed-vocabulary classification by the LLM → deterministic Python mapping of vocabulary to probabilities; per-language calibration). Everyone has weak supervision; the edge is *doing it better*. — reported in existentialistlogarithmic COMPETITIVE_ANALYSIS (third-party; technique is real, number is theirs)
2. **Self-supervised pretraining on the competition DICOMs themselves** (DINOv2-style on knee MRI, then fine-tune) rather than fine-tuning ImageNet weights. "Fine-tune a self-supervised backbone properly — worth ~5× what resolution is." — same source
3. **Noisy-student / self-distillation loops**: train student on teacher (Raptor CoAtNet) soft targets mixed with LLM labels (TianK003: 0.5·LLM + 0.5·quantile-matched Raptor teacher → 0.927 solo, +0.009 over LLM-only; Raptor-vs-LLM teacher AUC 0.9075). — `VERIFIED` (TianK003 experiments)
4. **Anatomical priors in the architecture**: per-slice CLS + patch-mean, plane and laterality embeddings, mean-within-plane pooling, label→series routing. — reported (Torres stack)
5. **More folds + SWA + rank-mean blending of *disagreeing* families** (blend gain comes from disagreement: correlation 0.542 between arms vs 0.905–0.986 within the CoAtNet family). 5-fold, seed variation, SWA last-3. — `VERIFIED` (TianK003, existentialistlogarithmic)
6. **Gold-58 OOF as the only offline signal**; report-label CV neither predicts absolute score nor reliably ranks models (a 288px model won CV 0.7282 → lost the board 0.688). Private teams don't LB-probe per-target weights (that's the public-fork game and risks the private shakeup). — `VERIFIED` (existentialistlogarithmic §11)
7. **Physical-scale sampling** (mm-based resampling, 130mm anatomical crop) to neutralize 16–19 sites of resolution variance. — `VERIFIED` (TianK003, existentialistlogarithmic)
8. **Disagreement-driven ensembling**: rank-mean, not probability-mean (AUC reads order only; probability-mean lets the most confident member dominate). — `VERIFIED` (existentialistlogarithmic COMPETITIVE_ANALYSIS)

### Measured dead ends (do not re-run)
| Route | Verdict |
|---|---|
| Test-time augmentation | 10/12 variants correlate >0.98 with parent — negative on the blend |
| Per-finding blend weights | Real headroom (+0.0076) provably unreachable |
| Narrower inference span | Looked +0.0055 on 58 studies; **−0.0068** on 4,349 — most instructive failure |
| More public/foreign models | 0.79–0.86 vs own CoAtNet 0.92; blends decline monotonically |
| pilkwang's 20 DINOv2 checkpoints | Its two published prediction files correlate 0.995 — already ensembled |
| RadImageNet | 0.8576, same dead band |
| Gated attention pooling | Lost to plain max-pooling in hyperparameter search (small backbone) |
| Self-distillation (naive) | Fold-0 gain was agreement with teacher, not truth — judge by gold-58 direction + solo LB only |

---

## 8. Recommended pipeline skeleton (for the custom build)

```
DICOMs (train_series/ / test_series/)
  │  pydicom + pylibjpeg; RescaleSlope/Intercept; MONOCHROME1 invert
  │  sort by IPP·(IOP_row × IOP_col)  [NEVER filename]
  ▼
Per-series: 1st–99th percentile window → resample to fixed mm/px → 130mm crop
  → laterality normalize (R→L mirror) → uint8 cache (sharded Kaggle Dataset)
  │  version string encodes EVERYTHING (px, bands, percentiles, crop, lat)
  ▼
Series selection: 1 fluid-sensitive / plane (sag, cor, ax); axial-FS guaranteed fallback
  │  recover contrast flags from headers (ScanningSequence/SeriesDescription/TR/TE)
  ▼
2.5D model: shared backbone per slice → per-slice CLS+patch-mean → attention/max pool
  → plane+laterality embeddings → 12 heads
  │  backbones: self-supervised-pretrained (DINOv2-style on knee MRI) + CoAtNet family
  │  aug: rotation/gamma/scale shared per series; laterality-aware flip or none
  ▼
Targets: LLM closed-vocab report labels (per-language calibrated) + gold override (8× weight)
  + teacher-table distillation (Raptor CoAtNet soft targets, quantile-matched)
  │  folds: GroupKFold on (scanner fingerprint + shared-report groups)
  ▼
Validation: gold-58 cross-fitted OOF ONLY (report-label CV is for ranking, never absolute)
  ▼
Inference: same selection fn, same pixels (byte-verified vs train); 16-thread header scan
  + 8 decode workers; incremental submission.csv every 25 studies; runtime projection
  + staged degradation; 0.5 on decode failure
  ▼
Ensemble: rank-mean over disagreeing families; SWA; 5-fold
```

## Key sources
- TianK003/RSNA-KneeMRI-kaggle-competition — CLAUDE.md + docs/traps.md (most detailed pipeline + failure log)
- existentialistlogarithmic/knee-abnormality — docs/FINDINGS.md (§§3,5,8,9,10,11,12), docs/COMPETITIVE_ANALYSIS.md, COMMENTS_FOR_MARTIJN.md
- liamd-schneider/RSNA-Knee-Abnormality-Detection — notebooks/reference-notes.md (0.903 public solution anatomy)
- tranbadat2607/RSNA-Knee-Abnormality-Detection — README (Raptor CoAtNet checkpoints, public label tables)
- aakashkavuru101/rsna — AGENTS.md (laterality-aware flip, CSV gotchas)
- limammohamedlimam/rsna-knee — CLAUDE.md (module layout: dicom_reader, preprocess, routing, augment)
