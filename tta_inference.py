"""
TTA inference for RSNA Knee submission notebook — drop-in code.

Paste into the submission notebook (replaces/augments the INFERENCE cell).
Assumes existing notebook globals:
    PLANES, N_SLICES, IMAGE_SIZE, N_FINDINGS, DEVICE, model (in eval mode),
    preprocess_study, sample_slices, study_to_batch pieces, SUB_COLS, OUT_CSV,
    SAVE_EVERY, RADIMAGENET_NORM (optional flag; default False).

Design decisions (see research doc for evidence):
- 8 views, intensity-only + horizontal flip. NO vertical flip, NO rotation:
  vertical flip breaks femur-above/tibia-below anatomy; rotation beyond what
  the model saw in training risks BN/stat mismatch. Intensity-only TTA is the
  safe, validated pattern for medical imaging.
- Transforms apply ONLY to present planes (plane_mask == 1). Zero-filled
  missing planes are left as zeros — brightening them would inject fake signal.
- All 8 views are stacked on the batch dim -> ONE forward pass per study.
  (8, S, 3, H, W) per plane ~ 63 MB fp32 at 288px; fine on a T4.
- Aggregation: mean of sigmoid probabilities (standard; median optional).
- model.eval() is enforced inside predict — BN running stats are frozen, so
  TTA views cannot corrupt normalization statistics.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- config ---
TTA_ENABLED = True          # master switch; False -> identical to single-view
TTA_AGG = "mean"            # "mean" | "median"
# Set True when the checkpoint is a RadImageNet model ([-1,1] input range).
# Falls back to the notebook global if defined there.
try:
    RADIMAGENET_NORM
except NameError:
    RADIMAGENET_NORM = False


# ------------------------------------------------------- view transforms ---
# Each takes vol: (S, 3, H, W) float32 in [0,1], returns same shape/range.
# Applied identically to every slice of a plane (no per-slice randomness).

def _tta_identity(vol):
    return vol

def _tta_hflip(vol):
    # Horizontal flip: SAFE for knee (L/R mirror-symmetric; finding presence
    # is flip-invariant). .copy() because [::-1] makes negative strides,
    # which torch.from_numpy rejects.
    return vol[:, :, :, ::-1].copy()

def _tta_bright_up(vol):
    return np.clip(vol + 0.10, 0.0, 1.0).astype(np.float32)

def _tta_bright_down(vol):
    return np.clip(vol - 0.10, 0.0, 1.0).astype(np.float32)

def _tta_gamma_down(vol):
    # gamma 0.90: lifts shadows, compresses highlights (contrast variant A)
    return np.clip(np.power(vol, 0.90), 0.0, 1.0).astype(np.float32)

def _tta_gamma_up(vol):
    # gamma 1.10: deepens shadows (contrast variant B)
    return np.clip(np.power(vol, 1.10), 0.0, 1.0).astype(np.float32)

def _tta_window_narrow(vol):
    # Narrow intensity window [0.15, 0.85] -> [0,1]: boosts mid-tone contrast,
    # mimics a tighter DICOM window/level. Mirrors the 0.910 team's
    # multi-window TTA idea.
    lo, hi = 0.15, 0.85
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

def _tta_window_soft(vol):
    # Soft/wide window: compress [0,1] -> [0.05, 0.95]. Gentle global
    # contrast reduction; catches over-confident edge responses.
    return (vol * 0.90 + 0.05).astype(np.float32)

# Order matters only for the ablation script; v0 must stay the identity.
TTA_VIEWS = [
    ("identity",    _tta_identity),
    ("hflip",       _tta_hflip),
    ("bright_up",   _tta_bright_up),
    ("bright_down", _tta_bright_down),
    ("gamma_090",   _tta_gamma_down),
    ("gamma_110",   _tta_gamma_up),
    ("window_narrow", _tta_window_narrow),
    ("window_soft",   _tta_window_soft),
]
N_TTA_VIEWS = len(TTA_VIEWS)
assert N_TTA_VIEWS == 8, f"expected 8 TTA views, got {N_TTA_VIEWS}"


# ------------------------------------------------------- batch building ----
def _to_model_range(vol):
    """[0,1] -> model input range. RadImageNet branch needs [-1,1]."""
    if RADIMAGENET_NORM:
        return (vol * 2.0 - 1.0).astype(np.float32)
    return vol

def make_tta_batch(planes_np, pmask_np, views=None):
    """planes_np: {plane: (S,3,H,W) float32 [0,1]}, pmask_np: (3,) float32.
    Returns (batch_dict, mask_t) with batch dim = n_views (or 1 if disabled).
    Transforms are applied ONLY to present planes (mask == 1).
    views: optional subset list of (name, fn) for ablation; default all views.
    """
    if views is None:
        views = TTA_VIEWS if TTA_ENABLED else [TTA_VIEWS[0]]
    per_plane_views = {p: [] for p in PLANES}
    for _name, fn in views:
        for pi, p in enumerate(PLANES):
            vol = planes_np[p]
            if pmask_np[pi] == 1.0:
                vol = fn(vol)
            # missing planes stay zero-filled: never transform zeros
            per_plane_views[p].append(_to_model_range(vol))
    batch = {
        p: torch.from_numpy(np.stack(per_plane_views[p], axis=0)).to(DEVICE)
        for p in PLANES
    }  # each (V,S,3,H,W)
    mask_t = torch.from_numpy(
        np.tile(pmask_np[None, :], (len(views), 1))
    ).to(DEVICE)  # (V,3)
    return batch, mask_t


# ------------------------------------------------------- inference ---------
@torch.no_grad()
def predict_one_study_tta(uid, study_dir, views=None, agg="mean"):
    """Single-study TTA predict. Returns probs (12,) or None on failure.
    views: optional subset of TTA_VIEWS for ablation (default: all/identity).
    """
    model.eval()
    sdir = os.path.join(study_dir, uid)
    if not os.path.isdir(sdir):
        return None
    planes_u8, pmask = preprocess_study(sdir)
    planes_np, pmask_np = study_to_batch(planes_u8, pmask)
    batch, mask_t = make_tta_batch(planes_np, pmask_np, views=views)
    logits = model(batch, mask_t)
    probs_v = torch.sigmoid(logits).double()
    if agg == "median":
        probs = probs_v.median(dim=0).values.cpu().numpy()
    else:
        probs = probs_v.mean(dim=0).cpu().numpy()
    if probs.shape != (N_FINDINGS,) or not np.all(np.isfinite(probs)):
        return None
    return probs


@torch.no_grad()
def predict_studies_tta(uids, test_dir):
    """TTA inference. Returns list of (uid, probs12). Failed study -> 0.5s.
    Drop-in replacement for predict_studies(); set TTA_ENABLED=False for
    bit-identical single-view behavior.
    """
    model.eval()  # freeze BN stats: TTA views must not update normalization
    results, failed = [], []
    t0 = time.time()
    n_views = N_TTA_VIEWS if TTA_ENABLED else 1
    for i, uid in enumerate(uids):
        _verbose = (i < 3)
        try:
            probs = predict_one_study_tta(uid, test_dir)
            if probs is None:
                raise ValueError("study dir missing or bad output")
            if _verbose:
                print(f"  [study {i+1}] uid={uid} views={n_views} "
                      f"probs={['%.3f' % v for v in probs]}")
        except Exception as e:
            probs = np.full(N_FINDINGS, 0.5)
            failed.append(uid)
            if len(failed) <= 5:
                import traceback
                print(f"  FAILED {uid}: {type(e).__name__}: {e}")
                traceback.print_exc()
        results.append((uid, probs))
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(uids):
            _write_csv(results)  # incremental save (existing helper)
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(uids) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(uids)} | {dt:.0f}s elapsed | {eta/60:.0f}min ETA | "
                  f"failed: {len(failed)}", flush=True)
    return results, failed
