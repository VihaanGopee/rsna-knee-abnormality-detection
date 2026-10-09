"""
TTA validation on the 58 gold studies — paste as a notebook cell AFTER the
model is loaded (needs: model, preprocess_study, study_to_batch, PLANES,
N_FINDINGS, DEVICE, and the functions from tta_inference.py).

Protocol:
  1. Baseline: identity view only  -> macro AUC on gold_58
  2. Full TTA: all 8 views         -> macro AUC on gold_58
  3. Ablation: identity + each single transform (2 views) -> per-view delta
Ship rule: keep TTA iff full-TTA macro - baseline macro >= +0.003.
           Drop any single view whose ablation delta < -0.001 (harmful).

Gold studies are TRAIN studies -> preprocessed from train_series/.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# ------------------------------------------------- locate gold + train ----
def _find_first(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

GOLD_CSV = _find_first([
    "/kaggle/input/gold-58/gold_58.csv",
    "/kaggle/working/gold_58.csv",
    "/kaggle/input/datasets/sridharshivashankar/gold-58/gold_58.csv",
])
assert GOLD_CSV, "FATAL: gold_58.csv not found — attach it as a Kaggle dataset"

COMP = _find_first([
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    "/kaggle/input/rsna-knee-abnormality-detection",
])
TRAIN_SERIES = _find_first([
    f"{COMP}/train_series", f"{COMP}/train", f"{COMP}/train_images",
]) if COMP else None
assert TRAIN_SERIES, "FATAL: train_series/ not found"

print(f"gold csv: {GOLD_CSV}\ntrain series: {TRAIN_SERIES}")

# ------------------------------------------------- labels -----------------
# gold_58.csv columns -> our 12 LABELS order. Adjust right-hand names if the
# csv uses different headers; matching is case/space-insensitive.
GOLD_COL_MAP = {
    'ACL': ['acl', 'acl_tear'],
    'MCL': ['mcl', 'mcl_tear'],
    'Medial Meniscus': ['medial_meniscus', 'medial meniscus', 'mm_tear'],
    'Lateral Meniscus': ['lateral_meniscus', 'lateral meniscus', 'lm_tear'],
    'Medial OA': ['medial_oa', 'medial oa', 'moa'],
    'Lateral OA': ['lateral_oa', 'lateral oa', 'loa'],
    'PF OA': ['pf_oa', 'pf oa', 'pfoa', 'patellofemoral_oa'],
    'Effusion': ['effusion'],
    'Synovitis': ['synovitis'],
    "Baker's": ['bakers', "baker's", 'baker_cyst', 'bakers_cyst'],
    'Contusion': ['contusion', 'bone_contusion'],
    'Fracture': ['fracture'],
}
LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

gold = pd.read_csv(GOLD_CSV)
_gcols = {c.lower().strip(): c for c in gold.columns}
uid_col = _gcols.get('studyinstanceuid') or _gcols.get('study_instance_uid')
assert uid_col, f"FATAL: no StudyInstanceUID column in {GOLD_CSV}"

y_true_cols = []
for lab in LABELS:
    hit = None
    for cand in GOLD_COL_MAP[lab]:
        if cand in _gcols:
            hit = _gcols[cand]; break
    assert hit, f"FATAL: no gold column for finding '{lab}'"
    y_true_cols.append(hit)
print(f"mapped {len(y_true_cols)} gold label columns OK")

uids = gold[uid_col].astype(str).tolist()
Y = gold[y_true_cols].to_numpy(dtype=np.float64)
assert Y.shape == (len(uids), 12), f"unexpected gold shape {Y.shape}"
print(f"{len(uids)} gold studies, label positives per finding: {(Y > 0.5).sum(0).tolist()}")

# ------------------------------------------------- run --------------------
def _run_views(views, tag):
    P = np.zeros((len(uids), N_FINDINGS))
    ok = np.ones(len(uids), dtype=bool)
    for i, uid in enumerate(uids):
        pr = predict_one_study_tta(uid, TRAIN_SERIES, views=views)
        if pr is None:
            ok[i] = False
            pr = np.full(N_FINDINGS, 0.5)
        P[i] = pr
        if (i + 1) % 10 == 0:
            print(f"  {tag}: {i+1}/{len(uids)}", flush=True)
    return P, ok

def _macro_auc(P, Y, ok):
    aucs = []
    for f in range(N_FINDINGS):
        yt = Y[ok, f]
        # skip degenerate findings (<2 positives or <2 negatives)
        if yt.min() == yt.max() or (yt > 0.5).sum() < 2 or (yt <= 0.5).sum() < 2:
            aucs.append(np.nan); continue
        try:
            aucs.append(roc_auc_score(yt, P[ok, f]))
        except Exception:
            aucs.append(np.nan)
    return float(np.nanmean(aucs)), aucs

model.eval()
IDENTITY = [TTA_VIEWS[0]]

print("\n[1/3] baseline (identity only)...")
P_base, ok = _run_views(IDENTITY, "baseline")
macro_base, aucs_base = _macro_auc(P_base, Y, ok)
print(f"  baseline macro AUC: {macro_base:.4f} (n_ok={ok.sum()}/{len(uids)})")

print("\n[2/3] full TTA (8 views)...")
P_tta, _ = _run_views(None, "tta8")  # None -> all views
macro_tta, aucs_tta = _macro_auc(P_tta, Y, ok)
print(f"  full-TTA macro AUC: {macro_tta:.4f}  delta={macro_tta-macro_base:+.4f}")

print("\n[3/3] per-view ablation (identity + one view)...")
print(f"  {'view':14s} {'macro':>7s} {'delta':>7s}  verdict")
ablation = {}
for name, fn in TTA_VIEWS[1:]:
    P_v, _ = _run_views([TTA_VIEWS[0], (name, fn)], f"abl_{name}")
    m_v, _ = _macro_auc(P_v, Y, ok)
    d = m_v - macro_base
    ablation[name] = d
    verdict = "DROP (harmful)" if d < -0.001 else ("keep" if d >= 0 else "neutral")
    print(f"  {name:14s} {m_v:7.4f} {d:+7.4f}  {verdict}")

print("\n===================== VERDICT =====================")
print(f"  baseline macro: {macro_base:.4f}")
print(f"  full TTA macro: {macro_tta:.4f}   (delta {macro_tta-macro_base:+.4f})")
harmful = [k for k, d in ablation.items() if d < -0.001]
if harmful:
    print(f"  HARMFUL views (drop before shipping): {harmful}")
if macro_tta - macro_base >= 0.003:
    print("  => SHIP TTA" + (f" (minus {harmful})" if harmful else ""))
else:
    print("  => DO NOT SHIP TTA (delta < +0.003)")
print("===================================================\n")

print("per-finding AUC (baseline -> full TTA):")
for lab, a0, a1 in zip(LABELS, aucs_base, aucs_tta):
    print(f"  {lab:16s} {a0:.4f} -> {a1:.4f}  ({a1-a0:+.4f})")
