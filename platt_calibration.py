"""
Per-finding Platt scaling: correct systematic report→image bias using 58 gold studies.

Why Platt (not isotonic): n=58 is 10x below isotonic's 500+ minimum. Isotonic
would overfit and make labels WORSE. Platt's 2-param logistic can't overfit as badly.

Usage (on Kaggle, where both files exist):
    python platt_calibration.py \
        --gold /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
        --weak /kaggle/input/datasets/sridharshivashankar/rsna-knee-phase1-weak-labels-v1/weak_labels_v1.csv \
        --out ./weak_labels_calibrated.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
          'Contusion', 'Fracture']

# Minimum positives to attempt calibration; below this, keep raw weak scores
MIN_POSITIVES = 15


def platt_scale(weak_scores, gold_binary):
    """
    Fit P(gold=1 | weak_score) via logistic regression.
    Returns calibrated scores for all input weak_scores.
    """
    X = np.asarray(weak_scores).reshape(-1, 1)
    y = np.asarray(gold_binary).astype(int)
    
    # Handle edge cases
    if y.sum() == 0 or y.sum() == len(y):
        return np.asarray(weak_scores)  # no signal, return as-is
    
    clf = LogisticRegression(C=1.0)  # mild regularization
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1]


def calibrate_labels(gold_df, weak_df, min_positives=MIN_POSITIVES):
    """
    For each finding:
    - If >= min_positives positives in gold: fit Platt scaling, apply to all weak
    - Else: keep raw weak scores (not enough data to calibrate reliably)
    
    Returns: calibrated DataFrame (same shape as weak_df)
    """
    # Merge on StudyInstanceUID to get paired (weak, gold)
    merged = weak_df.merge(
        gold_df[['StudyInstanceUID'] + LABELS],
        on='StudyInstanceUID', suffixes=('_weak', '_gold')
    )
    print(f"Paired {len(merged)} gold studies for calibration")
    
    calibrated = weak_df.copy()
    report = []
    
    for lab in LABELS:
        wcol = f"{lab}_weak"
        gcol = f"{lab}_gold"
        
        weak_scores = merged[wcol].values
        gold_binary = merged[gcol].values
        n_pos = int(gold_binary.sum())
        
        if n_pos < min_positives:
            report.append(f"  {lab}: SKIP (only {n_pos} positives < {min_positives})")
            continue
        
        # Fit on gold pairs, apply to ALL weak labels
        all_weak = weak_df[lab].values
        # Fit the calibrator on paired data
        X_pair = weak_scores.reshape(-1, 1)
        clf = LogisticRegression(C=1.0)
        clf.fit(X_pair, gold_binary.astype(int))
        # Apply to all
        calibrated[lab] = clf.predict_proba(all_weak.reshape(-1, 1))[:, 1]
        
        # Quick sanity: correlation before/after
        before = np.corrcoef(weak_scores, gold_binary)[0, 1]
        after_scores = clf.predict_proba(X_pair)[:, 1]
        after = np.corrcoef(after_scores, gold_binary)[0, 1]
        report.append(f"  {lab}: calibrated ({n_pos} pos), corr {before:.3f} → {after:.3f}")
    
    print("Calibration report:")
    print("\n".join(report))
    return calibrated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True, help='train.csv with 58 gold labels')
    ap.add_argument('--weak', required=True, help='weak_labels_v1.csv')
    ap.add_argument('--out', required=True, help='output calibrated CSV path')
    ap.add_argument('--min-positives', type=int, default=MIN_POSITIVES)
    args = ap.parse_args()
    
    gold_df = pd.read_csv(args.gold)
    # Gold rows are those with non-null label columns
    gold_df = gold_df.dropna(subset=LABELS).reset_index(drop=True)
    print(f"Gold studies: {len(gold_df)}")
    
    weak_df = pd.read_csv(args.weak)
    print(f"Weak studies: {len(weak_df)}")
    
    calibrated = calibrate_labels(gold_df, weak_df, args.min_positives)
    calibrated.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
