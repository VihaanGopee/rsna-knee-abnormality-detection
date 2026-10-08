#!/usr/bin/env python3
"""Validate severity labels against 58 gold binary labels.

Maps severity -> binary:
  none -> 0
  mild/moderate/severe -> 1

Reports per-finding accuracy, and shows disagreements.
"""
import json
import pandas as pd
from sklearn.metrics import accuracy_score

# Severity to binary mapping
def sev_to_bin(s):
    return 0 if s == "none" else 1

# Finding name mapping: severity key -> gold column
FINDING_MAP = {
    "synovitis": "Synovitis",
    "pf_oa": "PF OA",
    "medial_oa": "Medial OA",
    "lateral_oa": "Lateral OA",
}

def main():
    # Load severity labels
    with open("./graded_batch_1.json") as f:
        sev_data = json.load(f)
    # Handle both list and dict formats
    if isinstance(sev_data, list):
        sev = {d["StudyInstanceUID"]: d for d in sev_data}
    else:
        sev = sev_data
    
    print(f"Severity labels: {len(sev)}")
    
    # Load gold labels
    df = pd.read_csv("./train.csv")
    label_cols = list(FINDING_MAP.values())
    gold = df[df[label_cols].notna().any(axis=1)]
    print(f"Gold studies: {len(gold)}")
    
    # Compare
    for sev_key, gold_col in FINDING_MAP.items():
        y_true = []
        y_pred = []
        for _, row in gold.iterrows():
            uid = str(row["StudyInstanceUID"])
            if uid not in sev:
                continue
            gold_val = int(row[gold_col])
            sev_val = sev_to_bin(sev[uid][sev_key])
            y_true.append(gold_val)
            y_pred.append(sev_val)
        
        if y_true:
            acc = accuracy_score(y_true, y_pred)
            print(f"\n{gold_col}:")
            print(f"  Accuracy: {acc:.3f} ({sum(1 for t,p in zip(y_true,y_pred) if t==p)}/{len(y_true)})")
            # Show disagreements
            print(f"  Gold positive: {sum(y_true)}, Severity positive: {sum(y_pred)}")
        else:
            print(f"\n{gold_col}: no overlapping UIDs")

if __name__ == "__main__":
    main()
