#!/usr/bin/env python3
"""Strip expert label columns from gold_58.csv for blinded validation.

Creates gold_58_reports_only.csv (StudyInstanceUID + Report only)
so Gemini/Antigravity cannot see the answers during labeling.

Usage:
    python3 strip_gold.py [input_csv] [output_csv]
    python3 strip_gold.py  # defaults: gold_58.csv -> gold_58_reports_only.csv
"""
import csv
import sys

def strip_gold(in_path="gold_58.csv", out_path="gold_58_reports_only.csv"):
    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            {"StudyInstanceUID": r["StudyInstanceUID"], "Report": r["Report"]}
            for r in reader
        ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["StudyInstanceUID", "Report"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Stripped {len(rows)} rows: {in_path} -> {out_path}")

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "gold_58.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "gold_58_reports_only.csv"
    strip_gold(in_path, out_path)
