#!/usr/bin/env python3
"""Score a BLIND Gemini gold test against the expert labels.

Usage:
    python3 score_gold_test.py gold_58.csv graded_gold_structural.json [graded_gold_fluidtrauma.json]

- gold_58.csv: the ORIGINAL expert file (StudyInstanceUID, Report, ACL, MCL, ...).
  Run this script OUTSIDE Antigravity, after the blind grading JSONs are written.
- The graded JSONs come from the blind prompts (gold_58_reports_only.csv only).
  Each is a list of {"StudyInstanceUID": ..., <finding>: <term>, ...}.

Maps each graded term to binary and reports per-finding accuracy vs experts.
"""
import csv
import json
import sys

# graded-term -> binary, per finding key in the JSON
BIN_MAPS = {
    "acl":       {"intact": 0, "mucoid_degeneration": 0, "reconstructed": 0,
                  "sprain": 1, "partial_tear": 1, "complete_tear": 1},
    "mcl":       {"intact": 0, "sprain": 1, "complete_tear": 1},
    "mm":        {"intact": 0, "degenerative_signal": 0, "postoperative": 0,
                  "tear": 1, "macerated": 1},
    "lm":        {"intact": 0, "degenerative_signal": 0, "postoperative": 0,
                  "tear": 1, "macerated": 1},
    "effusion":  {"none": 0, "small": 1, "moderate": 1, "large": 1},
    "bakers":    {"none": 0, "present": 1, "ruptured": 1},
    "contusion": {"none": 0, "present": 1},
    "fracture":  {"none": 0, "present": 1, "suspected": 1},
}

# JSON finding key -> expert column in gold_58.csv
EXPERT_COL = {
    "acl": "ACL", "mcl": "MCL", "mm": "Medial Meniscus", "lm": "Lateral Meniscus",
    "effusion": "Effusion", "bakers": "Baker's",
    "contusion": "Contusion", "fracture": "Fracture",
}


def load_experts(path):
    experts = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            experts[str(r["StudyInstanceUID"])] = {
                k: int(float(r[col])) for k, col in EXPERT_COL.items()
            }
    return experts


def load_grades(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.values() if isinstance(data, dict) else data
    return {str(r["StudyInstanceUID"]): r for r in rows}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    experts = load_experts(sys.argv[1])
    print(f"expert studies: {len(experts)}")
    total_c, total_n = 0, 0
    for gpath in sys.argv[2:]:
        grades = load_grades(gpath)
        print(f"\n{gpath}: {len(grades)} graded studies")
        for finding, bmap in BIN_MAPS.items():
            col = EXPERT_COL[finding]
            c = n = 0
            skipped_terms = {}
            for uid, g in grades.items():
                if uid not in experts:
                    continue
                term = str(g.get(finding, "")).strip().lower()
                if not term:
                    continue  # finding not in this JSON (e.g. Prompt A file has no effusion)
                if term not in bmap:
                    skipped_terms[term] = skipped_terms.get(term, 0) + 1
                    continue
                n += 1
                if bmap[term] == experts[uid][finding]:
                    c += 1
            if n:
                print(f"  {finding:9s} {c}/{n} = {c/n:.3f}  (vs expert {col})")
                total_c += c
                total_n += n
            if skipped_terms:
                print(f"    !! unmapped terms skipped: {skipped_terms}")
    if total_n:
        print(f"\nOVERALL: {total_c}/{total_n} = {total_c/total_n:.4f}")


if __name__ == "__main__":
    main()
