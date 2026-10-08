#!/usr/bin/env python3
"""Verify a severity-graded pilot batch BEFORE committing to the full relabel run.

Reads the pilot batch (e.g. graded_batch_1.json: {uid: {finding: grade}}),
then checks:

  1. Coverage & hygiene: N graded, all-4-findings completeness, invalid grades.
  2. Grade distributions per finding + mode-collapse / severity-skew warnings.
  3. Presence agreement vs the 58 competition gold labels in train.csv
     (presence = grade != 'none').
  4. Presence agreement vs the teacher label table (optional --teacher
     llm_labels_v4_blend.csv); 0.5 "don't know" cells are excluded.
  5. Spot-check shortlist: extreme grades + gold/teacher disagreements, with
     report text, so a human can eyeball the suspicious ones.

Usage:
    python3 severity_batch_verify.py --batch graded_batch_1.json \
        [--train train.csv] [--teacher llm_labels_v4_blend.csv]

Exit 0 = checks ran (warnings are printed, not fatal). Exit 2 = usage error.
"""

import argparse
import csv
import json
import sys

GRADES = ("none", "mild", "moderate", "severe")

# canonical finding -> accepted key spellings (case-insensitive)
FINDING_ALIASES = {
    "synovitis": {"synovitis"},
    "pfoa": {"pfoa", "pf_oa", "pf oa"},
    "medial_oa": {"medial_oa", "medial oa"},
    "lateral_oa": {"lateral_oa", "lateral oa"},
}

# canonical finding -> gold-label column in competition train.csv
GOLD_COLS = {
    "synovitis": "Synovitis",
    "pfoa": "PF OA",
    "medial_oa": "Medial OA",
    "lateral_oa": "Lateral OA",
}

DEFAULT_TRAIN = "/home/hatch/workspace/user/files/train.csv"


def normalize_keys(entry):
    """Map whatever key spellings a batch uses onto canonical findings.

    Returns (grades, unknown_keys, conflicts): conflicts lists canonical
    findings matched by 2+ different spellings in the same entry — those are
    treated as ungraded (grade=None) rather than silently picking one.
    """
    out, unknown, conflicts = {}, [], []
    lowered = {str(k).strip().lower(): v for k, v in entry.items()}
    for canon, aliases in FINDING_ALIASES.items():
        hits = [a for a in aliases if a in lowered]
        if len(hits) > 1:
            conflicts.append(canon)
        elif hits:
            out[canon] = lowered[hits[0]]
    for k in lowered:
        if not any(k in aliases for aliases in FINDING_ALIASES.values()):
            unknown.append(k)
    return out, unknown, conflicts


def normalize_grade(g):
    g = str(g).strip().lower()
    return g if g in GRADES else None


def load_gold(train_path):
    """{uid: {canon_finding: 0/1}} for the 58 gold-labeled rows; plus reports."""
    gold, reports = {}, {}
    with open(train_path, newline="") as f:
        for row in csv.DictReader(f):
            uid = str(row.get("StudyInstanceUID", "")).strip()
            reports[uid] = row.get("Report", "") or ""
            if all((row.get(c) or "").strip() in ("0", "1") for c in GOLD_COLS.values()):
                if any(row.get(c) for c in GOLD_COLS.values()):
                    gold[uid] = {canon: int(row[col]) for canon, col in GOLD_COLS.items()}
    return gold, reports


def load_teacher(path):
    """{uid: {canon_finding: prob}}; tolerates column-name variants."""
    t = {}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in rdr.fieldnames or []}
        uid_col = next((cols[c] for c in cols if "uid" in c or "study" in c), None)
        colmap = {}
        for canon, aliases in FINDING_ALIASES.items():
            for a in aliases:
                if a in cols:
                    colmap[canon] = cols[a]
                    break
        for row in rdr:
            uid = str(row.get(uid_col, "")).strip() if uid_col else ""
            if not uid:
                continue
            probs = {}
            for canon, col in colmap.items():
                try:
                    probs[canon] = float(row[col])
                except (ValueError, TypeError):
                    pass
            if probs:
                t[uid] = probs
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--teacher", default=None)
    args = ap.parse_args()

    with open(args.batch) as f:
        batch = json.load(f)
    if not isinstance(batch, dict):
        print("ERROR: batch JSON must be an object {uid: {finding: grade}}", file=sys.stderr)
        sys.exit(2)

    gold, reports = load_gold(args.train)
    teacher = load_teacher(args.teacher) if args.teacher else {}

    findings = list(FINDING_ALIASES)
    dist = {f: {g: 0 for g in GRADES} for f in findings}
    invalid = {f: 0 for f in findings}
    n_full = 0
    weird_keys = {}
    graded = {}  # uid -> {canon: grade or None}
    warnings = []

    for uid, entry in batch.items():
        if not isinstance(entry, dict):
            continue
        norm, unknown, conflicts = normalize_keys(entry)
        if unknown:
            weird_keys[uid] = unknown
        if conflicts:
            warnings.append(f"UID {uid[:24]}…: conflicting keys for {conflicts} (left ungraded)")
        grades = {}
        for f in findings:
            g = normalize_grade(norm[f]) if f in norm else None
            grades[f] = g
            if f in norm and g is None:
                invalid[f] += 1
            elif g is not None:
                dist[f][g] += 1
        if all(grades[f] is not None for f in findings):
            n_full += 1
        graded[uid] = grades

    n = len(graded)
    if n == 0:
        print(f"== severity pilot check: {args.batch}\ngraded UIDs: 0 — batch is empty, nothing to verify.")
        sys.exit(0)
    full_pct = 100.0 * n_full / n
    print(f"== severity pilot check: {args.batch}")
    print(f"graded UIDs: {n} | complete (all 4 findings): {n_full} ({full_pct:.1f}%)")
    for f in findings:
        tot = sum(dist[f].values())
        cells = "  ".join(f"{g}:{dist[f][g]} ({100.0*dist[f][g]/tot:.0f}%)" if tot else f"{g}:0"
                            for g in GRADES)
        print(f"  {f:10s} {cells}")
        if invalid[f]:
            print(f"    !! {invalid[f]} INVALID grade strings (not none/mild/moderate/severe)")
            warnings.append(f"{f}: {invalid[f]} invalid grades")
        if tot:
            top = max(dist[f].values())
            if top / tot > 0.85:
                print(f"    !! MODE COLLAPSE: one grade covers {100.0*top/tot:.0f}% of {f}")
                warnings.append(f"{f}: mode collapse ({100.0*top/tot:.0f}% one grade)")
            if (dist[f]["severe"] + dist[f]["moderate"]) / tot > 0.60:
                print(f"    !! SEVERITY SKEW: {(dist[f]['severe']+dist[f]['moderate'])/tot:.0f}% "
                      f"moderate+severe — MRI cohorts skew none/mild; eyeball these")
                warnings.append(f"{f}: severity skew high")
    if weird_keys:
        print(f"  !! {len(weird_keys)} UIDs had unrecognized finding keys "
              f"(e.g. {next(iter(weird_keys.values()))}); ignored)")

    # --- gold agreement ---
    overlap = [uid for uid in graded if uid in gold]
    print(f"\ngold overlap: {len(overlap)} of {len(gold)} gold studies in this batch")
    disagree = []
    for f in findings:
        tp = tn = fp = fn = 0
        for uid in overlap:
            g = graded[uid][f]
            if g is None:
                continue
            pred = 1 if g != "none" else 0
            true = gold[uid][f]
            if pred and true:
                tp += 1
            elif not pred and not true:
                tn += 1
            elif pred and not true:
                fp += 1
                disagree.append((uid, f, g, true, "gold=0 graded=" + g))
            else:
                fn += 1
                disagree.append((uid, f, g, true, "gold=1 graded=none"))
        tot = tp + tn + fp + fn
        if tot:
            acc = (tp + tn) / tot
            print(f"  {f:10s} presence-acc vs gold: {acc:.2f} (n={tot}) "
                  f"[tp={tp} tn={tn} fp={fp} fn={fn}]")
            if acc < 0.70 and tot >= 3:
                warnings.append(f"{f}: gold presence agreement only {acc:.2f} (n={tot})")
    if overlap:
        print("  (gold prevalence in full 58 for context:) ", end="")
        for f in findings:
            prev = sum(gold[u][f] for u in gold) / len(gold)
            print(f"{f}={prev:.2f} ", end="")
        print()

    # --- teacher agreement ---
    if teacher:
        toverlap = [u for u in graded if u in teacher]
        print(f"\nteacher overlap: {len(toverlap)} batch UIDs in teacher table")
        tdis = []
        for f in findings:
            tp = tn = fp = fn = 0
            for uid in toverlap:
                if f not in teacher[uid]:
                    continue
                p = teacher[uid][f]
                if p == 0.5:
                    continue  # author's own "don't know" semantics
                g = graded[uid][f]
                if g is None:
                    continue
                pred = 1 if g != "none" else 0
                true = 1 if p > 0.5 else 0
                if pred and true:
                    tp += 1
                elif not pred and not true:
                    tn += 1
                elif pred and not true:
                    fp += 1
                else:
                    fn += 1
                # strong teacher conviction vs grade: worth an eyeball
                if (p >= 0.8 and g == "none") or (p <= 0.2 and g == "severe"):
                    tdis.append((uid, f, g, p))
            tot = tp + tn + fp + fn
            if tot:
                print(f"  {f:10s} presence-agree vs teacher: {(tp+tn)/tot:.2f} (n={tot})")
                if (tp + tn) / tot < 0.70 and tot >= 20:
                    warnings.append(f"{f}: teacher presence agreement {(tp+tn)/tot:.2f}")
    else:
        tdis = []
        print("\nteacher table not given (--teacher); skipping teacher comparison")

    # --- spot-check shortlist ---
    spots = []
    for uid, f, g, true, why in disagree:
        spots.append((uid, f"GOLD-DISAGREE {why}", g))
    for uid, f, g, p in tdis[:10]:
        spots.append((uid, f"TEACHER-DISAGREE p={p:.2f} graded={g}", g))
    extremes = [uid for uid, gs in graded.items()
                if sum(1 for f in findings if gs[f] == "severe") >= 2]
    for uid in extremes[:5]:
        sev = ", ".join(f"{f}={graded[uid][f]}" for f in findings)
        spots.append((uid, "EXTREME (2+ severe)", sev))

    print(f"\nspot-check shortlist ({len(spots)} shown, max 15):")
    for uid, why, detail in spots[:15]:
        rep = (reports.get(uid, "") or "").replace("\n", " ")[:400]
        print(f"  - {uid}\n    {why} | {detail}\n    report: {rep}")

    print("\n== verdict ==")
    if not warnings:
        print("PILOT LOOKS HEALTHY — no red flags. Safe to run the remaining ~4,026.")
    else:
        print(f"{len(warnings)} flag(s) to clear before the full run:")
        for w in warnings:
            print(f"  * {w}")
        print("Eyeball the spot-check list above, then re-run this on the fixed batch.")


if __name__ == "__main__":
    main()
