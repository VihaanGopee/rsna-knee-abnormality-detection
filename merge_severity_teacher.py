#!/usr/bin/env python3
"""Merge Gemini severity grades (9 Antigravity batch JSONs) into the teacher label table.

Replaces the teacher's 4 weakest findings (Synovitis, PF OA, Medial OA, Lateral OA)
with binary targets derived from severity grades:
    none -> 0 ; mild / moderate / severe -> 1

The training notebook (phase2-effnet-b3-v2.2) consumes the merged CSV as a DROP-IN
replacement for llm_labels_v4_blend.csv: identical 13-column layout
(StudyInstanceUID + 12 findings), values in [0,1], no NaNs. The notebook's own
2*|p-0.5| confidence mask then assigns weight 1.0 to every replaced cell, so
former 0.5 "report does not address" cells become real supervision instead of
being ignored.

Also writes severity_ordinal_v1.csv (uid + 4 ordinal columns 0-3) for the future
severity-auxiliary-head training run, and merge_report.json with full stats.

Usage:
    python3 merge_severity_teacher.py \
        --teacher llm_labels_v4_blend.csv \
        --batches 'graded_batch_*.json' \
        --outdir ./severity_merge_v1

Fails LOUD on: invalid grades, duplicate UIDs across batches, batch UIDs that
do not exist in the teacher table (row-shift symptom), and a batch-file count
mismatch (default: exactly 9 files must match the glob — catches a forgotten
batch or a stray old probe JSON like the earlier graded_batch_2.json). Teacher
UIDs with no severity grade keep their teacher values and are reported
(fallback, not fatal).
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
          'Contusion', 'Fracture']

SEV_TO_BIN = {"none": 0, "mild": 1, "moderate": 1, "severe": 1}
SEV_TO_ORD = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}

# Defensive normalization: Antigravity batches should use these exact keys,
# but accept common variants rather than dying on a spelling.
SEV_KEY_NORM = {
    "synovitis": "synovitis",
    "pfoa": "pfoa", "pf_oa": "pfoa", "pf oa": "pfoa",
    "medial_oa": "medial_oa", "medial oa": "medial_oa", "medialoa": "medial_oa",
    "lateral_oa": "lateral_oa", "lateral oa": "lateral_oa", "lateral_oa ": "lateral_oa",
    "lateraloa": "lateral_oa",
}
SEV_TO_LABEL_COL = {
    "synovitis": "Synovitis",
    "pfoa": "PF OA",
    "medial_oa": "Medial OA",
    "lateral_oa": "Lateral OA",
}


def fail(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_batches(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        fail(f"no batch files matched {pattern!r}")
    sev = {}          # uid -> {sevkey: grade}
    uid_src = {}      # uid -> batch file (for duplicate reports)
    uid_batch = {}    # uid -> batch basename (for per-batch distribution tables)
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        rows = (data if isinstance(data, list)
                else [{"StudyInstanceUID": k, **v} for k, v in data.items()])
        n = 0
        for r in rows:
            if not isinstance(r, dict) or "StudyInstanceUID" not in r:
                fail(f"{p}: row without StudyInstanceUID: {str(r)[:120]}")
            uid = str(r["StudyInstanceUID"])
            if uid in sev:
                fail(f"duplicate UID {uid} in {p} (first seen in {uid_src[uid]}) "
                     f"— batches must not overlap; check Antigravity row ranges")
            grades = {}
            for k, v in r.items():
                if k == "StudyInstanceUID":
                    continue
                nk = SEV_KEY_NORM.get(str(k).strip().casefold())
                if nk is None:
                    fail(f"{p}: unexpected severity key {k!r} for UID {uid}")
                gv = str(v).strip().casefold()
                if gv not in SEV_TO_BIN:
                    fail(f"{p}: invalid grade {v!r} for {k}/{uid} "
                         f"(expected one of none/mild/moderate/severe)")
                grades[nk] = gv
            missing = [k for k in SEV_TO_LABEL_COL if k not in grades]
            if missing:
                fail(f"{p}: UID {uid} missing grades for {missing}")
            sev[uid] = grades
            uid_src[uid] = p
            uid_batch[uid] = os.path.basename(p)
            n += 1
        print(f"  {p}: {n} rows")
    return sev, paths, uid_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True,
                    help="teacher CSV (llm_labels_v4_blend.csv)")
    ap.add_argument("--batches", required=True,
                    help="glob for severity batch JSONs, e.g. 'graded_batch_*.json'")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--expect-batches", type=int, default=9,
                    help="fail unless exactly this many batch files match the "
                         "glob (default 9: the full 9-batch Antigravity run)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---- teacher table (same normalization as the training notebook) ----
    lab = pd.read_csv(args.teacher, encoding="latin-1")
    norm = {str(c).strip().casefold(): c for c in lab.columns}
    expected = ["StudyInstanceUID"] + LABELS
    missing = [e for e in expected if e.strip().casefold() not in norm]
    if missing:
        fail(f"teacher CSV missing columns {missing}; header={list(lab.columns)}")
    lab = lab[[norm[e.strip().casefold()] for e in expected]]
    lab.columns = expected
    lab["StudyInstanceUID"] = lab["StudyInstanceUID"].astype(str)
    if lab[LABELS].isna().any().any():
        fail("NaN in teacher label values")
    if not ((lab[LABELS] >= 0) & (lab[LABELS] <= 1)).all().all():
        fail("teacher label values outside [0,1]")
    print(f"teacher: {lab.shape[0]} rows x {lab.shape[1]} cols")

    # ---- severity batches ----
    # Count check BEFORE loading: a missing batch or a stray file (e.g. an old
    # probe JSON) must fail here with a clear message, not silently change the
    # merge or trip a downstream duplicate-UID error.
    if args.expect_batches is not None:
        n_files = len(glob.glob(args.batches))
        if n_files != args.expect_batches:
            fail(f"expected {args.expect_batches} batch files, got {n_files} "
                 f"for {args.batches!r} — a batch is missing or a stray file "
                 f"(e.g. an old probe JSON) matched the glob")
    print("batches:")
    sev, paths, uid_batch = load_batches(args.batches)
    print(f"severity grades: {len(sev)} unique UIDs from {len(paths)} files")

    teacher_uids = set(lab["StudyInstanceUID"])
    sev_uids = set(sev)
    orphans = sev_uids - teacher_uids
    if orphans:
        fail(f"{len(orphans)} batch UIDs not in teacher table "
             f"(e.g. {sorted(orphans)[:5]}) — row-shift? check train.csv alignment")
    uncovered = teacher_uids - sev_uids
    print(f"coverage: {len(sev_uids)}/{len(teacher_uids)} teacher UIDs graded; "
          f"{len(uncovered)} fall back to teacher values")

    # ---- build merged + ordinal tables ----
    merged = lab.copy()
    ord_rows = []
    stats = {"n_teacher_rows": len(lab), "n_graded": len(sev_uids),
             "n_uncovered": len(uncovered), "batch_files": paths,
             "per_finding": {}, "per_batch": {}}
    # Per-batch severity distributions: the 9 batches were graded in separate
    # Antigravity chats and the grading rules evolved slightly between batches
    # (e.g. broader fat-pad mapping for synovitis in later batches), so a
    # batch-by-batch table in the report catches label drift before training.
    batch_names = [os.path.basename(p) for p in paths]
    for col in SEV_TO_LABEL_COL.values():
        stats["per_batch"][col] = {bn: Counter() for bn in batch_names}
    for skey, col in SEV_TO_LABEL_COL.items():
        bin_vals, ord_vals = [], []
        n_replaced_half = 0
        n_disagree_confident = 0
        n_confident = 0
        dist = Counter()
        for _, row in merged.iterrows():
            uid = row["StudyInstanceUID"]
            tval = float(row[col])
            if uid in sev:
                g = sev[uid][skey]
                dist[g] += 1
                stats["per_batch"][col][uid_batch[uid]][g] += 1
                b = SEV_TO_BIN[g]
                if abs(tval - 0.5) < 1e-9:
                    n_replaced_half += 1
                elif abs(tval - 0.5) > 0.25:
                    n_confident += 1
                    if int(round(tval)) != b:
                        n_disagree_confident += 1
                bin_vals.append(float(b))
                ord_vals.append(SEV_TO_ORD[g])
            else:
                bin_vals.append(tval)      # fallback: keep teacher
                ord_vals.append(-1)        # -1 = ungraded sentinel
        merged[col] = bin_vals
        stats["per_finding"][col] = {
            "severity_dist": dict(dist),
            "teacher_0.5_cells_replaced": n_replaced_half,
            "teacher_confident_cells": n_confident,
            "disagree_with_confident_teacher": n_disagree_confident,
            "ungraded": int((np.array(ord_vals) == -1).sum()),
        }
        ord_rows.append((skey, ord_vals))

    # ---- per-batch distribution report (drift check) ----
    for col, by_batch in stats["per_batch"].items():
        for bn, c in by_batch.items():
            n = sum(c.values())
            by_batch[bn] = {"n": n, "counts": dict(c),
                            "rates": ({g: round(v / n, 4) for g, v in c.items()}
                                      if n else {})}

    # ordinal table: uid + 4 ordinal columns (0-3, -1 = ungraded)
    odf = pd.DataFrame({"StudyInstanceUID": merged["StudyInstanceUID"]})
    for skey, vals in ord_rows:
        odf[skey] = vals

    # ---- verify drop-in contract (mirrors notebook asserts) ----
    assert merged[LABELS].notna().all().all(), "NaN in merged labels"
    assert ((merged[LABELS] >= 0) & (merged[LABELS] <= 1)).all().all(), \
        "merged labels outside [0,1]"
    # notebook mask math: replaced cells must get weight 1.0
    for skey, col in SEV_TO_LABEL_COL.items():
        w = 2.0 * np.abs(merged[col].values.astype(np.float32) - 0.5)
        graded_mask = merged["StudyInstanceUID"].isin(sev_uids).values
        assert (w[graded_mask] == 1.0).all(), \
            f"weight != 1.0 on graded {col} cells"

    out_csv = os.path.join(args.outdir, "labels_v1_severity_merge.csv")
    out_ord = os.path.join(args.outdir, "severity_ordinal_v1.csv")
    out_rep = os.path.join(args.outdir, "merge_report.json")
    merged.to_csv(out_csv, index=False)
    odf.to_csv(out_ord, index=False)
    with open(out_rep, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote:\n  {out_csv}\n  {out_ord}\n  {out_rep}")
    print("\nper-finding summary:")
    for col, s in stats["per_finding"].items():
        print(f"  {col}: dist={s['severity_dist']} | "
              f"0.5-cells replaced={s['teacher_0.5_cells_replaced']} | "
              f"disagree w/ confident teacher={s['disagree_with_confident_teacher']}/"
              f"{s['teacher_confident_cells']} | ungraded={s['ungraded']}")

    print("\nper-batch severity rates (drift check — rows should be similar "
          "if train.csv is shuffled; systematic drift means grading rules "
          "changed between Antigravity chats):")
    for col in SEV_TO_LABEL_COL.values():
        print(f"  {col}:")
        for bn in batch_names:
            r = stats["per_batch"][col][bn]["rates"]
            n = stats["per_batch"][col][bn]["n"]
            cells = " ".join(f"{g}:{r.get(g, 0.0):.1%}"
                             for g in ("none", "mild", "moderate", "severe"))
            print(f"    {bn} (n={n}): {cells}")


if __name__ == "__main__":
    main()
