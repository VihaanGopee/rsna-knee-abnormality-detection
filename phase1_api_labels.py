#!/usr/bin/env python3
"""Phase 1 weak-label extraction via Gemini API (replaces the Kaggle-GPU LLM path).

Pipeline:
  1. Load train.csv, dedupe to 4,276 unique reports by hash.
  2. PROBE mode: label 100 reports (58 gold + 42 extra), check gates
     (macro AUC >= 0.80 vs gold, parse-fail < 15%).
  3. FULL mode: label all unique reports, isotonic-calibrate each label on the
     58 gold studies, write weak_labels_v1.csv + descriptors.jsonl +
     calibration.json + manifest.json.

Usage:
    python3 phase1_api_labels.py --mode probe   # 100 reports, gate check
    python3 phase1_api_labels.py --mode full    # all 4,276 reports

Outputs land in ./phase1_api_out/.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, "/home/hatch/workspace/rsna-knee")
sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
import dynamic_credentials as dc
import label_core
import pandas as pd

MODEL = "gemini-2.5-flash"
ALLOWED_HOSTS = ["generativelanguage.googleapis.com"]
CREDENTIAL = "custom.gemini"
OUT_DIR = "/home/hatch/workspace/rsna-knee/phase1_api_out"
CSV_PATH = "/home/hatch/workspace/user/files/train.csv"

# Extra guidance appended to the standard prompt: severity defaults so the
# model never invents "present" (which is not in the vocab).
SEVERITY_NOTE = (
    "\nSeverity default: if a finding is stated as present but the report gives "
    "no grade/severity, use the mildest positive term "
    "(mild for osteoarthritis, trace_small for effusion, sprain for ligament "
    "injury, degenerative_signal for meniscus signal change)."
)


def gemini_extract(prompt: str, max_tokens: int = 1024) -> dict:
    """Call Gemini, return the raw JSON dict (or raise)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    url = dc.url_with_surrogate_query_param(
        url, CREDENTIAL, allowed_hosts=ALLOWED_HOSTS)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
            "response_mime_type": "application/json",
        },
    }
    last_err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = dc.read_json_response(resp)
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            obj = label_core.extract_json(text)
            if obj is None:
                raise ValueError(f"no JSON parsed from: {text[:200]}")
            usage = data.get("usageMetadata", {})
            return obj, usage
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503):
                time.sleep(2 ** attempt + random.random())
                continue
            raise
        except (ValueError, KeyError, IndexError) as e:
            last_err = e
            time.sleep(1 + random.random())
    raise RuntimeError(f"extract failed after retries: {last_err}")


def label_one(args):
    """Worker: (hash, report) -> record dict."""
    h, report = args
    prompt = label_core.build_prompt(report) + SEVERITY_NOTE
    t0 = time.time()
    try:
        raw, usage = gemini_extract(prompt)
        terms, issues = label_core.normalize_descriptors(raw)
        parse_ok = not issues
        return {
            "hash": h, "terms": terms, "issues": issues,
            "parse_ok": parse_ok, "seconds": time.time() - t0,
            "usage": usage,
        }
    except Exception as e:
        return {
            "hash": h, "terms": {k: "not_mentioned" for k in label_core.KEYS},
            "issues": [f"api_error: {e}"], "parse_ok": False,
            "seconds": time.time() - t0, "usage": {},
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "full"], required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    df["_hash"] = df["Report"].map(
        lambda r: hashlib.sha256(r.encode("utf-8", "replace")).hexdigest()[:16])
    uniq = df.drop_duplicates("_hash").set_index("_hash")["Report"].to_dict()
    print(f"studies: {len(df)} | unique reports: {len(uniq)}")

    label_cols = list(label_core.KEY2LABEL.values())
    gold_mask = df[label_cols].notna().all(axis=1)
    gold_df = df[gold_mask]
    gold_hashes = gold_df["_hash"].tolist()
    print(f"gold studies: {len(gold_df)}")
    hash_to_report = uniq

    if args.mode == "probe":
        random.seed(0)
        extra = [h for h in uniq if h not in set(gold_hashes)]
        targets = gold_hashes + random.sample(extra, 42)
        print(f"probe: {len(targets)} reports (58 gold + 42 extra)")
    else:
        targets = list(uniq.keys())
        print(f"full: {len(targets)} reports")

    t_start = time.time()
    records = []
    total_in, total_out = 0, 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(label_one, (h, hash_to_report[h])): h for h in targets}
        done = 0
        for fut in cf.as_completed(futs):
            rec = fut.result()
            records.append(rec)
            u = rec.get("usage", {})
            total_in += u.get("promptTokenCount", 0)
            total_out += u.get("candidatesTokenCount", 0)
            done += 1
            if done % 50 == 0 or done == len(targets):
                el = time.time() - t_start
                print(f"  {done}/{len(targets)}  ({el:.0f}s, "
                      f"{el/max(done,1):.1f}s/report)", flush=True)

    rec_by_hash = {r["hash"]: r for r in records}
    n_fail = sum(1 for r in records if not r["parse_ok"])
    print(f"\nparse failures: {n_fail}/{len(records)} ({n_fail/len(records):.1%})")

    # ---- validation vs gold ----
    def study_terms(uid):
        h = df.loc[df["StudyInstanceUID"] == uid, "_hash"].iloc[0]
        return rec_by_hash[h]["terms"]

    gold_scores, gold_true = {}, {}
    for k in label_core.KEYS:
        lab = label_core.KEY2LABEL[k]
        ranks, trues = [], []
        for _, row in gold_df.iterrows():
            t = study_terms(row["StudyInstanceUID"])
            ranks.append(label_core.rank_of(k, t[k]))
            trues.append(int(row[lab]))
        cal = label_core.calibrate_label(ranks, trues, sum(trues) / len(trues))
        gold_scores[lab] = [label_core.apply_calibration(k, t, cal)
                            for t in (study_terms(u)[k]
                                      for u in gold_df["StudyInstanceUID"])]
        gold_true[lab] = trues
        globals()[f"CAL_{k}"] = cal

    macro, per = label_core.macro_auc(gold_true, gold_scores)
    print(f"\nmacro AUC vs gold ({len(gold_df)}): {macro:.4f}")
    for lab in label_core.KEY2LABEL.values():
        a, n1, n = per[lab]
        print(f"  {lab:16s} AUC={a if a is None else f'{a:.3f}'}  positives={n1}/{n}")

    el = time.time() - t_start
    per_report = el / len(records)
    proj_h = per_report * len(uniq) / 3600
    print(f"\nparse-fail: {n_fail/len(records):.1%} | per-report: {per_report:.2f}s | "
          f"projected full: {proj_h:.2f}h")
    print(f"tokens: {total_in:,} in / {total_out:,} out")

    gates = {
        "macro_auc>=0.80": macro is not None and macro >= 0.80,
        "parse_fail<15%": n_fail / len(records) < 0.15,
    }
    print("\nGATES:", gates)
    if args.mode == "probe":
        assert all(gates.values()), "PROBE GATE FAILED"
        print("PROBE PASSED — full mode is cleared.")
        # save probe records for reuse
        json.dump(records, open(f"{OUT_DIR}/probe_records.json", "w"))
        return

    # ---- full mode outputs ----
    # descriptors.jsonl (one per unique report)
    with open(f"{OUT_DIR}/descriptors.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps({"hash": r["hash"], "terms": r["terms"],
                                "issues": r["issues"]}) + "\n")
    # calibration.json
    cal_json = {k: {"thresholds": cal[0], "values": cal[1]}
                for k, cal in ((k, globals()[f"CAL_{k}"]) for k in label_core.KEYS)}
    json.dump(cal_json, open(f"{OUT_DIR}/calibration.json", "w"), indent=1)
    # weak_labels_v1.csv (one row per study)
    rows = []
    for _, row in df.iterrows():
        terms = rec_by_hash[row["_hash"]]["terms"]
        out = {"StudyInstanceUID": row["StudyInstanceUID"]}
        for k in label_core.KEYS:
            lab = label_core.KEY2LABEL[k]
            out[lab] = label_core.apply_calibration(
                k, terms[k], globals()[f"CAL_{k}"])
        rows.append(out)
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/weak_labels_v1.csv", index=False)
    # manifest.json
    manifest = {
        "model": MODEL, "mode": "full",
        "n_studies": len(df), "n_unique_reports": len(uniq),
        "n_gold": len(gold_df), "macro_auc_vs_gold": macro,
        "parse_fail_rate": n_fail / len(records),
        "elapsed_s": el, "tokens_in": total_in, "tokens_out": total_out,
    }
    json.dump(manifest, open(f"{OUT_DIR}/manifest.json", "w"), indent=1)
    print(f"\nwrote: {OUT_DIR}/weak_labels_v1.csv, descriptors.jsonl, "
          f"calibration.json, manifest.json")


if __name__ == "__main__":
    main()
