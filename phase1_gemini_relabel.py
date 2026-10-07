#!/usr/bin/env python3
"""Phase 1 RE-LABEL via Gemini 3.8 Flash: generate a fresh, higher-quality
label set for all 4,276 reports to ensemble with (or replace) the teacher labels.

Why: teacher labels hit 0.8927 vs gold. Our GPT-4.1 labels hit 0.8473.
A Gemini 3.8 Flash label set gives us a third source to ensemble, potentially
pushing past 0.8927.

- Reads ./train.csv (4,276 unique reports)
- Labels ALL reports via Gemini (same prompt/parsing as Phase 1)
- Rotates across multiple API keys
- Resume-safe: progress in ./phase1_gemini_relabel_out/records.jsonl
- Output: ./phase1_gemini_relabel_out/gemini_38_labels.csv (StudyInstanceUID + 12 cols)

Usage:
    export GEMINI_API_KEYS="key1,key2,key3"   # comma-separated, no spaces
    python3 phase1_gemini_relabel.py --csv ./train.csv --model gemini-3.8-flash
    python3 phase1_gemini_relabel.py --csv ./train.csv --model gemini-3.8-flash-thinking

~4,276 reports. At ~40s/report (from v13 probe), this is ~47h single-threaded.
Use multiple keys and run in background.
"""
import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

import pandas as pd

BATCH = 1  # single report per call for reliability
TIMEOUT_S = 120
OUT_DIR = "./phase1_gemini_relabel_out"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "mm", "lm", "moa", "loa", "pfoa", "effusion",
        "synovitis", "bakers", "contusion", "fracture"]

# Maps closed-vocab terms to binary labels (same as Phase 1)
TERM_TO_LABEL = {
    "acl": {"intact": 0, "mucoid_degeneration": 0, "reconstructed": 0,
            "sprain": 1, "partial_tear": 1, "complete_tear": 1},
    "mcl": {"intact": 0, "sprain": 1, "complete_tear": 1},
    "mm": {"intact": 0, "degenerative_signal": 0, "postoperative": 0,
           "tear": 1, "macerated": 1},
    "lm": {"intact": 0, "degenerative_signal": 0, "postoperative": 0,
           "tear": 1, "macerated": 1},
    "moa": {"none": 0, "mild": 1, "moderate": 1, "severe": 1},
    "loa": {"none": 0, "mild": 1, "moderate": 1, "severe": 1},
    "pfoa": {"none": 0, "mild": 1, "moderate": 1, "severe": 1},
    "effusion": {"none": 0, "small": 1, "moderate": 1, "large": 1},
    "synovitis": {"none": 0, "mild": 1, "moderate": 1, "severe": 1, "suspected": 0.5},
    "bakers": {"none": 0, "present": 1, "ruptured": 1},
    "contusion": {"none": 0, "present": 1},
    "fracture": {"none": 0, "present": 1, "suspected": 0.5},
}

VOCAB = {
    "acl": {"finding": "anterior cruciate ligament (ACL)",
            "terms": ["intact", "mucoid_degeneration", "reconstructed",
                      "sprain", "partial_tear", "complete_tear"]},
    "mcl": {"finding": "medial collateral ligament (MCL)",
            "terms": ["intact", "sprain", "complete_tear"]},
    "mm": {"finding": "medial meniscus",
           "terms": ["intact", "degenerative_signal", "postoperative",
                     "tear", "macerated"]},
    "lm": {"finding": "lateral meniscus",
           "terms": ["intact", "degenerative_signal", "postoperative",
                     "tear", "macerated"]},
    "moa": {"finding": "medial compartment osteoarthritis",
            "terms": ["none", "mild", "moderate", "severe"]},
    "loa": {"finding": "lateral compartment osteoarthritis",
            "terms": ["none", "mild", "moderate", "severe"]},
    "pfoa": {"finding": "patellofemoral osteoarthritis",
            "terms": ["none", "mild", "moderate", "severe"]},
    "effusion": {"finding": "joint effusion",
            "terms": ["none", "small", "moderate", "large"]},
    "synovitis": {"finding": "synovitis",
            "terms": ["none", "mild", "moderate", "severe", "suspected"]},
    "bakers": {"finding": "Baker's cyst",
            "terms": ["none", "present", "ruptured"]},
    "contusion": {"finding": "bone contusion",
            "terms": ["none", "present"]},
    "fracture": {"finding": "fracture",
            "terms": ["none", "present", "suspected"]},
}


def build_prompt(report):
    lines = [
        "You are a musculoskeletal radiologist. Read the knee MRI report below",
        "(it may be in English, Spanish, German, Portuguese, or French).",
        "For each of the 12 findings, output EXACTLY ONE term from the allowed list.",
        "If the report does not address a finding, use the most neutral term",
        "(typically 'intact' or 'none').",
        "",
        "IMPORTANT - extract SEVERITY where the report provides it:",
        "- For OA (moa/loa/pfoa): use none/mild/moderate/severe based on described severity",
        "- For synovitis: use none/mild/moderate/severe; include Hoffa fat pad impingement",
        "  and plica syndrome as synovitis",
        "- For effusion: count even 'trace' amounts as small (not none)",
        "- For contusion: only mark present for traumatic bone marrow edema pattern,",
        "  not ordinary degenerative changes",
        "- For fracture: include avulsion and insufficiency fractures",
        "",
        "Output format: 12 lines, each 'key: term' (e.g. 'acl: partial_tear').",
        "No other text.",
        "",
    ]
    for k in KEYS:
        v = VOCAB[k]
        lines.append(f"{k} ({v['finding']}): {', '.join(v['terms'])}")
    lines += ["", "REPORT:", report, "", "LABELS:"]
    return "\n".join(lines)


def parse_response(text):
    out = {}
    for line in text.strip().split("\n"):
        m = re.match(r"^\s*(\w+)\s*:\s*([\w_/]+)\s*$", line)
        if m:
            k, v = m.group(1).lower(), m.group(2).lower()
            if k in KEYS:
                # canonicalize
                v = v.replace("-", "_").replace(" ", "_").replace("/", "_")
                if v in VOCAB[k]["terms"]:
                    out[k] = v
    return out


class KeyRotator:
    def __init__(self, keys):
        self.keys = keys
        self._cycle = itertools.cycle(range(len(keys)))
        self.exhausted = set()
        self.calls = [0] * len(keys)

    def next(self):
        for _ in range(len(self.keys)):
            i = next(self._cycle)
            if i not in self.exhausted:
                return i, self.keys[i]
        return None, None

    def mark_exhausted(self, i):
        self.exhausted.add(i)

    @property
    def live(self):
        return len(self.keys) - len(self.exhausted)


def gemini_chat(prompt, key, model):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    # Model-aware config: lite models reject temperature/thinkingConfig
    gen_config = {"maxOutputTokens": 2048}
    if "lite" not in model:
        gen_config["temperature"] = 0.0
        # 3.8 Flash may not support thinkingBudget=0; try without first
        # (add only if needed)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                data = json.load(resp)
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            if not text.strip():
                raise ValueError("empty response text")
            return text
        except urllib.error.HTTPError as e:
            eb = e.read().decode()[:400]
            last_err = f"HTTP {e.code}: {eb}"
            if e.code in (429, 500, 502, 503):
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"Gemini {last_err}")
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gemini failed after 3 attempts: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="train.csv with reports")
    ap.add_argument("--model", default="gemini-3.8-flash",
                    help="gemini-3.8-flash or gemini-3.8-pro")
    args = ap.parse_args()

    keys = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        print("ERROR: set GEMINI_API_KEYS env var (comma-separated)", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    records_path = os.path.join(OUT_DIR, "records.jsonl")

    # Load already-done UIDs for resume
    done = set()
    if os.path.exists(records_path):
        with open(records_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:
                    pass
    print(f"Resuming: {len(done)} already done", flush=True)

    df = pd.read_csv(args.csv)
    # Get unique reports (StudyInstanceUID + report text)
    # train.csv has one row per study; report column name may vary
    report_col = None
    for c in df.columns:
        if "report" in c.lower() or "text" in c.lower() or "finding" in c.lower():
            report_col = c
            break
    if report_col is None:
        # Try common names
        for c in ["Report", "report_text", "Findings", "findings"]:
            if c in df.columns:
                report_col = c
                break
    if report_col is None:
        print(f"ERROR: cannot find report column. Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"Using report column: {report_col}", flush=True)
    print(f"Total studies: {len(df)}", flush=True)

    rotator = KeyRotator(keys)
    fout = open(records_path, "a")
    n_new = 0
    n_fail = 0

    for idx, row in df.iterrows():
        uid = str(row["StudyInstanceUID"])
        if uid in done:
            continue
        report = str(row[report_col])
        if not report or report == "nan":
            # Empty report -> all neutral
            labels = {k: 0.5 for k in KEYS}
        else:
            prompt = build_prompt(report)
            try:
                # Rotate keys
                text = None
                tried = set()
                while len(tried) < rotator.live:
                    i, key = rotator.next()
                    if i is None or i in tried:
                        break
                    tried.add(i)
                    try:
                        text = gemini_chat(prompt, key, args.model)
                        rotator.calls[i] += 1
                        break
                    except RuntimeError as e:
                        msg = str(e)
                        if any(s in msg for s in ("429", "quota", "RESOURCE_EXHAUSTED", "rate")):
                            print(f"  key {i+1} hit limit, rotating...", flush=True)
                            rotator.mark_exhausted(i)
                            continue
                        raise
                if text is None:
                    raise RuntimeError("all keys exhausted")

                parsed = parse_response(text)
                # Convert terms to binary labels
                labels = {}
                for k in KEYS:
                    term = parsed.get(k)
                    if term and term in TERM_TO_LABEL[k]:
                        labels[k] = TERM_TO_LABEL[k][term]
                    else:
                        labels[k] = 0.5  # abstain if parse failed
            except Exception as e:
                print(f"  FAILED {uid}: {e} — will retry on next run", flush=True)
                n_fail += 1
                continue  # Don't mark as done; retry on next run

        rec = {"uid": uid, "labels": labels, "model": args.model}
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        done.add(uid)
        n_new += 1
        if n_new % 10 == 0:
            print(f"  {n_new} new ({len(done)} total), {n_fail} failed, keys live: {rotator.live}/{len(keys)}", flush=True)

    fout.close()

    # Write final CSV
    rows = []
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            row = {"StudyInstanceUID": r["uid"]}
            for lk, lbl in zip(LABELS, KEYS):
                row[lk] = r["labels"].get(lbl, 0.5)
            rows.append(row)
    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, f"gemini_{args.model.replace('-', '_')}_labels.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nDone: {len(rows)} studies -> {out_csv}", flush=True)
    print(f"Failed: {n_fail}", flush=True)


if __name__ == "__main__":
    main()
