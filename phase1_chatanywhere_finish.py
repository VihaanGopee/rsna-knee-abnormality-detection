#!/usr/bin/env python3
"""Phase 1 FINISH via ChatAnywhere (gpt-4.1): label the 956 reports that
OpenAI credits ran out on, then merge with the 3,320 OpenAI GPT-4.1 labels.

- Reads ./phase1_openai_out/records.jsonl (3,320 done)
- Labels the remaining 956 via ChatAnywhere gpt-4.1 (same prompt/parsing)
- Resume-safe: ChatAnywhere progress in ./phase1_hybrid_out/ca_records.jsonl
- Final merged outputs in ./phase1_hybrid_out/: weak_labels_v1.csv (4,407 rows),
  descriptors.jsonl (4,276 unique), manifest.json

Note: ChatAnywhere free tier = 100 req/day. 956 reports = ~192 calls.
Run day 1 (~100 calls), rerun day 2 to finish. Resume picks up automatically.

Usage:
    export CHATANYWHERE_API_KEY="sk-..."
    python3 phase1_chatanywhere_finish.py --csv ./train.csv
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

import pandas as pd

MODEL = "gpt-4.1"
API_URL = "https://api.chatanywhere.tech/v1/chat/completions"
BATCH = 5
TIMEOUT_S = 120
OPENAI_RECORDS = "./phase1_openai_out/records.jsonl"
OUT_DIR = "./phase1_hybrid_out"

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "mm", "lm", "moa", "loa", "pfoa", "effusion",
        "synovitis", "bakers", "contusion", "fracture"]

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
                 "terms": ["none", "trace_small", "moderate", "large"]},
    "synovitis": {"finding": "synovitis",
                  "terms": ["none", "mild", "moderate", "severe"]},
    "bakers": {"finding": "Baker's cyst",
               "terms": ["none", "small", "moderate", "large"]},
    "contusion": {"finding": "bone contusion / bone marrow edema",
                  "terms": ["none", "mild", "moderate", "severe"]},
    "fracture": {"finding": "fracture",
                 "terms": ["none", "nondisplaced", "displaced"]},
}
RANK = {k: {t: i for i, t in enumerate(v["terms"])}
        for k, v in VOCAB.items()}


def vocab_block():
    lines = []
    for k in KEYS:
        v = VOCAB[k]
        lines.append(f"- {k} ({v['finding']}): "
                     + " | ".join(v["terms"]) + " | not_mentioned")
    return "\n".join(lines)


def build_batch_prompt(reports):
    n = len(reports)
    body = []
    for i, r in enumerate(reports, 1):
        body.append(f'Report {i}:\n"""{r}"""')
    return f"""You are a musculoskeletal radiology reader. Read each knee MRI report below. Reports may be in Spanish, French, English, German, or another language — read them regardless of language.

For each report and each of the 12 findings, choose the SINGLE allowed term that best matches what the report STATES. Rules:
- If the report does not mention the finding at all, use "not_mentioned".
- Do NOT infer findings that are not stated.
- Negated findings ("no tear", "sin rotura", "pas de déchirure", "kein Riss") map to the intact/absent/none term.
- Uncertain/hedged language ("possible", "cannot exclude", "suggestive of") still maps to the matching positive term.
- If a finding is present without a grade, use the mildest positive term.

Findings and allowed terms:
{vocab_block()}

Output exactly {n} lines, one per report, in order. Each line: 12 comma-separated terms in this exact order:
acl,mcl,mm,lm,moa,loa,pfoa,effusion,synovitis,bakers,contusion,fracture
No other text. No explanations. No markdown.

""" + "\n\n".join(body)


def ca_chat(prompt, api_key):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    last_err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                API_URL, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}",
                         "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
                method="POST")
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                data = json.load(resp)
            return (data["choices"][0]["message"]["content"],
                    data.get("usage", {}))
        except urllib.error.HTTPError as e:
            eb = e.read().decode()[:300]
            last_err = f"HTTP {e.code}: {eb}"
            if e.code == 429:
                wait = 60 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"ChatAnywhere {last_err}")
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ChatAnywhere failed after 4 attempts: {last_err}")


def canon_term(key, raw):
    t = raw.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    t = re.sub(r"_+", "_", t).strip("_").strip("`").strip('"').strip("'")
    if t in VOCAB[key]["terms"] or t == "not_mentioned":
        return t
    aliases = {"rupture": "tear", "torn": "tear", "normal": "intact",
               "absent": "none", "no": "none"}
    if t in aliases:
        a = aliases[t]
        if a in VOCAB[key]["terms"]:
            return a
    return None


def parse_line(line):
    parts = [p for p in line.strip().strip("`").split(",") if p.strip()]
    if len(parts) != 12:
        return None
    terms = {}
    for k, raw in zip(KEYS, parts):
        c = canon_term(k, raw)
        terms[k] = c if c is not None else "not_mentioned"
    return terms


def parse_batch_output(text, n):
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("```")]
    out = []
    for ln in lines[:n]:
        out.append(parse_line(ln))
    while len(out) < n:
        out.append(None)
    return out


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./train.csv")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--openai-records", default=OPENAI_RECORDS)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    MODEL = args.model

    api_key = os.environ.get("CHATANYWHERE_API_KEY", "").strip()
    if not api_key:
        print('ERROR: export CHATANYWHERE_API_KEY="sk-..." first')
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)

    # load OpenAI progress
    openai_recs = {}
    if os.path.exists(args.openai_records):
        with open(args.openai_records) as f:
            for line in f:
                r = json.loads(line)
                # only keep successful parses; failed (no-credit) ones get redone
                if r.get("parse_ok"):
                    openai_recs[r["hash"]] = r
    print(f"OpenAI done: {len(openai_recs)} reports", flush=True)

    df = pd.read_csv(args.csv, encoding="latin-1")
    df["_hash"] = df["Report"].map(
        lambda r: hashlib.sha256(str(r).encode("utf-8", "replace")).hexdigest()[:16])
    uniq = df.drop_duplicates("_hash").set_index("_hash")["Report"].to_dict()
    print(f"unique reports: {len(uniq)}", flush=True)

    gold_mask = df[LABELS].notna().all(axis=1)
    gold_df = df[gold_mask]
    priors = {k: float(gold_df[lab].mean()) for k, lab in zip(KEYS, LABELS)}

    # ChatAnywhere resume
    ca_path = os.path.join(args.out, "ca_records.jsonl")
    ca_recs = {}
    if os.path.exists(ca_path):
        with open(ca_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("parse_ok"):
                    ca_recs[r["hash"]] = r
    print(f"ChatAnywhere done: {len(ca_recs)} reports", flush=True)

    done_hashes = set(openai_recs) | set(ca_recs)
    targets = [h for h in uniq if h not in done_hashes]
    print(f"todo: {len(targets)} reports (~{len(targets)//BATCH+1} API calls)",
          flush=True)
    if len(targets) > 500:
        print("NOTE: free tier is ~100 req/day — expect to resume tomorrow.",
              flush=True)

    # ---- label missing via ChatAnywhere ----
    t0 = time.time()
    n_fail = 0
    batches = [targets[i:i + BATCH] for i in range(0, len(targets), BATCH)]
    with open(ca_path, "a") as fout:
        for bi, bh in enumerate(batches):
            prompt = build_batch_prompt([uniq[h] for h in bh])
            try:
                text, _ = ca_chat(prompt, api_key)
                parsed = parse_batch_output(text, len(bh))
            except Exception as e:
                print(f"  batch {bi+1} ERROR: {e}", flush=True)
                parsed = [None] * len(bh)
            # single fallback for failures
            for i, (h, t) in enumerate(zip(bh, parsed)):
                if t is None:
                    try:
                        t2, _ = ca_chat(build_batch_prompt([uniq[h]]), api_key)
                        parsed[i] = parse_batch_output(t2, 1)[0]
                    except Exception:
                        pass
            for h, t in zip(bh, parsed):
                rec = {"hash": h, "terms": t, "parse_ok": t is not None,
                       "source": "chatanywhere"}
                if t is not None:
                    ca_recs[h] = rec
                else:
                    n_fail += 1
                fout.write(json.dumps(rec) + "\n")
            fout.flush()
            el = time.time() - t0
            print(f"  batch {bi+1}/{len(batches)} | {el:.0f}s | fails: {n_fail}",
                  flush=True)

    # ---- merge + outputs ----
    print("\nmerging...", flush=True)
    combined = dict(openai_recs)
    for h, r in ca_recs.items():
        combined[h] = r
    print(f"combined: {len(combined)}/{len(uniq)} unique reports", flush=True)
    missing = [h for h in uniq if h not in combined]
    if missing:
        print(f"WARNING: {len(missing)} reports still missing!", flush=True)

    rows = []
    for _, row in df.iterrows():
        rec = combined.get(row["_hash"], {})
        terms = rec.get("terms") or {}
        out = {"StudyInstanceUID": row["StudyInstanceUID"]}
        for k, lab in zip(KEYS, LABELS):
            t = terms.get(k)
            rank = RANK[k].get(t) if t else None
            out[lab] = (rank / (len(VOCAB[k]["terms"]) - 1)
                        if rank is not None else priors[k])
        rows.append(out)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "weak_labels_v1.csv"),
                              index=False)
    with open(os.path.join(args.out, "descriptors.jsonl"), "w") as f:
        for h in uniq:
            r = combined.get(h, {"terms": None})
            f.write(json.dumps({"hash": h, "terms": r.get("terms")}) + "\n")
    n_openai = sum(1 for r in combined.values()
                   if r.get("source") != "chatanywhere")
    manifest = {
        "model": f"gpt-4.1 (openai:{n_openai} + chatanywhere:{len(ca_recs)})",
        "probe_macro_auc": 0.8473,
        "n_reports": len(uniq),
        "n_studies": len(df),
        "n_missing": len(missing),
        "calibration": "rank_normalized_no_isotonic",
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1)
    print(f"wrote {args.out}/weak_labels_v1.csv ({len(rows)} rows), "
          f"descriptors.jsonl, manifest.json", flush=True)


if __name__ == "__main__":
    main()
