#!/usr/bin/env python3
"""Phase 1 FINISH via Mac Ollama (qwen3:14b): label the remaining ~596 reports
that OpenAI credits + ChatAnywhere points ran out on, then merge all three
sources into the final weak_labels_v1.csv.

- Reads ./phase1_openai_out/records.jsonl (3,320 GPT-4.1 via OpenAI)
- Reads ./phase1_hybrid_out/ca_records.jsonl (360 GPT-4.1 via ChatAnywhere)
- Labels the remaining ~596 via Ollama qwen3:14b (same prompt/parsing)
- Resume-safe: Mac progress in ./phase1_final_out/mac_records.jsonl
- Final merged outputs in ./phase1_final_out/

Usage:
    python3 phase1_mac_finish.py --csv ./train.csv
Requires: ollama serve running, qwen3:14b pulled.
~596 reports at ~7.4s each = ~75 min.
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

MODEL = "qwen3:14b"
OLLAMA_URL = "http://localhost:11434/api/chat"
BATCH = 5
TIMEOUT_S = 300
OPENAI_RECORDS = "./phase1_openai_out/records.jsonl"
CA_RECORDS = "./phase1_hybrid_out/ca_records.jsonl"
OUT_DIR = "./phase1_final_out"

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


def ollama_chat(prompt):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192},
    }
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                data = json.load(resp)
            content = data.get("message", {}).get("content", "")
            if not content.strip():
                content = data.get("message", {}).get("thinking", "")
            if not content.strip():
                raise ValueError("empty response content")
            return content
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ollama failed after 3 tries: {last}")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./train.csv")
    ap.add_argument("--openai-records", default=OPENAI_RECORDS)
    ap.add_argument("--ca-records", default=CA_RECORDS)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # load prior progress (only successful parses)
    prior_recs = {}
    for path, src in [(args.openai_records, "openai"),
                      (args.ca_records, "chatanywhere")]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("parse_ok"):
                        r["source"] = src
                        prior_recs[r["hash"]] = r
    print(f"prior done: {len(prior_recs)} reports", flush=True)

    df = pd.read_csv(args.csv, encoding="latin-1")
    df["_hash"] = df["Report"].map(
        lambda r: hashlib.sha256(str(r).encode("utf-8", "replace")).hexdigest()[:16])
    uniq = df.drop_duplicates("_hash").set_index("_hash")["Report"].to_dict()

    gold_mask = df[LABELS].notna().all(axis=1)
    gold_df = df[gold_mask]
    priors = {k: float(gold_df[lab].mean()) for k, lab in zip(KEYS, LABELS)}

    mac_path = os.path.join(args.out, "mac_records.jsonl")
    mac_recs = {}
    if os.path.exists(mac_path):
        with open(mac_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("parse_ok"):
                    mac_recs[r["hash"]] = r
    print(f"mac done: {len(mac_recs)} reports", flush=True)

    done = set(prior_recs) | set(mac_recs)
    targets = [h for h in uniq if h not in done]
    print(f"todo: {len(targets)} reports (~{len(targets)*7.4/60:.0f} min)",
          flush=True)

    t0 = time.time()
    n_fail = 0
    batches = [targets[i:i + BATCH] for i in range(0, len(targets), BATCH)]
    with open(mac_path, "a") as fout:
        for bi, bh in enumerate(batches):
            try:
                text = ollama_chat(build_batch_prompt([uniq[h] for h in bh]))
                parsed = parse_batch_output(text, len(bh))
            except Exception as e:
                print(f"  batch {bi+1} ERROR: {e}", flush=True)
                parsed = [None] * len(bh)
            for i, (h, t) in enumerate(zip(bh, parsed)):
                if t is None:  # single fallback
                    try:
                        t2 = ollama_chat(build_batch_prompt([uniq[h]]))
                        parsed[i] = parse_batch_output(t2, 1)[0]
                    except Exception:
                        pass
            for h, t in zip(bh, parsed):
                rec = {"hash": h, "terms": t, "parse_ok": t is not None,
                       "source": "mac"}
                if t is not None:
                    mac_recs[h] = rec
                else:
                    n_fail += 1
                fout.write(json.dumps(rec) + "\n")
            fout.flush()
            el = time.time() - t0
            print(f"  batch {bi+1}/{len(batches)} | {el:.0f}s | fails: {n_fail}",
                  flush=True)

    # ---- merge all three sources ----
    print("\nmerging...", flush=True)
    combined = dict(prior_recs)
    for h, r in mac_recs.items():
        combined[h] = r
    print(f"combined: {len(combined)}/{len(uniq)}", flush=True)
    missing = [h for h in uniq if h not in combined]
    if missing:
        print(f"WARNING: {len(missing)} still missing", flush=True)

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
            r = combined.get(h, {"terms": None, "source": "missing"})
            f.write(json.dumps({"hash": h, "terms": r.get("terms"),
                                "source": r.get("source")}) + "\n")
    src_counts = {}
    for r in combined.values():
        s = r.get("source", "?")
        src_counts[s] = src_counts.get(s, 0) + 1
    manifest = {
        "sources": src_counts,
        "probe_macro_auc_gpt41": 0.8473,
        "probe_macro_auc_mac": 0.7530,
        "n_reports": len(uniq),
        "n_studies": len(df),
        "n_missing": len(missing),
        "calibration": "rank_normalized_no_isotonic",
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1)
    print(f"wrote {args.out}/weak_labels_v1.csv ({len(rows)} rows)", flush=True)
    print(f"sources: {src_counts}", flush=True)


if __name__ == "__main__":
    main()
