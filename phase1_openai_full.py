#!/usr/bin/env python3
"""Phase 1 FULL run via OpenAI API (gpt-4.1): label all 4,276 unique reports.

Probe-validated: macro AUC 0.8473, 3% parse failures (vs Mac 0.753/5%).

- Batches 5 reports/call; single-report fallback for batch parse failures
- Resume-safe: incremental JSONL checkpoint, safe to interrupt and rerun
- Rank-normalized scores (rank/max_rank); abstain/not_mentioned -> gold prior
  (NO isotonic calibration — probe showed it hurts: 0.753 -> 0.738 on Mac)
- Outputs in ./phase1_openai_out/: weak_labels_v1.csv, descriptors.jsonl,
  manifest.json

Usage:
    export OPENAI_API_KEY="sk-..."
    python3 phase1_openai_full.py --csv ./train.csv [--model gpt-4.1]

Estimated: ~856 API calls, ~2.7M input tokens, ~$6-7 at gpt-4.1 rates.
Takes ~30-60 min depending on rate limits.
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
API_URL = "https://api.openai.com/v1/chat/completions"
BATCH = 5
TIMEOUT_S = 120
OUT_DIR = "./phase1_openai_out"

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

RATES = {  # $ per 1M tokens (approximate — verify current pricing)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
}


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


def openai_chat(prompt, api_key):
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
                         "Authorization": f"Bearer {api_key}"},
                method="POST")
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                data = json.load(resp)
            return (data["choices"][0]["message"]["content"],
                    data.get("usage", {}))
        except urllib.error.HTTPError as e:
            eb = e.read().decode()[:300]
            last_err = f"HTTP {e.code}: {eb}"
            if e.code == 429:
                wait = 10 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenAI {last_err}")
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"OpenAI failed after 4 attempts: {last_err}")


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


def label_batch(items, api_key, usage):
    """items: [(hash, report)]. Returns [(hash, terms_or_None)]. Falls back
    to single-report calls for batch items that fail to parse."""
    hashes = [h for h, _ in items]
    prompt = build_batch_prompt([r for _, r in items])
    text, u = openai_chat(prompt, api_key)
    usage["in"] += u.get("prompt_tokens", 0)
    usage["out"] += u.get("completion_tokens", 0)
    parsed = parse_batch_output(text, len(items))
    results = []
    for h, terms in zip(hashes, parsed):
        results.append([h, terms])
    # single-report fallback for failures
    failed_idx = [i for i, (_, t) in enumerate(results) if t is None]
    for i in failed_idx:
        h, r = items[i]
        try:
            t2, u2 = openai_chat(build_batch_prompt([r]), api_key)
            usage["in"] += u2.get("prompt_tokens", 0)
            usage["out"] += u2.get("completion_tokens", 0)
            p2 = parse_batch_output(t2, 1)[0]
            results[i][1] = p2  # may still be None -> abstain
        except Exception as e:
            print(f"    single fallback failed for {h[:8]}: {e}", flush=True)
    return [(h, t) for h, t in results]


def main():
    global MODEL, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./train.csv")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    MODEL = args.model
    OUT_DIR = args.out

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print('ERROR: export OPENAI_API_KEY="sk-..." first')
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(args.csv, encoding="latin-1")
    df["_hash"] = df["Report"].map(
        lambda r: hashlib.sha256(str(r).encode("utf-8", "replace")).hexdigest()[:16])
    uniq = df.drop_duplicates("_hash").set_index("_hash")["Report"].to_dict()
    print(f"studies: {len(df)} | unique reports: {len(uniq)}", flush=True)

    gold_mask = df[LABELS].notna().all(axis=1)
    gold_df = df[gold_mask]
    print(f"gold studies: {len(gold_df)}", flush=True)
    priors = {k: float(gold_df[lab].mean()) for k, lab in zip(KEYS, LABELS)}

    # resume
    done_path = os.path.join(OUT_DIR, "records.jsonl")
    rec_by_hash = {}
    if os.path.exists(done_path):
        with open(done_path) as f:
            for line in f:
                r = json.loads(line)
                rec_by_hash[r["hash"]] = r
        print(f"resuming: {len(rec_by_hash)} already done", flush=True)
    targets = [h for h in uniq if h not in rec_by_hash]
    print(f"todo: {len(targets)} reports", flush=True)
    r_in, r_out = RATES.get(MODEL, (2.00, 8.00))
    print(f"estimated cost for remaining: "
          f"~${len(targets)*622/1e6*r_in + len(targets)*35/1e6*r_out:.2f}",
          flush=True)

    usage = {"in": 0, "out": 0}
    n_fail = 0
    t0 = time.time()
    batches = [targets[i:i + BATCH] for i in range(0, len(targets), BATCH)]
    with open(done_path, "a") as fout:
        for bi, bh in enumerate(batches):
            try:
                res = label_batch([(h, uniq[h]) for h in bh], api_key, usage)
            except Exception as e:
                print(f"  batch {bi+1} ERROR: {e} — marking failed",
                      flush=True)
                res = [(h, None) for h in bh]
            for h, terms in res:
                rec = {"hash": h, "terms": terms,
                       "parse_ok": terms is not None}
                rec_by_hash[h] = rec
                fout.write(json.dumps(rec) + "\n")
                if terms is None:
                    n_fail += 1
            fout.flush()
            el = time.time() - t0
            ndone = len(rec_by_hash)
            cost = usage["in"] / 1e6 * r_in + usage["out"] / 1e6 * r_out
            print(f"  batch {bi+1}/{len(batches)} | {ndone}/{len(uniq)} "
                  f"| {el:.0f}s | fails: {n_fail} | "
                  f"tok {usage['in']}/{usage['out']} | ~${cost:.2f}",
                  flush=True)

    # ---- outputs ----
    print("\nbuilding outputs...", flush=True)
    rows = []
    for _, row in df.iterrows():
        rec = rec_by_hash[row["_hash"]]
        terms = rec["terms"] or {}
        out = {"StudyInstanceUID": row["StudyInstanceUID"]}
        for k, lab in zip(KEYS, LABELS):
            t = terms.get(k)
            rank = RANK[k].get(t) if t else None
            if rank is None:
                out[lab] = priors[k]  # abstain/not_mentioned -> gold prior
            else:
                out[lab] = rank / (len(VOCAB[k]["terms"]) - 1)
        rows.append(out)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "weak_labels_v1.csv"),
                              index=False)
    with open(os.path.join(OUT_DIR, "descriptors.jsonl"), "w") as f:
        for h in uniq:
            r = rec_by_hash[h]
            f.write(json.dumps({"hash": h, "terms": r["terms"]}) + "\n")
    total_cost = usage["in"] / 1e6 * r_in + usage["out"] / 1e6 * r_out
    manifest = {
        "model": MODEL,
        "probe_macro_auc": 0.8473,
        "probe_parse_fail_rate": 0.03,
        "parse_fail_rate": n_fail / max(len(rec_by_hash), 1),
        "n_reports": len(uniq),
        "n_studies": len(df),
        "tokens_in": usage["in"],
        "tokens_out": usage["out"],
        "approx_cost_usd": round(total_cost, 2),
        "calibration": "rank_normalized_no_isotonic",
    }
    json.dump(manifest, open(os.path.join(OUT_DIR, "manifest.json"), "w"),
              indent=1)
    print(f"wrote {OUT_DIR}/weak_labels_v1.csv, descriptors.jsonl, manifest.json",
          flush=True)
    print(f"total cost this run: ~${total_cost:.2f}", flush=True)


if __name__ == "__main__":
    main()
