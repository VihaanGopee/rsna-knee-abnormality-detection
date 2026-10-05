#!/usr/bin/env python3
"""Phase 1 FINISH via Gemini 3.8 Flash: label the remaining ~222 reports that
OpenAI credits + ChatAnywhere points ran out on, then merge all three
sources into the final weak_labels_v1.csv.

- Reads ./phase1_openai_out/records.jsonl (3,320 GPT-4.1 via OpenAI)
- Reads ./phase1_hybrid_out/ca_records.jsonl (734 GPT-4.1 via ChatAnywhere)
- Labels the remaining ~222 via Gemini 3.8 Flash (same prompt/parsing)
- Rotates across multiple API keys to stay under free-tier limits
- Resume-safe: Gemini progress in ./phase1_final_out/gemini_records.jsonl
- Final merged outputs in ./phase1_final_out/

Usage:
    export GEMINI_API_KEYS="key1,key2,key3,key4"   # comma-separated, no spaces
    python3 phase1_gemini_finish.py --csv ./train.csv

~222 reports / 5 per batch = ~45 API calls. With 4 keys at ~20 req/day each
(80/day total), this finishes in one run.
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

MODEL = "gemini-3.8-flash"
BATCH = 5
TIMEOUT_S = 120
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


class KeyRotator:
    """Round-robin across API keys; marks a key exhausted on 429/quota."""

    def __init__(self, keys):
        self.keys = keys
        self.exhausted = set()
        self._cycle = itertools.cycle(range(len(keys)))
        self.calls = {i: 0 for i in range(len(keys))}

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


def gemini_chat(prompt, key):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODEL}:generateContent?key={key}")
    # Lite models (gemini-*-flash-lite) reject temperature and numeric
    # thinking budgets — strip them, keep only maxOutputTokens.
    gen_config = {"maxOutputTokens": 2048}
    if "lite" not in MODEL:
        gen_config["temperature"] = 0.0
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}
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


def gemini_call(prompt, rotator):
    """Try keys in rotation; returns text or raises if all exhausted."""
    tried = set()
    while len(tried) < rotator.live:
        i, key = rotator.next()
        if i is None or i in tried:
            break
        tried.add(i)
        try:
            text = gemini_chat(prompt, key)
            rotator.calls[i] += 1
            return text
        except RuntimeError as e:
            msg = str(e)
            # quota / rate-limit signals -> try next key
            if any(s in msg for s in ("429", "quota", "RESOURCE_EXHAUSTED",
                                      "rate", "RATE_LIMIT")):
                print(f"    key {i+1} hit limit, rotating...", flush=True)
                rotator.mark_exhausted(i)
                continue
            raise
    raise RuntimeError("all Gemini keys exhausted or errored")


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
    ap.add_argument("--ca-records", default=CA_RECORDS)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    MODEL = args.model
    print(f"model: {MODEL}", flush=True)

    raw_keys = os.environ.get("GEMINI_API_KEYS", "").strip()
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        print('ERROR: export GEMINI_API_KEYS="key1,key2,key3,key4" first '
              '(comma-separated, no spaces)')
        sys.exit(1)
    print(f"using {len(keys)} Gemini API keys", flush=True)
    rotator = KeyRotator(keys)
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

    gem_path = os.path.join(args.out, "gemini_records.jsonl")
    gem_recs = {}
    if os.path.exists(gem_path):
        with open(gem_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("parse_ok"):
                    gem_recs[r["hash"]] = r
    print(f"gemini done: {len(gem_recs)} reports", flush=True)

    done = set(prior_recs) | set(gem_recs)
    targets = [h for h in uniq if h not in done]
    print(f"todo: {len(targets)} reports (~{len(targets)//BATCH+1} API calls)",
          flush=True)

    t0 = time.time()
    n_fail = 0
    batches = [targets[i:i + BATCH] for i in range(0, len(targets), BATCH)]
    with open(gem_path, "a") as fout:
        for bi, bh in enumerate(batches):
            try:
                text = gemini_call(build_batch_prompt([uniq[h] for h in bh]),
                                   rotator)
                parsed = parse_batch_output(text, len(bh))
            except Exception as e:
                print(f"  batch {bi+1} ERROR: {e}", flush=True)
                parsed = [None] * len(bh)
            for i, (h, t) in enumerate(zip(bh, parsed)):
                if t is None:  # single fallback
                    try:
                        t2 = gemini_call(build_batch_prompt([uniq[h]]),
                                         rotator)
                        parsed[i] = parse_batch_output(t2, 1)[0]
                    except Exception:
                        pass
            for h, t in zip(bh, parsed):
                rec = {"hash": h, "terms": t, "parse_ok": t is not None,
                       "source": "gemini"}
                if t is not None:
                    gem_recs[h] = rec
                else:
                    n_fail += 1
                fout.write(json.dumps(rec) + "\n")
            fout.flush()
            el = time.time() - t0
            print(f"  batch {bi+1}/{len(batches)} | {el:.0f}s | fails: {n_fail} "
                  f"| keys live: {rotator.live}/{len(keys)}", flush=True)
            time.sleep(2)  # gentle pacing for free tier

    # ---- merge all sources ----
    print("\nmerging...", flush=True)
    combined = dict(prior_recs)
    for h, r in gem_recs.items():
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
        "gemini_model": MODEL,
        "gemini_quality_note": "unvalidated vs gold; 222/4276 reports (~5%)",
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
