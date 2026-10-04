#!/usr/bin/env python3
"""Phase 1 probe via ChatAnywhere API: quality check for remaining 956 reports.

Same 100 reports, same prompt, same rank-AUC vs 58 gold as the OpenAI probe.
Tests gpt-4o-mini (120B open-weight) as a cheap alternative to
finish the 956 reports that lost OpenAI credits.

Cost: ~$0.01 for the 100-report probe.

Usage:
    export CHATANYWHERE_API_KEY="gsk-..."
    python3 phase1_chatanywhere_probe.py --csv ./train.csv [--model gpt-4o-mini]

ChatAnywhere API is OpenAI-compatible: https://api.chatanywhere.tech/v1/chat/completions
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error

import pandas as pd

MODEL = "gpt-4o-mini"
API_URL = "https://api.chatanywhere.tech/v1/chat/completions"
BATCH = 5
TIMEOUT_S = 120

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
ALLOWED = {k: set(v["terms"]) | {"not_mentioned"} for k, v in VOCAB.items()}

RATES = {  # $ per 1M tokens (approximate)
    "gpt-4o-mini": (0.15, 0.60),
    
    
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


def groq_chat(prompt, api_key):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
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
                wait = 10 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Groq {last_err}")
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"ChatAnywhere failed after 4 attempts: {last_err}")


def parse_batch(text, n):
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("```")]
    out = []
    for ln in lines[:n]:
        parts = [p.strip().strip('"').strip("'") for p in ln.split(",")]
        if len(parts) != 12:
            out.append(None)
            continue
        terms = {}
        ok = True
        for k, p in zip(KEYS, parts):
            if p not in ALLOWED[k]:
                ok = False
                break
            terms[k] = p
        out.append(terms if ok else None)
    while len(out) < n:
        out.append(None)
    return out


def auc_score(y_true, y_score):
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return None
    n1, n0 = len(pos), len(neg)
    alls = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    ranks, i = [], 0
    while i < len(alls):
        j = i
        while j < len(alls) and alls[j][0] == alls[i][0]:
            j += 1
        avg = (i + 1 + j) / 2
        ranks.extend([avg] * (j - i))
        i = j
    r1 = sum(r for (_, t), r in zip(alls, ranks) if t == 1)
    return (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./train.csv")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    MODEL = args.model

    api_key = os.environ.get("CHATANYWHERE_API_KEY", "").strip()
    if not api_key:
        print('ERROR: export CHATANYWHERE_API_KEY="gsk-..." first')
        sys.exit(1)

    df = pd.read_csv(args.csv, encoding="latin-1")
    df["_hash"] = df["Report"].map(
        lambda r: hashlib.sha256(str(r).encode("utf-8", "replace")).hexdigest()[:16])
    uniq = df.drop_duplicates("_hash").set_index("_hash")["Report"].to_dict()
    print(f"studies: {len(df)} | unique reports: {len(uniq)}", flush=True)

    gold_mask = df[LABELS].notna().all(axis=1)
    gold_df = df[gold_mask]
    gold_hashes = gold_df["_hash"].tolist()
    print(f"gold studies: {len(gold_df)}", flush=True)

    random.seed(0)
    extra = [h for h in uniq if h not in set(gold_hashes)]
    targets = gold_hashes + random.sample(extra, min(42, len(extra)))
    print(f"probe: {len(targets)} reports (same seed)", flush=True)

    rec_by_hash = {}
    n_fail = 0
    tot_in, tot_out = 0, 0
    t0 = time.time()
    batches = [targets[i:i + BATCH] for i in range(0, len(targets), BATCH)]
    for bi, bh in enumerate(batches):
        prompt = build_batch_prompt([uniq[h] for h in bh])
        try:
            text, usage = groq_chat(prompt, api_key)
        except Exception as e:
            print(f"  batch {bi+1} ERROR: {e}", flush=True)
            for h in bh:
                rec_by_hash[h] = {"terms": None, "parse_ok": False}
            n_fail += len(bh)
            continue
        tot_in += usage.get("prompt_tokens", 0)
        tot_out += usage.get("completion_tokens", 0)
        parsed = parse_batch(text, len(bh))
        for h, terms in zip(bh, parsed):
            ok = terms is not None
            rec_by_hash[h] = {"terms": terms, "parse_ok": ok}
            if not ok:
                n_fail += 1
        el = time.time() - t0
        print(f"  batch {bi+1}/{len(batches)} | {el:.0f}s | "
              f"fails: {n_fail} | tok in/out: {tot_in}/{tot_out}", flush=True)

    aucs = {}
    for k, lab in zip(KEYS, LABELS):
        scores, trues = [], []
        max_rank = len(VOCAB[k]["terms"]) - 1
        for _, row in gold_df.iterrows():
            rec = rec_by_hash.get(row["_hash"], {})
            t = (rec.get("terms") or {}).get(k)
            trues.append(int(row[lab]))
        prior = sum(trues) / len(trues)
        for _, row in gold_df.iterrows():
            rec = rec_by_hash.get(row["_hash"], {})
            t = (rec.get("terms") or {}).get(k)
            rank = RANK[k].get(t)
            scores.append(rank / max_rank if rank is not None else prior)
        a = auc_score(trues, scores)
        aucs[lab] = a
        n1 = sum(trues)
        print(f"  {lab:16s} AUC={a:.3f}" if a else f"  {lab:16s} AUC=n/a",
              f" positives={n1}/{len(gold_df)}", flush=True)
    valid = [a for a in aucs.values() if a is not None]
    macro = sum(valid) / len(valid) if valid else None
    pf = n_fail / max(len(targets), 1)
    print(f"\nmacro AUC (rank-based) vs gold: {macro:.4f}" if macro else
          "\nmacro AUC: undefined", flush=True)
    print(f"parse-fail rate: {pf:.1%}", flush=True)
    r_in, r_out = RATES.get(MODEL, (0.15, 0.60))
    print(f"tokens used: {tot_in} in / {tot_out} out "
          f"(~${tot_in/1e6*r_in + tot_out/1e6*r_out:.4f})", flush=True)
    print("\nComparison: GPT-4.1 0.8473 (3% fails) | Mac qwen3:14b 0.7530 (5%)",
          flush=True)


if __name__ == "__main__":
    main()
