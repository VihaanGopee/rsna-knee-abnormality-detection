#!/usr/bin/env python3
"""Phase 1 weak-label extraction on Justin's Mac via Ollama (qwen3:14b).

Batches 5 reports per Ollama call for speed (~7h for 4,276 reports).
Saves progress incrementally — safe to interrupt and resume.

Usage:
    python3 phase1_mac_batch.py --mode probe   # 100 reports, quality gate
    python3 phase1_mac_batch.py --mode full    # all 4,276 reports

Requires: ollama serve running, qwen3:14b pulled.
CSV: ./train.csv (competition file) or --csv PATH
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

# ---------------------------------------------------------------- config
MODEL = "qwen3:14b"
OLLAMA_URL = "http://localhost:11434/api/chat"
BATCH = 5
CSV_DEFAULT = "./train.csv"
OUT_DIR = "./phase1_mac_out"
TIMEOUT_S = 300

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "mm", "lm", "moa", "loa", "pfoa", "effusion",
        "synovitis", "bakers", "contusion", "fracture"]

# term -> rank per key (higher = more abnormal); mirrors label_core.VOCAB
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


def build_single_prompt(report):
    return build_batch_prompt([report]).replace(
        "Output exactly 1 lines, one per report, in order.",
        "Output exactly 1 line.")


# ---------------------------------------------------------------- ollama
def ollama_chat(prompt):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "think": False,          # critical: Qwen3 thinking -> empty content
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
            # defensive: if content empty but thinking present, use it
            if not content.strip():
                content = data.get("message", {}).get("thinking", "")
            if not content.strip():
                raise ValueError("empty response content")
            return content
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ollama failed after 3 tries: {last}")


# ---------------------------------------------------------------- parsing
def canon_term(key, raw):
    t = raw.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    import re
    t = re.sub(r"_+", "_", t).strip("_")
    if t in VOCAB[key]["terms"] or t == "not_mentioned":
        return t
    # light aliases
    aliases = {"rupture": "tear", "torn": "tear", "normal": "intact",
               "present": None, "absent": "none", "no": "none"}
    if t in aliases and aliases[t]:
        a = aliases[t]
        # map generic alias into this key's vocab if plausible
        if a in VOCAB[key]["terms"]:
            return a
        if a == "none" and "none" in VOCAB[key]["terms"]:
            return "none"
        if a == "intact" and "intact" in VOCAB[key]["terms"]:
            return "intact"
    return None


def parse_line(line):
    """'t1,...,t12' -> (terms dict, issues)."""
    parts = [p.strip() for p in line.strip().strip("`").split(",")]
    # tolerate trailing comma / empty parts
    parts = [p for p in parts if p]
    terms, issues = {}, []
    if len(parts) != 12:
        return None, [f"expected 12 terms, got {len(parts)}"]
    for k, raw in zip(KEYS, parts):
        c = canon_term(k, raw)
        if c is None:
            terms[k] = "not_mentioned"
            issues.append(f"unknown term for {k}: {raw!r}")
        else:
            terms[k] = c
    return terms, issues


def parse_batch_output(text, n):
    """Returns list of (terms, issues) per report, length n (None on failure)."""
    import re
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    # strip leading "1:" / "Report 1:" style prefixes models sometimes add
    clean = []
    for l in lines:
        l2 = re.sub(r"^(report\s*\d+\s*:\s*|\d+\s*[:.)]\s*)", "", l,
                    flags=re.IGNORECASE).strip()
        clean.append(l2)
    # drop markdown fences / prose lines that aren't 12-term lines
    cand = [l for l in clean if l.count(",") >= 11]
    if len(cand) != n:
        return None
    out = []
    for l in cand:
        terms, issues = parse_line(l)
        if terms is None:
            return None
        out.append((terms, issues))
    return out


# ---------------------------------------------------------------- pipeline
def label_batch(hashes_reports):
    """hashes_reports: list of (hash, report). Returns list of record dicts."""
    hashes = [h for h, _ in hashes_reports]
    reports = [r for _, r in hashes_reports]
    prompt = build_batch_prompt(reports)
    try:
        text = ollama_chat(prompt)
        parsed = parse_batch_output(text, len(reports))
    except Exception as e:
        parsed = None
        text = f"<error: {e}>"
    if parsed is None:
        # fallback: one call per report
        recs = []
        for h, r in hashes_reports:
            try:
                t2 = ollama_chat(build_single_prompt(r))
                p2 = parse_batch_output(t2, 1)
                if p2 is None:
                    raise ValueError("single parse failed")
                terms, issues = p2[0]
                recs.append({"hash": h, "terms": terms, "issues": issues,
                             "parse_ok": not issues, "fallback_single": True})
            except Exception as e2:
                recs.append({"hash": h,
                             "terms": {k: "not_mentioned" for k in KEYS},
                             "issues": [f"failed: {e2}"], "parse_ok": False,
                             "fallback_single": True})
        return recs
    return [{"hash": h, "terms": t, "issues": i, "parse_ok": not i,
             "fallback_single": False}
            for (h, _), (t, i) in zip(hashes_reports, parsed)]


def auc_score(y_true, y_score):
    # Mann-Whitney U; None if degenerate
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return None
    n1, n0 = len(pos), len(neg)
    # rank with ties (average)
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
    ap.add_argument("--mode", choices=["probe", "full"], required=True)
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    MODEL = args.model
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(args.csv, encoding="latin-1")
    df["_hash"] = df["Report"].map(
        lambda r: hashlib.sha256(str(r).encode("utf-8", "replace")).hexdigest()[:16])
    uniq = df.drop_duplicates("_hash").set_index("_hash")["Report"].to_dict()
    print(f"studies: {len(df)} | unique reports: {len(uniq)}", flush=True)

    gold_mask = df[LABELS].notna().all(axis=1)
    gold_df = df[gold_mask]
    gold_hashes = gold_df["_hash"].tolist()
    print(f"gold studies: {len(gold_df)}", flush=True)

    if args.mode == "probe":
        random.seed(0)
        extra = [h for h in uniq if h not in set(gold_hashes)]
        n_extra = min(42, len(extra))
        targets = gold_hashes + random.sample(extra, n_extra)
    else:
        targets = list(uniq.keys())
    print(f"{args.mode}: {len(targets)} reports", flush=True)

    # resume: load already-done hashes
    done_path = os.path.join(OUT_DIR, f"{args.mode}_records.jsonl")
    done = {}
    if os.path.exists(done_path):
        with open(done_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["hash"]] = r
        print(f"resuming: {len(done)} already done", flush=True)
    todo = [h for h in targets if h not in done]
    print(f"todo: {len(todo)}", flush=True)

    # batch the todo list
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    t0 = time.time()
    n_fail = 0
    with open(done_path, "a") as fout:
        for bi, bh in enumerate(batches):
            recs = label_batch([(h, uniq[h]) for h in bh])
            for r in recs:
                done[r["hash"]] = r
                fout.write(json.dumps(r) + "\n")
                if not r["parse_ok"]:
                    n_fail += 1
            fout.flush()
            el = time.time() - t0
            ndone = len(done)
            print(f"  batch {bi+1}/{len(batches)} | {ndone}/{len(targets)} "
                  f"| {el:.0f}s | {el/max(ndone,1):.1f}s/report "
                  f"| fails: {n_fail}", flush=True)

    # ---- probe validation ----
    rec_by_hash = done
    gold_scores, gold_true = {}, {}
    for k, lab in zip(KEYS, LABELS):
        scores, trues = [], []
        max_rank = len(VOCAB[k]["terms"]) - 1
        for _, row in gold_df.iterrows():
            t = rec_by_hash[row["_hash"]]["terms"][k]
            rank = RANK[k].get(t)
            trues.append(int(row[lab]))
        prior = sum(trues) / len(trues)
        for _, row in gold_df.iterrows():
            t = rec_by_hash[row["_hash"]]["terms"][k]
            rank = RANK[k].get(t)
            # rank -> 0..1 score; abstain (None) -> prior, like calibration
            scores.append(rank / max_rank if rank is not None else prior)
        gold_scores[lab] = scores
        gold_true[lab] = trues
    aucs = {}
    for lab in LABELS:
        a = auc_score(gold_true[lab], gold_scores[lab])
        n1 = sum(gold_true[lab])
        aucs[lab] = a
        print(f"  {lab:16s} AUC={a:.3f}" if a else f"  {lab:16s} AUC=n/a",
              f" positives={n1}/{len(gold_df)}", flush=True)
    valid = [a for a in aucs.values() if a is not None]
    macro = sum(valid) / len(valid) if valid else None
    print(f"\nmacro AUC (rank-based) vs gold: {macro:.4f}" if macro else
          "\nmacro AUC: undefined", flush=True)
    pf = n_fail / max(len(done), 1)
    print(f"parse-fail rate: {pf:.1%}", flush=True)

    if args.mode == "probe":
        ok = (macro is not None and macro >= 0.78 and pf < 0.15)
        print("PROBE PASSED — run --mode full" if ok else
              "PROBE FAILED — do not run full", flush=True)
        sys.exit(0 if ok else 1)

    # ---- full mode outputs ----
    # isotonic calibration on gold (simple PAVA)
    def pava_fit(xs, ys):
        order = sorted(range(len(xs)), key=lambda i: (xs[i] is None, xs[i]))
        xs = [xs[i] for i in order]
        ys = [float(ys[i]) for i in order]
        # drop None ranks for fitting, map them to prior later
        xyn = [(x, y) for x, y in zip(xs, ys) if x is not None]
        if not xyn:
            return None
        xs, ys = zip(*xyn)
        sums, counts, xm = [], [], []
        for x, y in zip(xs, ys):
            sums.append(y); counts.append(1); xm.append(x)
            while (len(sums) >= 2 and
                   sums[-2] / counts[-2] > sums[-1] / counts[-1] + 1e-12):
                sums[-2] += sums[-1]; counts[-2] += counts[-1]
                sums.pop(); counts.pop(); xm.pop()
        return (xm, [s / c for s, c in zip(sums, counts)])

    def pava_predict(cal, x, prior):
        if cal is None or x is None:
            return prior
        xm, vals = cal
        if x <= xm[0]:
            return vals[0]
        if x >= xm[-1]:
            return vals[-1]
        for i in range(len(xm) - 1):
            if xm[i] <= x <= xm[i + 1]:
                w = 0 if xm[i+1] == xm[i] else (x - xm[i]) / (xm[i+1] - xm[i])
                return vals[i] * (1 - w) + vals[i+1] * w
        return vals[-1]

    cals = {}
    for k, lab in zip(KEYS, LABELS):
        ranks = [RANK[k].get(rec_by_hash[u["_hash"]]["terms"][k])
                 for _, u in gold_df.iterrows()]
        trues = [int(v) for v in gold_df[lab]]
        prior = sum(trues) / len(trues)
        cals[k] = (pava_fit(ranks, trues), prior)

    rows = []
    for _, row in df.iterrows():
        terms = rec_by_hash[row["_hash"]]["terms"]
        out = {"StudyInstanceUID": row["StudyInstanceUID"]}
        for k, lab in zip(KEYS, LABELS):
            cal, prior = cals[k]
            out[lab] = pava_predict(cal, RANK[k].get(terms[k]), prior)
        rows.append(out)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "weak_labels_v1.csv"),
                              index=False)
    with open(os.path.join(OUT_DIR, "descriptors.jsonl"), "w") as f:
        for h in uniq:
            r = rec_by_hash[h]
            f.write(json.dumps({"hash": h, "terms": r["terms"]}) + "\n")
    json.dump({"model": MODEL, "macro_auc_rank": macro,
               "parse_fail_rate": pf, "n_reports": len(uniq)},
              open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=1)
    print(f"\nwrote {OUT_DIR}/weak_labels_v1.csv, descriptors.jsonl, manifest.json",
          flush=True)


if __name__ == "__main__":
    main()
