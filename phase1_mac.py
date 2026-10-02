#!/usr/bin/env python3
"""Phase 1 weak-label extraction for M1 Pro Mac via Ollama.

Uses the 35B MoE (qwen35-35b-a3b-full) or 14B (qwen3:14b) through Ollama's
OpenAI-compatible API. Pure Python, no GPU deps beyond Ollama.

Usage:
  1. Ensure Ollama is running: `ollama serve` (or the Ollama app)
  2. Download train.csv from Kaggle competition page to ./train.csv
  3. python3 phase1_mac.py --mode probe
  4. If probe passes gates: python3 phase1_mac.py --mode full

Outputs to ./phase1_labels/
"""

import argparse, csv, json, os, re, sys, time, urllib.request

# ---------------------------------------------------------------- config
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen35-35b-a3b-full"  # or "qwen3:14b"
TRAIN_CSV = "./train.csv"
OUT_DIR = "./phase1_labels"
SEED = 0
PROBE_N_OTHER = 42
MAX_NEW_TOKENS = 320
GATE_MIN_MACRO_AUC = 0.86
GATE_MAX_PARSE_FAIL = 0.15

# ---------------------------------------------------------------- vocab (from phase1-labels-v8)
KEY2LABEL = {
    "acl": "ACL", "mcl": "MCL", "mm": "Medial Meniscus", "lm": "Lateral Meniscus",
    "moa": "Medial OA", "loa": "Lateral OA", "pfoa": "PF OA",
    "effusion": "Effusion", "synovitis": "Synovitis", "bakers": "Baker's",
    "contusion": "Contusion", "fracture": "Fracture",
}
KEYS = list(KEY2LABEL.keys())
LABELS = [KEY2LABEL[k] for k in KEYS]

VOCAB = {
    "acl": {"finding": "anterior cruciate ligament (ACL)",
            "terms": ["intact", "mucoid_degeneration", "reconstructed", "sprain", "partial_tear", "complete_tear"],
            "notes": "sprain = low-grade injury, fibers continuous; complete_tear = full-thickness disruption/rupture; reconstructed = ACL graft present (post-surgical)"},
    "mcl": {"finding": "medial collateral ligament (MCL)",
            "terms": ["intact", "sprain", "complete_tear"],
            "notes": "sprain covers grade 1-2 / partial injury; complete_tear = grade 3"},
    "mm": {"finding": "medial meniscus",
           "terms": ["intact", "degenerative_signal", "postoperative", "tear", "macerated"],
           "notes": "degenerative_signal = intrasubstance signal WITHOUT a morphologic tear; tear = any morphologic tear; postoperative = partial meniscectomy or repair"},
    "lm": {"finding": "lateral meniscus",
           "terms": ["intact", "degenerative_signal", "postoperative", "tear", "macerated"],
           "notes": "degenerative_signal = intrasubstance signal WITHOUT a morphologic tear; tear = any morphologic tear; postoperative = partial meniscectomy or repair"},
    "moa": {"finding": "medial compartment osteoarthritis / cartilage loss",
            "terms": ["none", "mild", "moderate", "severe"],
            "notes": "grade the described cartilage loss / osteoarthritic change"},
    "loa": {"finding": "lateral compartment osteoarthritis / cartilage loss",
            "terms": ["none", "mild", "moderate", "severe"],
            "notes": "grade the described cartilage loss / osteoarthritic change"},
    "pfoa": {"finding": "patellofemoral osteoarthritis / cartilage loss",
             "terms": ["none", "mild", "moderate", "severe"],
             "notes": "Grade the described cartilage loss. Includes patellar and trochlear cartilage. Multilingual: rotula/femororrotuliano (ES), femoro-patellaire (FR)."},
    "effusion": {"finding": "joint effusion (fluid)",
                 "terms": ["none", "trace_small", "moderate", "large"],
                 "notes": "'trace' or 'minimal' fluid = trace_small (present). Multilingual: derrame (ES), epanchement (FR), Erguss (DE)."},
    "synovitis": {"finding": "synovitis / synovial proliferation / inflamed synovium",
                  "terms": ["absent", "present"],
                  "notes": "present ONLY if explicitly described. Multilingual: sinovitis (ES), synovite (FR), Synovitis (DE). Do NOT infer from effusion alone."},
    "bakers": {"finding": "Baker's (popliteal) cyst",
               "terms": ["absent", "present", "ruptured"], "notes": ""},
    "contusion": {"finding": "bone contusion / traumatic bone marrow edema",
                  "terms": ["absent", "present"],
                  "notes": "Multilingual: contusion (ES/FR), Kontusion (DE). Traumatic only; NOT degenerative edema."},
    "fracture": {"finding": "fracture (cortical break / fracture line)",
                 "terms": ["absent", "present"],
                 "notes": "Multilingual: fractura (ES), fracture (FR), Fraktur (DE). Includes occult, stress, insufficiency, avulsion."},
}
TERM_RANK = {k: {t: i for i, t in enumerate(v["terms"])} for k, v in VOCAB.items()}

def build_prompt(report):
    lines = []
    for k in KEYS:
        v = VOCAB[k]
        terms = " | ".join(v["terms"])
        lines.append(f"- {k} ({v['finding']}): {terms} | not_mentioned")
        if v["notes"]:
            lines.append(f"  ({v['notes']})")
    vocab_block = "\n".join(lines)
    return f"""You are a musculoskeletal radiology reader. Read the knee MRI report below. It may be written in Spanish, French, English, German, or another language — read it regardless of language.

For each of the 12 findings, choose the SINGLE allowed term that best matches what the report STATES. Rules:
- If the report does not mention the finding at all, use "not_mentioned".
- Do NOT infer findings that are not stated.
- Negated findings ("no tear", "sin rotura", "pas de déchirure", "kein Riss") map to the intact/absent/none term.
- Uncertain/hedged language ("possible", "cannot exclude", "suggestive of") still maps to the matching positive term — choose the term, not the hedge.

Findings and allowed terms:
{vocab_block}

Your ENTIRE response must be exactly the JSON object and nothing else. Do not add any text before or after. Return ONLY a JSON object with exactly these 12 keys, in this order:
{json.dumps(KEYS)}
Example: {{"acl": "intact", "mcl": "not_mentioned", ...}}
No explanations. No markdown. No extra text.

Report:
\"\"\"{report}\"\"\""""

# ---------------------------------------------------------------- ollama
def call_ollama(prompt, retries=3):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "stream": False,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(OLLAMA_URL, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(5)
    return None

# ---------------------------------------------------------------- parsing
def extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    m = re.search(r"\{", text)
    if not m:
        return None
    depth, start = 0, m.start()
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i+1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None

def canon_term(key, raw):
    t = raw.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    t = re.sub(r"_+", "_", t).strip("_")
    if t in VOCAB[key]["terms"] or t == "not_mentioned":
        return t
    return None

def normalize(raw):
    terms, issues = {}, []
    for k in KEYS:
        if k not in raw or not isinstance(raw[k], str):
            terms[k] = "not_mentioned"
            issues.append(f"missing/invalid: {k}")
            continue
        c = canon_term(k, raw[k])
        if c is None:
            terms[k] = "not_mentioned"
            issues.append(f"unknown term for {k}: {raw[k]!r}")
        else:
            terms[k] = c
    return terms, issues

# ---------------------------------------------------------------- calibration (PAVA isotonic)
def _pava_fit(xs, ys):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [float(ys[i]) for i in order]
    sums, counts, xmins = [], [], []
    for x, y in zip(xs, ys):
        sums.append(y); counts.append(1); xmins.append(x)
        while len(sums) >= 2 and sums[-2]/counts[-2] > sums[-1]/counts[-1] + 1e-12:
            sums[-2] += sums[-1]; counts[-2] += counts[-1]
            sums.pop(); counts.pop(); xmins.pop()
    merged = []
    for xm, s, c in zip(xmins, sums, counts):
        if merged and merged[-1][0] == xm:
            px, ps, pc = merged[-1]
            merged[-1] = (px, ps+s, pc+c)
        else:
            merged.append((xm, s, c))
    return [(xm, s/c) for xm, s, c in merged]

def _pava_predict(levels, x):
    v = levels[0][1]
    for xmin, val in levels:
        if x >= xmin: v = val
        else: break
    return v

# ---------------------------------------------------------------- main
def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "full"], required=True)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    MODEL = args.model

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Model: {MODEL} | Mode: {args.mode}")

    # Load train.csv (columns: StudyInstanceUID, Report, 12 label columns; latin-1 encoding)
    studies = {}  # study_id -> (report, labels dict)
    with open(TRAIN_CSV, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["StudyInstanceUID"]
            report = row.get("Report", "") or ""
            labels = {}
            for lbl in LABELS:
                v = row.get(lbl, "")
                if v not in ("", None):
                    try:
                        labels[lbl] = int(float(v))
                    except ValueError:
                        pass
            studies[sid] = (report, labels)
    print(f"studies: {len(studies)}")

    # Identify gold (all 12 labels present)
    gold_ids = [sid for sid, (_, lbls) in studies.items() if len(lbls) == 12]
    print(f"gold studies: {len(gold_ids)}")

    # Unique reports
    report_to_sids = {}
    for sid, (rep, _) in studies.items():
        report_to_sids.setdefault(rep, []).append(sid)
    unique_reports = list(report_to_sids.keys())
    print(f"unique reports: {len(unique_reports)}")

    # Probe set: gold reports + 42 others
    gold_reports = list({studies[sid][0] for sid in gold_ids})
    other_reports = [r for r in unique_reports if r not in set(gold_reports)]
    import random
    random.seed(SEED)
    probe_extra = random.sample(other_reports, min(PROBE_N_OTHER, len(other_reports)))

    if args.mode == "probe":
        target_reports = gold_reports + probe_extra
        print(f"probe: {len(gold_reports)} gold + {len(probe_extra)} extra = {len(target_reports)}")
    else:
        target_reports = unique_reports
        print(f"full: {len(target_reports)} reports")

    # Extract
    results = {}
    parse_fails = 0
    t0 = time.time()
    for i, rep in enumerate(target_reports):
        prompt = build_prompt(rep[:8000])  # truncate very long reports
        try:
            out = call_ollama(prompt)
            obj = extract_json(out)
            if obj is None:
                if i < 3:  # debug: show raw output for first 3 failures
                    print(f"\n--- RAW OUTPUT (report {i}) ---\n{out[:800]}\n--- END ---\n")
                parse_fails += 1
                # retry once
                out = call_ollama(prompt)
                obj = extract_json(out)
                if obj is None:
                    parse_fails += 1
                    results[rep] = ({k: "not_mentioned" for k in KEYS}, ["parse fail"])
                    continue
            terms, issues = normalize(obj)
            results[rep] = (terms, issues)
        except Exception as e:
            parse_fails += 1
            results[rep] = ({k: "not_mentioned" for k in KEYS}, [f"error: {e}"])
        if (i+1) % 10 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{len(target_reports)} ({dt/(i+1):.1f}s/report, {parse_fails} parse fails)")

    dt = time.time() - t0
    print(f"\ndone: {len(target_reports)} reports in {dt:.0f}s ({dt/len(target_reports):.1f}s/report)")
    print(f"parse failures: {parse_fails}/{len(target_reports)} ({100*parse_fails/len(target_reports):.1f}%)")

    # Calibrate on gold and compute AUC (probe only)
    if args.mode == "probe":
        from collections import defaultdict
        # ranks per label
        label_ranks = defaultdict(list)
        label_golds = defaultdict(list)
        for sid in gold_ids:
            rep, lbls = studies[sid]
            terms, _ = results.get(rep, ({k: "not_mentioned" for k in KEYS}, []))
            for k in KEYS:
                lbl = KEY2LABEL[k]
                rank = TERM_RANK[k].get(terms[k])
                label_ranks[k].append(rank)
                label_golds[k].append(lbls[lbl])

        # Fit isotonic and compute AUC
        import math
        aucs = {}
        for k in KEYS:
            ranks = label_ranks[k]
            golds = label_golds[k]
            pairs = [(r, g) for r, g in zip(ranks, golds) if r is not None]
            if not pairs:
                aucs[k] = 0.5
                continue
            # Fit PAVA
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            levels = _pava_fit(xs, ys)
            # Predict scores
            scores = [_pava_predict(levels, r) if r is not None else sum(ys)/len(ys)
                      for r in ranks]
            # AUC via Mann-Whitney
            pos = [s for s, g in zip(scores, golds) if g == 1]
            neg = [s for s, g in zip(scores, golds) if g == 0]
            if not pos or not neg:
                aucs[k] = 0.5
                continue
            # rank-based AUC
            all_scores = sorted([(s, g) for s, g in zip(scores, golds)])
            rank_sum = 0
            for i, (s, g) in enumerate(all_scores):
                if g == 1:
                    rank_sum += i + 1
            n_pos, n_neg = len(pos), len(neg)
            auc = (rank_sum - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
            aucs[k] = auc
            print(f"  {KEY2LABEL[k]:15s} AUC={auc:.3f} (pos {n_pos}/{n_pos+n_neg})")

        macro = sum(aucs.values()) / len(aucs)
        print(f"\nmacro AUC: {macro:.4f} (gate: >={GATE_MIN_MACRO_AUC})")
        pf_rate = parse_fails / len(target_reports)
        print(f"parse fail: {100*pf_rate:.1f}% (gate: <{100*GATE_MAX_PARSE_FAIL:.0f}%)")

        gates = {
            "macro_auc>=0.86": macro >= GATE_MIN_MACRO_AUC,
            "parse_fail<15%": pf_rate < GATE_MAX_PARSE_FAIL,
        }
        print(f"GATES: {gates}")
        if not all(gates.values()):
            print("GATE FAILED — do not run full mode.")
            sys.exit(1)
        print("PROBE PASS — ready for full mode.")

    # Save results
    out_path = os.path.join(OUT_DIR, f"descriptors_{args.mode}.jsonl")
    with open(out_path, "w") as f:
        for rep, (terms, issues) in results.items():
            f.write(json.dumps({"report_hash": hash(rep), "terms": terms,
                                "issues": issues}) + "\n")
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()
