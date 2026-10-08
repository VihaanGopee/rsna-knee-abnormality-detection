#!/usr/bin/env python3
"""Full 12-finding extraction via Google Antigravity (Gemini 3.8 Flash).

Based on 100% validation accuracy on 58 gold studies for the 4 severity findings.
Now extending to all 12 findings.

Findings (12):
- ACL, MCL: ligament tears (binary: intact vs tear)
- Medial/Lateral Meniscus: tears (binary)
- Medial/Lateral/PF OA: severity (none/mild/moderate/severe)
- Effusion: severity (none/small/moderate/large)
- Synovitis: severity (none/mild/moderate/severe)
- Baker's cyst: present/absent
- Contusion: present/absent
- Fracture: present/absent

Usage:
    python3 phase1_full_antigravity.py --csv ./train.csv

Output: ./phase1_full_out/full_labels.json
Resume-safe.
"""
import argparse
import asyncio
import json
import os
import sys

try:
    from google.antigravity import Agent, LocalAgentConfig
except ImportError:
    print("ERROR: google.antigravity not installed", file=sys.stderr)
    sys.exit(1)

import pandas as pd

SYSTEM_PROMPT = """You are an expert musculoskeletal radiologist.
Read the knee MRI report below (may be English, Spanish, German, French, or Portuguese).

For each finding, output EXACTLY ONE term from the allowed list.

1. acl: intact, mucoid_degeneration, reconstructed, sprain, partial_tear, complete_tear
2. mcl: intact, sprain, complete_tear
3. mm (medial meniscus): intact, degenerative_signal, postoperative, tear, macerated
4. lm (lateral meniscus): intact, degenerative_signal, postoperative, tear, macerated
5. moa (medial OA): none, mild, moderate, severe
6. loa (lateral OA): none, mild, moderate, severe
7. pfoa (PF OA): none, mild, moderate, severe
8. effusion: none, small, moderate, large (count "trace" as small)
9. synovitis: none, mild, moderate, severe (include Hoffa edema, plica syndrome)
10. bakers (Baker's cyst): none, present, ruptured
11. contusion: none, present (only traumatic bone marrow edema, not degenerative)
12. fracture: none, present, suspected (include avulsion, insufficiency)

Rules:
- If not mentioned → use "intact" or "none" (most neutral)
- For OA: grade by described severity
- For synovitis: include Hoffa fat pad impingement

Respond with ONLY valid JSON, no other text:
{"acl": "...", "mcl": "...", "mm": "...", "lm": "...", "moa": "...", "loa": "...", "pfoa": "...", "effusion": "...", "synovitis": "...", "bakers": "...", "contusion": "...", "fracture": "..."}"""

FINDINGS = ["acl", "mcl", "mm", "lm", "moa", "loa", "pfoa", "effusion",
            "synovitis", "bakers", "contusion", "fracture"]

# Valid terms per finding (for validation)
VALID_TERMS = {
    "acl": {"intact", "mucoid_degeneration", "reconstructed", "sprain", "partial_tear", "complete_tear"},
    "mcl": {"intact", "sprain", "complete_tear"},
    "mm": {"intact", "degenerative_signal", "postoperative", "tear", "macerated"},
    "lm": {"intact", "degenerative_signal", "postoperative", "tear", "macerated"},
    "moa": {"none", "mild", "moderate", "severe"},
    "loa": {"none", "mild", "moderate", "severe"},
    "pfoa": {"none", "mild", "moderate", "severe"},
    "effusion": {"none", "small", "moderate", "large"},
    "synovitis": {"none", "mild", "moderate", "severe", "suspected"},
    "bakers": {"none", "present", "ruptured"},
    "contusion": {"none", "present"},
    "fracture": {"none", "present", "suspected"},
}

DEFAULTS = {
    "acl": "intact", "mcl": "intact", "mm": "intact", "lm": "intact",
    "moa": "none", "loa": "none", "pfoa": "none", "effusion": "none",
    "synovitis": "none", "bakers": "none", "contusion": "none", "fracture": "none",
}


def parse_response(text):
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return None
    result = {}
    for f in FINDINGS:
        term = str(data.get(f, DEFAULTS[f])).lower().strip().replace(" ", "_").replace("-", "_")
        result[f] = term if term in VALID_TERMS[f] else DEFAULTS[f]
    return result


async def process_reports(reports, output_path, completed):
    config = LocalAgentConfig(system_instructions=SYSTEM_PROMPT)
    async with Agent(config) as agent:
        for idx, (uid, report_text) in enumerate(reports):
            if uid in completed:
                continue
            if not report_text or str(report_text) == "nan":
                completed[uid] = {f: DEFAULTS[f] for f in FINDINGS}
                continue
            success = False
            retries = 0
            while not success and retries < 5:
                try:
                    response = await agent.chat(f"Report:\n{report_text}")
                    result_text = await response.text()
                    parsed = parse_response(result_text)
                    if parsed is None:
                        raise ValueError(f"Parse failed: {result_text[:200]}")
                    completed[uid] = parsed
                    success = True
                    with open(output_path, "w") as out:
                        json.dump(completed, out, indent=1)
                    n = len(completed)
                    if n % 10 == 0:
                        print(f"[{n}/{len(reports)}] done", flush=True)
                    await asyncio.sleep(2)
                except Exception as e:
                    retries += 1
                    msg = str(e).lower()
                    if any(s in msg for s in ("quota", "rate", "429", "resource_exhausted")):
                        print(f"\nQuota hit at {idx}. Waiting 10 min...", flush=True)
                        await asyncio.sleep(600)
                    elif retries >= 5:
                        print(f"SKIP {uid} after 5 failures: {e}", flush=True)
                        completed[uid] = {f: DEFAULTS[f] for f in FINDINGS}
                        break
                    else:
                        print(f"Retry {retries}/5 for {uid}", flush=True)
                        await asyncio.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="./phase1_full_out/full_labels.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        with open(args.out) as f:
            completed = json.load(f)
    else:
        completed = {}
    print(f"Resuming: {len(completed)} already done", flush=True)
    df = pd.read_csv(args.csv)
    report_col = None
    for c in df.columns:
        if c.lower() in ("report", "report_text", "findings"):
            report_col = c
            break
    if not report_col:
        for c in df.columns:
            if "report" in c.lower():
                report_col = c
                break
    reports = [(str(r["StudyInstanceUID"]), str(r[report_col])) for _, r in df.iterrows()]
    print(f"Total: {len(reports)} reports", flush=True)
    asyncio.run(process_reports(reports, args.out, completed))
    print(f"\nDone: {len(completed)}/{len(reports)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
