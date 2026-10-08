#!/usr/bin/env python3
"""Severity extraction for 4 weak findings via Google Antigravity (Gemini 3.8 Flash).

Extracts severity grades (none/mild/moderate/severe) for:
- Synovitis
- PF OA (patellofemoral)
- Medial OA
- Lateral OA

These are the 4 findings where teacher labels are weakest (0.76-0.88).
We keep teacher labels for the other 8, patch only these 4.

Usage:
    python3 phase1_severity_antigravity.py --csv ./train.csv

Output: ./phase1_severity_out/severity_labels.json
    {uid: {"synovitis": "mild", "pfoa": "moderate", ...}, ...}

Resume-safe: skips UIDs already in output file.

VERSION v2 (2026-10-07): fixed empty-report checkpointing (was never written),
no poison-on-skip (failed UIDs left out for resume instead of "none" labels),
quota waits no longer burn the 5-retry budget, quota gives up after ~6h.
"""
VERSION = "v2"
import argparse
import asyncio
import json
import os
import sys
import time

try:
    from google.antigravity import Agent, LocalAgentConfig
except ImportError:
    print("ERROR: google.antigravity not installed", file=sys.stderr)
    sys.exit(1)

import pandas as pd

SYSTEM_PROMPT = """You are an expert musculoskeletal radiologist.
Read the knee MRI report below (may be English, Spanish, German, French, or Portuguese).

Grade severity as EXACTLY ONE of: none, mild, moderate, severe

Findings to grade:
1. synovitis — includes Hoffa fat pad edema/impingement, plica syndrome, synovial thickening
2. pfoa — patellofemoral compartment cartilage loss / osteoarthritis
3. medial_oa — medial compartment cartilage loss / osteoarthritis  
4. lateral_oa — lateral compartment cartilage loss / osteoarthritis

Rules:
- "trace" or "minimal" → mild (not none)
- If not mentioned → none
- If uncertain ("possible", "cannot exclude") → use your best judgment, prefer mild over none

Respond with ONLY valid JSON, no other text:
{"synovitis": "...", "pfoa": "...", "medial_oa": "...", "lateral_oa": "..."}"""

VALID_GRADES = {"none", "mild", "moderate", "severe"}
FINDINGS = ["synovitis", "pfoa", "medial_oa", "lateral_oa"]


def parse_response(text):
    """Extract JSON from response, validate grades."""
    # Strip markdown code blocks
    t = text.strip()
    if t.startswith("```"):
        # Remove first line (```json) and last line (```)
        lines = t.split("\n")
        t = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return None
    
    # Validate
    result = {}
    for f in FINDINGS:
        grade = str(data.get(f, "none")).lower().strip()
        result[f] = grade if grade in VALID_GRADES else "none"
    return result


def checkpoint(completed, output_path):
    """Write progress. Called after EVERY uid so no work is lost."""
    with open(output_path, "w") as out:
        json.dump(completed, out, indent=1)


async def process_reports(reports, output_path, completed):
    config = LocalAgentConfig(system_instructions=SYSTEM_PROMPT)
    quota_waits = 0
    MAX_QUOTA_WAITS = 36  # 36 x 10min = ~6h, then stop (resume later)

    async with Agent(config) as agent:
        for idx, (uid, report_text) in enumerate(reports):
            if uid in completed:
                continue

            if not report_text or str(report_text) == "nan":
                completed[uid] = {f: "none" for f in FINDINGS}
                checkpoint(completed, output_path)
                continue

            success = False
            retries = 0

            while not success and retries < 5:
                try:
                    response = await agent.chat(f"Report:\n{report_text}")
                    result_text = await response.text()

                    parsed = parse_response(result_text)
                    if parsed is None:
                        raise ValueError(f"Could not parse JSON from: {result_text[:200]}")

                    completed[uid] = parsed
                    success = True
                    checkpoint(completed, output_path)

                    n = len(completed)
                    if n % 10 == 0:
                        print(f"[{n}/{len(reports)}] done", flush=True)

                    # Rate limit: 2s between requests
                    await asyncio.sleep(2)

                except Exception as e:
                    msg = str(e).lower()
                    if any(s in msg for s in ("quota", "rate", "429", "resource_exhausted")):
                        # Quota resets with time, not attempts: don't burn the retry budget.
                        quota_waits += 1
                        if quota_waits > MAX_QUOTA_WAITS:
                            print(f"\nQuota exhausted for ~6h. Stopping; "
                                  f"{len(completed)}/{len(reports)} done, resume later.", flush=True)
                            return
                        print(f"\nQuota hit at {idx}. Waiting 10 min "
                              f"(wait {quota_waits}/{MAX_QUOTA_WAITS})...", flush=True)
                        await asyncio.sleep(600)
                        continue
                    retries += 1
                    if retries >= 5:
                        # Do NOT write fallback "none" labels: leave the uid out of
                        # completed so the next resume retries it. Writing "none"
                        # would silently poison the label set.
                        print(f"SKIP {uid} after 5 failures (left for resume): {e}", flush=True)
                        break
                    print(f"Retry {retries}/5 for {uid}: {e}", flush=True)
                    await asyncio.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="train.csv with StudyInstanceUID and Report columns")
    ap.add_argument("--out", default="./phase1_severity_out/severity_labels.json")
    args = ap.parse_args()

    print(f"phase1_severity_antigravity {VERSION}", flush=True)
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    
    # Load progress
    if os.path.exists(args.out):
        with open(args.out) as f:
            completed = json.load(f)
    else:
        completed = {}
    print(f"Resuming: {len(completed)} already done", flush=True)
    
    # Load reports
    df = pd.read_csv(args.csv)
    # Find report column
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
    if not report_col:
        print(f"ERROR: no report column found in {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    
    reports = [(str(r["StudyInstanceUID"]), str(r[report_col])) for _, r in df.iterrows()]
    print(f"Total: {len(reports)} reports", flush=True)
    
    asyncio.run(process_reports(reports, args.out, completed))
    
    print(f"\nDone: {len(completed)}/{len(reports)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
