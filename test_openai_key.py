#!/usr/bin/env python3
"""Test if your OpenAI API key works. Key is read from env var, never printed.

Usage:
    export OPENAI_API_KEY="sk-..."
    python3 test_openai_key.py
"""
import os
import sys
import urllib.request
import urllib.error
import json


def main():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print("ERROR: Set OPENAI_API_KEY env var first:")
        print('  export OPENAI_API_KEY="sk-..."')
        sys.exit(1)
    if not key.startswith("sk-"):
        print("WARNING: Key doesn't start with 'sk-' — may be invalid format.")

    # Use /v1/models (free, no tokens consumed) to validate the key.
    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
            models = [m["id"] for m in data.get("data", [])]
            print(f"OK: Key works! {len(models)} models accessible.")
            # Show a few relevant ones without dumping everything
            interesting = [m for m in models if any(
                k in m for k in ("gpt-4o", "gpt-4.1", "o1", "o3"))][:10]
            if interesting:
                print("Examples: " + ", ".join(interesting)
                      )
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        if e.code == 401:
            print("FAIL: Invalid API key (401 Unauthorized).")
        elif e.code == 429:
            print("FAIL: Rate limited or quota exceeded (429).")
            print(body)
        else:
            print(f"FAIL: HTTP {e.code}: {body}")
        sys.exit(1)
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
