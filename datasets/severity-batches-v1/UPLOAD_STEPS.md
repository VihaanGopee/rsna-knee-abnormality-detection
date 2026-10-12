# severity-batches-v1 — upload steps (Kaggle web UI)

The CORAL notebook (phase2-coral-v1.2) needs these 9 JSON files as a Kaggle
dataset named `severity-batches-v1`. Its fail-fast died in 14s on the last
run because the dataset wasn't attached ("found 0 severity batch JSONs,
need exactly 9").

**IMPORTANT:** upload ONLY these 9 files. The folder they came from also
has `graded_batch_1.json` and `graded_batch_2.json` — those are old test
files. Including them (11 files instead of 9) trips the "need exactly 9"
check, and they contain duplicate StudyInstanceUIDs, which trips a second
fatal check. They are NOT in this bundle.

## Steps

1. Kaggle → Datasets → **New Dataset**
2. Title: `severity-batches-v1`
3. Drag in the 9 `graded_batch_*.json` files from this folder (856 KB total)
4. Create (private is fine)
5. Open the CORAL notebook → right panel → **Add Data** → attach
   `severity-batches-v1` (alongside phase0runA/B/C + severity-merged-labels-v1)
6. Re-run with `PROBE = True`. Expect:
   `CORAL severity batches: 9 files OK`

## Verified

- 4,276 unique StudyInstanceUIDs across the 9 files (500×8 + 276), no overlaps
- Every record has StudyInstanceUID + synovitis/pfoa/medial_oa/lateral_oa
- All grades are strings in {none, mild, moderate, severe} — exactly what the
  notebook's GRADE_TO_ORD mapper expects
- Simulated the notebook's fail-fast + parsing checks against this exact
  bundle: all pass (see below)
