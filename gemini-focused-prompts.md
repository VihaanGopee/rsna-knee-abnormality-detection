# Gemini Focused Labeling — Antigravity Prompts

**Style:** Match the proven 100% severity prompt. 4 findings per prompt. Batch by 500 rows.

---

## Prompt A: Structural (ACL, MCL, MM, LM)

```
I have a dataset train.csv in this folder. Read rows [START] to [END], evaluate each knee MRI report, grade acl, mcl, mm, lm, and write the results into graded_structural_[START]_[END].json.

Terms:
- acl: intact, mucoid_degeneration, reconstructed, sprain, partial_tear, complete_tear
- mcl: intact, sprain, complete_tear
- mm: intact, degenerative_signal, postoperative, tear, macerated
- lm: intact, degenerative_signal, postoperative, tear, macerated

Rules: If not mentioned → intact. mucoid_degeneration/reconstructed → intact. degenerative_signal/postoperative → intact (not tears). Only sprain/tear/macerated/complete_tear are positive.

Output: [{"StudyInstanceUID": "...", "acl": "...", "mcl": "...", "mm": "...", "lm": "..."}, ...]
```

**Batches:** [0-500], [500-1000], [1000-1500], [1500-2000], [2000-2500], [2500-3000], [3000-3500], [3500-4000], [4000-4276]

---

## Prompt B: Fluid/Trauma (Effusion, Baker's, Contusion, Fracture)

```
I have a dataset train.csv in this folder. Read rows [START] to [END], evaluate each knee MRI report, grade effusion, bakers, contusion, fracture, and write the results into graded_fluidtrauma_[START]_[END].json.

Terms:
- effusion: none, small, moderate, large
- bakers: none, present, ruptured
- contusion: none, present
- fracture: none, present, suspected

Rules: If not mentioned → none. "trace"/"minimal" → small. "popliteal cyst" = bakers. For contusion: ONLY traumatic edema (not degenerative). For fracture: include avulsion/insufficiency.

Output: [{"StudyInstanceUID": "...", "effusion": "...", "bakers": "...", "contusion": "...", "fracture": "..."}, ...]
```

**Batches:** [0-500], [500-1000], [1000-1500], [1500-2000], [2000-2500], [2500-3000], [3000-3500], [3500-4000], [4000-4276]

---

## Gold test (58 studies) — MUST BE BLIND

The grading step must NEVER see the expert labels. A previous test was contaminated because the agent found the answers in the same folder.

**Clean folder checklist** — the Antigravity folder must contain ONLY:
- `gold_58_reports_only.csv` (58 rows: StudyInstanceUID + Report, no expert columns)
- the prompt you paste

It must NOT contain: `gold_58.csv`, `train.csv` (has the 58 expert rows), `all_reports.json`, any `graded_*.json` from earlier runs, or any other file with expert labels.

**Step 1 — Prompt A (structural), paste:**
```
I have gold_58_reports_only.csv in this folder. It has 58 rows with columns: StudyInstanceUID, Report. For each row, read ONLY the Report text and grade acl, mcl, mm, lm using these terms: acl: intact, mucoid_degeneration, reconstructed, sprain, partial_tear, complete_tear | mcl: intact, sprain, complete_tear | mm: intact, degenerative_signal, postoperative, tear, macerated | lm: intact, degenerative_signal, postoperative, tear, macerated. Rules: If not mentioned → intact. mucoid_degeneration/reconstructed → intact. degenerative_signal/postoperative → intact (not tears). Only sprain/tear/macerated/complete_tear are positive. Write results into graded_gold_structural.json as [{"StudyInstanceUID": "...", "acl": "...", "mcl": "...", "mm": "...", "lm": "..."}, ...]. Grade ONLY from the Report text. Do not read any other file in this folder.
```

**Step 2 — Prompt B (fluid/trauma), paste:**
```
I have gold_58_reports_only.csv in this folder. It has 58 rows with columns: StudyInstanceUID, Report. For each row, read ONLY the Report text and grade effusion, bakers, contusion, fracture using these terms: effusion: none, small, moderate, large | bakers: none, present, ruptured | contusion: none, present | fracture: none, present, suspected. Rules: If not mentioned → none. "trace"/"minimal" → small. "popliteal cyst" = bakers. For contusion: ONLY traumatic edema (not degenerative). For fracture: include avulsion/insufficiency. Write results into graded_gold_fluidtrauma.json as [{"StudyInstanceUID": "...", "effusion": "...", "bakers": "...", "contusion": "...", "fracture": "..."}, ...]. Grade ONLY from the Report text. Do not read any other file in this folder.
```

**Step 3 — scoring (OUTSIDE Antigravity, after both JSONs are written), run locally:**
```
python3 score_gold_test.py gold_58.csv graded_gold_structural.json graded_gold_fluidtrauma.json
```
Binary mappings: acl intact/mucoid_degeneration/reconstructed=0 else 1; mcl intact=0 else 1; mm/lm intact/degenerative_signal/postoperative=0 else 1; effusion none=0 else 1; bakers none=0 else 1; contusion none=0 else 1; fracture none=0, present/suspected=1.

**Decision rule:** if a prompt's findings beat the teacher's accuracy on those findings, run the full 4,276 with that prompt (500-row batches). Otherwise stop.
