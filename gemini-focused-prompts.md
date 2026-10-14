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

## Gold test (58 studies)

```
I have gold_58.csv in this folder. It has 58 rows with columns: StudyInstanceUID, Report, ACL, MCL, Medial Meniscus, Lateral Meniscus, Effusion, Baker's, Contusion, Fracture.

For each row, read the Report text, grade [acl, mcl, mm, lm] using the terms above, then compare to the expert columns. Report accuracy per finding.

Map to binary: acl intact/mucoid_degeneration/reconstructed=0, sprain/partial_tear/complete_tear=1. mcl intact=0, sprain/complete_tear=1. mm/lm intact/degenerative_signal/postoperative=0, tear/macerated=1.
```

Do the same for [effusion, bakers, contusion, fracture] vs expert columns.
