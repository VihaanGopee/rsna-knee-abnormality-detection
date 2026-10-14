# Gemini Focused Labeling Prompts — Remaining 8 Findings

**Why focused:** The 4-finding severity prompt hit 100% (58/58) vs gold. The 12-finding all-in-one dropped to 0.8319. Focus wins.

**How to use:** Copy each prompt, paste into Antigravity Gemini 3.8 Flash, paste the report text where indicated. Output is 2 lines per report.

**Batch files:** Same as severity — graded_batch_N_M.json format, but for these findings.

---

## Prompt 1: Ligaments (ACL, MCL)

```
You are a musculoskeletal radiologist. Read the knee MRI report below
(it may be in English, Spanish, German, Portuguese, or French).
For each of the 2 findings, output EXACTLY ONE term from the allowed list.
If the report does not address a finding, use 'intact'.

IMPORTANT:
- For ACL: 'mucoid_degeneration' and 'reconstructed' count as intact (not tears)
- For ACL: 'sprain', 'partial_tear', 'complete_tear' are all positive
- For MCL: 'sprain' and 'complete_tear' are positive

Output format: 2 lines, each 'key: term' (e.g. 'acl: partial_tear').
No other text.

acl (anterior cruciate ligament (ACL)): intact, mucoid_degeneration, reconstructed, sprain, partial_tear, complete_tear
mcl (medial collateral ligament (MCL)): intact, sprain, complete_tear

REPORT:
[PASTE REPORT HERE]

LABELS:
```

---

## Prompt 2: Menisci (Medial, Lateral)

```
You are a musculoskeletal radiologist. Read the knee MRI report below
(it may be in English, Spanish, German, Portuguese, or French).
For each of the 2 findings, output EXACTLY ONE term from the allowed list.
If the report does not address a finding, use 'intact'.

IMPORTANT:
- 'degenerative_signal' (intrasubstance signal, not reaching surface) counts as intact, NOT a tear
- 'postoperative' (prior meniscectomy/meniscal surgery) counts as intact
- Only 'tear' and 'macerated' are positive
- A "complex tear" or "bucket-handle tear" = tear

Output format: 2 lines, each 'key: term' (e.g. 'mm: tear').
No other text.

mm (medial meniscus): intact, degenerative_signal, postoperative, tear, macerated
lm (lateral meniscus): intact, degenerative_signal, postoperative, tear, macerated

REPORT:
[PASTE REPORT HERE]

LABELS:
```

---

## Prompt 3: Fluid (Effusion, Baker's Cyst)

```
You are a musculoskeletal radiologist. Read the knee MRI report below
(it may be in English, Spanish, German, Portuguese, or French).
For each of the 2 findings, output EXACTLY ONE term from the allowed list.
If the report does not address a finding, use 'none'.

IMPORTANT:
- For effusion: count even 'trace' or 'minimal' fluid as 'small' (not none)
- For effusion: 'small', 'moderate', 'large' are all positive
- For Baker's cyst: 'present' and 'ruptured' are both positive
- A "popliteal cyst" = Baker's cyst

Output format: 2 lines, each 'key: term' (e.g. 'effusion: small').
No other text.

effusion (joint effusion): none, small, moderate, large
bakers (Baker's cyst): none, present, ruptured

REPORT:
[PASTE REPORT HERE]

LABELS:
```

---

## Prompt 4: Trauma (Contusion, Fracture)

```
You are a musculoskeletal radiologist. Read the knee MRI report below
(it may be in English, Spanish, German, Portuguese, or French).
For each of the 2 findings, output EXACTLY ONE term from the allowed list.
If the report does not address a finding, use 'none'.

IMPORTANT:
- For contusion: ONLY mark 'present' for traumatic bone marrow edema pattern
  (e.g., "kissing contusions", post-traumatic edema). Do NOT mark for
  ordinary degenerative changes or osteoarthritis-related edema.
- For fracture: include avulsion fractures and insufficiency fractures
- For fracture: 'suspected' (e.g., "cannot exclude fracture") counts as positive
  in the binary mapping but use the term 'suspected' in output

Output format: 2 lines, each 'key: term' (e.g. 'contusion: present').
No other text.

contusion (bone contusion): none, present
fracture (fracture): none, present, suspected

REPORT:
[PASTE REPORT HERE]

LABELS:
```

---

## Binary mapping (for merge)

After collecting terms, map to binary using:

- acl: intact/mucoid_degeneration/reconstructed=0; sprain/partial_tear/complete_tear=1
- mcl: intact=0; sprain/complete_tear=1
- mm/lm: intact/degenerative_signal/postoperative=0; tear/macerated=1
- effusion: none=0; small/moderate/large=1
- bakers: none=0; present/ruptured=1
- contusion: none=0; present=1
- fracture: none=0; present/suspected=1
