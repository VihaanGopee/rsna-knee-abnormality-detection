# RSNA Knee Abnormality Detection

Kaggle competition: [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

## Task
Detect 12 knee abnormalities from MRI studies (DICOM). One prediction per **study** (not per slice): 12 confidence scores.

**Labels:** ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture

## Metric
Macro-averaged AUC ROC across the 12 targets.

## Key Dates
- Entry / team merger deadline: **October 15, 2026**
- Final submission deadline: **October 22, 2026 11:59 PM UTC** (~22 days)

## Prizes — $77,000 total
- **Main leaderboard:** 1st $9K, 2nd $7K, 3rd $6.5K, 4th $6K, 5th $5.5K, 6th–10th $5K each
- **Efficiency track:** ~$18K for most efficient models (separate leaderboard)

## Data
- **~570 GB, 819,640 DICOM files** — one of Kaggle's largest datasets
- Train: 4,407 studies, 24,371 series (Sagittal / Coronal / Axial)
- ⚠️ **Only 58 studies have all 12 labels.** The other 4,349 have DICOM + radiology report (multilingual, ~9-12 languages) but blank labels.
- Test: ~1,300 studies (hidden; visible test.csv is a 3-row stub)
- Reports are **NOT available at test time**

## The Core Challenge
This is a **weak-supervision problem**: 58 gold labels + 4,349 reports → extract training signal.
Standard approach: LLM label extraction from reports → train image models → fine-tune on 58.

## Leaderboard (Sep 2026 snapshots)
- Top: ~0.957 | Dense wall at 0.943–0.945 (~81 teams ≥0.945)
- Top 10 within ~0.006 band — heavily-forked community ensembles
- Analysts warn of public-LB overfitting; private shakeup expected

## Submission Format
```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
```

## Constraints
- Notebook-only, internet OFF, ≤9h runtime
- GPU: 1× P100 or 2× T4 (~30h/week quota)
- 5 submissions/day

## Structure
- `notebooks/` — Kaggle submission notebooks (canonical versions)
- `src/` — shared Python modules
- `data/` — local data notes (no large files)
- `docs/` — research notes and findings

## Workflow
1. Muse builds notebooks, pushes to this repo
2. Justin downloads from GitHub → uploads to Kaggle → runs → submits
3. Justin sends back logs, Muse debugs

## Rules
- Every submission artifact validated locally before Justin runs anything
- One canonical notebook per drop, stale versions explicitly marked
- No submission ships without passing all validation gates
- Test set size never hardcoded (visible test.csv is a stub)
