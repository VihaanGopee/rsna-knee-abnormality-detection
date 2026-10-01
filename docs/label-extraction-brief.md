# Weak Supervision / Label Extraction from Radiology Reports
## RSNA Knee Abnormality Detection — Research Brief

**Date:** 2026-09-30
**Status:** Research synthesis from public repos, notebooks, and literature. Facts marked [VERIFIED] come from primary sources read directly; [INFERENCE] marks analyst conclusions.

---

## 1. The problem setup

- 4,407 train studies. Only **58** have expert gold labels for the 12 targets.
- The other 4,349 have free-text radiology reports (multilingual: Spanish, French, English, others — ~9–12 languages) and full DICOMs, but blank label columns.
- Reports are **not available at test time** — they can only be used to manufacture training targets.
- Metric is macro-AUC, so per-label label quality matters independently for all 12 targets.

The entire competition reduces to: **how good a teacher can you build from the reports, and how well can the imaging model learn from that teacher without memorizing its mistakes.**

---

## 2. Approaches in use by public teams

### 2a. LLM-based extraction (dominant approach)

**Sadam Torres — "Domain adaptation beats resolution: DINOv2 on knee" (public notebook, LB 0.883)** [VERIFIED]
- Local LLM: **Qwen3-14B-AWQ** served with **vLLM**, fixed JSON schema output.
- Reads each report into a **closed vocabulary**, then a deterministic map converts to probabilities.
- **Report-label quality: 0.881 against the 58 gold studies** — the best public number found.
- Source: existentialistlogarithmic/knee-abnormality `docs/COMPETITIVE_ANALYSIS.md` (author's stated numbers, credited).

**tranbadat2607 (public repo, actively developed)** [VERIFIED — read CLAUDE.md directly]
- LLM: **`gpt-5.4-mini`** (switched from `gpt-5.6-sol`; same quality, ~80x cheaper).
- Outputs **graded 0–1 labels** per target, not binary — soft targets as a drop-in training signal.
- Full corpus done: all 4,407 studies labeled, **2,415,243 tokens total** (~600 tokens/study), resumable + token-budget-aware pipeline (`scripts/llm_label_gold.py`).
- Training integration: gold studies use real labels at **sample weight 3.0** (`GOLD_WEIGHT`), everything else uses LLM labels.
- Anti-leakage: **folds grouped by report-text hash**, not random — studies sharing an identical report (→ identical derived label) stay in the same fold.
- First submission off this pipeline: **0.813 LB** (OOF macro AUC 0.7675 on the 58).

**liamd-schneider (public repo)** [VERIFIED — read README directly]
- LLM: **`Qwen/Qwen2.5-3B-Instruct`**, batched, greedy-decoded — sized to run on Kaggle's free GPU.
- Hand-written prompt with clinical definitions per finding (e.g., Synovitis broadened to include Hoffa fat-pad impingement/plica; "trace" effusion counts as present; Contusion requires traumatic pattern, not degenerative marrow edema; Fracture includes avulsion/insufficiency).
- Validation protocol (the right one): run the extractor on the 58 gold studies first, score against real labels before trusting it on the 4,349.
- **Reference solution's 35B model: 83.3% agreement**; their 3B model: **75.9% mean accuracy, 0.64 mean F1** across the 12 labels.
- 11/12 labels beat the majority-class baseline. MCL is the exception (81.0% vs 84.5% majority baseline — rare enough that always-negative wins on accuracy, though F1 0.52 shows real positives found).
- **~10% of reports fail JSON parsing** → conservatively defaulted to all-negative.
- Full run: 4,349/4,349 processed in ~4.7h (~0.24 studies/s on P100), 369 (8.5%) parse failures.
- Practical gotcha found: **`train.csv` is latin-1 encoded, not UTF-8** — pandas' default silently mangles accented characters in the Report column.

**jlawnat (public repo)** [VERIFIED — README]
- LLM soft labels + 2.5D inputs + ResNet18 → **0.789 public LB**.
- Uses 5-fold gold-only cross-validation and "TripletMean" study aggregation.

### 2b. Rule-based / regex extractors

**soumic28 (public repo)** [VERIFIED — README]
- Negation-aware multilingual regex extractor with Spanish negation filtering (`sin`, `no se observa`, `descartad[ao]`, `ausencia de`).
- Claimed 76x training-set expansion (58 → 4,407).

**tranbadat2607's rule-based baseline** [VERIFIED]: **0.814 mean agreement AUC** on the 58 gold — the number every LLM approach is measured against.

**existentialistlogarithmic's hand-built multilingual lexicon** [VERIFIED]: **0.769** vs Torres' LLM at **0.881** — "the label gap is the headline. Everything the imaging model learns is bounded by what its teacher knows."

### 2c. Measured label-quality leaderboard (vs 58 gold studies)

| Extractor | Score vs gold | Metric | Source |
|---|---|---|---|
| Qwen3-14B-AWQ (Torres) | **0.881** | agreement | COMPETITIVE_ANALYSIS.md |
| gpt-5.4-mini / gpt-5.6-sol (tranbadat2607) | **0.869** | mean agreement AUC | repo CLAUDE.md |
| 35B reference model | **83.3%** | accuracy | liamd-schneider README |
| Rule-based baseline | **0.814** | mean agreement AUC | tranbadat2607 |
| Qwen2.5-3B-Instruct | 75.9% acc / 0.64 F1 | accuracy / F1 | liamd-schneider |
| Hand-built multilingual lexicon | 0.769 | agreement | existentialistlogarithmic |
| Report-derived ≈ image labels | ~82% | agreement | existentialistlogarithmic (UNVERIFIED) |

Per-target extremes (tranbadat2607, gpt-5.4-mini): **ACL strongest (~0.97–0.99), Synovitis weakest (~0.68).**

---

## 3. Key technical details that decide quality

1. **Graded (soft) labels beat binary.** tranbadat2607's LLM outputs 0–1 graded scores per target. This preserves the model's uncertainty instead of thresholding it away, and trains better under BCE. [VERIFIED]
2. **Validate the extractor on the 58 before trusting it.** Both serious teams (liamd-schneider, tranbadat2607) run extraction on the gold 58 and score against truth first. This is the single most important methodological rule. [VERIFIED]
3. **Fold by report-text hash, not randomly.** Identical reports → identical derived labels; random folds leak these across train/val. tranbadat2607 groups folds by report hash. Related: existentialistlogarithmic measured **random K-fold inflates AUC by 0.05–0.14 (measured 0.087)** — use scanner/grouped splits. [VERIFIED]
4. **Upweight gold.** tranbadat2607 uses GOLD_WEIGHT=3.0 for the 58 real labels vs LLM labels. [VERIFIED]
5. **Handle parse failures conservatively.** ~8–10% of LLM outputs fail JSON parsing; defaulting to all-negative is the safe choice, not silent guessing. [VERIFIED]
6. **Closed vocabulary + deterministic mapping.** Torres reads into a closed vocabulary then maps deterministically to probabilities, rather than asking the LLM for raw probabilities — constrains hallucination. [VERIFIED]
7. **Clinical definitions in the prompt matter.** The liamd-schneider prompt's per-finding definitions (trace effusion = present, contusion = traumatic pattern only, etc.) are doing real work — these encode the annotation conventions of the gold labels. [VERIFIED]
8. **Report-label CV ranks models; it never estimates the leaderboard.** existentialistlogarithmic's working rule. A metadata-only model scored 0.669 in scanner-grouped CV against report labels but 0.531 on the LB against expert labels — a **0.138 gap** measuring how far report-derived targets sit from the scored truth. [VERIFIED]
9. **Beware leaky public label sets.** "Three of four public label sets [surveyed 2026-09-07] are answer keys" — i.e., contaminated. Only use label sets you generate yourself or have verified. [VERIFIED — existentialistlogarithmic PATH.md]

---

## 4. The fundamental ceiling: report labels ≠ image labels

This is the most important conceptual finding:

- **VisualCheXbert** (Jain et al., arXiv 2102.11467): a biomedically-pretrained BERT that maps radiology reports **directly to image labels** (not report labels) agrees with radiologists labeling images better than radiologists labeling the corresponding reports do — by **F1 0.12–0.21**, and beats the CheXpert rule-based report labeler by **F1 0.14**. The report is a lossy, incomplete proxy for what's visible in the image. [VERIFIED — literature]
- **Synovitis is the concrete case:** existentialistlogarithmic found only **13 of 27 true Synovitis cases are written about at all** — a perfect report reader caps at **0.8076** on that label, and their model already scores 0.790 there. No label-extraction improvement can fix findings the radiologist didn't dictate. [VERIFIED]
- Implication: label extraction has a **per-label ceiling set by report completeness**, not just extractor accuracy. The labels most worth improving are the ones that are both frequently written about AND poorly extracted (ACL/Meniscus-type findings), not the ones that are rarely mentioned.

---

## 5. SOTA literature: rule-based → BERT → LLM labelers

Chest X-ray labeling (MIMIC-500 benchmark), the closest studied analogue:

| Labeler | Macro F1 | Source |
|---|---|---|
| CheXpert (rule-based) | 0.8864 | preprints.org 2025 |
| CheXbert (BERT fine-tuned on 1,000 radiologist labels + back-translation) | 0.9047 | Smit et al. |
| GPT-4 direct | 0.9014 | preprints.org 2025 |
| CheX-LLM (LLM fine-tuned + RLHF on labeling) | **0.9115** | preprints.org 2025 |

Key points:
- CheXbert: F1 0.798 on MIMIC-CXR, +0.045 over the CheXpert rule-based labeler, 0.007 short of a board-certified radiologist. Fine-tuning included **back-translation augmentation** (English→German→English rephrasing). [VERIFIED — deeplearning.ai summary of Smit et al.]
- The fine-tuned LLM (CheX-LLM) beats GPT-4 direct — **specialized training on the labeling task beats a bigger general model**. Its biggest gains were on findings with complex descriptions and frequent uncertainty/negation. [VERIFIED — preprints.org]
- No MSK-MRI-specific labeler exists in the literature — CheXpert/CheXbert/NegBio are all chest X-ray. The knee competition is the first large-scale test of these ideas on MSK MRI. [INFERENCE]

---

## 6. Multilingual handling

- Reports span ~9–12 languages (Spanish, French, English confirmed; others per RSNA).
- **LLM approaches handle multilinguality natively** — no per-language engineering. This is the decisive advantage of LLM extraction over rules here. [VERIFIED — all LLM teams use one prompt]
- Rule-based approaches need per-language lexicons + negation lists (soumic28's Spanish negation patterns; existentialistlogarithmic's hand-built multilingual lexicon at 0.769). [VERIFIED]
- Practical: `train.csv` is **latin-1 encoded** — must be read with `encoding='latin-1'` or accented characters (common in Spanish/French reports) are silently mangled. [VERIFIED — liamd-schneider]
- TianK003's methodology note: label changes are judged on **coverage per language + OOF over all 4,407**, never on the 58 gold alone — i.e., check the extractor doesn't collapse on some language subset. [VERIFIED]

---

## 7. Negation and uncertainty handling

- Rule-based: explicit negation token lists per language (`sin`, `no se observa`, `descartad[ao]`, `ausencia de` for Spanish). Fragile across 9–12 languages. [VERIFIED — soumic28]
- LLM: negation/hedging handled implicitly by the model; the CheX-LLM paper shows the largest LLM gains over rules are exactly on **negated and uncertain mentions**. [VERIFIED — literature]
- Uncertainty → graded labels: tranbadat2607's 0–1 graded outputs naturally encode "possible"/"probable"/"cannot exclude" hedging as intermediate scores instead of forcing a binary choice. This is preferable to CheXpert-style U-Zeros/U-Ones collapsing. [VERIFIED + INFERENCE]
- MIT 2026 study (Alhamoud et al.): vision-language models often **ignore negation** (~25% retrieval drop, near-random on negated MCQs) — a warning if any VLM-based labeling is attempted; text-only LLM extraction doesn't share this failure mode. [VERIFIED — dotmed summary]

---

## 8. What private / top teams likely do differently [mostly INFERENCE]

Verified context:
- The public top (~0.942–0.957) is **one shared, heavily-forked community ensemble**; its own author warns it is "likely overfit to the public leaderboard" — expect a private shakeup. (TianK003, VERIFIED)
- TianK003's decomposition of a 0.936 notebook: **"It trains nothing: every member is a mounted public checkpoint."** DINOv2-S ≈ 0.899 → +16-slice ViT + RadImageNet R50 + stacking ≈ 0.920 → +CoAtNet-2@384 ≈ 0.935 → +gold-58-tuned weights → 0.936. (VERIFIED)
- ~81 teams ≥0.945 in a dense wall — everyone holds the same ensemble members.

Where a private team buys separation on the label side:
1. **Better teacher, not just bigger ensemble.** Best public teacher = 0.881 (Qwen3-14B). A 35B-class model hits 83.3% accuracy; a fine-tuned labeler (CheX-LLM pattern: fine-tune on the 58 gold + back-translation augmentation) could push past both. Nobody public has done the fine-tuned-labeler step. [INFERENCE]
2. **Actually training on 4,349 studies** instead of mounting checkpoints trained by others. The public 0.936 stack trains nothing — its members' teachers are fixed. A custom-trained model with a better teacher + noise-aware loss is the main open lane. [INFERENCE]
3. **Labeler ensembles + confidence weighting.** Multiple diverse labelers (different LLMs/prompts) → per-study/per-label agreement as a confidence weight in the imaging loss. Downweight studies where labelers disagree; this is standard noisy-label practice and absent from public repos. [INFERENCE]
4. **Gold-anchored calibration.** 58 gold labels are enough to fit per-label calibration (Platt/isotonic) mapping LLM graded scores → gold-conformant probabilities, correcting systematic over/under-calling per finding. [INFERENCE]
5. **Exploiting the report↔image gap (VisualCheXbert direction).** Train the imaging model with an auxiliary objective that predicts image-grounded labels, or distill from a model trained gold-only — partially correcting for findings the reports omit (Synovitis-type incompleteness). [INFERENCE]
6. **Scanner-grouped validation.** Public forks mostly use random folds (inflates AUC 0.05–0.14). A team validating honestly is less likely to overfit the public LB and more likely to survive the private shakeup. [VERIFIED measurement, INFERENCE on who does it]

---

## 9. Recommendations for our pipeline

1. **Build our own LLM labeler; validate on the 58 first.** Target: beat 0.881 (Torres/Qwen3-14B). Use a 30B+ class model or an API model for the one-time labeling run; graded 0–1 outputs; closed vocabulary + deterministic mapping; clinical definitions per finding in the prompt.
2. **Consider fine-tuning the labeler** (CheX-LLM pattern): fine-tune a small LLM on the 58 gold reports with back-translation augmentation. Highest-upside, lowest-explored step.
3. **Generate the full-corpus label set once, offline** (Kaggle rules: internet off at scoring; labels must be precomputed and mounted, or extracted with a local model inside the notebook).
4. **Train custom imaging models on the 4,349** with noise-aware training: gold upweighting (3.0x), labeler-confidence sample weights, label smoothing on weak labels, report-hash-grouped folds.
5. **Per-label calibration** of LLM scores against the 58 gold before training.
6. **Never trust report-label CV for absolute estimates** — it ranks, it doesn't estimate. Keep a gold-only holdout for honest numbers.
7. **Track coverage per language** for any label-set change (TianK003's rule).
8. **Treat Synovitis-type labels as completeness-capped** — don't burn labeler effort where the report itself is the ceiling; spend it on frequently-mentioned, poorly-extracted findings.

---

## 10. Open questions

- Exact per-language distribution of the 4,349 reports (needed to check extractor coverage per language).
- Whether the hidden test reports come from the same sites/languages as train (site shift would degrade a report-tuned pipeline — but reports aren't used at test time, so this only affects teacher quality, not inference).
- The efficiency-track formula and whether a custom-trained single model can compete there (a distilled single model could win efficiency while the ensemble chases accuracy).
- What the actual private-LB shakeup looks like — unknowable until winners are announced (early November 2026).

---

## Sources

- https://github.com/tranbadat2607/rsna-knee-abnormality-detection/blob/HEAD/CLAUDE.md
- https://github.com/liamd-schneider/rsna-knee-abnormality-detection/blob/HEAD/README.md
- https://github.com/existentialistlogarithmic/knee-abnormality/blob/HEAD/docs/COMPETITIVE_ANALYSIS.md
- https://github.com/existentialistlogarithmic/knee-abnormality/blob/HEAD/docs/PATH.md
- https://github.com/existentialistlogarithmic/knee-abnormality/blob/HEAD/docs/FINDINGS.md
- https://github.com/tiank003/rsna-kneemri-kaggle-competition/blob/HEAD/CLAUDE.md
- https://github.com/jlawnat/knee-mri-abnormality-detection/blob/HEAD/README.md
- https://github.com/soumic28/rsna-knee-abnormality-predictin (README via search)
- Jain et al., VisualCheXbert, arXiv:2102.11467
- Smit et al., CheXbert (via deeplearning.ai summary)
- CheX-LLM, preprints.org/manuscript/202504.1668 (2025)
- Alhamoud et al., MIT VLM negation study 2026 (via es.dotmed.com)
