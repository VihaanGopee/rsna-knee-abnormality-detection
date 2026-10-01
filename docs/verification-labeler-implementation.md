# Verification Brief: Torres's Labeler (Qwen3-14B-AWQ) — Implementation Details
**Date:** 2026-10-01 (research run 2026-09-30 ~19:00 PDT)
**Question:** What are the exact mechanics of the best-measured public label extractor (Sadam Torres, Qwen3-14B-AWQ + vLLM, closed vocabulary → deterministic probability map, 0.881 agreement vs gold-58, feeding a 0.883-LB system), and how do public teams adapt grayscale MRI slices to DINOv2?

## What was verified (primary sources read 2026-10-01)

### Source 1: Torres's own repo — `existentialistlogarithmic/knee-abnormality`
- README documents the full system: **DINOv2 ViT-S/14 backbone + v2.5D max/mean/std triplet aggregation**, RadImageNet ResNet-50 cross-plane attention arm, 336px, rank-mean fusion. LB **0.883** public. Inference ~25 s/study.
- **Data sourcing stated plainly:** "Labels: LLM-generated from radiology reports (Qwen3-14B-AWQ) on 4,407 studies + 58 gold studies, abstain-masked on missing mentions, gold 8× upweight."
- Training recipe: 40 epochs, asymmetric focal loss, cosine anneal, EMA; DINOv2 backbone **frozen** for first 5 epochs, then full fine-tune; early stopping on gold-58 fold val AUC; fold-by-scanner; 2.5D slice aggregation (max/mean/std triplet per orientation, concatenation, attention over slices); fold averaging; rank-mean fusion.
- **Key training signal detail:** they use the 58 gold studies for validation, "frozen DINOv2 weights for first 5 epochs then full fine-tune" — the weak-label noise is why the backbone stays frozen early.

### Source 2: COMPETITIVE_ANALYSIS.md in that repo (author: the "Domain adaptation beats resolution" notebook author — Torres himself)
The labeler paragraph, verbatim (the only public statement of the mechanics):

> "The labels this trains on are LLM-extracted from the DICOM radiology reports. The labeler that created them: a local LLM (Qwen3-14B-AWQ, vLLM, fixed JSON schema) reading into a closed vocabulary, then a deterministic map to probabilities. … The closed-vocabulary-then-deterministic-map design is the part worth copying: ask the model to pick from a closed vocabulary, then map vocabulary to probabilities in deterministic Python, because a model is good at the first job and bad at the second."

And the probability table:

> "The probability values are measured positive rates (an effusion described as mild is positive about 45% of the time) — ask your LLM to emit them and you're measuring your model's calibration bias, not the data."

FINDINGS.md in the same repo adds one design constraint: the author **deliberately did not re-run** Torres's labeler — "re-running a large LLM inference job … is beyond the scope of this analysis." Nobody has independently reproduced the labeler; the 0.881 number comes from Torres's own report of it (referenced in our label-extraction brief as "Torres's Qwen3-14B-AWQ labeler: 0.881 agreement vs gold-58").

### Source 3: tranbadat2607/rsna-knee-abnormality-detection — `scripts/llm_label_gold.py` (771 lines, read in full)
A complete, working, replicable implementation of the *same task* (grade 12 findings 0–1 from a multilingual report). Different engine (OpenAI gpt-5.4-mini API, not local Qwen) and different design (direct graded output, not closed-vocab), but every engineering detail is exposed:
- **System prompt:** "expert musculoskeletal radiologist assistant," 12 findings with one-line definitions in fixed order, reports in any of ~10 languages.
- **Scoring rubric (3 bands):**
  - 0.0–0.2: explicitly normal/intact or explicitly negated
  - 0.3–0.5: not mentioned at all, or hedged ("possible," "cannot exclude")
  - 0.6–1.0: explicitly asserted; scaled by severity language ("large/complete tear" high, "trace/small/mild" low)
- 6 multilingual few-shot examples (EN, FR, ES, TR, EL) — also serve as prompt-cache padding
- Structured output: `{"results": [{"i": N, "s": [12 numbers]}]}`, strict JSON schema, **positional arrays** (no repeated key strings) to save output tokens
- `max_tokens = batch_size * 70 + 200`; missing reports → fill 0.5s; clamp [0,1]; whitespace normalization; batched requests; resumable (skips done StudyInstanceUIDs); `reasoning_effort="none"` (measured 3× token cut, no quality drop on spot check vs gold)
- Their measured numbers: rule-based extractor 0.814 vs gold-58; LLM direct-grading **0.869** vs gold-58.

## Question 1: Torres's closed-vocab mechanics — what we know and what's missing

**Known:**
- Engine: Qwen3-14B-AWQ served with vLLM, locally. Fixed JSON schema output.
- Architecture: model outputs a **closed-vocabulary descriptor** per finding (e.g. effusion → one of {absent, trace, mild, moderate, large, …}); **deterministic Python** maps descriptor → probability using **measured positive rates** per descriptor (e.g. "mild effusion" → 0.45).
- Rationale (Torres): the model is good at *classifying language* (pick the descriptor) and bad at *calibrating numbers*; don't ask it to emit probabilities.
- Validation: 0.881 agreement vs gold-58 (Torres's own measurement). System built on these labels: 0.883 public LB.

**Not recoverable from public sources (genuine gaps):**
- The exact closed vocabulary per finding (descriptor list)
- The exact prompt template
- The full descriptor→probability table (only the "mild effusion ≈ 0.45" example is quoted)
- vLLM config (tensor parallel, batch size, measured throughput/timing)
- How the "measured positive rates" were estimated (gold-58? spot checks? corpus statistics?) — see calibration note below
- Their gold-58 validation methodology (macro-AUC of extracted labels vs gold? per-label?)
- The Kaggle notebook itself ("Domain adaptation beats resolution: DINOv2 on knee") is JS-rendered and not fetchable via text tools; no mirror of its code exists. The one repo that studied Torres (above) explicitly declined to re-run the labeler.

## Question 2: closed-vocab + deterministic map vs direct graded output

This is now answerable from two public data points plus engineering reasoning:

| | Torres: closed-vocab → deterministic map | tranbadat2607: direct graded output |
|---|---|---|
| Engine | Qwen3-14B-AWQ (local, vLLM) | gpt-5.4-mini (API) |
| Model's job | classify language → descriptor | classify language AND calibrate number |
| Mapping | deterministic Python table (measured rates) | model emits 0–1 directly |
| vs gold-58 | **0.881** | **0.869** |
| Reproducibility | high — mapping is code, not vibes | prompt/model-version sensitive |

**Verdict for our Phase 1: closed-vocab + deterministic map is the right design, and the comparison is (mildly) confounded.** The 0.881 vs 0.869 gap favors closed-vocab but the engines differ (Qwen3-14B vs gpt-5.4-mini), so it is not a clean A/B. The real argument is structural:
1. It removes the model's calibration bias from the loop — Torres's explicit point, and a known LLM weakness (models emit 0.7-ish mush regardless of true rates).
2. It is **less prompt-sensitive** than direct grading, which directly answers "is 0.881 replicable without their exact prompt": the number we must reproduce is not in the prompt, it's in the calibration table — which we build ourselves from data.
3. Deterministic mapping makes the label set auditable (we can print the full table; a reviewer/Justin can sanity-check "mild effusion → 0.45").

**Recommended Phase 1 design (combines both sources):**
- Engine: Qwen3-14B-AWQ + vLLM on Colab T4 (16 GB fits the AWQ quant; ~7–9 GB). 4,407 reports is a batch job — a few hours, well inside the ~20u label-extraction budget. T4 lacks flash-attention; slower than A100/L4 but fine for an offline labeling job.
- Prompt: tranbadat2607's system-prompt skeleton (role, 12 definitions, multilingual handling, few-shot) adapted to emit **descriptors** instead of numbers.
- Mapping: deterministic table; **calibrate the table ourselves** on our own gold-58 agreement runs (see below) — do not hand-tune to Torres's 0.45 example.
- Engineering: batch reports per call, strict JSON schema, positional arrays, resumable output keyed by StudyInstanceUID, missing → abstain (not 0.5!) for the *training* labels per our abstain-masking doctrine; 0.5-fill only as a parse fallback inside the labeler.

## Calibration-table provenance (important subtlety)

"Measured positive rates" needs a denominator. With only 58 gold studies, a per-descriptor rate like P(gold=1 | report says "mild effusion") is estimated on a handful of samples — noisy. Options for how Torres may have done it, and what we should do:
1. **Estimate from gold-58** (noisy but unbiased; ~58 studies × mention-rate).
2. **Estimate from the LLM-labeled corpus itself** (large-N but circular — the labels you're calibrating become the ground truth).
3. **Hybrid:** LLM corpus rates as prior, gold-58 as the check; shrink noisy cells toward the corpus rate.

Our call: **option 3**, with the table construction logged and the gold-58 agreement number reported per label. Note the contamination warning from FINDINGS.md: 3 of 4 surveyed public label sets **copy gold-58 labels into their gold rows** — if we ever download Torres's labels for comparison, never evaluate our labeler on their gold rows, and never calibrate our table on them. Our gold-58 ground truth is train.csv only.

## Replicability verdict: is 0.881 reachable without Torres's exact prompt?

**Yes, with the right expectations.** The 0.881 is a property of (a) the closed-vocab architecture, (b) a decent multilingual MSK prompt (tranbadat2607's is public and complete — we can start from it), and (c) a calibration table measured on our own data. None of these require Torres's exact tokens. What we should NOT expect: to hit exactly 0.881 on the first run. Target: **≥0.86 agreement vs gold-58** as the Phase 1 exit gate (beats the best public direct-grading number 0.869 only if the architecture advantage is real; 0.86 clears "best measured method" bar with margin for our engine differing). If we land 0.86–0.88, the labels are Tier-1 committed quality.

## Question 3: grayscale MRI → DINOv2 adaptation

**What the public knee repos do:** nothing exotic is documented anywhere. Torres's README specifies the backbone and aggregation but not the channel handling; the analysis repo's FINDINGS.md doesn't mention it either. No public knee repo documents a learned stem, cross-attention adapter, or 1-channel patch embed.

**What DINOv2 expects (verified from HF docs/config):** `num_channels=3`, ImageNet preprocessing — resize shortest edge 256, center crop 224, rescale [0,1], normalize with ImageNet mean/std. The model takes already-normalized input.

**Standard practice (universal across medical-imaging DINOv2 work, and the only sane default):** replicate the grayscale slice to 3 channels, then apply the standard ImageNet normalization. This keeps the pretrained patch-embed weights valid (their 3 input channels all see the same signal; the learned filters respond to intensity patterns as if they were luminance).

**Alternatives considered and rejected for Phase 2a:**
- Learned 1→3 conv stem: adds parameters to fit on weak labels; no public knee team reports needing it; DINOv2's patch embed is part of what we want to keep frozen early anyway.
- Feeding single channel with weight surgery (summing/averaging the 3-channel patch-embed weights): mathematically equivalent-ish to replication at init but diverges under fine-tuning in unprincipled ways; zero public evidence of benefit.
- True 3-channel stacking (e.g. adjacent slices as channels): that's the 2.5D-triplet design operating *above* the encoder, not a stem change — Torres already does max/mean/std triplets per orientation at the aggregation layer. Don't conflate the two.

**Decision:** replicate grayscale → 3 channels, ImageNet mean/std, standard DINOv2 preprocessing. If a Phase-2 ablation slot is free, a 10-minute probe comparing replication vs patch-embed weight-averaging is cheap — but replication is the default, not the experiment.

## Timing / compute notes for Phase 1

- No public throughput numbers exist for Torres's labeling run (gap noted above). Estimate: with vLLM on a Colab T4, batched multilingual reports with short JSON outputs, 4,407 reports should complete in roughly **3–8 hours** — a single session, inside the ~20u budget. **Phase 1 must start with a 100-report timing probe** before the full run (per EXECUTION_PROTOCOL doctrine: measure, then commit).
- vLLM on T4: use `--enforce-eager` if CUDA-graph memory issues appear (T4 + AWQ is a known finicky combo); keep batch sizes modest; the task is throughput-bound on output tokens (short), so this is not a quota risk.

## Bottom line

- Torres's exact prompt/vocab/table are **not public and not recoverable** — but they are also **not needed**. The transferable invention is the architecture (closed-vocab → deterministic measured-rate map), which is fully described, plus a complete public graded-labeling implementation (tranbadat2607's script) we can adapt as the prompt skeleton.
- Grayscale → DINOv2: replicate to 3 channels + ImageNet norm. No evidence any public team does otherwise.
- Phase 1 exit gate: **≥0.86 macro-agreement vs gold-58** (train.csv ground truth, never Torres's label files for gold rows), timing probe first, abstain-preserving outputs for training.
