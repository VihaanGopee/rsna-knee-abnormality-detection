# Training Strategy Brief — RSNA Knee Abnormality Detection (2026)

**Date:** 2026-09-30
**Author:** research subagent
**Status:** synthesis of verified competition facts + community experiment logs + ML literature

> **Tag legend** (borrowed from existentialistlogarithmic/knee-abnormality, the most rigorous public experiment log):
> `VERIFIED` = read directly from competition files/API/pages or measured in a logged experiment.
> `INFERRED` = strong community consensus, not directly measured. `LITERATURE` = from published papers, not this competition.

---

## 0. Executive summary

The competition is a **weak-supervision problem by design** (the host's data description explicitly says reports exist "from which you may wish to derive the labels for the remaining studies"). The training-strategy game has three layers:

1. **Label quality** — how good your report-derived labels are (agreement with image labels ~82% is the `UNVERIFIED` community estimate; best clean teacher measured 0.899 gold agreement).
2. **Model architecture + slice aggregation** — CoAtNet-2.5D with per-finding attention-MIL is the strongest measured family (0.912–0.920 gold-58 per checkpoint); DINOv2/RadImageNet arms are weaker (0.840/0.854) but add diversity.
3. **Ensemble scale + blend discipline** — "adding members is the only operation that has ever moved this board" (existentialistlogarithmic, from a 2×2 submission experiment). But blend weights tuned on the public LB are **LB-probing, not learning** — the 0.939→0.941 public gains came from per-target weights fit to the public split, and analysts warn of a private shakeup.

The gap between the best reproduced public recipe (**0.943**) and the board top (**0.957**) is 0.014. The dense wall (837 teams ≥0.938, only 81 ≥0.945) is consistent with hundreds of teams forking the same 2–3 public notebooks (492 teams sit at *exactly* 0.936, 163 at *exactly* 0.937 — fork fingerprints).

---

## 1. Combining 58 gold labels with 4,349 weak labels

### 1.1 The label landscape (`VERIFIED` facts)

- Exactly 58 studies have all 12 expert labels; the other 4,349 have reports only. No partial middle ground.
- Reports are multilingual: en 39.3%, es 15.6%, tr 12.4%, el 7.4%, hr 7.3%, de 5.9%, bg 5.0%, nl 3.5%, fr 1.9%, bs 1.8% (py3langid detection).
- **Five public label tables copy the gold-58 labels verbatim**: barun2104 stratified folds, rayanbabur calibrated targets, tasmeemreza refined labels, zaidaliiq1000, yunusgmsoy 4-source-merged. Using any of them means your "weak" labels contain the gold labels — a leakage-shaped advantage on public LB that will not transfer to private.
- Best clean teacher measured: **flight0234 hybrid, 0.899 gold agreement** (tranbadat's ledger). Their own labeler: 0.855.
- Publicly shared CC0 report labels exist: `stevenleehans/rsna-knee-llm-report-labels` (repackaged by existentialistlogarithmic as `knee-phase1-public`).

### 1.2 How top teams weight gold vs weak (community practice, `INFERRED` from logs)

- **Gold override + upweighting:** existentialistlogarithmic's training rules: *"Gold studies carry 8x weight and override report-derived targets, being the only labels known to match what is scored."*
- **Abstain-masking:** their labeler emits a five-channel target; when a report is silent on a finding, that finding contributes **no gradient** (masked loss) rather than teaching a negative. Rationale: absence of mention ≠ absence of finding, especially in terse multilingual reports. This is one of the highest-leverage design decisions in the weak-supervision layer.
- **Soft targets:** jlawnat's pipeline (0.789 LB) trains ResNet18 on LLM-derived **soft labels** (probabilities, not hard 0/1) with 5-fold gold-only CV. Soft labels preserve the labeler's uncertainty; hard thresholding at 0.5 discards it.
- **Two-stage schedule** (standard, from Kaggle SSL practice + the AES-2 1st-place writeup pattern):
  1. **Stage 1:** train on all 4,407 with weak/soft labels (possibly abstain-masked), heavy augmentation.
  2. **Stage 2:** fine-tune on the 58 gold (upweighted) with light augmentation and low LR.
  The AES-2 winner's two-staging (+0.015) is the closest public precedent for "pretrain on noisy, fine-tune on clean" winning a competition.

### 1.3 What NOT to do

- Do not train on the five public tables that embed gold-58 verbatim and believe your CV. Your validation is contaminated.
- Do not treat "report silent" as negative without evidence — the abstain-mask result says this teaches false negatives.

---

## 2. Semi-supervised techniques applicable here

### 2.1 Noisy Student / self-training (`LITERATURE`, Xie et al. 2019, arXiv:1911.04252)

The canonical recipe, directly applicable:
1. Train a teacher on labeled data (here: 58 gold + weak labels).
2. Teacher generates **soft** pseudo-labels on unlabeled data (soft > hard when teacher is imperfect — the paper shows hard pseudo-labels hurt with out-of-domain data).
3. Train an **equal-or-larger student** on labeled + pseudo-labeled data with **noise**: input noise (RandAugment) + model noise (dropout, stochastic depth). The noised student is forced to learn harder than the clean teacher.
4. Iterate: student becomes teacher.

Medical-imaging precedent: DenseUNet teacher-student with linearly-decayed MixUp for ultrasound landmark detection (IUGC 2025); nnU-Net + Noisy Student (OpenReview). Both report gains in scarce-annotation regimes.

**Competition mapping:** the "unlabeled" pool here is the 4,349 report-labeled studies (weak labels ≈ noisy pseudo-labels) — Noisy Student formalizes what teams are already doing ad hoc, with the critical additions of (a) soft targets, (b) student noise, (c) iteration.

### 2.2 Multi-label FixMatch (`LITERATURE`, Ihler et al., CVPR 2024 workshops)

**Distribution-Aware Multi-Label FixMatch (ML-FixMatch+DA)** adapts FixMatch to multi-label CheXpert:
- Generates pseudo-labels from weakly-augmented views, enforces consistency on strongly-augmented views, confidence-masked.
- **Key finding: distribution alignment is ESSENTIAL for multi-label** (optional in multiclass) — it handles class imbalance and generates reliable positive AND negative pseudo-labels with a single threshold and no prior on label distributions.
- Result: **+2.6% AUC** on CheXpert SSL tasks, +1.9% in missing-label scenarios.

**Competition mapping:** our label prevalence is imbalanced (Effusion 60% → MCL 16% on gold-58). A naive per-class threshold will starve rare classes of pseudo-labels; distribution alignment fixes this without hand-tuned per-class thresholds.

### 2.3 Teacher-student with confidence filtering (community baseline)

The simple version teams actually run: teacher → pseudo-labels → keep top-K by confidence → retrain. Guardrails from the kagglecoach playbook: **cap pseudo-labeled data at 2–3× the labeled set, keep threshold ≥0.9, always evaluate on held-out labeled data, never on pseudo-labels.** Failure mode: "pseudo-labeling improves CV but hurts leaderboard — the base model's confident predictions are biased in a way that CV doesn't detect."

### 2.4 Test-set pseudo-labeling (`INFERRED` — private-team territory)

Standard Kaggle endgame move: predict on the hidden test set (inside the submission kernel, or via a prior submission's OOF), keep high-confidence predictions, retrain, resubmit. It adapts the model to the test distribution. **Risk:** it amplifies the model's own biases and is pure public-LB fuel if the confidence selection is tuned against the public split. No public log in this competition admits to it; the top-0.957 teams' methods are unpublished. Treat as a plausible component of the 0.943→0.957 gap, not a verified one.

---

## 3. Avoiding overfitting the 58 gold studies

### 3.1 Scanner-grouped CV is mandatory (`VERIFIED`)

- **Random K-fold inflates macro AUC by 0.087** (measured, FINDINGS.md §9). The DICOM headers are de-identified (no site/institution/dates), but a strong **scanner fingerprint** survives and must be the fold-grouping key.
- Reference implementation: fold 0 splits 3,525 train / 882 val across **35 validation scanner groups** (existentialistlogarithmic's cache build).

### 3.2 The gold-58 as regression guard, not tuner (`VERIFIED` practice)

- tranbadat's ledger: **"Gold-58 cannot resolve ±0.01 fusion effects (SE ~0.02): use it as a regression guard, never to tune weights."**
- msmile-shiny's note (translated): gold-58 has been used repeatedly in development and **cannot be treated as an independent test set**.
- Practical rule: if a change moves gold-58 OOF by <0.02, it is noise; if it *drops* gold-58 by >0.02 while LB rises, you are LB-probing — revert.

### 3.3 Full-fit vs folds (`VERIFIED` measurement)

- Five-fold ensemble gold OOF: 0.8980. Full-fit (all 4,407 incl. 58 gold) priced at **+0.003 over folds** (E083) — real but small, and a full-fit model **cannot be scored offline by construction**.
- E064/E087: mixing folds + full-fit members gave exactly linear-interpolation results — **nothing super-additive**. Ship one or the other, not a blend of both, unless you have evidence.

### 3.4 The reseed floor (`VERIFIED`)

- Pure reseed of the same ensemble: **±0.003 spread** (E092). **Read no board difference under 0.003 as real.** Four of existentialistlogarithmic's last five submissions landed inside this floor — "we are no longer short of board attempts, we are short of things worth putting on the board."

---

## 4. Loss functions for multi-label AUC

### 4.1 Asymmetric Loss (ASL) — the community default (`LITERATURE`, Ridnik et al. 2021, arXiv:2009.14119)

Decouples focusing for positives and negatives:
- `L+ = (1−p)^γ+ · log(p)`, `L− = p^γ− · log(1−p)`, typically **γ− > γ+** (standard: γ+=0 or 1, γ−=3 or 4).
- Plus an asymmetric **probability margin** that discards very-hard negatives (likely mislabeled): best combined COCO result γ+=0, γ−=4, m=0.05.
- Chest-X-ray precedent (NIH, 0.769 mean AUC): ASL γ−=4.0/γ+=1.0 beat focal loss for extreme multi-label imbalance; stacking focal + pos_weight creates conflicting corrections — use ASL alone.
- **Why it fits here:** 12 imbalanced heads, noisy weak labels (the probability margin rejects likely-mislabeled hard negatives — exactly the report-label noise), no static pos_weight needed.

### 4.2 Direct AUC optimization — LibAUC (`LITERATURE`, Yuan et al.)

- `AUCMLoss(imratio=...)` with the **PESG** optimizer directly optimizes a squared-hinge AUC surrogate; `APLoss_SH` + SOAP_ADAM for AUPRC.
- Checklist: binary 0/1 labels, compute imratio from train, lr ∈ [0.05, 0.1] (much higher than Adam norms), `optimizer.update_regularizer()`, sigmoid before the loss, reshape to (N,1).
- **Caveat for this competition:** AUCMLoss is pairwise — O(batch²) memory per head × 12 heads × 2.5D slices. Feasible as a **fine-tuning-stage loss** on the 58 gold (or on frozen-backbone heads), not as the from-scratch training loss on 2×T4. No public log in this competition reports using it — it is an under-explored lever, which is exactly why it's interesting.

### 4.3 Supporting pieces

- **Label smoothing (0.05–0.1)** on weak-label heads — standard with noisy labels (ZooCAM 1st-place recipe).
- **BCEWithLogitsLoss** remains the base for most public notebooks; ASL is the upgrade.
- **Ranking losses** (e.g., pairwise hinge on pos/neg pairs) are the manual alternative to LibAUC when the library won't fit in the kernel.

---

## 5. Ensembling — what the top ensembles look like

### 5.1 Measured anatomy of strong ensembles (`VERIFIED` from logs)

| Ensemble | Composition | Score |
|---|---|---|
| tranbadat `rsna-knee-abnormality` (0.910 LB) | 24-member DINOv2 TTA ensemble + 5-fold DINOv3-small + RadImageNet ResNet-50; weights 0.65/0.35 then 0.75/0.25; "V11 challenger" recipe submitted | 0.910 |
| existentialistlogarithmic `knee-infer-raptorv1` (0.940) | 4 public CC0 CoAtNet arms + own resnet34 lineage, rank-mean | 0.940 |
| msmile-shiny 4-member builder (0.943) | community Speedy/D4 recipe reproduction | 0.943 |
| Speedy/D4 recipe | public notebook lineage | 0.942 |

### 5.2 What actually adds diversity (`VERIFIED` / `INFERRED`)

- **Pipeline/label diversity adds; more checkpoints of the same pipeline do not.** tranbadat: residual-gated CoAtNet +0.007 on gold-58; extra checkpoints of the same pipeline and backbone swaps on identical labels/views: ~zero.
- **Member count moved the board +0.002 twice; blend weight moved it 0.000 twice** (2×2 submission experiment). Scale of *diverse* members > cleverness of weights.
- Architectural diversity (CoAtNet vs ViT vs ResNet) beats depth uniformity — errors must be uncorrelated for variance reduction to work (general Kaggle finding, confirmed here by the arm ablations).
- **Rank-mean / rank averaging** is the dominant fusion (used by both 0.940 and 0.943 lines) — robust to miscalibrated members, which matters because members trained on different label tables are differently calibrated.

### 5.3 TTA (test-time augmentation)

- tranbadat's DINOv2 arm: 24 members from **two physical-scale crop configs × jittered multi-window TTA** — TTA is baked into member generation, not just final averaging.
- **Caution (`LITERATURE`):** TTA can hurt. A controlled study on chest X-rays: rotation/scaling TTA *dropped* accuracy 76.6%→65.2% (hurt 11.7%, helped 0.3%). For knee MRI, flips are anatomically safe; rotations/scales are not obviously safe. Validate TTA per-transform on scanner-grouped OOF, don't assume.
- **Budget:** hidden test ≈ 215k slices / 9 h ≈ **24 s/study end-to-end incl. DICOM reads**. Every TTA view and every ensemble member spends this budget. This is the hard constraint that kills naive "100-member" plans — design the inference kernel around it.

### 5.4 The tie-penalty detail (`VERIFIED` reasoning, berattcelikk's log)

AUC counts tied pairs as 0.5. Rounding probabilities (e.g., `np.round(p, 4)`) creates artificial ties and leaks score. **Retain full float64 precision** from logits through submission CSV. Free, no downside.

---

## 6. What separates private-team solutions from public forks

(`INFERRED` — assembled from the 0.943→0.957 gap, fork fingerprints, and experiment logs. No top-team writeup is public.)

1. **Custom label tables, not the five public ones.** The public tables embed gold-58 verbatim and cap at ~0.899 teacher agreement. A better multilingual report labeler (or a vision-language cross-check) is the most plausible single source of the gap — labels are the only training signal 4,349/4,407 studies have.
2. **Ensemble scale with real diversity.** The 0.943 line is a 4-member public-recipe blend. The 0.957 teams are widely believed (forum consensus, `INFERRED`) to run larger ensembles of self-supervised ViTs (DINOv2/DINOv3 family) — but note the measured counter-evidence: DINOv2 scored 0.840 on gold-58 vs CoAtNet's 0.912–0.920. If the top is ViT-heavy, their *labels or training* differ from the public DINOv2 arms, not just the backbone.
3. **No LB-probed blend weights.** The public 0.939→0.941 gains are documented as per-target weights fit to the public split. Private teams that tuned this way will shake; teams that used rank-mean with fixed weights are robust.
4. **Test-set adaptation.** Pseudo-labeling the hidden test inside the kernel (or via OOF from earlier submissions) is the standard unspoken endgame. It cannot be verified from public logs.
5. **Training longer / more seeds.** GPU quota (~30 h/week) binds everyone equally; private teams with 5 members × quota have 5× the training budget of a solo team — team size is a legitimate, rules-sanctioned compute multiplier (max team size 5, merger deadline 2026-10-15).
6. **What they do NOT have:** more data (everyone has the same 4,407), the reports at test time (nobody does), or a different metric.

**Bottom line:** the reproducible public ceiling is 0.943. The next 0.014 is some combination of better weak labels, bigger diverse ensembles, and disciplined (non-LB-probed) fusion. None of it is magic; all of it is expensive.

---

## 7. Public-LB overfitting: the shakeup risk and how robust teams protect

### 7.1 Why analysts expect a shakeup (`INFERRED` from board structure)

- **Fork fingerprints:** 492 teams at exactly 0.936 + 163 at exactly 0.937 = forks of two public notebooks, not independent solutions. The public LB is measuring *one* solution family hundreds of times.
- **LB-probed weights:** documented 0.939→0.941 gains from per-target blend weights fit against the public split — textbook public overfitting.
- **The Kaggle Book's meta-analysis** (Roelofs et al., 120 competitions): most shakeups come from overcrowded rankings where competitors sit within noise of each other — exactly this board's 0.006-wide top-10 band. Shakeups concentrate where the training set is tiny or non-i.i.d. — here the *gold* set is 58 studies.
- Counterpoint: public = ~33%? (unverified split for this competition — do not assume; the split fraction was not confirmed in any source I reached).

### 7.2 The robust-team playbook (synthesized)

1. **Decide by scanner-grouped OOF, not by LB.** Random K-fold inflates 0.087; gold-58 alone has SE ~0.02. The only trustworthy offline number is scanner-grouped CV on weak labels + gold-58 as a guard.
2. **Fixed fusion weights** (rank-mean, equal weights) over LB-tuned per-target weights. If per-target weights can't beat equal weights on grouped OOF, they're probing.
3. **Diversity over tuning:** new members from different pipelines/labels/views; never "one more seed of the same config."
4. **Reseed discipline:** nothing under ±0.003 off the board is a signal. Budget submissions accordingly (5/day is not the constraint — GPU hours are).
5. **Full precision** through the submission write (tie penalty).
6. **Bank a robust 0.94x early**, then take exactly one or two well-reasoned shots at the wall — the board keeps your best, so the downside of a challenger is only the GPU time.
7. **Assume the private set is drawn from the same study distribution** (train/test studies are disjoint, `VERIFIED`; sites span 5 continents) — scanner-grouped CV is the closest available proxy for "unseen scanners," which is what the private set tests.

---

## 8. Concrete technique checklist for the build

**Labels**
- [ ] Custom multilingual report labeler (LLM, abstain-aware: silence → mask, not negative)
- [ ] Soft targets (probabilities), not hard 0/1
- [ ] Gold-58 override + upweight (8× measured as working)
- [ ] Never train on the five public tables that embed gold-58 verbatim

**Training**
- [ ] Scanner-fingerprint-grouped folds (random K-fold inflates 0.087)
- [ ] Stage 1: weak labels, heavy aug, ASL loss (γ−=4, γ+=0–1, probability margin)
- [ ] Stage 2: gold fine-tune, low LR; optional LibAUC/PESG head-only pass
- [ ] Label smoothing 0.05–0.1 on weak heads
- [ ] Noisy-Student iteration: soft pseudo-labels, larger noised student, ≥0.9 confidence, ≤2–3× data cap
- [ ] (Optional) ML-FixMatch+DA with distribution alignment for rare classes

**Models**
- [ ] CoAtNet-2.5D + per-finding attention-MIL @336–384px (strongest measured family)
- [ ] DINOv2/DINOv3 ViT arm (diversity, not strength)
- [ ] RadImageNet ResNet-50 arm (diversity)
- [ ] Attention pooling over slices (not max — max blinds diffuse findings like effusion/synovitis/OA; not mean — mean drowns focal tears)

**Ensemble & inference**
- [ ] Rank-mean fusion, fixed weights
- [ ] TTA limited to safe transforms (flips), validated per-transform
- [ ] float64 precision end-to-end
- [ ] Inference kernel budgeted at ≤24 s/study (215k slices / 9 h)
- [ ] Submission kernel: image-only, internet off, mounts weights as datasets

**Validation discipline**
- [ ] All decisions on scanner-grouped OOF; gold-58 = regression guard (SE ~0.02)
- [ ] Nothing under ±0.003 off the public board counts as signal
- [ ] Bank a robust score early; challengers must beat it on OOF first

---

## Sources

**Competition-verified logs & repos**
- https://github.com/existentialistlogarithmic/knee-abnormality/blob/HEAD/docs/FINDINGS.md (FINDINGS.md — Phase 0 verification ledger, tags every claim)
- https://github.com/existentialistlogarithmic/knee-abnormality/blob/HEAD/docs/PATH.md ("what 0.95 actually requires", 0.940 standing, reseed floor ±0.003)
- https://github.com/existentialistlogarithmic/knee-abnormality/blob/HEAD/docs/HANDOFF.md (public-notebook route, fork fingerprints: 492 @ 0.936, 163 @ 0.937)
- https://github.com/existentialistlogarithmic/knee-abnormality/blob/HEAD/docs/COMMENTS_FOR_MARTIJN.md (member-count 2×2 experiment, wall/cliff analysis)
- https://github.com/existentialistlogarithmic/knee-abnormality/blob/HEAD/kaggle/04_train/README.md (8× gold weight, abstain masking, scanner-grouped folds)
- https://github.com/msmile-shiny/cvproject_rsna-knee-detection (0.943 reproduction, Speedy/D4 recipe 0.942, gold-58 reuse warning)
- https://github.com/tranbadat2607/rsna-knee-abnormality-detection/blob/HEAD/CLAUDE.md (0.910 blend anatomy, CoAtNet vs DINOv2 vs RadImageNet gold-58 numbers, LB-probed weights warning)
- https://github.com/jlawnat/knee-mri-abnormality-detection (LLM soft-label pipeline, 0.789)
- https://github.com/Arjun-Bhattarai/rsna-knee-abnormality-detection (weak-supervision staging 0.644→0.763)
- https://github.com/berattcelikk/daily-engineering-log/blob/HEAD/docs/2026/day_121_multimodal_ml_milestones.md (AUC tie-penalty, float64)

**Papers**
- FixMatch, Sohn et al. 2020 — https://arxiv.org/abs/2001.07685
- Distribution-Aware Multi-Label FixMatch (CheXpert), Ihler et al., CVPRW 2024 — https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/html/Ihler_Distribution-Aware_Multi-Label_FixMatch_for_Semi-Supervised_Learning_on_CheXpert._CVPRW_2024_paper.html
- Noisy Student, Xie et al. 2019 — http://arxiv.org/pdf/1911.04252
- Asymmetric Loss for Multi-Label Classification, Ridnik et al. 2021 — http://arxiv.org/pdf/2009.14119
- Knowledge Distillation: A Good Teacher Is Patient and Consistent, Beyer et al., CVPR 2022 — https://openaccess.thecvf.com/content/CVPR2022/papers/Beyer_Knowledge_Distillation_A_Good_Teacher_Is_Patient_and_Consistent_CVPR_2022_paper.pdf
- LibAUC (direct AUC optimization) — https://github.com/sparsel/libauc
