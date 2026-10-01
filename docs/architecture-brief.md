# Architecture Brief: RSNA Knee Abnormality Detection (2026)

Researched 2026-09-30. Sources: public Kaggle notebooks/writeups, competitor GitHub repos
(TianK003, tranbadat2607, existentialistlogarithmic, liamd-schneider, sh0ch), SOTA literature
(MRNet 2018, CoPAS/Nature Commun 2024, EClinicalMedicine 2025, ELNet/MIDL 2020), timm/HF model zoos.

**Convention:** ✅ = verified from a primary source (repo/notebook/paper). 🔮 = inference from
evidence. LB = public leaderboard. All LB numbers are snapshots, not live.

---

## 1. What dominates the public leaderboard

The consensus top-of-LB recipe is **2.5D self-supervised ViT + attention pooling**, not 3D CNNs:

| Component | Consensus choice | Notes |
|---|---|---|
| Per-slice backbone | **DINOv2 ViT-S/14** (workhorse), ViT-B/14 | ✅ TianK003: "DINOv2 ViT-S/14 as the workhorse"; existentialistlogarithmic: DINOv2 ViT-B/14 fully fine-tuned |
| Blend partners | **DINOv3 ViT-S/16** + **RadImageNet ResNet-50** | ✅ tranbadat2607 0.910 LB: 24-member DINOv2 TTA + 5-fold DINOv3-small + RadImageNet ResNet-50, fixed-weight blend |
| Hybrid contender | **CoAtNet** (conv-attention hybrid) | ✅ TianK003 best solo 0.932 = c03 CoAtNet pair member; best LB 0.942 |
| Slice→series | attention pooling over slices (gated or windowed) | ✅ per-label window attention measured +0.007 (TianK003) |
| Series→study | one-series-per-slot with presence mask; mean within plane → concat planes; or cross-attention over all series | ✅ Mattia Angeli "Bend the Knee to DinoV3": cross-attention over all series |
| Slice features | CLS token **and** patch-token mean | ✅ existentialistlogarithmic gap analysis |
| Resolution | 224 → 336 px (+0.017 measured); **384 px does NOT help** | ✅ TianK003: CoAtNet-1@224 = 0.8683 beats CoAtNet-2@384 = 0.8641 at ⅓ cost |
| Labels | LLM-read report labels as the de-facto standard target source | ✅ "LLM-read report labels as the de-facto standard target source" |

**Key negative results (measured, don't re-run):**
- ❌ Three backbones on the *same* input blended to +0.001 — "diversity has to come from the
  input representation and pretraining regime" (TianK003)
- ❌ A second backbone on the same teacher table adds nothing measurable (P-42: 0.927 = solo)
- ❌ 384px input: no gain, 3× cost
- ❌ EfficientNet era is over — "Not EfficientNet — that was the early-baseline era"

**Approximate public-LB ladder (snapshots, Sep 2026):**
~0.88 (Torres DINOv2 writeup) → 0.899 (AADIGUPTA "Let me Cook") → 0.903 (liamd-schneider ref,
rank 18/792) → 0.910 (tranbadat 3-arm blend) → 0.927 (TianK003 best solo) → 0.937 (tonylica
DINOv3 repro) → 0.942 (TianK003 best blend) → ~0.957 (top, heavily-forked community ensembles 🔮)

---

## 2. How top solutions fuse multiple series/planes

Five fusion patterns seen in the wild, ranked by evidence:

1. **Mean-within-plane → concat planes + embeddings** (existentialistlogarithmic/Torres):
   mean-pool slices within each plane, concatenate plane embeddings, add laterality + plane
   embeddings. Simple, strong.
2. **Cross-attention over all series** (Mattia Angeli "Bend the Knee to DinoV3"): a transformer
   layer attends over every series' pooled embedding jointly. More expressive; their ensembled
   notebook is a cited top solution.
3. **Per-series head → mean across series** (MRNet-style, liamd-schneider 0.903): shared 2D CNN
   per slice → pool to per-series embedding → linear head → 12 logits per series → mean across
   the study's series. Their hyperparameter search picked plain **max-pooling** over gated
   attention for their backbone/data scale — "the fancy option didn't actually win."
4. **2.5D one-series-per-slot + presence mask** (TianK003): fixed number of series slots, a
   presence mask for missing series, attention pooling over slices. Handles variable series
   counts cleanly.
5. **Per-finding pooling rules** (AADIGUPTA 0.899 "Let me Cook"): different abnormalities get
   different aggregation — focal findings (fracture, contusion) need max-like pooling over
   slices; diffuse findings (effusion, synovitis) need mean-like pooling. "Evidence-gated
   promotion." 🔮 This is under-exploited and a natural private-team edge.

**CoPAS (HKUST, Nature Communications 2024)** — the closest literature to this exact task
(12 knee abnormalities): "Co-Plane Attention across MRI Sequences" — learns attention *across*
planes/sequences, explicitly capturing intensity variations across sequences and decoupling
spatial features. Model beat junior radiologists, matched seniors. This is the architectural
prior the competition's top end is rediscovering.

---

## 3. Self-supervised pretraining: the real battleground

**Publicly available, competition-legal pretrained weights:**

| Weights | Source | Params | Access | Notes |
|---|---|---|---|---|
| DINOv2 ViT-S/14 | Meta, LVD-142M | 22M | timm `vit_small_patch14_dinov2.lvd142m` | ✅ the workhorse |
| DINOv2 ViT-B/14 | Meta, LVD-142M | 86M | timm `vit_base_patch14_dinov2.lvd142m` | fully fine-tuned by leaders |
| DINOv3 ViT-S/16 | Meta, LVD-1689M | 22M | timm `vit_small_patch16_dinov3.lvd1689m` | ✅ used by tranbadat + tonylica 0.937 |
| DINOv3 ViT-B/16 | Meta, LVD-1689M | 86M | timm `vit_base_patch16_dinov3.lvd1689m` / HF `facebook/dinov3-vitb16-pretrain-lvd1689m` | gated HF download |
| RadImageNet ResNet-50 | 1.35M radiology images | 25M | HF `Lab-Rasool/RadImageNet` (`ResNet50.pt`) or official GDrive | medical-domain pretraining; ✅ used as blend arm |
| CoAtNet | ImageNet | 25–74M | timm (`coatnet_*`) | TianK003's best solo family |

**What the literature says about SSL for medical imaging:**
- ✅ Frozen DINO-family features transfer to MRI classification without fine-tuning (brain tumor
  MRI study: DINOv1/v2/v3 all strong; partial fine-tuning of last block did NOT help there —
  though knee leaders DO fully fine-tune, so treat as task-dependent)
- ✅ DINOv3-style teacher-student pretraining works slice-wise on MRI (brain MRI foundation
  model paper, 2026: ViT-B/16, multi-crop 2 global + 8 local, EMA teacher)
- ✅ 2.5D CNN beat 3D CNN on stability for a 2026 lung-CT resource-performance study
  (3D showed threshold instability; transformers degenerated to all-positive)

**🔮 The private-team moat — custom SSL on the 570GB corpus:**
Nobody in public has published DINO-style pretraining *on the competition's own 819K DICOM
slices*. Natural-image DINOv2 is the public ceiling; domain-specific SSL on knee MRI is the
textbook private-team move (cf. every recent medical Kaggle). With ~22 days left this is
feasible but must be weighed against the label-quality lever (§6), which measured larger.

**License check needed:** DINOv3 ships under Meta's custom DINOv3 license (not plain Apache).
Verify competition legality before shipping DINOv3 weights. (One competitor repo did a full
license-tier analysis and excluded some assets on licensing grounds.)

---

## 4. SOTA literature for knee MRI specifically

| Work | Key idea | Result |
|---|---|---|
| **MRNet** (Bien et al., PLOS Med 2018) | AlexNet per-slice → max-pool → logistic regression over 3 series | AUC 0.937 / 0.965 / 0.847 (abnormal / ACL / meniscus) |
| **ELNet** (Tsai et al., MIDL 2020) | Lightweight multi-slice CNN; **coronal best for meniscus, axial best for ACL/abnormality** | Beat MRNet on meniscus (0.904) and abnormal (0.941) |
| **CoPAS** (Qiu/Chen et al., Nature Commun 2024) | Co-Plane Attention across MRI sequences, 12 abnormalities | ≥ junior radiologists, ≈ senior radiologists |
| **EClinicalMedicine 2025** (HKUST/SMU) | Attention-guided coarse-to-fine, 9 abnormalities, 13,419 patients, 5 centers | Robust multi-center; global→local zoom helps subtle pathology |
| 2026 review (Cureus) | Multichannel CNNs (axial+sagittal+coronal) | >92% accuracy, "significantly outperforming single-plane" |

**Takeaways:** (a) plane choice matters per finding (ELNet) — a per-label/plane routing is
principled, not a hack; (b) coarse-to-fine / zoom mechanisms help subtle pathology;
(c) multi-plane always beats single-plane; (d) 12-label multi-task is the published SOTA
shape (CoPAS), matching this competition exactly.

---

## 5. Handling variable series/slice counts and sequence types

Measured practices (✅ unless marked):

- **Never sort DICOM slices by filename** — measured ρ = −0.012 vs true spatial order, fails
  silently. Sort by `InstanceNumber`. (TianK003 traps)
- **Per-series 1st–99th percentile intensity windowing**, not a global window (liamd-schneider)
- Apply `RescaleSlope`/`RescaleIntercept`; invert `MONOCHROME1`
- Center-crop depth to ~16–64 slices; pad-to-square; resize 224–336; uint8 cache
- **Series selection:** prefer `Fluid_Sensitive=1` series; take one series per plane first,
  then fill remaining slots — same function at train and inference (no train/serve drift)
- **Presence masks** for missing planes/series (2.5D one-series-per-slot design)
- **Laterality normalisation** — 5 of 12 labels are laterality-specific (medial vs lateral
  meniscus/OA, MCL). **Never use horizontal flips** — a flip silently swaps medial/lateral
  without updating the label, and the loss still goes down (silent bug). Rotation/gamma/scale
  jitter shared across a whole series is safe.
- **Plane + laterality embeddings** added to pooled features (Torres design)

---

## 6. What private/top teams likely do that public forks don't

Ranked by expected value (🔮 = inference, but each is grounded in cited evidence):

1. **Teacher-student label refinement (the biggest lever).** ✅ TianK003 measured: training on
   0.5·LLM + 0.5·quantile-matched "Raptor" teacher table = +0.009 over LLM-only, "the first
   change of training targets that transfers to the LB." Their Raptor pass: full inference over
   4,349 studies at 5–7 s/study (~6 GPU-h). Self-distillation gave +0.0109 on fold-0 but did
   NOT transfer to production — target changes must be judged by gold-58 direction + solo LB
   only. "The label gap is the headline" — everything the image model learns is bounded by
   what its teacher knows.
2. **Custom SSL pretraining on the 570GB corpus** (§3) — the architectural moat nobody has
   published.
3. **Input-representation diversity, not backbone diversity.** ✅ Measured: 3 backbones same
   input = +0.001; but wide slice band + per-label window attention = +0.007, hybrid backbone
   with different meniscus errors = +0.004. Ensemble members must differ in *input* and
   *pretraining regime*.
4. **Per-finding pooling/fusion** (AADIGUPTA's evidence-gated promotion; ELNet's
   plane-per-finding) — focal vs diffuse findings need different aggregation.
5. **External GPU training** (RunPod A100/4090) to escape Kaggle's ~30h/week quota, then
   inference-only kernels on Kaggle. ✅ TianK003 trains CoAtNet arms on RunPod (~35–48 min
   per arm on 4090).
6. **Rank-based blending** over folds, not probability averaging (+0.006…+0.03 measured).
7. **Full-precision outputs** — don't round probabilities (ties cost AUC; one team measured
   the tie penalty explicitly).
8. **Validation discipline:** scanner-grouped folds (✅ random K-fold inflates macro AUC by
   0.087!), cross-fitted gold-58 scoring (each gold study scored only by folds that never saw
   it), LB noise floor 0.005 (gold-58 AUC noise floor ~0.05 — nearly useless for selection).
9. **Efficiency-track arbitrage:** a fast single model (0.8h) can beat a 0.883 public model
   on the efficiency board — separate config, not an afterthought.

---

## 7. Inference-time budget: what fits in 9h on 2×T4 for ~1,300 studies

Budget: 32,400 s / ~1,300 studies ≈ **25 s/study**.

Measured anchors (✅ TianK003 / existentialistlogarithmic):
- Single CoAtNet/DINOv2 member, full hidden test: **≤42 min** (~2 s/study on T4)
- Two-member solo: ~29 min
- Multi-member fork (their LB 0.942 blend): **≤8h 06min** — near the cap
- CPU-only inference: 19.7 s/study vs 2.2 s/study on T4 → **GPU is mandatory**
- Teacher pass (Raptor): 5.1–7.5 s/study

Design rules:
- **Decode each study once, share across members** — a fold adds one forward pass, no re-decode
- ~2 s/study/member → ~12–15 members fit in 9h with shared decode; TTA multiplies members
- Survival engineering (liamd-schneider): write `submission.csv` every 25 studies; project total
  runtime and degrade in stages (fewer slices → fewer members); failed studies → 0.5, never crash
- ⚠️ **NEVER select the P100 accelerator** — Kaggle's PyTorch ships no Pascal CUDA kernels;
  the session dies at the first convolution. T4 only. (✅ TianK003 hard constraint)
- Internet OFF at scoring → all weights must be mounted as Kaggle datasets, never downloaded
  at runtime. Training kernels may use internet ON to fetch weights, then save to outputs.

**Rough capacity plan:** 2×T4, shared decode, 224–336px, DINOv2-S/CoAtNet-1 class members at
~2 s/study → a 10-member diverse ensemble ≈ 1.5–3h. A 24-member TTA ensemble (tranbadat-style)
is the expensive end (~6–8h). Keep a 30–40% time buffer for queue variance and the
staged-degradation fallback.

---

## 8. Recommended architecture shortlist (for the project scope)

| Slot | Role | Candidate | Params | Why |
|---|---|---|---|---|
| A | Workhorse | DINOv2 ViT-S/14, fully fine-tuned, 2.5D + attention pool | 22M | consensus best single |
| B | Hybrid | CoAtNet-1/2, 224px | 42–74M | best measured solo (0.932); different error profile on menisci |
| C | Next-gen SSL | DINOv3 ViT-S/16 (timm) | 22M | 0.937 notebook gate; license check first |
| D | Medical prior | RadImageNet ResNet-50 | 25M | domain pretraining; cheap diversity |
| E | Moat | Custom DINO-pretrained ViT-S on competition slices | 22M | unpublished; biggest private-team lever |
| Fusion | — | Cross-attention over series (Angeli) + per-finding pooling (AADIGUPTA) + plane/laterality embeddings | — | combine, don't pick one blindly |
| Blend | — | Rank-mean over members; diversity from input representation + pretraining regime | — | +0.006…+0.03 measured |

## Open questions for the parent agent
1. DINOv3 license vs competition rules — verify before building on it.
2. Live LB top-10 check (my fetches of kaggle.com failed) — needed to confirm the 0.957 figure
   and whether the top is still fork-dominated.
3. Whether external GPU (RunPod etc.) is acceptable for training — top teams use it; without
   it we're capped at ~30h/week Kaggle GPU.
4. Efficiency track: pursue as a separate config (fast single model) or ignore.
