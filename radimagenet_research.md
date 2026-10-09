# RadImageNet Deep-Dive: Implementation Plan for RSNA Knee

**Date:** 2026-10-08
**Status:** Research complete. Weights confirmed downloadable. Ready to implement.

---

## 1. Weight Availability — CONFIRMED

### PyTorch weights (primary)
- **URL:** https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing
- **Size:** 186 MB zip, contains `ResNet50.pt`
- **Gating:** NONE for weights. The Google Form only gates the training *dataset*. Weights are open download.
- **Source repo:** https://github.com/teslib/radimagenet (official, MIT license)

### Kaggle dataset (easiest for our pipeline)
- **URL:** https://www.kaggle.com/datasets/ipythonx/notop-wg-radimagenet
- Mount as a Kaggle dataset — no manual download needed.

### TensorFlow weights (alternative)
- **URL:** https://drive.google.com/drive/folders/1Es7cK1hv7zNHJoUW0tI0e6nLFVYTqPqK?usp=sharing

### Available architectures
| Architecture | PyTorch | TensorFlow | Top-1 on RadImageNet |
|---|---|---|---|
| ResNet-50 | ✅ | ✅ | 72.3% |
| DenseNet-121 | ✅ | ✅ | 73.1% |
| InceptionV3 | ✅ | ✅ | 73.2% |
| InceptionResNetV2 | ✅ | ✅ | 74.0% |
| EfficientNet (any) | ❌ | ❌ | — |
| Swin Transformer | 🔜 "later" | 🔜 | — |

**No EfficientNet RadImageNet weights exist.** Cannot convert — different architecture entirely.

### Critical loader quirk
The PyTorch weights are stored in `nn.Sequential` indexing, NOT torchvision naming:
- File uses: `backbone.0.*` through `backbone.8.*`
- torchvision expects: `conv1.*`, `bn1.*`, `layer1.*` ... `layer4.*`

**Rename map required:**
```
backbone.0 → conv1
backbone.1 → bn1
backbone.2 → relu
backbone.3 → maxpool
backbone.4 → layer1
backbone.5 → layer2
backbone.6 → layer3
backbone.7 → layer4
backbone.8 → avgpool
```

This matches how the official `pytorch_example.ipynb` constructs the model:
```python
base_model = resnet50(pretrained=False)
encoder_layers = list(base_model.children())
self.backbone = nn.Sequential(*encoder_layers[:9])  # all except final fc
backbone.load_state_dict(torch.load("resnet50_torch.pt"))  # loads directly
```

**Recommendation:** Follow the notebook's approach — wrap torchvision's ResNet-50 layers in `nn.Sequential` and load directly. No rename map needed.

---

## 2. Architecture Compatibility

### Our current setup
- EfficientNet-B3 @ 288px, feature dim 1536, attention MIL head

### RadImageNet option
- ResNet-50, feature dim **2048** (not 1536)
- Must update MIL head input dim: 1536 → 2048

### Can we use ResNet-50 + our MIL head directly?
**Yes.** The MIL head is architecture-agnostic — it takes a feature vector per slice. Swap:
```python
# Before
backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
feature_dim = 1536

# After
backbone = load_radimagenet_resnet50()  # custom loader, see Section 1
feature_dim = 2048
```

### ResNet-50 vs EfficientNet-B3 on medical MRI
| Study | ResNet-50 | EfficientNet-B3 | Winner |
|---|---|---|---|
| Brain tumor MRI (2,114 scans) | 99.34% | **99.48%** | B3 (small gap) |
| Alzheimer's MRI (33,984 scans) | 0.87 precision | **0.95 precision** | B3 (clear) |
| General transfer learning | baseline | +0.5-1% | B3 |

**B3 is the better architecture.** But the domain pretraining gain (+4.5-4.8% relative) is **5-10× larger** than the architecture gap. RadImageNet-ResNet50 should beat ImageNet-EfficientNetB3.

### Best strategy: ENSEMBLE both
Different pretraining regime = maximum diversity. The kapasique playbook measured **+0.0167 macro AUC on 58 gold** from adding a different-regime member (vs same-family reseeds which *hurt*).
- Member 1: ImageNet EfficientNet-B3 (current v2.3, in training)
- Member 2: RadImageNet ResNet-50 (new)
- Blend by rank averaging

---

## 3. Input Preprocessing

### What RadImageNet was trained with

**PyTorch example (`pytorch_example.ipynb`):**
```python
image = cv2.imread(image)
image = (image - 127.5) * 2 / 255   # → [-1, 1] range
image = cv2.resize(image, (224, 224))
```

**TensorFlow script (`acl_train.py`):**
- `image_size = 256` (default, configurable)
- Keras `preprocess_input` (ImageNet-style for the TF path)

### Our pipeline
- 288×288, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Recommendation
**Keep 288×288.** ResNet-50 is fully convolutional — it handles any input size. The features were learned at 224, but fine-tuning at 288 is standard practice and works well.

**Normalization:** Two options:
1. **Match RadImageNet:** Use `(x - 127.5) * 2 / 255` → [-1, 1]. Most faithful to pretraining.
2. **Keep ImageNet norm:** The network will adapt during fine-tuning since we're unfreezing all layers.

**Recommendation:** Start with option 1 (RadImageNet's own normalization) for the RadImageNet branch. It's a one-line change in the data loader. If results are poor, fall back to ImageNet norm.

**No Phase 0 changes needed.** Our cached 288×288 images work as-is. Only the normalization in the training data loader changes.

---

## 4. Fine-Tuning Strategy

### What the paper tested (24 scenarios)
For each of 4 architectures, they tested:
| Strategy | Learning rates tested |
|---|---|
| Unfreeze all layers | 0.001, 0.0001 |
| Freeze all (train head only) | 0.01, 0.001 |
| Unfreeze top 10 layers | 0.01, 0.001 |

Results were **averaged across all 24 settings** — the paper reports that RadImageNet beat ImageNet on average, not just in the best setting. This means the gain is robust, not cherry-picked.

### Recommended for OUR setup

We have 4,407 studies (large for medical imaging). We don't need to freeze.

```
Strategy: Unfreeze ALL layers from epoch 0
Optimizer: AdamW (match our v2.3 setup)
Backbone LR: 1e-4 (lower than our current 3e-4, because features are already domain-adapted)
Head LR: 3e-4 (keep current — the MIL head is randomly initialized)
Epochs: 20-25 (expect faster convergence than 30; domain pretraining = head start)
Scheduler: Same cosine schedule as v2.3
Loss: Same ASL (γ_neg=4.0) + teacher confidence weighting + gold override
EMA: Same 0.998
```

### Why lower LR for the backbone?
ImageNet features need large updates to adapt to MRI. RadImageNet features are already MRI-adapted — large updates would destroy useful features. 1e-4 is the paper's tested value for unfreeze-all.

### Alternative: gradual unfreezing (if 1e-4 underperforms)
```
Epochs 1-5:  Freeze backbone, train MIL head only @ 3e-4
Epochs 6-25: Unfreeze all @ 1e-4 (backbone) / 3e-4 (head)
```

---

## 5. Expected Gain for OUR Setup

### Published numbers (exact)
From the paper's Figure 3 caption (small datasets, 5-fold CV):
- **ACL MRI: +4.8% relative gain** vs ImageNet (same architecture)
- **Meniscus MRI: +4.5% relative gain** vs ImageNet (same architecture)

These are **relative** gains: `(RadImageNet_AUC - ImageNet_AUC) / ImageNet_AUC`.

### Translating to our setup
Our baseline (ImageNet B3): ~0.84 val macro AUC.

Naive translation: 0.84 × 1.045 = 0.878 (+0.038 absolute).

**But** — three reasons the real gain will be smaller:
1. **We have more data** (4,407 vs their small datasets). Transfer learning helps most when data is scarce.
2. **Multi-label dilutes** — the +4.5% was for binary meniscus. We have 12 findings; not all benefit equally.
3. **Our baseline is already high** — going from 0.70→0.73 is easier than 0.84→0.88.

### Realistic estimate for our setup
| Scenario | Expected val macro gain |
|---|---|
| Pessimistic | +0.010 |
| **Realistic** | **+0.015 to +0.025** |
| Optimistic | +0.035 |

### As an ensemble member (most likely first use)
The kapasique playbook measured **+0.0167 on 58 gold** from adding a different-pretraining-regime model to an ensemble. This is the most directly applicable number — it's on RSNA Knee, with 58 gold, measuring exactly what we'd do.

### Bottom line
- Solo RadImageNet-ResNet50: likely **+0.015 to +0.025** over our ImageNet-B3 solo
- As ensemble member with B3: **+0.010 to +0.017** on top of B3 solo (diversity gain)
- Combined realistic: B3 solo (~0.85?) + RadImageNet member → **~0.865-0.870 val macro**

---

## 6. DINOv2 Alternative

### timm model names (verified)
```
vit_small_patch14_dinov2.lvd142m      # 22M params, smallest
vit_base_patch14_dinov2.lvd142m       # 86.6M params, standard
vit_small_patch14_reg4_dinov2.lvd142m # with registers
vit_small_patch16_dinov3.lvd1689m     # DINOv3, newest
```
All available via `timm.create_model(name, pretrained=True)` — downloads automatically, no manual steps.

### DINOv2 vs RadImageNet for knee MRI

| Factor | RadImageNet | DINOv2 |
|---|---|---|
| **Direct knee MRI evidence** | ✅ +4.8% ACL, +4.5% meniscus | ❌ None found |
| **Domain** | Radiology-specific (CT/MRI/US) | General (LVD-142M natural images) |
| **Pretraining method** | Supervised (165 pathology labels) | Self-supervised |
| **Medical evidence** | Strong, peer-reviewed | Mixed |
| **On clinical MRI** | Designed for this | One study: weaker than ImageNet on brain MRI |
| **RAD-DINO** (medical DINOv2) | N/A | Beats plain DINOv2, but chest X-ray only |
| **Integration cost** | Medium (manual download + loader) | Trivial (timm one-liner) |
| **Params (smallest)** | 25.6M (ResNet-50) | 22M (ViT-S) |
| **CNN vs ViT** | CNN (fits our pipeline) | ViT (needs pipeline changes) |

### Key evidence against DINOv2 for MRI
A 2024 comparative study on **brain MRI glioma grading** (clinical data):
> "DINOv2's performance was not as strong as ImageNet-based pre-trained models... its effectiveness may vary with data that significantly differs from natural images such as MRI."

DINOv2 was trained on natural images (LVD-142M). MRI looks nothing like natural images. Self-supervised features don't automatically transfer to MRI.

### Recommendation
**RadImageNet first.** It has:
1. Direct knee MRI evidence (+4.8% ACL)
2. Radiology-specific pretraining (not natural images)
3. CNN architecture (drop-in for our pipeline)
4. Peer-reviewed publication

**DINOv2 later** as a third ensemble member for diversity, IF budget allows. The 0.910 team uses it as primary, but they also use massive TTA (24 views) which we don't. Their DINOv2 success may depend on that TTA.

**Do not replace RadImageNet with DINOv2.** Use both if possible: RadImageNet (domain) + DINOv2 (self-supervised) + ImageNet-B3 (current) = three pretraining paradigms = maximum ensemble diversity.

---

## 7. Why We Missed RadImageNet

Honest answer:

1. **We followed the public trajectory.** TianK003's EfficientNet-B3 @ 288px (ImageNet) is the best-documented public approach. We optimized within that paradigm instead of questioning the paradigm.

2. **Friction.** ImageNet weights are one line in timm: `timm.create_model('efficientnet_b3', pretrained=True)`. RadImageNet requires: manual Google Drive download → unzip → custom loader with Sequential-indexing quirk → no timm integration. Nobody built the easy path.

3. **The paper is from 2022.** It's well-cited in academia but didn't penetrate the Kaggle community. The Kaggle winners who use it (0.910 team) don't publish detailed writeups.

4. **"Good enough" trap.** ImageNet pretraining got us to 0.89 LB. When something works, you optimize it instead of replacing it. The +0.086 from teacher labels masked the pretraining inefficiency.

5. **No head-to-head on Kaggle.** Nobody published "I swapped ImageNet for RadImageNet on RSNA Knee and gained +0.02." Without that social proof, the community stuck with the default.

**The 0.910 team figured it out.** Their stack is DINOv2 (0.65) + DINOv3 (0.35) + RadImageNet-ResNet50 blended in. Zero ImageNet. Zero EfficientNet.

---

## 8. Concrete Implementation Plan

### Phase A: Download and loader (0 GPU hours, ~1 hour wall clock)
1. Add Kaggle dataset `ipythonx/notop-wg-radimagenet` to the training notebook
2. Write `load_radimagenet_resnet50()`:
   ```python
   import torch
   from torch import nn
   from torchvision.models import resnet50

   def load_radimagenet_resnet50(weights_path, feature_dim_only=False):
       # Build backbone as Sequential (matches weight file format)
       base = resnet50(pretrained=False)
       layers = list(base.children())
       backbone = nn.Sequential(*layers[:9])  # exclude final fc
       state = torch.load(weights_path, map_location='cpu')
       backbone.load_state_dict(state, strict=True)
       return backbone  # output: [B, 2048, 7, 7] at 224px
   ```
3. Test loader locally: verify `strict=True` passes, output shape correct

### Phase B: Notebook surgery (0 GPU hours)
1. Copy `phase2-effnet-b3-v2.3.ipynb` → `phase2-rad50-v1.ipynb`
2. Replace backbone creation with RadImageNet loader
3. Change `feature_dim` 1536 → 2048
4. Change normalization to `[-1, 1]` scaling: `(x - 127.5) * 2 / 255`
5. Set backbone LR to 1e-4 (keep head at 3e-4 — may need per-param-group LRs)
6. Reduce epochs 30 → 25
7. Update VERSION line: `rad50-v1: RadImageNet ResNet-50 backbone, 2048-dim, [-1,1] norm`

### Phase C: Probe (0.1 GPU hours)
Run with `PROBE=True`. Verify:
- Weights load with `strict=True` (no shape mismatches)
- Forward pass works at 288×288
- Feature dim is 2048
- Loss decreases on probe batches

### Phase D: Full run (~7 GPU hours)
Launch 25-epoch run. Expected:
- Faster early convergence (domain features = head start)
- Best val macro: **0.855-0.870** (vs B3's projected ~0.845-0.855)
- Best gold macro: **0.870-0.890**

### Phase E: Ensemble (~0 GPU hours)
Rank-average B3 (v2.3) + Rad50 (v1) predictions per finding.
Expected ensemble gain: **+0.010 to +0.017** over best solo.

### Total cost
| Phase | GPU hours |
|---|---|
| A-C (prep + probe) | 0.1 |
| D (full run) | 7 |
| E (ensemble) | 0 |
| **Total** | **~7.1** |

Well within the 45/week budget. Can run in parallel with B3 seed-2.

---

## 9. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Weights don't load (format issue) | Low | `strict=True` fails loudly; official notebook shows exact method |
| 288px hurts (trained at 224) | Low | CNNs are resolution-robust; paper used 256px variant successfully |
| [-1,1] norm underperforms ImageNet norm | Medium | Easy ablation: try both, keep winner |
| Gain smaller than +0.015 | Medium | Even +0.010 justifies 7 GPU-hours; ensemble diversity alone is valuable |
| ResNet-50 slower than B3 | Low | R50 is actually faster than B3 (fewer FLOPs at 288px) |
| Kaggle dataset `ipythonx/notop-wg-radimagenet` is stale/removed | Low | Fallback: Google Drive direct download in notebook |

---

## 10. Sources

- RadImageNet paper: Mei et al., Radiology: Artificial Intelligence, DOI: 10.1148/ryai.210315
- Official repo: https://github.com/teslib/radimagenet
- PyTorch weights: https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing
- Kaggle weights mirror: https://www.kaggle.com/datasets/ipythonx/notop-wg-radimagenet
- PyTorch example: https://github.com/teslib/radimagenet/blob/main/pytorch_example.ipynb
- ACL training script: https://github.com/teslib/radimagenet/blob/main/acl/acl_train.py
- Figure with +4.8%/+4.5%: https://www.researchgate.net/figure/Performance-of-the-RadImageNet-pretrained-models-and-ImageNet-pretrained-models-on-small_fig3_362308151
- DINOv2 medical comparison: Huang et al., arXiv:2402.07595
- RAD-DINO: arXiv:2401.10815
- Ensemble diversity evidence: https://github.com/kapasique/kaggle-dominator/blob/HEAD/references/learned-playbook.md
- 0.910 team CLAUDE.md: https://github.com/tranbadat2607/rsna-knee-abnormality-detection
