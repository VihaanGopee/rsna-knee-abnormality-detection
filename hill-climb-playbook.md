# Hill-Climb Playbook — phase2-0956-plus-ours-v1

Target: 0.97 public LB (0.964 milestone on the way). Method: overfit the blend
weights to public LB feedback, as authorized. 5 submissions/day, ~2h15m per score.

## How to run a variant

Open the notebook, find `# === TUNE THESE WEIGHTS VIA LB FEEDBACK ===`,
set `USE_GLOBAL = True`, replace `W_GLOBAL` with one of the blocks below,
PROBE=True first (sanity), then PROBE=False, run, submit.

## Variants (all rows sum to 1.0 — paste as-is)

Baseline (already running, S0):
```python
W_GLOBAL = {'raptor384': 0.30, 'raptor224': 0.15, 'r7': 0.35, 'convnext': 0.12, 'coral': 0.08}
```

V1 — pure 0.956 recipe (exact public ratios: raptor side 0.65x384+0.35x224, then 0.5xraptor+0.5xr7):
```python
W_GLOBAL = {'raptor384': 0.325, 'raptor224': 0.175, 'r7': 0.50, 'convnext': 0.0, 'coral': 0.0}
```

V2 — ours out, current public ratios kept (ablation: do Justin's models dilute?):
```python
W_GLOBAL = {'raptor384': 0.375, 'raptor224': 0.1875, 'r7': 0.4375, 'convnext': 0.0, 'coral': 0.0}
```

V3 — ours up (diversity push):
```python
W_GLOBAL = {'raptor384': 0.25, 'raptor224': 0.12, 'r7': 0.28, 'convnext': 0.20, 'coral': 0.15}
```

V4 — convnext only:
```python
W_GLOBAL = {'raptor384': 0.30, 'raptor224': 0.15, 'r7': 0.35, 'convnext': 0.20, 'coral': 0.0}
```

V5 — coral only:
```python
W_GLOBAL = {'raptor384': 0.30, 'raptor224': 0.15, 'r7': 0.35, 'convnext': 0.0, 'coral': 0.20}
```

V6 — public ratio: heavier 384:
```python
W_GLOBAL = {'raptor384': 0.40, 'raptor224': 0.10, 'r7': 0.50, 'convnext': 0.0, 'coral': 0.0}
```

V7 — public ratio: heavier 224:
```python
W_GLOBAL = {'raptor384': 0.25, 'raptor224': 0.25, 'r7': 0.50, 'convnext': 0.0, 'coral': 0.0}
```

## Decision tree

1. **S0 (baseline).** If S0 >= 0.957: Justin's models add real diversity -> V3, then
   push the winning component further (V4 vs V5 to isolate which one helps).
2. If S0 < 0.956: the models dilute -> drop them (V1 or V2). Hill-climb only the
   public trio: V6 vs V7, then refine around the winner in 0.05 steps.
3. If S0 lands 0.956-0.957: run V2 (ablation). V2 > S0 means dilute; V2 <= S0
   means the models are neutral-to-helpful and V3 is worth a shot.
4. **Per-finding overfit (last resort, highest variance):** if the global best
   plateaus, go back to per-finding weights (USE_GLOBAL=False) and tune only the
   OA findings + Synovitis, where CORAL's ordinal heads are the moat. Change one
   finding at a time; anything under +0.001 is noise.

## Honest notes

- 0.956 -> 0.964 is +0.008. The unused-checkpoint trick bought +0.002. Weight
  tuning on rank-averaged blends typically moves ~0.001-0.003 per real signal.
  Per-finding overfitting is the only lever with +0.008 headroom, and it is
  noise-chasing: public-LB gains past ~0.960 may not survive the private shakeup.
  The target is the public score, so that tradeoff is yours to take.
- Log every score next to its variant: date, variant, weights, LB. Without the
  log, 5/day is wasted.
- Scores take ~2h15m. Submit, walk away, come back to a number, pick the next
  variant. The tree above is the whole strategy — no new training needed.
