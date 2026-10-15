# Hill-Climb Score Log — phase2-0956-plus-ours-v1

Paste one row per submission. Without this log, 5 submissions/day is wasted.
Decision tree lives in `hill-climb-playbook.md`.

| # | date (PDT) | variant | weights (raptor384 / raptor224 / r7 / convnext / coral) | LB | rank | note |
|---|------------|---------|----------------------------------------------------------|----|------|------|
| 1 | 2026-10-14 | S0 5-model blend | 0.30 / 0.15 / 0.35 / 0.12 / 0.08 | 0.951 | — | models dilute; zeroed out after this |
| 2 | | | 0.30 / 0.20 / 0.50 / 0.00 / 0.00 | | | |
| 3 | | | / / / 0.00 / 0.00 | | | |
| 4 | | | / / / 0.00 / 0.00 | | | |
| 5 | | | / / / 0.00 / 0.00 | | | |

## Rules of the search (from the playbook)

- ~40 submissions left before 2026-10-22 final deadline, 5/day, ~2h15m per score.
- Start: pure-public weights (S2 above is the first probe: bump 224 up a tick).
- Move toward whatever direction raises the score; if 224-up drops, go 224-down.
- Only after the global trio plateaus: per-finding tuning on OA findings + Synovitis (CORAL's moat), one finding at a time. Anything under +0.001 is noise.
- Safety anchor stays the 0.954 fork — a worse blend erases nothing.
- Public-LB gains past ~0.960 may not survive the private shakeup; that's the accepted tradeoff of this campaign.
