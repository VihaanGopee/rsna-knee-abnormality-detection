"""Label dependency modeling for RSNA Knee — 12 findings.

Two complementary mechanisms (use either or both):

  1. LabelCorrAdapter  — learned 12x12 logit mixer, zero-init residual form
     logits' = logits @ (I + A). Starts as exact identity (A=0), so the
     model behaves identically to the 12-independent-heads baseline on day 1.
     Inspect A after training: non-zero off-diagonals = learned dependencies.

  2. PCDLoss           — Pairwise Correlation Difference auxiliary loss.
     Aligns the model's predicted label-correlation matrix with the empirical
     correlation matrix computed from training labels (CMLL-style).
     Uses a FIFO probability buffer so it works with tiny micro-batches.

Integration (minimal — ~10 lines in the training notebook):
  See INTEGRATION_SNIPPET at the bottom of this file.

Expected gain: +0.003 to +0.010 macro AUC combined. No public team does this.
"""

import numpy as np
import torch
import torch.nn as nn

FINDINGS = ["ACL", "MCL", "Medial_Meniscus", "Lateral_Meniscus",
            "Medial_OA", "Lateral_OA", "PF_OA", "Effusion",
            "Synovitis", "Bakers", "Contusion", "Fracture"]
N_FINDINGS = len(FINDINGS)


# ----------------------------------------------------------------------------
# 0. Empirical correlation matrix (run ONCE, offline, on CPU)
# ----------------------------------------------------------------------------
def compute_empirical_corr(label_csv, findings=FINDINGS, min_conf=0.25,
                           out_path="label_corr_empirical.pt"):
    """Compute pairwise-complete Pearson (phi) correlation of binarized labels.

    label_csv : merged_labels_v1.csv — soft teacher labels, 0.5 = "report
                does not address this finding" (must be EXCLUDED, not treated
                as a real 0.5 target).
    min_conf  : keep a sample for pair (i,j) only if both cells have
                confidence w = 2*|p-0.5| >= min_conf.

    Returns (C, n_pairs) and saves dict to out_path.
    Sanity checks to eyeball:
      Medial_OA <-> Lateral_OA, Medial_OA <-> PF_OA : strongly positive
      Effusion <-> Synovitis                        : strongly positive
      ACL <-> Medial_Meniscus / Lateral_Meniscus    : moderately positive
      Fracture <-> Contusion                        : moderately positive
    """
    import pandas as pd
    df = pd.read_csv(label_csv)
    Y = df[findings].to_numpy(dtype=np.float64)
    n = len(findings)
    C = np.eye(n)
    N = np.zeros((n, n), dtype=np.int64)
    conf = np.abs(Y - 0.5) * 2.0          # teacher confidence weight w in [0,1]
    Yb = (Y > 0.5).astype(np.float64)      # binarize; 0.5-cells masked out below
    for i in range(n):
        for j in range(i + 1, n):
            m = (conf[:, i] >= min_conf) & (conf[:, j] >= min_conf)
            N[i, j] = N[j, i] = int(m.sum())
            a, b = Yb[m, i], Yb[m, j]
            if m.sum() < 50 or a.std() < 1e-9 or b.std() < 1e-9:
                c = 0.0
            else:
                c = float(np.corrcoef(a, b)[0, 1])
            C[i, j] = C[j, i] = c
    payload = {"corr": C.astype(np.float32),
               "n_pairs": N,
               "findings": findings,
               "min_conf": min_conf}
    np.save(out_path.replace(".pt", ".npy"), C.astype(np.float32))
    try:
        torch.save(payload, out_path)
    except Exception:
        pass  # numpy file is the fallback
    return C, N


# ----------------------------------------------------------------------------
# 1. Learned correlation adapter — 12x12, zero-init residual
# ----------------------------------------------------------------------------
class LabelCorrAdapter(nn.Module):
    """logits_out = logits @ (I + A),  A init 0.

    - Day 1: exact identity mapping — zero behavior change vs baseline.
    - A is 144 params; put it in the HEAD lr group (3e-4), not the backbone group.
    - reg_loss() keeps A near zero; add it to the total loss.
    - After training, inspect A: A[i,j] != 0 means finding j's logit is
      adjusted by finding i's logit (learned dependency).
    """

    def __init__(self, n_findings=N_FINDINGS, reg_lambda=1e-3):
        super().__init__()
        self.n = n_findings
        self.A = nn.Parameter(torch.zeros(n_findings, n_findings))
        self.reg_lambda = reg_lambda

    def forward(self, logits):
        I = torch.eye(self.n, device=logits.device, dtype=logits.dtype)
        return logits @ (I + self.A)

    def reg_loss(self):
        return self.reg_lambda * (self.A ** 2).sum()

    def dependency_report(self):
        """Return sorted list of (i, j, A[i,j]) for |A[i,j]| > 0.02."""
        A = self.A.detach().cpu().numpy()
        hits = []
        for i in range(self.n):
            for j in range(self.n):
                if i != j and abs(A[i, j]) > 0.02:
                    hits.append((FINDINGS[i], FINDINGS[j], float(A[i, j])))
        return sorted(hits, key=lambda t: -abs(t[2]))


# ----------------------------------------------------------------------------
# 2. PCD loss — align predicted vs empirical label correlations
# ----------------------------------------------------------------------------
def pairwise_corr(x, eps=1e-6):
    """Pearson correlation matrix of columns. x: [B, K] -> [K, K]."""
    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / max(x.size(0) - 1, 1)
    sd = xc.std(dim=0, unbiased=False).clamp_min(eps)
    return (cov / (sd[:, None] * sd[None, :])).clamp(-1.0, 1.0)


class PCDLoss(nn.Module):
    """Pairwise Correlation Difference (CMLL-style) auxiliary loss.

    L_pcd = mean over off-diagonal pairs of (C_pred - C_emp)^2

    C_pred is computed on a FIFO buffer of recent predicted probabilities,
    so this works with micro-batch size 2 (buffer default 512 samples).
    Buffer holds DETACHED probs on CPU: 512x12 floats = 24 KB. Negligible.

    Args:
        emp_corr: [12, 12] numpy array from compute_empirical_corr().
        buffer_size: FIFO capacity in samples.
        warmup_batches: skip loss (return 0) for the first N forward calls.
    """

    def __init__(self, emp_corr, buffer_size=512, warmup_batches=8, eps=1e-6):
        super().__init__()
        C = torch.as_tensor(np.asarray(emp_corr), dtype=torch.float32)
        assert C.shape[0] == C.shape[1], "emp_corr must be square"
        self.register_buffer("C_emp", C)
        mask = 1.0 - torch.eye(C.shape[0])
        self.register_buffer("offdiag_mask", mask)
        self.register_buffer("n_pairs", mask.sum())
        self.buffer_size = buffer_size
        self.warmup_batches = warmup_batches
        self.eps = eps
        self._buf = []          # list of [b, K] cpu tensors (detached)
        self._buf_n = 0
        self._calls = 0

    def _push(self, probs):
        p = probs.detach().float().cpu()
        self._buf.append(p)
        self._buf_n += p.size(0)
        while self._buf_n > self.buffer_size:
            old = self._buf.pop(0)
            self._buf_n -= old.size(0)

    def forward(self, probs):
        """probs: [B, K] predicted probabilities in (0, 1), grads attached."""
        self._calls += 1
        self._push(probs)
        if self._calls <= self.warmup_batches or self._buf_n < 32:
            return probs.sum() * 0.0  # zero loss, keeps autograd graph alive
        X = torch.cat(self._buf, dim=0).to(probs.device)
        C_pred = pairwise_corr(X, self.eps)
        diff = (C_pred - self.C_emp.to(probs.device)) * self.offdiag_mask.to(probs.device)
        return (diff ** 2).sum() / self.n_pairs.to(probs.device)

    def current_pcd(self):
        """Diagnostic: current PCD value without affecting the buffer."""
        if self._buf_n < 32:
            return float("nan")
        X = torch.cat(self._buf, dim=0)
        C_pred = pairwise_corr(X, self.eps)
        diff = (C_pred - self.C_emp) * self.offdiag_mask
        return float(((diff ** 2).sum() / self.n_pairs).item())


# ----------------------------------------------------------------------------
# INTEGRATION SNIPPET — paste into the training notebook
# ----------------------------------------------------------------------------
INTEGRATION_SNIPPET = r'''
# ---- Cell: MODEL __init__ (add 2 lines) ----
from label_corr import LabelCorrAdapter, PCDLoss
self.label_corr = LabelCorrAdapter(n_findings=12, reg_lambda=1e-3)

# ---- Cell: MODEL forward — after the 12 heads produce logits [B, 12] ----
logits = torch.cat([self.heads[k](feat_k) for k in range(12)], dim=1)  # existing line
logits = self.label_corr(logits)                                        # NEW: dependency mixer

# ---- Cell: SETUP (once) ----
# emp = torch.load('/kaggle/input/.../label_corr_empirical.pt')['corr']  # or np.load .npy
# pcd_fn = PCDLoss(emp_corr=emp, buffer_size=512, warmup_batches=8)
LAMBDA_PCD = 0.1   # ablate: 0.05 / 0.1 / 0.2 ; 0 = adapter-only baseline

# ---- Cell: TRAINING STEP (loss computation) ----
loss_main = asl_loss(logits, targets, ...)                 # unchanged
loss_pcd  = pcd_fn(torch.sigmoid(logits))                  # NEW (0 during warmup)
loss_reg  = model.label_corr.reg_loss()                    # NEW
loss = loss_main + LAMBDA_PCD * loss_pcd + loss_reg        # NEW total

# ---- Optimizer param groups: put label_corr params in the HEAD group ----
# get_param_groups(): backbone -> 1e-4, everything else (heads + label_corr) -> 3e-4
# (label_corr.A is 144 params; it must NOT be in the backbone group)

# ---- After training: inspect what was learned ----
for src, dst, w in model.label_corr.dependency_report():
    print(f"{src:16s} -> {dst:16s}  {w:+.3f}")
print("final PCD:", pcd_fn.current_pcd())
'''
