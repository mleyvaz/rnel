"""Neural RNEL (requires PyTorch): typed evidential head and per-source evidential network.

RNELHead: features -> evidence (e_T, e_F, e_C, e_U, e_N) -> tuple of Definition 8.1 (typed labels).
PerSourceEvidential: one binary opinion per piece of evidence, trained with true/false claim labels
only; contradiction is read between sources (Definition 8.4), before fusion (Proposition 8.3).
"""
from __future__ import annotations

import torch
from torch import nn

TYPES = ("T", "F", "C", "U", "N")


def rnel_from_evidence(e: torch.Tensor, W: float = 2.0) -> torch.Tensor:
    """Definition 8.1: (T, F, C, U, N, G) = (e_T, e_F, e_C, e_U, e_N, W) / (sum e + W). e: (..., 5)."""
    S = e.sum(-1, keepdim=True) + W
    return torch.cat([e / S, torch.full_like(S, W) / S], dim=-1)


class RNELHead(nn.Module):
    """Linear map from features to non-negative typed evidence (softplus), order T, F, C, U, N."""

    def __init__(self, in_dim: int, n_types: int = 5):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_types)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.linear(h))


def typed_evidential_loss(e: torch.Tensor, y: torch.Tensor, W: float = 2.0, reg: float = 0.0,
                          weight: torch.Tensor | None = None) -> torch.Tensor:
    """Negative log-likelihood of the observed type under alpha/A, alpha = e + W/M, plus reg * wrong evidence."""
    M = e.shape[-1]
    alpha = e + W / M
    nll = torch.log(alpha.sum(-1)) - torch.log(alpha.gather(1, y.unsqueeze(1)).squeeze(1))
    if reg > 0:
        mask = torch.ones_like(e).scatter_(1, y.unsqueeze(1), 0.0)
        nll = nll + reg * (e * mask).sum(-1) / M
    if weight is None:
        return nll.mean()
    w = weight[y]
    return (w * nll).sum() / w.sum()


class PerSourceEvidential(nn.Module):
    """Shared network phi applied to every (evidence, claim) pair -> binary evidence (e_T, e_F).

    forward(x, mask): x (B, P, d) pieces, mask (B, P). Returns claim evidence (mean over pieces) and
    per-piece evidence. Train with typed_evidential_loss on the claim evidence and labels T=0, F=1.
    """

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        e = nn.functional.softplus(self.out(self.phi(x))) * mask.unsqueeze(-1)
        return e.sum(1) / mask.sum(1, keepdim=True).clamp_min(1), e


def source_conflict(e: torch.Tensor, mask: torch.Tensor, W: float = 2.0) -> torch.Tensor:
    """Largest pairwise conflict b_i d_j + d_i b_j between the per-piece opinions (i < j). e: (B, P, 2)."""
    S = e.sum(-1) + W
    b, d = e[..., 0] / S, e[..., 1] / S
    c = b.unsqueeze(2) * d.unsqueeze(1) + d.unsqueeze(2) * b.unsqueeze(1)  # (B, P, P)
    P = e.shape[1]
    upper = torch.triu(torch.ones(P, P, dtype=torch.bool), diagonal=1)
    valid = upper & (mask.unsqueeze(2) * mask.unsqueeze(1)).bool()
    c = torch.where(valid, c, torch.zeros_like(c))
    return c.flatten(1).max(1).values


def fused_dissonance(E: torch.Tensor, W: float = 2.0) -> torch.Tensor:
    """Dissonance of the fused binary evidence: 2 min(b, d) = (1 - u)(1 - |b - d| / (b + d))."""
    S = E.sum(-1) + W
    return 2 * torch.minimum(E[..., 0], E[..., 1]) / S
