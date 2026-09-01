"""Evidential Dirichlet Learning utilities for GeoRectNetPlus."""

import torch
import torch.nn.functional as F

def kl_dirichlet(alpha: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """KL divergence between Dir(α) and Dir(1,...,1)."""
    ones = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_ones = ones.sum(dim=1, keepdim=True)
    kl = (torch.lgamma(S_alpha) - torch.lgamma(S_ones)
          - (torch.lgamma(alpha) - torch.lgamma(ones)).sum(dim=1, keepdim=True)
          + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=1, keepdim=True))
    return kl.mean()

def edl_loss(alpha, target, epoch, total_epochs, annealing_epochs=10):
    """EDL loss = Bayes risk + annealed KL."""
    target_1h = torch.cat([1.0 - target, target], dim=1)
    S = alpha.sum(dim=1, keepdim=True)
    bayes_risk = (target_1h * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    annealing_coef = min(1.0, epoch / max(1, annealing_epochs))
    alpha_tilde = target_1h + (1.0 - target_1h) * (alpha - 1.0) + 1.0
    kl = kl_dirichlet(alpha_tilde, num_classes=2)
    return bayes_risk.mean() + annealing_coef * kl


# ═══════════════════════════════════════════════════════════════════
#  CONTRIBUTION 2: Cross-Layer Attention Agreement Map (CLAAM / MLAA)
# ═══════════════════════════════════════════════════════════════════
