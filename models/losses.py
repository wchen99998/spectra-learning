"""JEPA losses for peak-set pretraining.

BCS (Batched Characteristic Slicing) regularizer adapted from
eb_jepa/losses.py for collapse prevention via SIGReg.
"""

from __future__ import annotations

import torch
from torch import nn


def _epps_pulley(
    x: torch.Tensor,
    t_min: float = -3.0,
    t_max: float = 3.0,
    n_points: int = 10,
) -> torch.Tensor:
    """Epps-Pulley test statistic measuring deviation from N(0,1).

    Args:
        x: [N, M] projected embeddings (N samples, M slices).
    Returns:
        T: [M] per-slice test statistic.
    """
    t = torch.linspace(t_min, t_max, n_points, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)
    x_t = x.unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)  # (M, T)
    err = exp_f * (ecf - exp_f).abs() ** 2
    return torch.trapezoid(err, t, dim=1)  # (M,)


class BCSLoss(nn.Module):
    """BCS (Batched Characteristic Slicing) regularizer for SIGReg.

    Projects embeddings onto random 1-D slices and penalises deviation
    from a standard Gaussian via the Epps-Pulley characteristic-function
    test.  Deterministic per-step seeding keeps behaviour reproducible
    and torch.compile-friendly.
    """

    def __init__(self, num_slices: int = 256, lmbd: float = 10.0):
        super().__init__()
        self.num_slices = num_slices
        self.lmbd = lmbd
        self.step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: [N, D] embeddings.  Returns: scalar BCS loss."""
        dev = x.device
        with torch.no_grad():
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            A = torch.randn(x.size(1), self.num_slices, device=dev, generator=g)
            A = A / A.norm(p=2, dim=0)
        projected = x @ A  # [N, num_slices]
        self.step += 1
        return self.lmbd * _epps_pulley(projected).mean()


def squared_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE prediction loss with optional valid mask.

    Args:
        pred: [B, N, D] predicted embeddings.
        target: [B, N, D] target embeddings.
        valid_mask: [B, N] bool mask of valid positions.
    """
    diff = (pred - target).pow(2)
    per_token = diff.mean(dim=-1)
    if valid_mask is not None:
        per_token = per_token * valid_mask.float()
        return per_token.sum() / valid_mask.float().sum().clamp(min=1.0)
    return per_token.mean()
