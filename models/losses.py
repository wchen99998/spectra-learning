"""Loss modules for peak-set pretraining.

SIGReg: Epps-Pulley Gaussianity statistic with random slicing
directions sampled inside forward().
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SIGReg(nn.Module):
    """Epps-Pulley Gaussianity regularizer.

    Takes a single tensor ``proj [..., D]`` and computes
    the characteristic-function distance from a standard Gaussian.  Random
    projection directions are sampled *inside* ``forward()`` — no
    pre-computed projection needed.
    """

    def __init__(self, knots: int = 17, num_slices: int = 256):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(
        self,
        proj: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Epps-Pulley statistic.

        Parameters
        ----------
        proj : Tensor [..., D]
            Projected embeddings with feature dimension in the last axis.
        valid_mask : Tensor [...], optional
            Boolean mask aligned with ``proj`` leading dimensions.
            When provided, statistics are computed over valid positions only
            without changing tensor shape.

        Returns
        -------
        Scalar — mean statistic across slicing directions.
        """
        flat = proj.reshape(-1, proj.size(-1))
        A = torch.randn(flat.size(-1), self.num_slices, device=flat.device, dtype=flat.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (flat @ A).unsqueeze(-1) * self.t  # [N, num_slices, knots]
        if valid_mask is None:
            cos_mean = x_t.cos().mean(0)
            sin_mean = x_t.sin().mean(0)
            sample_count = torch.tensor(
                float(flat.size(0)),
                device=flat.device,
                dtype=flat.dtype,
            )
        else:
            weights = valid_mask.reshape(-1).to(dtype=flat.dtype, device=flat.device)
            sample_count = weights.sum().clamp_min(1.0)
            weight_view = weights.unsqueeze(-1).unsqueeze(-1)
            cos_mean = (x_t.cos() * weight_view).sum(0) / sample_count
            sin_mean = (x_t.sin() * weight_view).sum(0) / sample_count
        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = (err @ self.weights) * sample_count
        return statistic.mean()


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance regularization loss."""

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    @staticmethod
    def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
        dim = matrix.size(0)
        return matrix.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()

    def forward(
        self,
        proj_a: torch.Tensor,
        proj_b: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = proj_a.reshape(-1, proj_a.size(-1))
        b = proj_b.reshape(-1, proj_b.size(-1))

        if valid_mask is not None:
            keep = valid_mask.reshape(-1)
            a = a[keep]
            b = b[keep]

        repr_loss = F.mse_loss(a, b)

        a_centered = a - a.mean(dim=0, keepdim=True)
        b_centered = b - b.mean(dim=0, keepdim=True)

        std_a = torch.sqrt(a_centered.var(dim=0, unbiased=False) + self.eps)
        std_b = torch.sqrt(b_centered.var(dim=0, unbiased=False) + self.eps)
        std_loss = 0.5 * (
            F.relu(self.gamma - std_a).mean()
            + F.relu(self.gamma - std_b).mean()
        )

        cov_a = (a_centered.T @ a_centered) / a_centered.size(0)
        cov_b = (b_centered.T @ b_centered) / b_centered.size(0)
        cov_loss = (
            self._off_diagonal(cov_a).pow(2).sum() / cov_a.size(0)
            + self._off_diagonal(cov_b).pow(2).sum() / cov_b.size(0)
        )

        return (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
