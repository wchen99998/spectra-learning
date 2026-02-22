"""SIGReg losses for peak-set pretraining.

LeJEPA-style SIGReg: Epps-Pulley Gaussianity statistic with random slicing
directions sampled inside forward().
"""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Epps-Pulley Gaussianity regularizer (LeJEPA-style).

    Takes a single tensor ``proj [V, B, D]`` (all views stacked) and computes
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

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute the Epps-Pulley statistic.

        Parameters
        ----------
        proj : Tensor [V, B, D]
            Projected embeddings for all views.

        Returns
        -------
        Scalar — mean statistic across slicing directions.
        """
        A = torch.randn(proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t  # [V, B, num_slices, knots]
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
