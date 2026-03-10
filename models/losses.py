from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
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
        flat = proj.reshape(-1, proj.size(-1))
        A = torch.randn(
            flat.size(-1), self.num_slices, device=flat.device, dtype=flat.dtype
        )
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
        statistic = err @ self.weights
        return statistic.mean()
