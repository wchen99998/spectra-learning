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
        A = torch.randn(
            proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype
        )
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t  # [B, N, num_slices, knots]
        if valid_mask is None:
            cos_mean = x_t.cos().mean(dim=0)
            sin_mean = x_t.sin().mean(dim=0)
            position_weights = torch.ones(
                proj.shape[1], device=proj.device, dtype=proj.dtype
            )
        else:
            weights = valid_mask.to(dtype=proj.dtype, device=proj.device)
            sample_count = weights.sum(dim=0).clamp_min(1.0)
            weight_view = weights.unsqueeze(-1).unsqueeze(-1)
            cos_mean = (x_t.cos() * weight_view).sum(dim=0) / sample_count.view(
                -1, 1, 1
            )
            sin_mean = (x_t.sin() * weight_view).sum(dim=0) / sample_count.view(
                -1, 1, 1
            )
            position_weights = valid_mask.any(dim=0).to(
                dtype=proj.dtype, device=proj.device
            )
        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = err @ self.weights
        per_position = statistic.mean(dim=-1)
        return (per_position * position_weights).sum() / position_weights.sum().clamp_min(1.0)
