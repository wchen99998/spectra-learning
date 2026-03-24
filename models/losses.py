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
        batch_size, num_tokens, dim = proj.shape
        A = torch.randn(
            dim, self.num_slices, device=proj.device, dtype=proj.dtype
        )
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t  # [B, N, num_slices, knots]
        if valid_mask is None:
            cos_mean = x_t.cos().mean(0)
            sin_mean = x_t.sin().mean(0)
            sample_count = torch.full(
                (num_tokens,), batch_size, device=proj.device, dtype=proj.dtype
            )
            position_mask = torch.ones(
                num_tokens, device=proj.device, dtype=torch.bool
            )
        else:
            weights = valid_mask.to(dtype=proj.dtype, device=proj.device)
            sample_count = weights.sum(dim=0)
            denom = sample_count.clamp_min(1.0).unsqueeze(-1).unsqueeze(-1)
            weight_view = weights.unsqueeze(-1).unsqueeze(-1)
            cos_mean = (x_t.cos() * weight_view).sum(0) / denom
            sin_mean = (x_t.sin() * weight_view).sum(0) / denom
            position_mask = valid_mask.any(dim=0)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = (err @ self.weights) * sample_count.unsqueeze(-1)
        if not position_mask.any():
            return proj.new_tensor(0.0)
        return statistic[position_mask].mean()
