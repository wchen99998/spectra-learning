"""SIGReg losses for peak-set pretraining.

BCS (Batched Characteristic Slicing) regularizer mirrors
~/eb_jepa/eb_jepa/losses.py behavior.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch import nn


def _all_reduce_avg(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        return dist_nn.all_reduce(x, op=dist.ReduceOp.SUM) / dist.get_world_size()
    return x


def _epps_pulley(
    x: torch.Tensor,
    t: torch.Tensor,
    exp_f: torch.Tensor,
) -> torch.Tensor:
    x = x.float()
    t = t.float()
    exp_f = exp_f.float()
    x_t = x.unsqueeze(2) * t
    ecf_real = _all_reduce_avg(torch.cos(x_t).mean(0))
    ecf_imag = _all_reduce_avg(torch.sin(x_t).mean(0))
    err = exp_f * ((ecf_real - exp_f).square() + ecf_imag.square())
    return torch.trapezoid(err, t, dim=1)


class BCSLoss(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg."""

    def __init__(
        self,
        num_slices: int = 256,
        lmbd: float = 10.0,
        t_min: float = -3.0,
        t_max: float = 3.0,
        n_points: int = 10,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.lmbd = lmbd
        t = torch.linspace(t_min, t_max, n_points)
        self.register_buffer("ep_t", t)
        self.register_buffer("ep_exp_f", torch.exp(-0.5 * t**2))

    def sample_projection(
        self,
        feature_dim: int,
        *,
        device: torch.device,
        seed: int | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            proj_shape = (feature_dim, self.num_slices)
            if seed is None:
                proj = torch.randn(proj_shape, device=device)
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                proj = torch.randn(proj_shape, device=device, generator=generator)
            proj = proj / proj.norm(p=2, dim=0)
        return proj

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        proj: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if proj is None:
            proj = self.sample_projection(z1.size(1), device=z1.device)
        proj = proj.to(device=z1.device, dtype=z1.dtype)

        z_both = torch.cat([z1, z2], dim=0)
        views = z_both @ proj
        view1, view2 = views.chunk(2, dim=0)

        bcs = (_epps_pulley(view1, self.ep_t, self.ep_exp_f).mean()
               + _epps_pulley(view2, self.ep_t, self.ep_exp_f).mean()) / 2
        invariance_loss = F.mse_loss(z1, z2)
        loss = invariance_loss + self.lmbd * bcs
        return {
            "loss": loss,
            "bcs_loss": bcs,
            "invariance_loss": invariance_loss,
        }
