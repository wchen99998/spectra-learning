import torch
import torch.nn.functional as F
from torch import nn


def _off_diagonal_squared_sum(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum() - x.diagonal().square().sum()


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
            sample_count = flat.new_tensor(float(flat.size(0)))
            cos_mean = x_t.cos().mean(0)
            sin_mean = x_t.sin().mean(0)
        else:
            weights = valid_mask.reshape(-1).to(dtype=flat.dtype, device=flat.device)
            sample_count = weights.sum().clamp_min(1.0)
            weight_view = weights.unsqueeze(-1).unsqueeze(-1)
            cos_mean = (x_t.cos() * weight_view).sum(0) / sample_count
            sin_mean = (x_t.sin() * weight_view).sum(0) / sample_count
        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = err @ self.weights
        return (statistic * sample_count).mean()


class VICReg(nn.Module):
    def __init__(
        self,
        *,
        inv_coeff: float = 0.0,
        var_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        variance_target: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.inv_coeff = float(inv_coeff)
        self.var_coeff = float(var_coeff)
        self.cov_coeff = float(cov_coeff)
        self.variance_target = float(variance_target)
        self.eps = float(eps)

    def _moments(
        self,
        z: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = z.float().reshape(-1, z.shape[-1])
        if valid_mask is None:
            weights = torch.ones(
                flat.shape[0], device=flat.device, dtype=flat.dtype
            )
        else:
            weights = valid_mask.reshape(-1).to(device=flat.device, dtype=flat.dtype)

        count = weights.sum().clamp_min(1.0)
        if bool((count <= 1.0).item()):
            zero = flat.new_zeros(())
            d = flat.shape[-1]
            return flat.new_zeros(d), flat.new_zeros(d, d), zero

        weights_col = weights.unsqueeze(-1)
        mean = (flat * weights_col).sum(dim=0, keepdim=True) / count
        centered = flat - mean
        cov = centered.transpose(0, 1) @ (centered * weights_col)
        cov = cov / (count - 1.0).clamp_min(1.0)
        var = cov.diagonal()
        return var, cov, count

    def _var_cov_loss(
        self,
        z: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var, cov, count = self._moments(z, valid_mask)
        if bool((count <= 1.0).item()):
            zero = z.new_zeros(())
            return zero, zero
        std = torch.sqrt(var.clamp_min(0.0) + self.eps)
        var_loss = F.relu(self.variance_target - std).mean()
        cov_loss = _off_diagonal_squared_sum(cov) / cov.shape[0]
        return var_loss, cov_loss

    def _invariance_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diff = (z1 - z2).float()
        if valid_mask is None:
            return diff.square().mean()
        token_loss = diff.square().mean(dim=-1)
        weights = valid_mask.reshape(-1).to(
            dtype=token_loss.dtype,
            device=token_loss.device,
        )
        token_loss = token_loss.reshape(-1)
        return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        valid_mask2: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        var1, cov1 = self._var_cov_loss(z1, valid_mask)

        if z2 is None:
            inv = z1.new_zeros(())
            loss = self.var_coeff * var1 + self.cov_coeff * cov1
            return {
                "loss": loss,
                "inv_loss": inv,
                "var_loss": var1,
                "cov_loss": cov1,
            }

        inv = self._invariance_loss(z1, z2, valid_mask)
        var2, cov2 = self._var_cov_loss(
            z2, valid_mask if valid_mask2 is None else valid_mask2
        )
        var = 0.5 * (var1 + var2)
        cov = 0.5 * (cov1 + cov2)
        loss = self.inv_coeff * inv + self.var_coeff * var + self.cov_coeff * cov
        return {
            "loss": loss,
            "inv_loss": inv,
            "var_loss": var,
            "cov_loss": cov,
        }
