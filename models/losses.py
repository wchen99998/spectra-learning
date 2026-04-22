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

    def _sample_directions(
        self,
        dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        directions = torch.randn(
            dim,
            self.num_slices,
            device=device,
            dtype=dtype,
        )
        return directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)

    def _loss_from_flat(
        self,
        flat: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        directions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if directions is None:
            directions = self._sample_directions(
                flat.size(-1),
                device=flat.device,
                dtype=flat.dtype,
            )
        x_t = (flat @ directions).unsqueeze(-1) * self.t.to(flat)
        if valid_mask is None:
            sample_count = flat.new_tensor(float(flat.size(0)))
            cos_mean = x_t.cos().mean(0)
            sin_mean = x_t.sin().mean(0)
        else:
            mask_weights = valid_mask.reshape(-1).to(dtype=flat.dtype, device=flat.device)
            sample_count = mask_weights.sum()
            if bool((sample_count == 0).item()):
                return flat.new_zeros(())
            weight_view = mask_weights.unsqueeze(-1).unsqueeze(-1)
            cos_mean = (x_t.cos() * weight_view).sum(0) / sample_count
            sin_mean = (x_t.sin() * weight_view).sum(0) / sample_count
        err = (cos_mean - self.phi.to(flat)).square() + sin_mean.square()
        statistic = err @ self.weights.to(flat)
        return statistic.mean() * sample_count

    def forward(
        self,
        proj: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat = proj.reshape(-1, proj.size(-1))
        flat_mask = None if valid_mask is None else valid_mask.reshape(-1)
        return self._loss_from_flat(flat, flat_mask)


class SlotwiseSIGReg(SIGReg):
    def forward(
        self,
        proj: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if proj.ndim == 3:
            proj = proj.unsqueeze(1)
            if valid_mask is not None:
                valid_mask = valid_mask.unsqueeze(1)
        batch_size, num_views, num_slots, dim = proj.shape
        slot_proj = proj.reshape(batch_size * num_views, num_slots, dim)
        slot_mask = (
            None
            if valid_mask is None
            else valid_mask.reshape(batch_size * num_views, num_slots)
        )
        directions = self._sample_directions(
            dim,
            device=proj.device,
            dtype=proj.dtype,
        )
        total = proj.new_zeros(())
        for slot_idx in range(num_slots):
            total = total + self._loss_from_flat(
                slot_proj[:, slot_idx, :],
                None if slot_mask is None else slot_mask[:, slot_idx],
                directions=directions,
            )
        return total


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
