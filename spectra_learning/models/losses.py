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
        with torch.autocast(device_type=flat.device.type, enabled=False):
            flat = flat.float()
            if directions is None:
                directions = self._sample_directions(
                    flat.size(-1),
                    device=flat.device,
                    dtype=flat.dtype,
                )
            else:
                directions = directions.to(device=flat.device, dtype=flat.dtype)
            x_t = (flat @ directions).unsqueeze(-1) * self.t.to(flat)
            if valid_mask is None:
                sample_count = flat.new_tensor(float(flat.size(0)))
                cos_mean = x_t.cos().mean(0)
                sin_mean = x_t.sin().mean(0)
            else:
                mask_weights = valid_mask.reshape(-1).to(
                    dtype=flat.dtype,
                    device=flat.device,
                )
                sample_count = mask_weights.sum()
                safe_count = sample_count.clamp_min(1.0)
                weight_view = mask_weights.unsqueeze(-1).unsqueeze(-1)
                cos_mean = (x_t.cos() * weight_view).sum(0) / safe_count
                sin_mean = (x_t.sin() * weight_view).sum(0) / safe_count
            err = (cos_mean - self.phi.to(flat)).square() + sin_mean.square()
            statistic = err @ self.weights.to(flat)
            return statistic.mean() * sample_count

    def forward(
        self,
        proj: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        directions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat = proj.reshape(-1, proj.size(-1))
        flat_mask = None if valid_mask is None else valid_mask.reshape(-1)
        return self._loss_from_flat(flat, flat_mask, directions=directions)


class SlotwiseSIGReg(SIGReg):
    def forward(
        self,
        proj: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        directions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.autocast(device_type=proj.device.type, enabled=False):
            proj = proj.float()
            if proj.ndim == 3:
                proj = proj.unsqueeze(1)
                if valid_mask is not None:
                    valid_mask = valid_mask.unsqueeze(1)
            batch_size, num_views, num_slots, dim = proj.shape
            slot_proj = proj.reshape(batch_size * num_views, num_slots, dim)
            if directions is None:
                directions = self._sample_directions(
                    dim,
                    device=proj.device,
                    dtype=proj.dtype,
                )
            else:
                directions = directions.to(device=proj.device, dtype=proj.dtype)
            projected = torch.einsum("bld,ds->bls", slot_proj, directions)
            x_t = projected.unsqueeze(-1) * self.t.to(projected)
            if valid_mask is None:
                sample_count = projected.new_full(
                    (num_slots,),
                    float(batch_size * num_views),
                )
                cos_mean = x_t.cos().mean(0)
                sin_mean = x_t.sin().mean(0)
            else:
                slot_mask = valid_mask.reshape(batch_size * num_views, num_slots).to(
                    dtype=proj.dtype,
                    device=proj.device,
                )
                sample_count = slot_mask.sum(0)
                safe_count = sample_count.clamp_min(1.0)
                weight_view = slot_mask.unsqueeze(-1).unsqueeze(-1)
                cos_mean = (x_t.cos() * weight_view).sum(0) / safe_count[:, None, None]
                sin_mean = (x_t.sin() * weight_view).sum(0) / safe_count[:, None, None]
            err = (cos_mean - self.phi.to(projected)).square() + sin_mean.square()
            statistic = err @ self.weights.to(projected)
            return (statistic.mean(-1) * sample_count).sum()
