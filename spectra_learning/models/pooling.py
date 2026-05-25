from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn


class CovariancePool(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        compressed_dim: int,
    ) -> None:
        super().__init__()
        self.left_proj = nn.Linear(input_dim, compressed_dim, bias=False)
        self.right_proj = nn.Linear(input_dim, compressed_dim, bias=False)
        nn.init.xavier_normal_(self.left_proj.weight)
        nn.init.xavier_normal_(self.right_proj.weight)

    @property
    def compressed_dim(self) -> int:
        return self.left_proj.out_features

    def covariance_matrix(
        self,
        peak_embeddings: Float[Tensor, "batch peaks dim"],
        valid_mask: Bool[Tensor, "batch peaks"],
    ) -> Float[Tensor, "batch compressed compressed"]:
        mask = valid_mask.unsqueeze(-1).to(dtype=peak_embeddings.dtype)
        left = self.left_proj(peak_embeddings) * mask
        right = self.right_proj(peak_embeddings) * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        covariance = left.transpose(1, 2) @ right
        return covariance / denom.unsqueeze(-1)

    def forward(
        self,
        peak_embeddings: Float[Tensor, "batch peaks dim"],
        valid_mask: Bool[Tensor, "batch peaks"],
    ) -> Float[Tensor, "batch flattened_covariance"]:
        with torch.autocast(device_type=peak_embeddings.device.type, enabled=False):
            covariance = self.covariance_matrix(peak_embeddings.float(), valid_mask)
        return covariance.flatten(start_dim=1)

    def reconstruction_loss(
        self,
        peak_embeddings: Float[Tensor, "batch peaks dim"],
        valid_mask: Bool[Tensor, "batch peaks"],
    ) -> Float[Tensor, ""]:
        with torch.autocast(device_type=peak_embeddings.device.type, enabled=False):
            x = peak_embeddings.detach().float()
            mask = valid_mask.unsqueeze(-1).to(dtype=x.dtype)
            x = x * mask
            denom = mask.sum(dim=1).clamp_min(1.0)

            covariance = self.covariance_matrix(x, valid_mask)
            left_gram = self.left_proj.weight.float() @ self.left_proj.weight.float().T
            right_gram = (
                self.right_proj.weight.float() @ self.right_proj.weight.float().T
            )
            projected_reconstruction = torch.einsum(
                "ij,bjk->bik",
                left_gram,
                covariance,
            )
            projected_reconstruction = torch.einsum(
                "bij,jk->bik",
                projected_reconstruction,
                right_gram,
            )
            reconstruction_norm_sq = (
                covariance * projected_reconstruction
            ).sum(dim=(1, 2))
            cross_term = covariance.square().sum(dim=(1, 2))

            token_gram = x @ x.transpose(1, 2) / denom.unsqueeze(-1)
            target_norm_sq = token_gram.square().sum(dim=(1, 2))
            loss_sq = (
                reconstruction_norm_sq
                - 2.0 * cross_term
                + target_norm_sq
            ).clamp_min(0.0)
            return torch.sqrt(loss_sq + 1e-12).mean()


def build_covariance_pooler_from_config(config: Any) -> CovariancePool | None:
    dim = int(_config_get(config, "covariance_pooling_dim", -1))
    if dim <= 0 or not bool(_config_get(config, "train_covariance_pooling", True)):
        return None
    pooler = CovariancePool(
        input_dim=int(_config_get(config, "model_dim", None)),
        compressed_dim=dim,
    )
    pooler.requires_grad_(True)
    return pooler


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)
