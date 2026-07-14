from __future__ import annotations

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn


def token_mask_with_optional_cls(
    valid_mask: Bool[Tensor, "batch peaks"],
    *,
    include_cls_token: bool,
) -> Bool[Tensor, "batch tokens"]:
    if not include_cls_token:
        return valid_mask
    cls_mask = torch.ones(
        valid_mask.shape[0],
        1,
        device=valid_mask.device,
        dtype=torch.bool,
    )
    return torch.cat([valid_mask, cls_mask], dim=1)


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
        peak_embeddings = peak_embeddings[:, : valid_mask.shape[1]]
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
            x = peak_embeddings[:, : valid_mask.shape[1]].detach().float()
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


class SinglePairCovariancePool(nn.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        compressed_dim: int,
        include_diagonal: bool = False,
        include_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.single_pool = CovariancePool(
            input_dim=single_dim,
            compressed_dim=compressed_dim,
        )
        self.pair_left_proj = nn.Linear(pair_dim, compressed_dim, bias=False)
        self.pair_right_proj = nn.Linear(pair_dim, compressed_dim, bias=False)
        self.output_proj = nn.Linear(
            2 * compressed_dim * compressed_dim,
            compressed_dim * compressed_dim,
        )
        self.include_diagonal = include_diagonal
        self.include_cls_token = include_cls_token
        nn.init.xavier_normal_(self.pair_left_proj.weight)
        nn.init.xavier_normal_(self.pair_right_proj.weight)
        nn.init.xavier_normal_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @property
    def compressed_dim(self) -> int:
        return self.single_pool.compressed_dim

    @property
    def output_dim(self) -> int:
        return self.compressed_dim * self.compressed_dim

    def pair_covariance_matrix(
        self,
        pair_embeddings: Float[Tensor, "batch peaks peaks pair"],
        valid_mask: Bool[Tensor, "batch peaks"],
    ) -> Float[Tensor, "batch compressed compressed"]:
        token_mask = token_mask_with_optional_cls(
            valid_mask,
            include_cls_token=self.include_cls_token,
        )
        num_tokens = token_mask.shape[1]
        batch_size = pair_embeddings.shape[0]
        if pair_embeddings.shape[1] >= num_tokens and pair_embeddings.shape[2] >= num_tokens:
            pair_embeddings = pair_embeddings[:, :num_tokens, :num_tokens]
            pair_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
        else:
            num_tokens = pair_embeddings.shape[1]
            pair_mask = torch.ones(
                batch_size,
                num_tokens,
                num_tokens,
                device=pair_embeddings.device,
                dtype=torch.bool,
            )
        if not self.include_diagonal:
            diagonal = torch.eye(
                num_tokens,
                device=pair_embeddings.device,
                dtype=torch.bool,
            ).view(1, num_tokens, num_tokens)
            pair_mask = pair_mask & ~diagonal
        pair_mask_f = pair_mask.unsqueeze(-1).to(dtype=pair_embeddings.dtype)
        left = self.pair_left_proj(pair_embeddings) * pair_mask_f
        right = self.pair_right_proj(pair_embeddings) * pair_mask_f
        left = left.reshape(batch_size, num_tokens * num_tokens, self.compressed_dim)
        right = right.reshape(batch_size, num_tokens * num_tokens, self.compressed_dim)
        denom = pair_mask_f.sum(dim=(1, 2)).clamp_min(1.0)
        covariance = left.transpose(1, 2) @ right
        return covariance / denom.unsqueeze(-1)

    def forward(
        self,
        peak_embeddings: Float[Tensor, "batch peaks dim"],
        valid_mask: Bool[Tensor, "batch peaks"],
        pair_embeddings: Float[Tensor, "batch peaks peaks pair"],
    ) -> Float[Tensor, "batch flattened_covariance"]:
        with torch.autocast(device_type=peak_embeddings.device.type, enabled=False):
            token_mask = token_mask_with_optional_cls(
                valid_mask,
                include_cls_token=self.include_cls_token,
            )
            single_covariance = self.single_pool.covariance_matrix(
                peak_embeddings[:, : token_mask.shape[1]].float(),
                token_mask,
            )
            pair_covariance = self.pair_covariance_matrix(
                pair_embeddings.float(),
                valid_mask,
            )
            single_features = single_covariance.flatten(start_dim=1)
            pair_features = pair_covariance.flatten(start_dim=1)
            single_features = torch.nn.functional.layer_norm(
                single_features,
                (self.output_dim,),
            )
            pair_features = torch.nn.functional.layer_norm(
                pair_features,
                (self.output_dim,),
            )
            pooled = torch.cat(
                [
                    single_features,
                    pair_features,
                ],
                dim=1,
            )
            return self.output_proj(pooled)
