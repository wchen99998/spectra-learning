import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spectra_learning.models.transformer import _build_norm
from spectra_learning.models.common import (
    _build_frozen_position_embedding,
    _merge_visible_mask,
)
from spectra_learning.models.pairformer import PairFeatureEmbedder, PairformerBlock
from spectra_learning.models.peak_features import PeakFeatureEmbedder


class PeakSetEncoder(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        embedder: PeakFeatureEmbedder,
        num_layers: int,
        num_heads: int,
        attention_mlp_multiple: float = 4.0,
        norm_eps: float = 1e-5,
        apply_final_norm: bool = True,
        num_peaks: int = 64,
        use_position_embedding: bool = True,
        pair_dim: int | None = None,
        pair_num_heads: int | None = None,
        pair_feature_hidden_dim: int = 128,
        pairformer_dropout: float = 0.0,
        pairformer_refresh_pair: bool = True,
        pairformer_use_cuequivariance: bool = True,
        pairformer_mz_scale: float = 1000.0,
        pairformer_precursor_mz_scale: float = 1000.0,
        pairformer_use_fourier_features: bool = True,
        pairformer_fourier_num_freqs: int = 16,
        pairformer_fourier_x_min: float = 1e-2,
        pairformer_fourier_x_max: float = 1000.0,
        pairformer_relative_fourier_x_min: float = 1e-3,
        pairformer_relative_fourier_x_max: float = 1.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_position_embedding = use_position_embedding
        self.embedder = embedder
        self.position_embedding = _build_frozen_position_embedding(
            num_peaks,
            model_dim,
        )
        pair_dim = model_dim if pair_dim is None else pair_dim
        pair_num_heads = num_heads if pair_num_heads is None else pair_num_heads
        self.pair_embedder = PairFeatureEmbedder(
            single_dim=model_dim,
            pair_dim=pair_dim,
            hidden_dim=pair_feature_hidden_dim,
            mz_scale=pairformer_mz_scale,
            precursor_mz_scale=pairformer_precursor_mz_scale,
            use_fourier_features=pairformer_use_fourier_features,
            fourier_num_freqs=pairformer_fourier_num_freqs,
            fourier_x_min=pairformer_fourier_x_min,
            fourier_x_max=pairformer_fourier_x_max,
            relative_fourier_x_min=pairformer_relative_fourier_x_min,
            relative_fourier_x_max=pairformer_relative_fourier_x_max,
        )
        self.blocks = nn.ModuleList(
            [
                PairformerBlock(
                    single_dim=model_dim,
                    pair_dim=pair_dim,
                    num_heads=num_heads,
                    pair_num_heads=pair_num_heads,
                    attention_mlp_multiple=attention_mlp_multiple,
                    pair_feature_hidden_dim=pair_feature_hidden_dim,
                    norm_eps=norm_eps,
                    dropout=pairformer_dropout,
                    refresh_pair=pairformer_refresh_pair,
                    use_cuequivariance=pairformer_use_cuequivariance,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = (
            _build_norm(model_dim, eps=norm_eps, affine=False)
            if apply_final_norm
            else nn.Identity()
        )

    def _add_positions(
        self,
        x: Float[Tensor, "batch peaks dim"],
    ) -> Float[Tensor, "batch peaks dim"]:
        if not self.use_position_embedding:
            return x
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.position_embedding(positions).to(dtype=x.dtype)

    def forward_with_block_outputs(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        valid_mask: Bool[Tensor, "batch peaks"] | None = None,
        visible_mask: Bool[Tensor, "batch peaks"] | None = None,
        block_indices: list[int] | tuple[int, ...] = (),
        precursor_mz: Float[Tensor, "batch"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch peaks dim"],
        list[Float[Tensor, "batch peaks dim"]],
    ]:
        block_indices = tuple(idx for idx in block_indices)
        peak_valid_mask = (
            torch.ones_like(peak_mz, dtype=torch.bool)
            if valid_mask is None
            else valid_mask
        )
        peak_visible_mask = _merge_visible_mask(peak_valid_mask, visible_mask)
        if peak_visible_mask is None:
            peak_visible_mask = peak_valid_mask
        x = self._add_positions(self.embedder(peak_mz, peak_intensity))
        # x: [B, N, D], z: [B, N, N, P], masks: [B, N]
        z = self.pair_embedder(
            peak_mz,
            peak_intensity,
            x,
            peak_visible_mask,
            precursor_mz=precursor_mz,
        )
        selected = set(block_indices)
        selected_peak_outputs: dict[int, Float[Tensor, "batch peaks dim"]] = {}
        for block_idx, block in enumerate(self.blocks, start=1):
            x, z = block(
                x,
                z,
                peak_visible_mask,
                peak_visible_mask,
            )
            if block_idx in selected and block_idx != self.num_layers:
                selected_peak_outputs[block_idx] = x
        x = self.final_norm(x)
        if self.num_layers in selected:
            selected_peak_outputs[self.num_layers] = x
        return x, [selected_peak_outputs[idx] for idx in block_indices]

    def forward_peak_block_outputs(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        valid_mask: Bool[Tensor, "batch peaks"] | None = None,
        visible_mask: Bool[Tensor, "batch peaks"] | None = None,
        block_indices: list[int] | tuple[int, ...] = (),
        precursor_mz: Float[Tensor, "batch"] | None = None,
    ) -> list[Float[Tensor, "batch peaks dim"]]:
        _, peak_block_outputs = self.forward_with_block_outputs(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            block_indices=block_indices,
            precursor_mz=precursor_mz,
        )
        return peak_block_outputs

    def forward(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        valid_mask: Bool[Tensor, "batch peaks"] | None = None,
        visible_mask: Bool[Tensor, "batch peaks"] | None = None,
        precursor_mz: Float[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch peaks dim"]:
        output, _ = self.forward_with_block_outputs(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            precursor_mz=precursor_mz,
        )
        return output
