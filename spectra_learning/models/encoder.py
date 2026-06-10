import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spectra_learning.models.transformer import _build_norm
from spectra_learning.models.common import (
    _build_frozen_position_embedding,
    _merge_visible_mask,
)
from spectra_learning.models.pairmixer import PairFeatureEmbedder, PairMixerBlock
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
        apply_final_pair_norm: bool = False,
        num_peaks: int = 64,
        use_position_embedding: bool = True,
        pair_dim: int | None = None,
        pair_feature_hidden_dim: int = 128,
        pairformer_dropout: float = 0.0,
        pairmixer_use_pair_bias_attention: bool = False,
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
        self.cls_token = nn.Parameter(torch.empty(model_dim))
        self.cls_to_peak_pair_token = nn.Parameter(torch.empty(pair_dim))
        self.peak_to_cls_pair_token = nn.Parameter(torch.empty(pair_dim))
        self.cls_cls_pair_token = nn.Parameter(torch.empty(pair_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.cls_to_peak_pair_token, std=0.02)
        nn.init.normal_(self.peak_to_cls_pair_token, std=0.02)
        nn.init.normal_(self.cls_cls_pair_token, std=0.02)
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
                PairMixerBlock(
                    single_dim=model_dim,
                    pair_dim=pair_dim,
                    num_heads=num_heads,
                    attention_mlp_multiple=attention_mlp_multiple,
                    norm_eps=norm_eps,
                    dropout=pairformer_dropout,
                    use_pair_bias_attention=pairmixer_use_pair_bias_attention,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = (
            _build_norm(model_dim, eps=norm_eps, affine=False)
            if apply_final_norm
            else nn.Identity()
        )
        self.final_pair_norm = (
            _build_norm(pair_dim, eps=norm_eps, affine=True)
            if apply_final_pair_norm
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

    def _append_cls_token(
        self,
        x: Float[Tensor, "batch peaks dim"],
    ) -> Float[Tensor, "batch tokens dim"]:
        cls = self.cls_token.view(1, 1, -1).expand(x.shape[0], 1, -1)
        cls = cls.to(dtype=x.dtype) + x[:, :1] * 0.0
        return torch.cat([x, cls], dim=1)

    def _append_cls_pair_tokens(
        self,
        pair: Float[Tensor, "batch peaks peaks pair"],
    ) -> Float[Tensor, "batch tokens tokens pair"]:
        peak_to_cls = self.peak_to_cls_pair_token.view(1, 1, 1, -1).to(
            dtype=pair.dtype
        )
        peak_to_cls = peak_to_cls + pair[:, :, :1] * 0.0
        with_cls_column = torch.cat([pair, peak_to_cls], dim=2)
        cls_to_peak = self.cls_to_peak_pair_token.view(1, 1, 1, -1).to(
            dtype=pair.dtype
        )
        cls_to_peak = cls_to_peak + pair[:, :1] * 0.0
        cls_cls = self.cls_cls_pair_token.view(1, 1, 1, -1).to(dtype=pair.dtype)
        cls_cls = cls_cls + pair[:, :1, :1] * 0.0
        cls_row = torch.cat([cls_to_peak, cls_cls], dim=2)
        return torch.cat([with_cls_column, cls_row], dim=1)

    def _append_cls_mask(
        self,
        peak_mask: Bool[Tensor, "batch peaks"],
    ) -> Bool[Tensor, "batch tokens"]:
        cls_mask = torch.ones_like(peak_mask[:, :1])
        return torch.cat([peak_mask, cls_mask], dim=1)

    def forward_with_block_outputs(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        valid_mask: Bool[Tensor, "batch peaks"] | None = None,
        visible_mask: Bool[Tensor, "batch peaks"] | None = None,
        block_indices: list[int] | tuple[int, ...] = (),
        precursor_mz: Float[Tensor, "batch"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch tokens dim"],
        list[Float[Tensor, "batch tokens dim"]],
        Float[Tensor, "batch tokens tokens pair"],
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
        x = self._append_cls_token(x)
        z = self._append_cls_pair_tokens(z)
        token_visible_mask = self._append_cls_mask(peak_visible_mask)
        selected = set(block_indices)
        selected_peak_outputs: dict[int, Float[Tensor, "batch tokens dim"]] = {}
        for block_idx, block in enumerate(self.blocks, start=1):
            x, z = block(
                x,
                z,
                token_visible_mask,
                token_visible_mask,
            )
            if block_idx in selected and block_idx != self.num_layers:
                selected_peak_outputs[block_idx] = x
        x = self.final_norm(x)
        z = self.final_pair_norm(z)
        pair_mask = token_visible_mask.unsqueeze(2) & token_visible_mask.unsqueeze(1)
        z = z * pair_mask.unsqueeze(-1).to(dtype=z.dtype)
        if self.num_layers in selected:
            selected_peak_outputs[self.num_layers] = x
        return x, [selected_peak_outputs[idx] for idx in block_indices], z

    def forward_peak_block_outputs(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        valid_mask: Bool[Tensor, "batch peaks"] | None = None,
        visible_mask: Bool[Tensor, "batch peaks"] | None = None,
        block_indices: list[int] | tuple[int, ...] = (),
        precursor_mz: Float[Tensor, "batch"] | None = None,
    ) -> list[Float[Tensor, "batch tokens dim"]]:
        _, peak_block_outputs, _ = self.forward_with_block_outputs(
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
    ) -> Float[Tensor, "batch tokens dim"]:
        output, _, _ = self.forward_with_block_outputs(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            precursor_mz=precursor_mz,
        )
        return output
