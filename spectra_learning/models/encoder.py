import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spectra_learning.data.spectra import DEFAULT_NUM_PEAKS, PEAK_MZ_MAX
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
        num_peaks: int = DEFAULT_NUM_PEAKS,
        use_position_embedding: bool = True,
        pairmixer_block_type: str = "dense",
        pairmixer_transition_type: str = "swiglu",
        pair_dim: int | None = None,
        pair_feature_hidden_dim: int = 128,
        pairmixer_dropout: float = 0.0,
        pairmixer_mz_scale: float = PEAK_MZ_MAX,
        pairmixer_precursor_mz_scale: float = PEAK_MZ_MAX,
        pairmixer_use_fourier_features: bool = True,
        pairmixer_fourier_num_freqs: int = 16,
        pairmixer_fourier_x_min: float = 1e-2,
        pairmixer_fourier_x_max: float = PEAK_MZ_MAX,
        pairmixer_relative_fourier_x_min: float = 1e-3,
        pairmixer_relative_fourier_x_max: float = 1.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_position_embedding = use_position_embedding
        self.pairmixer_block_type = pairmixer_block_type.lower()
        if self.pairmixer_block_type not in {
            "dense",
            "bi-dense",
            "fastmixer",
            "fastmixer-dense",
        }:
            raise ValueError(
                "pairmixer_block_type must be one of "
                "('dense', 'bi-dense', 'fastmixer', 'fastmixer-dense')"
            )
        self.use_bi_dense = self.pairmixer_block_type in {"bi-dense", "fastmixer"}
        self.embedder = embedder
        self.metadata_proj = nn.Linear(2, model_dim, bias=False)
        nn.init.xavier_normal_(self.metadata_proj.weight)
        self.position_embedding = _build_frozen_position_embedding(
            num_peaks,
            model_dim,
        )
        pair_dim = model_dim if pair_dim is None else pair_dim
        self.cls_token = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.cls_to_peak_pair_token = nn.Parameter(torch.empty(pair_dim))
        self.peak_to_cls_pair_token = nn.Parameter(torch.empty(pair_dim))
        self.cls_cls_pair_token = nn.Parameter(torch.empty(pair_dim))
        nn.init.normal_(self.cls_to_peak_pair_token, std=0.02)
        nn.init.normal_(self.peak_to_cls_pair_token, std=0.02)
        nn.init.normal_(self.cls_cls_pair_token, std=0.02)
        self.pair_embedder = PairFeatureEmbedder(
            single_dim=model_dim,
            pair_dim=pair_dim,
            hidden_dim=pair_feature_hidden_dim,
            mz_scale=pairmixer_mz_scale,
            precursor_mz_scale=pairmixer_precursor_mz_scale,
            use_fourier_features=pairmixer_use_fourier_features,
            fourier_num_freqs=pairmixer_fourier_num_freqs,
            fourier_x_min=pairmixer_fourier_x_min,
            fourier_x_max=pairmixer_fourier_x_max,
            relative_fourier_x_min=pairmixer_relative_fourier_x_min,
            relative_fourier_x_max=pairmixer_relative_fourier_x_max,
        )
        blocks = []
        for _ in range(self.num_layers):
            block = PairMixerBlock(
                single_dim=model_dim,
                pair_dim=pair_dim,
                num_heads=num_heads,
                attention_mlp_multiple=attention_mlp_multiple,
                norm_eps=norm_eps,
                dropout=pairmixer_dropout,
                use_single_to_pair_update=self.use_bi_dense,
                transition_type=pairmixer_transition_type,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
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
        metadata_embedding: Float[Tensor, "batch dim"] | None = None,
    ) -> Float[Tensor, "batch tokens dim"]:
        cls = self.cls_token.view(1, 1, -1).expand(x.shape[0], 1, -1)
        cls = cls.to(dtype=x.dtype) + x[:, :1] * 0.0
        if metadata_embedding is not None:
            cls = cls + metadata_embedding.unsqueeze(1).to(dtype=x.dtype)
        return torch.cat([x, cls], dim=1)

    def _metadata_embedding(
        self,
        spectrum_metadata: Float[Tensor, "batch metadata"] | None,
        dtype: torch.dtype,
    ) -> Float[Tensor, "batch dim"] | None:
        if spectrum_metadata is None:
            return None
        return self.metadata_proj(spectrum_metadata.to(dtype=dtype))

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

    def forward_with_pair(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        valid_mask: Bool[Tensor, "batch peaks"] | None = None,
        visible_mask: Bool[Tensor, "batch peaks"] | None = None,
        precursor_mz: Float[Tensor, "batch"] | None = None,
        spectrum_metadata: Float[Tensor, "batch metadata"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch tokens dim"],
        Float[Tensor, "batch tokens tokens pair"],
    ]:
        peak_valid_mask = (
            torch.ones_like(peak_mz, dtype=torch.bool)
            if valid_mask is None
            else valid_mask
        )
        peak_visible_mask = _merge_visible_mask(peak_valid_mask, visible_mask)
        if peak_visible_mask is None:
            peak_visible_mask = peak_valid_mask
        x = self.embedder(peak_mz, peak_intensity)
        metadata_embedding = self._metadata_embedding(spectrum_metadata, x.dtype)
        if metadata_embedding is not None:
            x = x + metadata_embedding.unsqueeze(1).to(dtype=x.dtype)
        x = self._add_positions(x)
        # x: [B, N, D], z: [B, N, N, P], masks: [B, N]
        z = self.pair_embedder(
            peak_mz,
            peak_intensity,
            x,
            peak_visible_mask,
            precursor_mz=precursor_mz,
        )
        x = self._append_cls_token(x, metadata_embedding)
        z = self._append_cls_pair_tokens(z)
        token_visible_mask = self._append_cls_mask(peak_visible_mask)
        for block in self.blocks:
            x, z = block(
                x,
                z,
                token_visible_mask,
                token_visible_mask,
            )
        x = self.final_norm(x)
        z = self.final_pair_norm(z)
        pair_mask = token_visible_mask.unsqueeze(2) & token_visible_mask.unsqueeze(1)
        z = z * pair_mask.unsqueeze(-1).to(dtype=z.dtype)
        return x, z

    def forward(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        valid_mask: Bool[Tensor, "batch peaks"] | None = None,
        visible_mask: Bool[Tensor, "batch peaks"] | None = None,
        precursor_mz: Float[Tensor, "batch"] | None = None,
        spectrum_metadata: Float[Tensor, "batch metadata"] | None = None,
    ) -> Float[Tensor, "batch tokens dim"]:
        output, _ = self.forward_with_pair(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        return output
