import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spectra_learning.models.transformer import _build_norm
from spectra_learning.models.common import (
    _build_frozen_position_embedding,
    _merge_visible_mask,
)
from spectra_learning.models.induced_pair import (
    InducedPairBlock,
    InducedPairState,
    TokenInducingAssignment,
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
        pairmixer_block_type: str = "dense",
        pair_dim: int | None = None,
        induced_pair_num_inducing: int = 8,
        pair_feature_hidden_dim: int = 128,
        pairmixer_dropout: float = 0.0,
        pairmixer_use_pair_bias_attention: bool = False,
        pairmixer_mz_scale: float = 1000.0,
        pairmixer_precursor_mz_scale: float = 1000.0,
        pairmixer_use_fourier_features: bool = True,
        pairmixer_fourier_num_freqs: int = 16,
        pairmixer_fourier_x_min: float = 1e-2,
        pairmixer_fourier_x_max: float = 1000.0,
        pairmixer_relative_fourier_x_min: float = 1e-3,
        pairmixer_relative_fourier_x_max: float = 1.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_position_embedding = use_position_embedding
        self.pairmixer_block_type = pairmixer_block_type.lower()
        self.use_induced_pair = self.pairmixer_block_type == "induced"
        self.embedder = embedder
        self.position_embedding = _build_frozen_position_embedding(
            num_peaks,
            model_dim,
        )
        pair_dim = model_dim if pair_dim is None else pair_dim
        self.cls_token = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        if self.use_induced_pair:
            self.inducing_token = nn.Parameter(
                torch.empty(induced_pair_num_inducing, model_dim)
            )
            self.latent_pair_token = nn.Parameter(
                torch.empty(induced_pair_num_inducing, induced_pair_num_inducing, pair_dim)
            )
            nn.init.normal_(self.inducing_token, std=0.02)
            nn.init.normal_(self.latent_pair_token, std=0.02)
            self.initial_left_assignment = TokenInducingAssignment(model_dim)
            self.initial_right_assignment = TokenInducingAssignment(model_dim)
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
        block_cls = InducedPairBlock if self.use_induced_pair else PairMixerBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    single_dim=model_dim,
                    pair_dim=pair_dim,
                    num_heads=num_heads,
                    attention_mlp_multiple=attention_mlp_multiple,
                    norm_eps=norm_eps,
                    dropout=pairmixer_dropout,
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

    def _initial_inducing_tokens(
        self,
        x: Float[Tensor, "batch tokens dim"],
    ) -> Float[Tensor, "batch inducing dim"]:
        batch_size = x.shape[0]
        return self.inducing_token.to(dtype=x.dtype).view(
            1,
            self.inducing_token.shape[0],
            -1,
        ).expand(batch_size, -1, -1)

    def _compress_induced_pair(
        self,
        pair: Float[Tensor, "batch tokens tokens pair"],
        left_assignment: Float[Tensor, "batch tokens inducing"],
        right_assignment: Float[Tensor, "batch tokens inducing"],
        token_visible_mask: Bool[Tensor, "batch tokens"],
    ) -> Float[Tensor, "batch inducing inducing pair"]:
        pair_mask = token_visible_mask.unsqueeze(2) & token_visible_mask.unsqueeze(1)
        pair_mask_f = pair_mask.unsqueeze(-1).to(dtype=pair.dtype)
        latent_pair = torch.einsum(
            "bia,bijp,bjc->bacp",
            left_assignment,
            pair * pair_mask_f,
            right_assignment,
        )
        denom = torch.einsum(
            "bia,bij,bjc->bac",
            left_assignment,
            pair_mask.to(dtype=left_assignment.dtype),
            right_assignment,
        ).clamp_min(1e-6)
        return latent_pair / denom.unsqueeze(-1).to(dtype=latent_pair.dtype)

    def _initial_induced_pair_state(
        self,
        x: Float[Tensor, "batch tokens dim"],
        dense_pair: Float[Tensor, "batch tokens tokens pair"],
        token_visible_mask: Bool[Tensor, "batch tokens"],
    ) -> InducedPairState:
        inducing = self._initial_inducing_tokens(x)
        left_assignment = self.initial_left_assignment(x, inducing)
        right_assignment = self.initial_right_assignment(x, inducing)
        token_mask_f = token_visible_mask.unsqueeze(-1).to(dtype=left_assignment.dtype)
        left_assignment = left_assignment * token_mask_f
        right_assignment = right_assignment * token_mask_f.to(dtype=right_assignment.dtype)
        pair = self._compress_induced_pair(
            dense_pair,
            left_assignment,
            right_assignment,
            token_visible_mask,
        )
        latent_pair_token = self.latent_pair_token.to(dtype=pair.dtype).view(
            1,
            self.latent_pair_token.shape[0],
            self.latent_pair_token.shape[1],
            -1,
        ).expand(pair.shape[0], -1, -1, -1)
        pair = pair + latent_pair_token
        assignment = 0.5 * (left_assignment + right_assignment)
        return InducedPairState(inducing, pair, assignment)

    def _forward_with_induced_pair(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        valid_mask: Bool[Tensor, "batch peaks"] | None = None,
        visible_mask: Bool[Tensor, "batch peaks"] | None = None,
        precursor_mz: Float[Tensor, "batch"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch tokens dim"],
        InducedPairState,
    ]:
        peak_valid_mask = (
            torch.ones_like(peak_mz, dtype=torch.bool)
            if valid_mask is None
            else valid_mask
        )
        peak_visible_mask = _merge_visible_mask(peak_valid_mask, visible_mask)
        if peak_visible_mask is None:
            peak_visible_mask = peak_valid_mask
        x = self._add_positions(self.embedder(peak_mz, peak_intensity))
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
        state = self._initial_induced_pair_state(x, z, token_visible_mask)
        for block in self.blocks:
            x, state = block(
                x,
                state,
                token_visible_mask,
                token_visible_mask,
            )
        x = self.final_norm(x)
        pair = self.final_pair_norm(state.pair)
        assignment = state.assignment * token_visible_mask.unsqueeze(-1).to(
            dtype=state.assignment.dtype
        )
        return x, InducedPairState(state.inducing, pair, assignment)

    def forward_with_pair(
        self,
        peak_mz: Float[Tensor, "batch peaks"],
        peak_intensity: Float[Tensor, "batch peaks"],
        valid_mask: Bool[Tensor, "batch peaks"] | None = None,
        visible_mask: Bool[Tensor, "batch peaks"] | None = None,
        precursor_mz: Float[Tensor, "batch"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch tokens dim"],
        Float[Tensor, "batch tokens tokens pair"] | InducedPairState,
    ]:
        if self.use_induced_pair:
            return self._forward_with_induced_pair(
                peak_mz,
                peak_intensity,
                valid_mask=valid_mask,
                visible_mask=visible_mask,
                precursor_mz=precursor_mz,
            )
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
    ) -> Float[Tensor, "batch tokens dim"]:
        output, _ = self.forward_with_pair(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            precursor_mz=precursor_mz,
        )
        return output
