import torch
from torch import nn

from spectra_learning.models.transformer import (
    _build_norm,
    create_visible_attention_mask,
)
from spectra_learning.models.common import (
    _build_frozen_position_embedding,
    _build_non_causal_blocks,
    _masked_mean_pool,
    _merge_visible_mask,
)
from spectra_learning.models.peak_features import PeakFeatureEmbedder
from spectra_learning.models.spectral_attention_bias import SpectralGraphormerBias


class PeakSetEncoder(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0,
        feature_mlp_hidden_dim: int = 128,
        fourier_mlp_hidden_dim: int | None = None,
        fourier_mlp_num_layers: int = 2,
        fourier_strategy: str = "log_spaced",
        fourier_x_min: float = 3e-3,
        fourier_x_max: float = 1000.0,
        fourier_funcs: str = "both",
        fourier_num_freqs: int = 256,
        fourier_sigma: float = 10.0,
        fourier_trainable: bool = False,
        fourier_input_scale: float = 1000.0,
        qk_norm: bool = False,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        apply_final_norm: bool = True,
        num_peaks: int = 64,
        use_position_embedding: bool = True,
        use_cls_token: bool = True,
        num_register_tokens: int = 0,
        use_precursor_token: bool = False,
        spectral_bias_relative_kind: str = "none",
        spectral_bias_use_precursor: bool = False,
        spectral_bias_use_intensity: bool = False,
        spectral_bias_num_freqs: int = 128,
        spectral_bias_fourier_strategy: str = "log_spaced",
        spectral_bias_fourier_x_min: float = 3e-3,
        spectral_bias_fourier_x_max: float = 1000.0,
        spectral_bias_fourier_sigma: float = 10.0,
        spectral_bias_fourier_trainable: bool = False,
        spectral_bias_mass_scale: float = 1000.0,
        spectral_bias_precursor_scale: float = 1000.0,
        spectral_bias_rbf_num_basis: int = 64,
        spectral_bias_rbf_delta_min: float = -1000.0,
        spectral_bias_rbf_delta_max: float = 1000.0,
        spectral_bias_rbf_use_absolute_delta: bool = False,
        spectral_bias_intensity_hidden_dim: int = 16,
        spectral_bias_init_std: float = 0.0,
        spectral_bias_clip: float | None = None,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        norm_type = str(norm_type).lower()
        self.use_cls_token = bool(use_cls_token)
        self.num_register_tokens = int(num_register_tokens)
        self.use_precursor_token = bool(use_precursor_token)
        self.use_position_embedding = bool(use_position_embedding)
        relative_kind = str(spectral_bias_relative_kind).lower()
        spectral_bias_enabled = (
            relative_kind not in {"", "none", "false", "off"}
            or bool(spectral_bias_use_precursor)
            or bool(spectral_bias_use_intensity)
        )
        self.embedder = PeakFeatureEmbedder(
            model_dim=model_dim,
            hidden_dim=feature_mlp_hidden_dim,
            fourier_mlp_hidden_dim=fourier_mlp_hidden_dim,
            fourier_mlp_num_layers=fourier_mlp_num_layers,
            fourier_strategy=fourier_strategy,
            fourier_x_min=fourier_x_min,
            fourier_x_max=fourier_x_max,
            fourier_funcs=fourier_funcs,
            fourier_num_freqs=fourier_num_freqs,
            fourier_sigma=fourier_sigma,
            fourier_trainable=fourier_trainable,
            fourier_input_scale=fourier_input_scale,
        )
        self.position_embedding = _build_frozen_position_embedding(
            int(num_peaks),
            model_dim,
        )
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.empty(model_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.empty(self.num_register_tokens, model_dim)
            )
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None
        self.blocks = _build_non_causal_blocks(
            dim=model_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            norm_type=norm_type,
        )
        self.final_norm = (
            _build_norm(model_dim, eps=norm_eps, norm_type=norm_type, affine=False)
            if apply_final_norm
            else nn.Identity()
        )
        if spectral_bias_enabled:
            self.spectral_attn_biases = nn.ModuleList(
                [
                    SpectralGraphormerBias(
                        num_heads=int(num_heads),
                        mass_scale=float(spectral_bias_mass_scale),
                        precursor_scale=float(spectral_bias_precursor_scale),
                        first_token_is_precursor=self.use_precursor_token,
                        relative_kind=spectral_bias_relative_kind,
                        num_freqs=int(spectral_bias_num_freqs),
                        fourier_strategy=str(spectral_bias_fourier_strategy),
                        fourier_x_min=float(spectral_bias_fourier_x_min),
                        fourier_x_max=float(spectral_bias_fourier_x_max),
                        fourier_sigma=float(spectral_bias_fourier_sigma),
                        fourier_trainable=bool(spectral_bias_fourier_trainable),
                        use_precursor_bias=bool(spectral_bias_use_precursor),
                        use_intensity_bias=bool(spectral_bias_use_intensity),
                        intensity_hidden_dim=int(spectral_bias_intensity_hidden_dim),
                        rbf_num_basis=int(spectral_bias_rbf_num_basis),
                        rbf_delta_min=float(spectral_bias_rbf_delta_min),
                        rbf_delta_max=float(spectral_bias_rbf_delta_max),
                        rbf_use_absolute_delta=bool(
                            spectral_bias_rbf_use_absolute_delta
                        ),
                        init_std=float(spectral_bias_init_std),
                        bias_clip=spectral_bias_clip,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            self.spectral_attn_biases = None

    def _add_positions(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_position_embedding:
            return x
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.position_embedding(positions).to(dtype=x.dtype)

    def _append_special_tokens(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        special_tokens = []
        if self.cls_token is not None:
            cls = self.cls_token.view(1, 1, -1).expand(x.shape[0], -1, -1)
            special_tokens.append(cls.to(dtype=x.dtype))
        if self.register_tokens is not None:
            registers = self.register_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
            special_tokens.append(registers.to(dtype=x.dtype))
        if not special_tokens:
            return x, attn_mask
        special = torch.cat(special_tokens, dim=1)
        x = torch.cat([x, special], dim=1)
        if attn_mask is None:
            return x, None
        special_mask = torch.ones(
            x.shape[0],
            special.shape[1],
            device=x.device,
            dtype=torch.bool,
        )
        return x, torch.cat([attn_mask, special_mask], dim=1)

    def _spectral_attn_bias(
        self,
        block_idx: int,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        precursor_mz: torch.Tensor | None,
        *,
        num_special_tokens: int,
    ) -> torch.Tensor | None:
        if self.spectral_attn_biases is None:
            return None
        return self.spectral_attn_biases[int(block_idx)](
            peak_mz,
            peak_intensity=peak_intensity,
            precursor_mz=precursor_mz,
            num_special_tokens=int(num_special_tokens),
        )

    def split_peak_and_cls(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_cls_token:
            return x[:, :-1], x[:, -1]
        return x, x.mean(dim=1)

    def forward_with_block_outputs(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        block_indices: list[int] | tuple[int, ...] = (),
        precursor_mz: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        block_indices = tuple(int(idx) for idx in block_indices)
        attn_mask = _merge_visible_mask(valid_mask, visible_mask)
        x = self._add_positions(self.embedder(peak_mz, peak_intensity))
        seq_len = peak_mz.shape[1]
        selected = set(block_indices)
        selected_peak_outputs: dict[int, torch.Tensor] = {}
        special_len = int(self.use_cls_token) + self.num_register_tokens
        x, visible_mask = self._append_special_tokens(x, attn_mask)
        attn_mask = (
            create_visible_attention_mask(visible_mask)
            if visible_mask is not None
            else None
        )
        for block_idx, block in enumerate(self.blocks, start=1):
            attn_bias = self._spectral_attn_bias(
                block_idx - 1,
                peak_mz,
                peak_intensity,
                precursor_mz,
                num_special_tokens=special_len,
            )
            x = block(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )
            if block_idx in selected and block_idx != self.num_layers:
                selected_peak_outputs[block_idx] = x[:, :seq_len]
        x = self.final_norm(x)
        if self.num_layers in selected:
            selected_peak_outputs[self.num_layers] = x[:, :seq_len]
        peak_x = x[:, :seq_len]
        if self.use_cls_token:
            cls_x = x[:, seq_len]
            output = torch.cat([peak_x, cls_x.unsqueeze(1)], dim=1)
        else:
            output = peak_x
        return output, [selected_peak_outputs[idx] for idx in block_indices]

    def forward_peak_block_outputs(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        block_indices: list[int] | tuple[int, ...] = (),
        precursor_mz: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
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
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        return_cls_token: bool = False,
        precursor_mz: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.forward_with_block_outputs(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            precursor_mz=precursor_mz,
        )
        if return_cls_token:
            peak_x, cls_x = self.split_peak_and_cls(output)
            if not self.use_cls_token and valid_mask is not None:
                cls_x = _masked_mean_pool(peak_x, valid_mask)
            return output, cls_x
        return output
