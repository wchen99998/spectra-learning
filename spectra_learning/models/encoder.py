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


class PeakSetEncoder(nn.Module):
    cls_token: nn.Parameter | None
    register_tokens: nn.Parameter | None
    num_cls_tokens: int

    def __init__(
        self,
        *,
        model_dim: int,
        embedder: PeakFeatureEmbedder,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        apply_final_norm: bool = True,
        num_peaks: int = 64,
        use_position_embedding: bool = True,
        num_cls_tokens: int = 1,
        num_register_tokens: int = 0,
        use_precursor_token: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        norm_type = norm_type.lower()
        self.num_cls_tokens = num_cls_tokens
        self.use_cls_token = self.num_cls_tokens > 0
        self.num_register_tokens = num_register_tokens
        self.use_precursor_token = use_precursor_token
        self.use_position_embedding = use_position_embedding
        self.embedder = embedder
        self.position_embedding = _build_frozen_position_embedding(
            num_peaks,
            model_dim,
        )
        if self.use_cls_token:
            cls_token_shape = (
                (model_dim,)
                if self.num_cls_tokens == 1
                else (self.num_cls_tokens, model_dim)
            )
            self.cls_token = nn.Parameter(torch.empty(cls_token_shape))
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
            norm_type=norm_type,
        )
        self.final_norm = (
            _build_norm(model_dim, eps=norm_eps, norm_type=norm_type, affine=False)
            if apply_final_norm
            else nn.Identity()
        )

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
            if self.cls_token.ndim == 1:
                cls = self.cls_token.view(1, 1, -1).expand(x.shape[0], -1, -1)
            else:
                cls = self.cls_token.unsqueeze(0).expand(x.shape[0], -1, -1)
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

    def split_peak_and_cls(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_cls_token:
            peak_x = x[:, :-self.num_cls_tokens]
            cls_x = x[:, -self.num_cls_tokens:]
            if self.num_cls_tokens == 1:
                return peak_x, cls_x[:, 0]
            return peak_x, cls_x
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
        block_indices = tuple(idx for idx in block_indices)
        attn_mask = _merge_visible_mask(valid_mask, visible_mask)
        x = self._add_positions(self.embedder(peak_mz, peak_intensity))
        seq_len = peak_mz.shape[1]
        selected = set(block_indices)
        selected_peak_outputs: dict[int, torch.Tensor] = {}
        x, visible_mask = self._append_special_tokens(x, attn_mask)
        attn_mask = (
            create_visible_attention_mask(visible_mask)
            if visible_mask is not None
            else None
        )
        for block_idx, block in enumerate(self.blocks, start=1):
            x = block(
                x,
                attn_mask=attn_mask,
            )
            if block_idx in selected and block_idx != self.num_layers:
                selected_peak_outputs[block_idx] = x[:, :seq_len]
        x = self.final_norm(x)
        if self.num_layers in selected:
            selected_peak_outputs[self.num_layers] = x[:, :seq_len]
        peak_x = x[:, :seq_len]
        if self.use_cls_token:
            cls_x = x[:, seq_len : seq_len + self.num_cls_tokens]
            output = torch.cat([peak_x, cls_x], dim=1)
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
