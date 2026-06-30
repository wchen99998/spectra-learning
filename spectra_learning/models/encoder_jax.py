from __future__ import annotations

import jax.numpy as jnp
import torch
from flax import nnx

from spectra_learning.models.common_jax import (
    Array,
    LayerNorm,
    Linear,
    activation_checkpoint_policy,
    assign_param,
    build_frozen_position_embedding,
    merge_visible_mask,
    should_activation_checkpoint,
)
from spectra_learning.models.pairmixer_jax import (
    PairFeatureEmbedder,
    PairMixerBlock,
    _active_indices,
    _gather_pair,
    _scatter_pair,
)
from spectra_learning.models.peak_features_jax import PeakFeatureEmbedder


def _normal_token_param(rngs: nnx.Rngs, shape: tuple[int, ...]) -> nnx.Param:
    return nnx.Param(rngs.params.normal(shape, dtype=jnp.float32) * 0.02)


class PeakSetEncoder(nnx.Module):
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
        pair_feature_hidden_dim: int = 128,
        pairmixer_dropout: float = 0.0,
        pairmixer_mz_scale: float = 1000.0,
        pairmixer_precursor_mz_scale: float = 1000.0,
        pairmixer_use_fourier_features: bool = True,
        pairmixer_fourier_num_freqs: int = 16,
        pairmixer_fourier_x_min: float = 1e-2,
        pairmixer_fourier_x_max: float = 1000.0,
        pairmixer_relative_fourier_x_min: float = 1e-3,
        pairmixer_relative_fourier_x_max: float = 1.0,
        pairmixer_fast_max_visible_tokens: int | None = None,
        activation_checkpoint_mode: str = "none",
        activation_checkpoint_every_n_layers: int = 1,
        activation_checkpoint_modules: tuple[str, ...] = ("encoder", "predictor"),
        rngs: nnx.Rngs | None = None,
        compute_dtype: object = jnp.float32,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.num_layers = num_layers
        self.use_position_embedding = use_position_embedding
        self.pairmixer_block_type = pairmixer_block_type.lower()
        if self.pairmixer_block_type not in {"dense", "bi-dense", "fastmixer"}:
            raise ValueError(
                "pairmixer_block_type must be one of "
                "('dense', 'bi-dense', 'fastmixer')"
            )
        self.use_bi_dense = self.pairmixer_block_type in {"bi-dense", "fastmixer"}
        self.use_fastmixer = self.pairmixer_block_type == "fastmixer"
        self.pairmixer_fast_max_visible_tokens = pairmixer_fast_max_visible_tokens
        self.activation_checkpoint_mode = activation_checkpoint_mode.lower()
        self.activation_checkpoint_every_n_layers = activation_checkpoint_every_n_layers
        self.activation_checkpoint_modules = activation_checkpoint_modules
        self.embedder = embedder
        self.metadata_proj = Linear(
            2,
            model_dim,
            bias=False,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        self.position_embedding = build_frozen_position_embedding(num_peaks, model_dim)
        pair_dim = model_dim if pair_dim is None else pair_dim
        self.cls_token = _normal_token_param(rngs, (model_dim,))
        self.cls_to_peak_pair_token = _normal_token_param(rngs, (pair_dim,))
        self.peak_to_cls_pair_token = _normal_token_param(rngs, (pair_dim,))
        self.cls_cls_pair_token = _normal_token_param(rngs, (pair_dim,))
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
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        blocks = []
        for _ in range(num_layers):
            block = PairMixerBlock(
                single_dim=model_dim,
                pair_dim=pair_dim,
                num_heads=num_heads,
                attention_mlp_multiple=attention_mlp_multiple,
                norm_eps=norm_eps,
                dropout=pairmixer_dropout,
                use_single_to_pair_update=self.use_bi_dense,
                use_fastmixer=self.use_fastmixer,
                fastmixer_max_visible_tokens=self.pairmixer_fast_max_visible_tokens,
                compute_dtype=compute_dtype,
                rngs=rngs,
            )
            blocks.append(block)
        self.blocks = nnx.List(blocks)
        self.final_norm = (
            LayerNorm(model_dim, eps=norm_eps, affine=False)
            if apply_final_norm
            else None
        )
        self.final_pair_norm = (
            LayerNorm(pair_dim, eps=norm_eps, affine=True)
            if apply_final_pair_norm
            else None
        )

    def _add_positions(self, x: Array) -> Array:
        if not self.use_position_embedding:
            return x
        positions = jnp.arange(x.shape[1])
        return x + self.position_embedding(positions).astype(x.dtype)

    def _append_cls_token(
        self,
        x: Array,
        metadata_embedding: Array | None = None,
    ) -> Array:
        cls = jnp.broadcast_to(self.cls_token[...], (x.shape[0], 1, x.shape[-1]))
        if metadata_embedding is not None:
            cls = cls + metadata_embedding[:, None, :].astype(cls.dtype)
        return jnp.concatenate([x, cls.astype(x.dtype)], axis=1)

    def _metadata_embedding(self, spectrum_metadata: Array | None, dtype: object) -> Array | None:
        if spectrum_metadata is None:
            return None
        return self.metadata_proj(spectrum_metadata.astype(dtype))

    def _append_cls_pair_tokens(self, pair: Array) -> Array:
        batch_size, num_peaks, _, pair_dim = pair.shape
        peak_to_cls = jnp.broadcast_to(
            self.peak_to_cls_pair_token[...],
            (batch_size, num_peaks, 1, pair_dim),
        )
        with_cls_column = jnp.concatenate([pair, peak_to_cls.astype(pair.dtype)], axis=2)
        cls_to_peak = jnp.broadcast_to(
            self.cls_to_peak_pair_token[...],
            (batch_size, 1, num_peaks, pair_dim),
        )
        cls_cls = jnp.broadcast_to(
            self.cls_cls_pair_token[...],
            (batch_size, 1, 1, pair_dim),
        )
        cls_row = jnp.concatenate(
            [cls_to_peak.astype(pair.dtype), cls_cls.astype(pair.dtype)],
            axis=2,
        )
        return jnp.concatenate([with_cls_column, cls_row], axis=1)

    def _append_cls_mask(self, peak_mask: Array) -> Array:
        cls_mask = jnp.ones_like(peak_mask[:, :1])
        return jnp.concatenate([peak_mask, cls_mask], axis=1)

    def forward_with_pair(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        *,
        valid_mask: Array | None = None,
        visible_mask: Array | None = None,
        precursor_mz: Array | None = None,
        spectrum_metadata: Array | None = None,
        deterministic: bool = True,
    ) -> tuple[Array, Array]:
        peak_valid_mask = (
            jnp.ones_like(peak_mz, dtype=jnp.bool_) if valid_mask is None else valid_mask
        )
        peak_visible_mask = merge_visible_mask(peak_valid_mask, visible_mask)
        if peak_visible_mask is None:
            peak_visible_mask = peak_valid_mask
        x = self.embedder(peak_mz, peak_intensity)
        metadata_embedding = self._metadata_embedding(spectrum_metadata, x.dtype)
        if metadata_embedding is not None:
            x = x + metadata_embedding[:, None, :].astype(x.dtype)
        x = self._add_positions(x)
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
        if self.use_fastmixer:
            x, z = self._forward_fastmixer_blocks(
                x,
                z,
                token_visible_mask,
            )
        else:
            for block_idx, block in enumerate(self.blocks, start=1):
                if should_activation_checkpoint(
                    mode=self.activation_checkpoint_mode,
                    modules=self.activation_checkpoint_modules,
                    module="encoder",
                    block_idx=block_idx,
                    every_n=self.activation_checkpoint_every_n_layers,
                ):
                    x, z = nnx.remat(
                        _call_pair_mixer_block,
                        policy=activation_checkpoint_policy(
                            self.activation_checkpoint_mode
                        ),
                    )(block, x, z, token_visible_mask, token_visible_mask)
                else:
                    x, z = block(
                        x,
                        z,
                        token_visible_mask,
                        token_visible_mask,
                        deterministic=deterministic,
                    )
        if self.final_norm is not None:
            x = self.final_norm(x)
        if self.final_pair_norm is not None and not self.use_fastmixer:
            z = self.final_pair_norm(z)
        pair_visible_mask = token_visible_mask[:, :, None] & token_visible_mask[:, None, :]
        z = z * pair_visible_mask[..., None].astype(z.dtype)
        return x, z

    def _forward_fastmixer_blocks(
        self,
        x: Array,
        z: Array,
        token_visible_mask: Array,
    ) -> tuple[Array, Array]:
        idx, compact_token_mask = _active_indices(
            token_visible_mask,
            self.pairmixer_fast_max_visible_tokens,
        )
        dense_pair_shape = z.shape
        z = _gather_pair(z, idx)
        for block_idx, block in enumerate(self.blocks, start=1):
            if should_activation_checkpoint(
                mode=self.activation_checkpoint_mode,
                modules=self.activation_checkpoint_modules,
                module="encoder",
                block_idx=block_idx,
                every_n=self.activation_checkpoint_every_n_layers,
            ):
                x, z = nnx.remat(
                    _call_fast_pair_mixer_block,
                    policy=activation_checkpoint_policy(
                        self.activation_checkpoint_mode
                    ),
                )(block, x, z, idx, compact_token_mask, token_visible_mask)
            else:
                x, z = block.fastmixer_compact_call(
                    x,
                    z,
                    idx,
                    compact_token_mask,
                    token_visible_mask,
                )
        if self.final_pair_norm is not None:
            z = self.final_pair_norm(z)
        return x, _scatter_pair(z, idx, dense_pair_shape)

    def __call__(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        *,
        valid_mask: Array | None = None,
        visible_mask: Array | None = None,
        precursor_mz: Array | None = None,
        spectrum_metadata: Array | None = None,
    ) -> Array:
        output, _ = self.forward_with_pair(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        return output

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        assign_param(self.cls_token, state_dict[f"{prefix}.cls_token"])
        self.embedder.load_torch_state_dict(state_dict, f"{prefix}.embedder")
        self.metadata_proj.load_torch_state_dict(state_dict, f"{prefix}.metadata_proj")
        self.position_embedding.load_torch_state_dict(
            state_dict,
            f"{prefix}.position_embedding",
        )
        assign_param(
            self.cls_to_peak_pair_token,
            state_dict[f"{prefix}.cls_to_peak_pair_token"],
        )
        assign_param(
            self.peak_to_cls_pair_token,
            state_dict[f"{prefix}.peak_to_cls_pair_token"],
        )
        assign_param(
            self.cls_cls_pair_token,
            state_dict[f"{prefix}.cls_cls_pair_token"],
        )
        self.pair_embedder.load_torch_state_dict(
            state_dict,
            f"{prefix}.pair_embedder",
        )
        for idx, block in enumerate(self.blocks):
            block.load_torch_state_dict(state_dict, f"{prefix}.blocks.{idx}")
        if self.final_norm is not None:
            self.final_norm.load_torch_state_dict(state_dict, f"{prefix}.final_norm")
        if self.final_pair_norm is not None:
            self.final_pair_norm.load_torch_state_dict(
                state_dict,
                f"{prefix}.final_pair_norm",
            )


def _call_pair_mixer_block(
    block: PairMixerBlock,
    single: Array,
    pair: Array,
    peak_mask: Array,
    token_mask: Array,
) -> tuple[Array, Array]:
    return block(single, pair, peak_mask, token_mask, deterministic=True)


def _call_fast_pair_mixer_block(
    block: PairMixerBlock,
    single: Array,
    pair: Array,
    idx: Array,
    compact_token_mask: Array,
    token_mask: Array,
) -> tuple[Array, Array]:
    return block.fastmixer_compact_call(
        single,
        pair,
        idx,
        compact_token_mask,
        token_mask,
    )
