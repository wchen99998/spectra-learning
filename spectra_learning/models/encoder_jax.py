from __future__ import annotations

import math
from typing import cast

import jax
import jax.numpy as jnp
import torch
from flax import nnx

from spectra_learning.data.spectra import DEFAULT_NUM_PEAKS, PEAK_MZ_MAX
from spectra_learning.models.common_jax import (
    Array,
    Linear,
    RMSNorm,
    activation_checkpoint,
    assign_param,
    merge_visible_mask,
    should_activation_checkpoint,
)
from spectra_learning.models.pairmixer_jax import (
    PairFeatureEmbedder,
    PairMixerBlock,
    PairOnlyBlock,
    run_pair_only_blocks,
    SingleMixerBlock,
    _active_indices,
    _apply_pair_rope,
    _gather_single,
    _gather_pair,
    _scatter_pair,
)
from spectra_learning.models.peak_features_jax import PeakFeatureEmbedder
from spectra_learning.models.precursor_edges_jax import PrecursorEdgeEmbedder
from spectra_learning.models.spectrum_metadata import MASSIVE_V2_CONDITION_DIM
from spectra_learning.models.settings import PeakSetJEPASettings


def _normal_token_param(rngs: nnx.Rngs, shape: tuple[int, ...]) -> nnx.Param:
    return nnx.Param(rngs.params.normal(shape, dtype=jnp.float32) * 0.02)


def _append_cls_pairs(
    pair: Array, peak_to_cls_token: Array, cls_to_peak_token: Array,
    cls_cls_token: Array,
) -> Array:
    batch_size, num_peaks, _, pair_dim = pair.shape
    peak_to_cls = jnp.broadcast_to(
        peak_to_cls_token,
        (batch_size, num_peaks, 1, pair_dim),
    )
    with_cls_column = jnp.concatenate([pair, peak_to_cls.astype(pair.dtype)], axis=2)
    cls_to_peak = jnp.broadcast_to(
        cls_to_peak_token,
        (batch_size, 1, num_peaks, pair_dim),
    )
    cls_cls = jnp.broadcast_to(
        cls_cls_token,
        (batch_size, 1, 1, pair_dim),
    )
    cls_row = jnp.concatenate(
        [cls_to_peak.astype(pair.dtype), cls_cls.astype(pair.dtype)],
        axis=2,
    )
    return jnp.concatenate([with_cls_column, cls_row], axis=1)


class MetadataEmbedder(nnx.Module):
    def __init__(
        self,
        condition_dim: int,
        *,
        compute_dtype: object,
        rngs: nnx.Rngs,
    ) -> None:
        self.linear0 = Linear(
            MASSIVE_V2_CONDITION_DIM,
            condition_dim,
            compute_dtype=compute_dtype,
            init="xavier_normal",
            rngs=rngs,
        )
        self.linear2 = Linear(
            condition_dim,
            condition_dim,
            compute_dtype=compute_dtype,
            init="xavier_normal",
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        return self.linear2(jax.nn.silu(self.linear0(x)))

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        self.linear0.load_torch_state_dict(state_dict, f"{prefix}.0")
        self.linear2.load_torch_state_dict(state_dict, f"{prefix}.2")


class PeakSetEncoder(nnx.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        embedder: PeakFeatureEmbedder,
        num_layers: int,
        num_heads: int,
        attention_mlp_multiple: float = 8 / 3,
        norm_eps: float = 1e-5,
        apply_final_norm: bool = True,
        apply_final_pair_norm: bool = False,
        num_peaks: int = DEFAULT_NUM_PEAKS,
        use_cls_token: bool = True,
        use_position_embedding: bool = True,
        use_pair_path: bool = True,
        pairmixer_block_type: str = "dense",
        pairmixer_transition_type: str = "swiglu",
        pair_dim: int | None = None,
        pair_feature_hidden_dim: int = 128,
        pairmixer_dropout: float = 0.0,
        pairmixer_use_pair_bias: bool = True,
        pairmixer_mz_scale: float = PEAK_MZ_MAX,
        pairmixer_precursor_mz_scale: float = PEAK_MZ_MAX,
        pairmixer_mz_embedding: str = "fourier",
        pairmixer_mz_token_bin_size: float = 0.1,
        pairmixer_mz_token_embedding_dim: int = 128,
        pairmixer_fourier_num_freqs: int = 16,
        pairmixer_fourier_x_min: float = 1e-2,
        pairmixer_fourier_x_max: float = PEAK_MZ_MAX,
        pairmixer_relative_fourier_x_min: float = 1e-3,
        pairmixer_relative_fourier_x_max: float = 1.0,
        pairmixer_fast_max_visible_tokens: int | None = None,
        activation_checkpoint_mode: str = "none",
        activation_checkpoint_every_n_layers: int = 1,
        activation_checkpoint_modules: tuple[str, ...] = ("encoder", "predictor"),
        metadata_schema: str | None = None,
        metadata_conditioning: str = "additive",
        metadata_condition_dim: int = 256,
        mz_quantization_bin_size: float | None = None,
        pair_to_single_rank: int | None = None,
        rngs: nnx.Rngs | None = None,
        compute_dtype: object = jnp.float32,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        if metadata_conditioning == "precursor_edge" and not (
            use_pair_path and use_cls_token
        ):
            raise ValueError("precursor_edge metadata requires the pair path and CLS token")
        self.num_layers = num_layers
        self.mz_quantization_bin_size = mz_quantization_bin_size
        self.use_cls_token = use_cls_token
        self.use_pair_path = use_pair_path
        self.metadata_schema = metadata_schema
        self.metadata_conditioning = metadata_conditioning
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
        self.use_fastmixer = self.pairmixer_block_type in {
            "fastmixer",
            "fastmixer-dense",
        }
        self.pairmixer_fast_max_visible_tokens = pairmixer_fast_max_visible_tokens
        self.activation_checkpoint_mode = activation_checkpoint_mode.lower()
        self.activation_checkpoint_every_n_layers = activation_checkpoint_every_n_layers
        self.activation_checkpoint_modules = activation_checkpoint_modules
        self.embedder = embedder
        if self.metadata_conditioning in {"adaln_zero", "precursor_edge"}:
            self.metadata_embedder = MetadataEmbedder(
                metadata_condition_dim,
                compute_dtype=compute_dtype,
                rngs=rngs,
            )
            self.metadata_proj = None
        else:
            self.metadata_proj = Linear(
                2,
                model_dim,
                bias=False,
                compute_dtype=compute_dtype,
                rngs=rngs,
            )
            self.metadata_embedder = None
        pair_dim = model_dim if pair_dim is None else pair_dim
        self.cls_token = (
            _normal_token_param(rngs, (model_dim,)) if self.use_cls_token else None
        )
        self.cls_to_peak_pair_token = (
            _normal_token_param(rngs, (pair_dim,))
            if self.use_pair_path and self.use_cls_token
            else None
        )
        self.peak_to_cls_pair_token = (
            _normal_token_param(rngs, (pair_dim,))
            if self.use_pair_path and self.use_cls_token
            else None
        )
        self.cls_cls_pair_token = (
            _normal_token_param(rngs, (pair_dim,))
            if self.use_pair_path and self.use_cls_token
            else None
        )
        self.pair_embedder = (
            PairFeatureEmbedder(
                single_dim=model_dim,
                pair_dim=pair_dim,
                hidden_dim=pair_feature_hidden_dim,
                mz_scale=pairmixer_mz_scale,
                precursor_mz_scale=pairmixer_precursor_mz_scale,
                mz_embedding=pairmixer_mz_embedding,
                token_bin_size=pairmixer_mz_token_bin_size,
                token_embedding_dim=pairmixer_mz_token_embedding_dim,
                fourier_num_freqs=pairmixer_fourier_num_freqs,
                fourier_x_min=pairmixer_fourier_x_min,
                fourier_x_max=pairmixer_fourier_x_max,
                relative_fourier_x_min=pairmixer_relative_fourier_x_min,
                relative_fourier_x_max=pairmixer_relative_fourier_x_max,
                compute_dtype=compute_dtype,
                rngs=rngs,
            )
            if self.use_pair_path
            else None
        )
        blocks = []
        for _ in range(num_layers):
            block = (
                PairMixerBlock(
                    single_dim=model_dim,
                    pair_dim=pair_dim,
                    num_heads=num_heads,
                    attention_mlp_multiple=attention_mlp_multiple,
                    norm_eps=norm_eps,
                    dropout=pairmixer_dropout,
                    use_single_to_pair_update=self.use_bi_dense,
                    use_pair_bias=pairmixer_use_pair_bias,
                    use_rope=use_position_embedding,
                    use_fastmixer=self.use_fastmixer,
                    fastmixer_max_visible_tokens=self.pairmixer_fast_max_visible_tokens,
                    transition_type=pairmixer_transition_type,
                    pair_to_single_rank=pair_to_single_rank,
                    compute_dtype=compute_dtype,
                    rngs=rngs,
                )
                if self.use_pair_path
                else SingleMixerBlock(
                    single_dim=model_dim,
                    num_heads=num_heads,
                    attention_mlp_multiple=attention_mlp_multiple,
                    norm_eps=norm_eps,
                    use_rope=use_position_embedding,
                    transition_type=pairmixer_transition_type,
                    condition_dim=(
                        metadata_condition_dim
                        if self.metadata_conditioning == "adaln_zero"
                        else None
                    ),
                    compute_dtype=compute_dtype,
                    rngs=rngs,
                )
            )
            blocks.append(block)
        self.blocks = nnx.List(blocks)
        self.final_norm = (
            RMSNorm(model_dim, eps=norm_eps, affine=False)
            if apply_final_norm
            else None
        )
        self.final_adaLN_modulation = (
            Linear(
                metadata_condition_dim,
                2 * model_dim,
                compute_dtype=compute_dtype,
                init="zeros",
                rngs=rngs,
            )
            if apply_final_norm and self.metadata_conditioning == "adaln_zero"
            else None
        )
        self.final_pair_norm = (
            RMSNorm(pair_dim, eps=norm_eps, affine=True)
            if self.use_pair_path and apply_final_pair_norm
            else None
        )
        self.precursor_edges = (
            PrecursorEdgeEmbedder(
                pair_dim,
                metadata_condition_dim,
                precursor_mz_scale=pairmixer_precursor_mz_scale,
                compute_dtype=compute_dtype,
                rngs=rngs,
            )
            if self.metadata_conditioning == "precursor_edge"
            else None
        )

    def _append_cls_token(
        self,
        x: Array,
        metadata_embedding: Array | None = None,
    ) -> Array:
        if not self.use_cls_token:
            return x
        cls = jnp.broadcast_to(self.cls_token[...], (x.shape[0], 1, x.shape[-1]))
        if metadata_embedding is not None:
            cls = cls + metadata_embedding[:, None, :].astype(cls.dtype)
        return jnp.concatenate([x, cls.astype(x.dtype)], axis=1)

    def _metadata_embedding(self, spectrum_metadata: Array | None, dtype: object) -> Array | None:
        # Precursor-edge metadata enters through pairs only (_embed_pairs).
        if spectrum_metadata is None or self.precursor_edges is not None:
            return None
        if self.metadata_conditioning == "adaln_zero":
            return self.metadata_embedder(spectrum_metadata.astype(dtype))
        return self.metadata_proj(spectrum_metadata.astype(dtype))

    def _finalize(self, x: Array, condition: Array | None) -> Array:
        if self.final_norm is not None:
            x = self.final_norm(x)
        if self.final_adaLN_modulation is not None:
            shift, scale = jnp.split(
                self.final_adaLN_modulation(condition),
                2,
                axis=-1,
            )
            x = x * (1 + scale[:, None]) + shift[:, None]
        return x

    def _embed_peaks(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        valid_mask: Array | None,
        visible_mask: Array | None,
        spectrum_metadata: Array | None,
    ) -> tuple[Array, Array, Array | None]:
        peak_valid_mask = (
            jnp.ones_like(peak_mz, dtype=jnp.bool_) if valid_mask is None else valid_mask
        )
        peak_visible_mask = merge_visible_mask(peak_valid_mask, visible_mask)
        if peak_visible_mask is None:
            peak_visible_mask = peak_valid_mask
        x = self.embedder(self._quantize_mz(peak_mz), peak_intensity)
        metadata_embedding = self._metadata_embedding(spectrum_metadata, x.dtype)
        if metadata_embedding is not None and self.metadata_conditioning != "adaln_zero":
            x = x + metadata_embedding[:, None, :].astype(x.dtype)
        return x, peak_visible_mask, metadata_embedding

    def _quantize_mz(self, peak_mz: Array) -> Array:
        """Use the same categorical state for denoising and clean embeddings."""
        if self.mz_quantization_bin_size is None:
            return peak_mz
        width = self.mz_quantization_bin_size
        bins = jnp.clip(
            jnp.floor(peak_mz.astype(jnp.float32) * PEAK_MZ_MAX / width),
            0,
            math.ceil(PEAK_MZ_MAX / width) - 1,
        )
        return (bins + 0.5) * (width / PEAK_MZ_MAX)

    def _append_cls_pair_tokens(self, pair: Array) -> Array:
        if not self.use_cls_token:
            return pair
        return _append_cls_pairs(
            pair, self.peak_to_cls_pair_token[...],
            self.cls_to_peak_pair_token[...], self.cls_cls_pair_token[...],
        )

    def _embed_pairs(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        x: Array,
        peak_visible_mask: Array,
        precursor_mz: Array | None,
        spectrum_metadata: Array | None,
    ) -> Array:
        z = cast(PairFeatureEmbedder, self.pair_embedder)(
            self._quantize_mz(peak_mz),
            peak_intensity,
            x,
            peak_visible_mask,
            precursor_mz=precursor_mz,
        )
        z = self._append_cls_pair_tokens(z)
        if self.precursor_edges is not None and spectrum_metadata is not None:
            condition = self.metadata_embedder(spectrum_metadata.astype(x.dtype))
            z = self.precursor_edges(
                z, peak_mz, spectrum_metadata, condition, peak_visible_mask
            )
        return z

    def _append_cls_mask(self, peak_mask: Array) -> Array:
        if not self.use_cls_token:
            return peak_mask
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
        assert self.use_pair_path
        x, peak_visible_mask, metadata_embedding = self._embed_peaks(
            peak_mz,
            peak_intensity,
            valid_mask,
            visible_mask,
            spectrum_metadata,
        )
        z = self._embed_pairs(
            peak_mz,
            peak_intensity,
            x,
            peak_visible_mask,
            precursor_mz,
            spectrum_metadata,
        )
        x = self._append_cls_token(
            x,
            metadata_embedding if self.metadata_conditioning != "adaln_zero" else None,
        )
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
                    x, z = activation_checkpoint(
                        _call_pair_mixer_block, self.activation_checkpoint_mode
                    )(block, x, z, token_visible_mask, token_visible_mask)
                else:
                    x, z = cast(PairMixerBlock, block)(
                        x,
                        z,
                        token_visible_mask,
                        token_visible_mask,
                        deterministic=deterministic,
                    )
        x = self._finalize(x, metadata_embedding)
        if self.final_pair_norm is not None and not self.use_fastmixer:
            z = self.final_pair_norm(z)
        pair_visible_mask = token_visible_mask[:, :, None] & token_visible_mask[:, None, :]
        z = z * pair_visible_mask[..., None].astype(z.dtype)
        return x, z

    def forward_with_pair_compact(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        *,
        valid_mask: Array | None = None,
        visible_mask: Array | None = None,
        max_visible_tokens: int | None = None,
        precursor_mz: Array | None = None,
        spectrum_metadata: Array | None = None,
    ) -> tuple[Array, Array, Array, Array]:
        assert self.use_pair_path
        x, peak_visible_mask, metadata_embedding = self._embed_peaks(
            peak_mz,
            peak_intensity,
            valid_mask,
            visible_mask,
            spectrum_metadata,
        )
        z = self._embed_pairs(
            peak_mz,
            peak_intensity,
            x,
            peak_visible_mask,
            precursor_mz,
            spectrum_metadata,
        )
        x = self._append_cls_token(
            x,
            metadata_embedding if self.metadata_conditioning != "adaln_zero" else None,
        )
        token_visible_mask = self._append_cls_mask(peak_visible_mask)
        x, z, idx, compact_token_mask = self._forward_fastmixer_blocks_compact(
            x,
            z,
            token_visible_mask,
            max_visible_tokens=max_visible_tokens,
        )
        x = self._finalize(x, metadata_embedding)
        x = x * compact_token_mask[..., None].astype(x.dtype)
        if self.final_pair_norm is not None:
            z = self.final_pair_norm(z)
        pair_visible_mask = compact_token_mask[:, :, None] & compact_token_mask[:, None, :]
        z = z * pair_visible_mask[..., None].astype(z.dtype)
        return x, z, idx, compact_token_mask

    def forward_compact(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        *,
        valid_mask: Array | None = None,
        visible_mask: Array | None = None,
        max_visible_tokens: int | None = None,
        precursor_mz: Array | None = None,
        spectrum_metadata: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        if self.use_pair_path:
            x, _, idx, compact_token_mask = self.forward_with_pair_compact(
                peak_mz,
                peak_intensity,
                valid_mask=valid_mask,
                visible_mask=visible_mask,
                max_visible_tokens=max_visible_tokens,
                precursor_mz=precursor_mz,
                spectrum_metadata=spectrum_metadata,
            )
            return x, idx, compact_token_mask
        x, peak_visible_mask, metadata_embedding = self._embed_peaks(
            peak_mz,
            peak_intensity,
            valid_mask,
            visible_mask,
            spectrum_metadata,
        )
        x = self._append_cls_token(
            x,
            metadata_embedding if self.metadata_conditioning != "adaln_zero" else None,
        )
        token_visible_mask = self._append_cls_mask(peak_visible_mask)
        compact_tokens = (
            self.pairmixer_fast_max_visible_tokens
            if max_visible_tokens is None
            else max_visible_tokens
        )
        idx, compact_token_mask = _active_indices(
            token_visible_mask,
            compact_tokens,
        )
        x = _gather_single(x, idx)
        x = x * compact_token_mask[..., None].astype(x.dtype)
        x = self._forward_single_blocks(
            x,
            compact_token_mask,
            idx,
            metadata_embedding,
        )
        x = self._finalize(x, metadata_embedding)
        return (
            x * compact_token_mask[..., None].astype(x.dtype),
            idx,
            compact_token_mask,
        )

    def _forward_single_blocks(
        self,
        x: Array,
        token_visible_mask: Array,
        token_positions: Array | None = None,
        condition: Array | None = None,
        num_blocks: int | None = None,
    ) -> Array:
        blocks = self.blocks if num_blocks is None else self.blocks[:num_blocks]
        for block_idx, block in enumerate(blocks, start=1):
            if should_activation_checkpoint(
                mode=self.activation_checkpoint_mode,
                modules=self.activation_checkpoint_modules,
                module="encoder",
                block_idx=block_idx,
                every_n=self.activation_checkpoint_every_n_layers,
            ):
                x = activation_checkpoint(
                    _call_single_mixer_block, self.activation_checkpoint_mode
                )(block, x, token_visible_mask, token_positions, condition)
            else:
                x = cast(SingleMixerBlock, block)(
                    x,
                    token_visible_mask,
                    token_positions,
                    condition,
                )
        return x

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
                x, z = activation_checkpoint(
                    _call_fast_pair_mixer_block, self.activation_checkpoint_mode
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

    def _forward_fastmixer_blocks_compact(
        self,
        x: Array,
        z: Array,
        token_visible_mask: Array,
        *,
        max_visible_tokens: int | None = None,
    ) -> tuple[Array, Array, Array, Array]:
        fast_max_visible_tokens = (
            self.pairmixer_fast_max_visible_tokens
            if max_visible_tokens is None
            else max_visible_tokens
        )
        idx, compact_token_mask = _active_indices(
            token_visible_mask,
            fast_max_visible_tokens,
        )
        x = _gather_single(x, idx)
        x = x * compact_token_mask[..., None].astype(x.dtype)
        z = _gather_pair(z, idx)
        for block_idx, block in enumerate(self.blocks, start=1):
            if should_activation_checkpoint(
                mode=self.activation_checkpoint_mode,
                modules=self.activation_checkpoint_modules,
                module="encoder",
                block_idx=block_idx,
                every_n=self.activation_checkpoint_every_n_layers,
            ):
                x, z = activation_checkpoint(
                    _call_fast_pair_mixer_block_compact_only,
                    self.activation_checkpoint_mode,
                )(block, x, z, compact_token_mask, idx)
            else:
                x, z = block.fastmixer_compact_only_call(
                    x,
                    z,
                    compact_token_mask,
                    idx,
                )
        return x, z, idx, compact_token_mask

    def __call__(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        *,
        valid_mask: Array | None = None,
        visible_mask: Array | None = None,
        precursor_mz: Array | None = None,
        spectrum_metadata: Array | None = None,
        num_blocks: int | None = None,
    ) -> Array:
        """Encodes all tokens; with ``num_blocks`` returns that block's raw
        residual stream instead of the final normed output."""
        if self.use_pair_path:
            if num_blocks is not None:
                raise ValueError("num_blocks is only supported without the pair path")
            output, _ = self.forward_with_pair(
                peak_mz,
                peak_intensity,
                valid_mask=valid_mask,
                visible_mask=visible_mask,
                precursor_mz=precursor_mz,
                spectrum_metadata=spectrum_metadata,
            )
            return output
        output, peak_visible_mask, metadata_embedding = self._embed_peaks(
            peak_mz,
            peak_intensity,
            valid_mask,
            visible_mask,
            spectrum_metadata,
        )
        output = self._append_cls_token(
            output,
            metadata_embedding if self.metadata_conditioning != "adaln_zero" else None,
        )
        token_visible_mask = self._append_cls_mask(peak_visible_mask)
        output = self._forward_single_blocks(
            output,
            token_visible_mask,
            condition=metadata_embedding,
            num_blocks=num_blocks,
        )
        if num_blocks is not None:
            return output
        return self._finalize(output, metadata_embedding)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        if self.use_cls_token:
            assign_param(self.cls_token, state_dict[f"{prefix}.cls_token"])
        self.embedder.load_torch_state_dict(state_dict, f"{prefix}.embedder")
        if self.metadata_conditioning == "adaln_zero":
            self.metadata_embedder.load_torch_state_dict(
                state_dict,
                f"{prefix}.metadata_embedder",
            )
        else:
            self.metadata_proj.load_torch_state_dict(state_dict, f"{prefix}.metadata_proj")
        if self.use_pair_path and self.use_cls_token:
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
        if self.use_pair_path:
            cast(PairFeatureEmbedder, self.pair_embedder).load_torch_state_dict(
                state_dict,
                f"{prefix}.pair_embedder",
            )
        for idx, block in enumerate(self.blocks):
            block.load_torch_state_dict(state_dict, f"{prefix}.blocks.{idx}")
        if self.final_norm is not None:
            self.final_norm.load_torch_state_dict(state_dict, f"{prefix}.final_norm")
        if self.final_adaLN_modulation is not None:
            self.final_adaLN_modulation.load_torch_state_dict(
                state_dict,
                f"{prefix}.final_adaLN_modulation",
            )
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


def _call_single_mixer_block(
    block: SingleMixerBlock,
    single: Array,
    token_mask: Array,
    token_positions: Array | None,
    condition: Array | None,
) -> Array:
    return block(single, token_mask, token_positions, condition)


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


def _call_fast_pair_mixer_block_compact_only(
    block: PairMixerBlock,
    single: Array,
    pair: Array,
    compact_token_mask: Array,
    token_positions: Array,
) -> tuple[Array, Array]:
    return block.fastmixer_compact_only_call(
        single,
        pair,
        compact_token_mask,
        token_positions,
    )


class PairOnlyEncoder(nnx.Module):
    """Embed compact peak pairs directly and update only the pair representation."""

    def __init__(
        self, cfg: PeakSetJEPASettings, *, compute_dtype: object, rngs: nnx.Rngs,
    ) -> None:
        pair_dim = cfg.model_dim if cfg.pairmixer_pair_dim is None else cfg.pairmixer_pair_dim
        if pair_dim % 4:
            raise ValueError("pair-only encoder requires pair width divisible by 4 for 2D RoPE")
        if cfg.pairmixer_triangle_attention_every_n_layers < 0:
            raise ValueError("pairmixer_triangle_attention_every_n_layers must be nonnegative")
        self.use_cls_token = cfg.encoder_use_cls_token
        self.peak_to_cls_pair_token = (
            _normal_token_param(rngs, (pair_dim,)) if self.use_cls_token else None
        )
        self.cls_to_peak_pair_token = (
            _normal_token_param(rngs, (pair_dim,)) if self.use_cls_token else None
        )
        self.cls_cls_pair_token = (
            _normal_token_param(rngs, (pair_dim,)) if self.use_cls_token else None
        )
        self.use_pair_path = True
        self.pairmixer_fast_max_visible_tokens = cfg.pairmixer_fast_encoder_max_visible_tokens
        self.activation_checkpoint_mode = cfg.activation_checkpoint_mode
        self.activation_checkpoint_modules = cfg.activation_checkpoint_modules
        self.activation_checkpoint_every_n_layers = cfg.activation_checkpoint_every_n_layers
        self.pair_embedder = PairFeatureEmbedder(
            single_dim=None,
            pair_dim=pair_dim,
            hidden_dim=cfg.pairmixer_pair_feature_hidden_dim,
            mz_scale=cfg.pairmixer_mz_scale,
            precursor_mz_scale=cfg.pairmixer_precursor_mz_scale,
            mz_embedding=cfg.pairmixer_mz_embedding,
            token_bin_size=cfg.pairmixer_mz_token_bin_size,
            token_embedding_dim=cfg.pairmixer_mz_token_embedding_dim,
            fourier_num_freqs=cfg.pairmixer_fourier_num_freqs,
            fourier_x_min=cfg.pairmixer_fourier_x_min,
            fourier_x_max=cfg.pairmixer_fourier_x_max,
            relative_fourier_x_min=cfg.pairmixer_relative_fourier_x_min,
            relative_fourier_x_max=cfg.pairmixer_relative_fourier_x_max,
            compute_dtype=compute_dtype,
            rngs=rngs,
        )
        condition_dim = (
            cfg.encoder_metadata_condition_dim if cfg.encoder_metadata_schema else None
        )
        self.metadata_embedder = (
            MetadataEmbedder(condition_dim, compute_dtype=compute_dtype, rngs=rngs)
            if condition_dim is not None else None
        )
        self.blocks = nnx.List([
            PairOnlyBlock(
                pair_dim=pair_dim,
                attention_mlp_multiple=cfg.attention_mlp_multiple,
                norm_eps=cfg.norm_eps,
                transition_type=cfg.pairmixer_transition_type,
                condition_dim=condition_dim,
                triangle_attention_num_heads=(
                    cfg.pairmixer_triangle_attention_num_heads
                    if cfg.pairmixer_triangle_attention_every_n_layers
                    and (index + 1) % cfg.pairmixer_triangle_attention_every_n_layers == 0
                    else None
                ),
                compute_dtype=compute_dtype,
                rngs=rngs,
            ) for index in range(cfg.encoder_num_layers)
        ])
        self.final_pair_norm = (
            RMSNorm(pair_dim, eps=cfg.norm_eps, affine=True)
            if cfg.encoder_apply_final_pair_norm else None
        )

    def __call__(
        self, peak_mz: Array, peak_intensity: Array, *,
        valid_mask: Array, visible_mask: Array,
        max_visible_tokens: int | None = None,
        precursor_mz: Array | None = None,
        spectrum_metadata: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        token_budget = (
            self.pairmixer_fast_max_visible_tokens
            if max_visible_tokens is None else max_visible_tokens
        )
        idx, token_mask = _active_indices(
            valid_mask & visible_mask,
            None if token_budget is None else token_budget - int(self.use_cls_token),
        )
        # Compact before embedding: never allocate a dense N x N pair tensor.
        mz = jnp.take_along_axis(peak_mz, idx, axis=1)
        intensity = jnp.take_along_axis(peak_intensity, idx, axis=1)
        mz = jnp.where(token_mask, mz, 0.0)
        intensity = jnp.where(token_mask, intensity, 0.0)
        pair = self.pair_embedder(mz, intensity, None, token_mask, precursor_mz=precursor_mz)
        if self.use_cls_token:
            pair = _append_cls_pairs(
                pair, self.peak_to_cls_pair_token[...],
                self.cls_to_peak_pair_token[...], self.cls_cls_pair_token[...],
            )
            # CLS uses the first position after the physical peaks, matching
            # the existing encoder convention. It never receives an m/z label.
            idx = jnp.concatenate([
                idx, jnp.full((idx.shape[0], 1), peak_mz.shape[1], dtype=idx.dtype),
            ], axis=1)
            token_mask = jnp.concatenate([
                token_mask, jnp.ones((token_mask.shape[0], 1), dtype=jnp.bool_),
            ], axis=1)
        # Preserve original peak positions across compaction, including CLS.
        # Rotate embedded pair features once, before the first triangle block.
        pair = _apply_pair_rope(pair, idx)
        valid_pair = token_mask[:, :, None] & token_mask[:, None, :]
        pair = pair * valid_pair[..., None].astype(pair.dtype)
        condition = (
            self.metadata_embedder(spectrum_metadata)
            if self.metadata_embedder is not None else None
        )
        pair = run_pair_only_blocks(
            self.blocks, pair, token_mask, condition,
            checkpoint_mode=(
                self.activation_checkpoint_mode
                if "encoder" in self.activation_checkpoint_modules else "none"
            ),
            checkpoint_every_n=self.activation_checkpoint_every_n_layers,
        )
        if self.final_pair_norm is not None:
            pair = self.final_pair_norm(pair)
        return pair, idx, token_mask
