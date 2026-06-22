from __future__ import annotations

import jax.numpy as jnp
import torch
from flax import nnx

from spectra_learning.models.common_jax import (
    Array,
    LayerNorm,
    activation_checkpoint_policy,
    assign_param,
    build_frozen_position_embedding,
    merge_visible_mask,
    should_activation_checkpoint,
)
from spectra_learning.models.induced_pair_jax import (
    InducedPairBlock,
    InducedPairState,
    TokenInducingAssignment,
)
from spectra_learning.models.pairmixer_jax import PairFeatureEmbedder, PairMixerBlock
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
        induced_pair_num_inducing: int = 8,
        pairmixer_triangle_mediator_num_mediators: int = 8,
        pairmixer_triangle_mediator_eps: float = 1e-4,
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
        self.use_induced_pair = self.pairmixer_block_type == "induced"
        self.use_triangle_mediator = self.pairmixer_block_type == "triangle_mediator"
        self.activation_checkpoint_mode = activation_checkpoint_mode.lower()
        self.activation_checkpoint_every_n_layers = activation_checkpoint_every_n_layers
        self.activation_checkpoint_modules = activation_checkpoint_modules
        self.embedder = embedder
        self.position_embedding = build_frozen_position_embedding(num_peaks, model_dim)
        pair_dim = model_dim if pair_dim is None else pair_dim
        self.cls_token = _normal_token_param(rngs, (model_dim,))
        if self.use_induced_pair:
            self.inducing_token = _normal_token_param(
                rngs,
                (induced_pair_num_inducing, model_dim),
            )
            self.latent_pair_token = _normal_token_param(
                rngs,
                (induced_pair_num_inducing, induced_pair_num_inducing, pair_dim),
            )
            self.initial_left_assignment = TokenInducingAssignment(
                model_dim,
                compute_dtype=compute_dtype,
            )
            self.initial_right_assignment = TokenInducingAssignment(
                model_dim,
                compute_dtype=compute_dtype,
            )
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
        )
        blocks = []
        for _ in range(num_layers):
            if self.use_induced_pair:
                block = InducedPairBlock(
                    single_dim=model_dim,
                    pair_dim=pair_dim,
                    num_heads=num_heads,
                    attention_mlp_multiple=attention_mlp_multiple,
                    norm_eps=norm_eps,
                    dropout=pairmixer_dropout,
                    use_pair_bias_attention=pairmixer_use_pair_bias_attention,
                    compute_dtype=compute_dtype,
                )
            else:
                block = PairMixerBlock(
                    single_dim=model_dim,
                    pair_dim=pair_dim,
                    num_heads=num_heads,
                    attention_mlp_multiple=attention_mlp_multiple,
                    norm_eps=norm_eps,
                    dropout=pairmixer_dropout,
                    use_pair_bias_attention=pairmixer_use_pair_bias_attention,
                    triangle_mediator_num_mediators=(
                        pairmixer_triangle_mediator_num_mediators
                        if self.use_triangle_mediator
                        else None
                    ),
                    triangle_mediator_eps=pairmixer_triangle_mediator_eps,
                    compute_dtype=compute_dtype,
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

    def _append_cls_token(self, x: Array) -> Array:
        cls = jnp.broadcast_to(self.cls_token[...], (x.shape[0], 1, x.shape[-1]))
        return jnp.concatenate([x, cls.astype(x.dtype)], axis=1)

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

    def _initial_inducing_tokens(self, x: Array) -> Array:
        batch_size = x.shape[0]
        return jnp.broadcast_to(
            self.inducing_token[...].astype(x.dtype),
            (batch_size, self.inducing_token[...].shape[0], self.inducing_token[...].shape[1]),
        )

    def _compress_induced_pair(
        self,
        pair: Array,
        left_assignment: Array,
        right_assignment: Array,
        token_visible_mask: Array,
    ) -> Array:
        pair_mask = token_visible_mask[:, :, None] & token_visible_mask[:, None, :]
        pair_mask_f = pair_mask[..., None].astype(pair.dtype)
        latent_pair = jnp.einsum(
            "bia,bijp,bjc->bacp",
            left_assignment,
            pair * pair_mask_f,
            right_assignment,
        )
        denom = jnp.einsum(
            "bia,bij,bjc->bac",
            left_assignment,
            pair_mask.astype(left_assignment.dtype),
            right_assignment,
        )
        denom = jnp.maximum(denom, jnp.asarray(1e-6, dtype=denom.dtype))
        return latent_pair / denom[..., None].astype(latent_pair.dtype)

    def _initial_induced_pair_state(
        self,
        x: Array,
        dense_pair: Array,
        token_visible_mask: Array,
    ) -> InducedPairState:
        inducing = self._initial_inducing_tokens(x)
        left_assignment = self.initial_left_assignment(x, inducing)
        right_assignment = self.initial_right_assignment(x, inducing)
        token_mask_f = token_visible_mask[..., None].astype(left_assignment.dtype)
        left_assignment = left_assignment * token_mask_f
        right_assignment = right_assignment * token_mask_f.astype(right_assignment.dtype)
        pair = self._compress_induced_pair(
            dense_pair,
            left_assignment,
            right_assignment,
            token_visible_mask,
        )
        latent_pair_token = jnp.broadcast_to(
            self.latent_pair_token[...].astype(pair.dtype),
            (
                pair.shape[0],
                self.latent_pair_token[...].shape[0],
                self.latent_pair_token[...].shape[1],
                self.latent_pair_token[...].shape[2],
            ),
        )
        pair = pair + latent_pair_token
        assignment = 0.5 * (left_assignment + right_assignment)
        return InducedPairState(inducing, pair, assignment)

    def _forward_with_induced_pair(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        *,
        valid_mask: Array | None = None,
        visible_mask: Array | None = None,
        precursor_mz: Array | None = None,
        deterministic: bool = True,
    ) -> tuple[Array, InducedPairState]:
        peak_valid_mask = (
            jnp.ones_like(peak_mz, dtype=jnp.bool_) if valid_mask is None else valid_mask
        )
        peak_visible_mask = merge_visible_mask(peak_valid_mask, visible_mask)
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
        for block_idx, block in enumerate(self.blocks, start=1):
            if should_activation_checkpoint(
                mode=self.activation_checkpoint_mode,
                modules=self.activation_checkpoint_modules,
                module="encoder",
                block_idx=block_idx,
                every_n=self.activation_checkpoint_every_n_layers,
            ):
                x, state = nnx.remat(
                    _call_pair_mixer_block,
                    policy=activation_checkpoint_policy(
                        self.activation_checkpoint_mode
                    ),
                )(block, x, state, token_visible_mask, token_visible_mask)
            else:
                x, state = block(
                    x,
                    state,
                    token_visible_mask,
                    token_visible_mask,
                    deterministic=deterministic,
                )
        if self.final_norm is not None:
            x = self.final_norm(x)
        pair = state.pair
        if self.final_pair_norm is not None:
            pair = self.final_pair_norm(pair)
        assignment = state.assignment * token_visible_mask[..., None].astype(
            state.assignment.dtype
        )
        return x, InducedPairState(state.inducing, pair, assignment)

    def forward_with_pair(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        *,
        valid_mask: Array | None = None,
        visible_mask: Array | None = None,
        precursor_mz: Array | None = None,
        deterministic: bool = True,
    ) -> tuple[Array, Array | InducedPairState]:
        if self.use_induced_pair:
            return self._forward_with_induced_pair(
                peak_mz,
                peak_intensity,
                valid_mask=valid_mask,
                visible_mask=visible_mask,
                precursor_mz=precursor_mz,
                deterministic=deterministic,
            )
        peak_valid_mask = (
            jnp.ones_like(peak_mz, dtype=jnp.bool_) if valid_mask is None else valid_mask
        )
        peak_visible_mask = merge_visible_mask(peak_valid_mask, visible_mask)
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
        if self.final_pair_norm is not None:
            z = self.final_pair_norm(z)
        pair_visible_mask = token_visible_mask[:, :, None] & token_visible_mask[:, None, :]
        z = z * pair_visible_mask[..., None].astype(z.dtype)
        return x, z

    def __call__(
        self,
        peak_mz: Array,
        peak_intensity: Array,
        *,
        valid_mask: Array | None = None,
        visible_mask: Array | None = None,
        precursor_mz: Array | None = None,
    ) -> Array:
        output, _ = self.forward_with_pair(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            visible_mask=visible_mask,
            precursor_mz=precursor_mz,
        )
        return output

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        assign_param(self.cls_token, state_dict[f"{prefix}.cls_token"])
        self.embedder.load_torch_state_dict(state_dict, f"{prefix}.embedder")
        self.position_embedding.load_torch_state_dict(
            state_dict,
            f"{prefix}.position_embedding",
        )
        if self.use_induced_pair:
            assign_param(self.inducing_token, state_dict[f"{prefix}.inducing_token"])
            assign_param(
                self.latent_pair_token,
                state_dict[f"{prefix}.latent_pair_token"],
            )
            self.initial_left_assignment.load_torch_state_dict(
                state_dict,
                f"{prefix}.initial_left_assignment",
            )
            self.initial_right_assignment.load_torch_state_dict(
                state_dict,
                f"{prefix}.initial_right_assignment",
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
