"""TPU fusion for the three PairMixer triangle input projections.

Calls are specialized by encoder/predictor role and compact token shape. Only
the projection is padded to an eight-token tile; the surrounding PairMixer
keeps its exact semantic shape.
"""

from __future__ import annotations

import math
from functools import cache

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from spectra_learning.models.common_jax import Array


def _tile_index_map(batch_index: int, row_block: int) -> tuple[int, int, int, int]:
    return batch_index, row_block, 0, 0


def _shared_matrix_index_map(
    batch_index: int,
    row_block: int,
) -> tuple[int, int]:
    del batch_index, row_block
    return 0, 0


def _shared_vector_index_map(
    batch_index: int,
    row_block: int,
) -> tuple[int]:
    del batch_index, row_block
    return (0,)


def _linear_reference(x: Array, weight: Array, bias: Array) -> Array:
    compute_dtype = x.dtype
    accumulator_dtype = jnp.float32 if compute_dtype == jnp.bfloat16 else None
    output = jax.lax.dot_general(
        x,
        weight.astype(compute_dtype),
        dimension_numbers=(((x.ndim - 1,), (1,)), ((), ())),
        precision=(
            jax.lax.Precision.DEFAULT
            if compute_dtype == jnp.bfloat16
            else jax.lax.Precision.HIGHEST
        ),
        preferred_element_type=accumulator_dtype,
    )
    output = output + bias.astype(output.dtype)
    if accumulator_dtype is not None:
        output = output.astype(compute_dtype)
    return output


def _layer_norm_reference(
    x: Array,
    scale: Array,
    bias: Array,
    *,
    eps: float,
) -> Array:
    x_float = x.astype(jnp.float32)
    mean = jnp.mean(x_float, axis=-1, keepdims=True)
    mean_square = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
    variance = jnp.maximum(mean_square - jnp.square(mean), 0.0)
    normalized = (x_float - mean) * jax.lax.rsqrt(variance + eps)
    return (normalized * scale + bias).astype(x.dtype)


def _triangle_input_projections_reference(
    pair: Array,
    norm_scale: Array,
    norm_bias: Array,
    projection_weight: Array,
    projection_bias: Array,
    input_gate_weight: Array,
    input_gate_bias: Array,
    output_gate_weight: Array,
    output_gate_bias: Array,
    *,
    norm_eps: float,
) -> tuple[Array, Array]:
    normalized = _layer_norm_reference(
        pair,
        norm_scale,
        norm_bias,
        eps=norm_eps,
    )
    projected = _linear_reference(
        normalized,
        projection_weight,
        projection_bias,
    ) * jax.nn.sigmoid(
        _linear_reference(normalized, input_gate_weight, input_gate_bias)
    )
    output_gate = jax.nn.sigmoid(
        _linear_reference(normalized, output_gate_weight, output_gate_bias)
    )
    return projected, output_gate


@cache
def _triangle_input_projection_call(
    batch_size: int,
    num_tokens: int,
    pair_dim: int,
    norm_eps: float,
    kernel_role: str,
):
    def kernel(
        pair_ref,
        norm_scale_ref,
        norm_bias_ref,
        projection_weight_ref,
        projection_bias_ref,
        input_gate_weight_ref,
        input_gate_bias_ref,
        output_gate_weight_ref,
        output_gate_bias_ref,
        projected_ref,
        output_gate_ref,
    ) -> None:
        pair_float = pair_ref[...].astype(jnp.float32)
        mean = jnp.mean(pair_float, axis=-1, keepdims=True)
        mean_square = jnp.mean(jnp.square(pair_float), axis=-1, keepdims=True)
        variance = jnp.maximum(
            mean_square - jnp.square(mean),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        normalized = (
            (pair_float - mean)
            * jax.lax.rsqrt(variance + norm_eps)
            * norm_scale_ref[...]
            + norm_bias_ref[...]
        ).astype(jnp.bfloat16)

        dimension_numbers = (((3,), (1,)), ((), ()))
        projected_accumulator = jax.lax.dot_general(
            normalized,
            projection_weight_ref[...],
            dimension_numbers=dimension_numbers,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )
        input_gate_accumulator = jax.lax.dot_general(
            normalized,
            input_gate_weight_ref[...],
            dimension_numbers=dimension_numbers,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )
        output_gate_accumulator = jax.lax.dot_general(
            normalized,
            output_gate_weight_ref[...],
            dimension_numbers=dimension_numbers,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )

        projection = (
            projected_accumulator + projection_bias_ref[...]
        ).astype(jnp.bfloat16)
        input_gate = (
            input_gate_accumulator + input_gate_bias_ref[...]
        ).astype(jnp.bfloat16)
        output_gate = (
            output_gate_accumulator + output_gate_bias_ref[...]
        ).astype(jnp.bfloat16)
        projected_ref[...] = (
            projection.astype(jnp.float32)
            * jax.nn.sigmoid(input_gate.astype(jnp.float32))
        ).astype(jnp.bfloat16)
        output_gate_ref[...] = jax.nn.sigmoid(
            output_gate.astype(jnp.float32)
        ).astype(jnp.bfloat16)

    def pallas_projection(
        pair: Array,
        norm_scale: Array,
        norm_bias: Array,
        projection_weight: Array,
        projection_bias: Array,
        input_gate_weight: Array,
        input_gate_bias: Array,
        output_gate_weight: Array,
        output_gate_bias: Array,
    ) -> tuple[Array, Array]:
        buffering = pl.Buffered(buffer_count=1)
        pair_spec = pl.BlockSpec(
            block_shape=(1, 8, num_tokens, pair_dim),
            index_map=_tile_index_map,
        )
        pair_vector_spec = pl.BlockSpec(
            block_shape=(pair_dim,),
            index_map=_shared_vector_index_map,
            pipeline_mode=buffering,
        )
        double_pair_vector_spec = pl.BlockSpec(
            block_shape=(2 * pair_dim,),
            index_map=_shared_vector_index_map,
            pipeline_mode=buffering,
        )
        double_pair_matrix_spec = pl.BlockSpec(
            block_shape=(2 * pair_dim, pair_dim),
            index_map=_shared_matrix_index_map,
            pipeline_mode=buffering,
        )
        pair_matrix_spec = pl.BlockSpec(
            block_shape=(pair_dim, pair_dim),
            index_map=_shared_matrix_index_map,
            pipeline_mode=buffering,
        )
        projected_spec = pl.BlockSpec(
            block_shape=(1, 8, num_tokens, 2 * pair_dim),
            index_map=_tile_index_map,
        )
        output_gate_spec = pl.BlockSpec(
            block_shape=(1, 8, num_tokens, pair_dim),
            index_map=_tile_index_map,
        )
        output_shapes = (
            jax.ShapeDtypeStruct(
                (batch_size, num_tokens, num_tokens, 2 * pair_dim),
                jnp.bfloat16,
            ),
            jax.ShapeDtypeStruct(
                (batch_size, num_tokens, num_tokens, pair_dim),
                jnp.bfloat16,
            ),
        )
        return pl.pallas_call(
            kernel,
            out_shape=output_shapes,
            grid=(batch_size, num_tokens // 8),
            in_specs=(
                pair_spec,
                pair_vector_spec,
                pair_vector_spec,
                double_pair_matrix_spec,
                double_pair_vector_spec,
                double_pair_matrix_spec,
                double_pair_vector_spec,
                pair_matrix_spec,
                pair_vector_spec,
            ),
            out_specs=(projected_spec, output_gate_spec),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel"),
            ),
            name=(
                f"pairmixer_{kernel_role}_triangle_projection_"
                f"b{batch_size}_n{num_tokens}_p{pair_dim}"
            ),
        )(
            pair,
            norm_scale,
            norm_bias,
            projection_weight.astype(jnp.bfloat16),
            projection_bias,
            input_gate_weight.astype(jnp.bfloat16),
            input_gate_bias,
            output_gate_weight.astype(jnp.bfloat16),
            output_gate_bias,
        )

    @jax.custom_vjp
    def projection(
        pair: Array,
        norm_scale: Array,
        norm_bias: Array,
        projection_weight: Array,
        projection_bias: Array,
        input_gate_weight: Array,
        input_gate_bias: Array,
        output_gate_weight: Array,
        output_gate_bias: Array,
    ) -> tuple[Array, Array]:
        return pallas_projection(
            pair,
            norm_scale,
            norm_bias,
            projection_weight,
            projection_bias,
            input_gate_weight,
            input_gate_bias,
            output_gate_weight,
            output_gate_bias,
        )

    def projection_forward(*primals):
        return pallas_projection(*primals), primals

    def projection_backward(primals, output_cotangents):
        _, pullback = jax.vjp(
            lambda *values: _triangle_input_projections_reference(
                *values,
                norm_eps=norm_eps,
            ),
            *primals,
        )
        return pullback(output_cotangents)

    projection.defvjp(projection_forward, projection_backward)
    return projection


def triangle_input_projections(
    pair: Array,
    norm_scale: Array,
    norm_bias: Array,
    projection_weight: Array,
    projection_bias: Array,
    input_gate_weight: Array,
    input_gate_bias: Array,
    output_gate_weight: Array,
    output_gate_bias: Array,
    *,
    norm_eps: float,
    kernel_role: str,
) -> tuple[Array, Array]:
    batch_size, num_tokens, _, pair_dim = pair.shape
    padded_tokens = math.ceil(num_tokens / 8) * 8
    token_padding = padded_tokens - num_tokens
    pair = jnp.pad(
        pair,
        (
            (0, 0),
            (0, token_padding),
            (0, token_padding),
            (0, 0),
        ),
    )
    projection = _triangle_input_projection_call(
        batch_size,
        padded_tokens,
        pair_dim,
        norm_eps,
        kernel_role,
    )
    projected, output_gate = projection(
        pair,
        norm_scale,
        norm_bias,
        projection_weight,
        projection_bias,
        input_gate_weight,
        input_gate_bias,
        output_gate_weight,
        output_gate_bias,
    )
    return (
        projected[:, :num_tokens, :num_tokens, :],
        output_gate[:, :num_tokens, :num_tokens, :],
    )
