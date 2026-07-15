from __future__ import annotations

import math
from functools import cache

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from spectra_learning.models.common_jax import Array


def pallas_causal_attention(
    query: Array,
    key: Array,
    value: Array,
    *,
    block_size: int = 128,
    batch_head_group: int = 8,
    query_is_scaled: bool = False,
) -> Array:
    """Causal attention with a fused Pallas forward and explicit backward."""
    batch_size, num_heads, sequence_length, head_dim = query.shape
    num_batch_heads = batch_size * num_heads
    group_size = math.gcd(num_batch_heads, batch_head_group)
    kernel_batch_size = batch_size
    kernel_num_heads = num_heads

    # Mosaic requires this block dimension to span the axis or be divisible by 8.
    if group_size % 8 and group_size != num_batch_heads:
        padded_num_batch_heads = 8 * math.ceil(num_batch_heads / 8)
        batch_head_padding = padded_num_batch_heads - num_batch_heads
        padding = ((0, batch_head_padding), (0, 0), (0, 0))
        query = jnp.pad(
            query.reshape(num_batch_heads, sequence_length, head_dim),
            padding,
        )[None, ...]
        key = jnp.pad(
            key.reshape(num_batch_heads, sequence_length, head_dim),
            padding,
        )[None, ...]
        value = jnp.pad(
            value.reshape(num_batch_heads, sequence_length, head_dim),
            padding,
        )[None, ...]
        kernel_batch_size = 1
        kernel_num_heads = padded_num_batch_heads
        group_size = 8

    kernel = _pallas_causal_attention_kernel(
        kernel_batch_size,
        kernel_num_heads,
        sequence_length,
        head_dim,
        block_size,
        group_size,
        query_is_scaled,
    )
    output = kernel(query, key, value)
    return output.reshape(
        kernel_batch_size * kernel_num_heads,
        sequence_length,
        head_dim,
    )[:num_batch_heads].reshape(
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
    )


@cache
def _pallas_causal_attention_kernel(
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    head_dim: int,
    block_size: int,
    group_size: int,
    query_is_scaled: bool,
):
    padded_sequence_length = block_size * math.ceil(sequence_length / block_size)
    num_query_blocks = padded_sequence_length // block_size
    num_full_query_blocks = sequence_length // block_size
    tail_query_rows = sequence_length % block_size
    last_query_block = num_query_blocks - 1
    num_batch_heads = batch_size * num_heads
    num_batch_head_groups = num_batch_heads // group_size
    attention_scale = 1.0 / math.sqrt(head_dim)

    def online_softmax_update(carry, scores_qk, value_block):
        accumulator, row_max, row_sum = carry
        block_max = jnp.max(scores_qk, axis=-1)
        new_max = jnp.maximum(row_max, block_max)
        old_scale = jnp.exp(row_max - new_max)
        probabilities_qk = jnp.exp(scores_qk - new_max[..., None])
        new_sum = row_sum * old_scale + jnp.sum(probabilities_qk, axis=-1)
        probability_value = jax.lax.dot_general(
            probabilities_qk.astype(jnp.bfloat16),
            value_block.astype(jnp.bfloat16),
            (((2,), (1,)), ((0,), (0,))),
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )
        new_accumulator = (
            accumulator * old_scale[..., None] + probability_value
        )
        return new_accumulator, new_max, new_sum

    def full_forward_body(
        query_block,
        query_ref,
        key_ref,
        value_ref,
        output_ref,
        logsumexp_ref,
    ):
        query_block_value = query_ref[...]
        carry = (
            jnp.zeros(
                (group_size, block_size, head_dim),
                dtype=jnp.float32,
            ),
            jnp.full((group_size, block_size), -jnp.inf, dtype=jnp.float32),
            jnp.zeros((group_size, block_size), dtype=jnp.float32),
        )

        def earlier_key_block(key_block, current_carry):
            key_slice = pl.ds(key_block * block_size, block_size)
            key_block_value = key_ref.at[:, key_slice, :][...]
            value_block = value_ref.at[:, key_slice, :][...]
            scores_qk = jax.lax.dot_general(
                query_block_value,
                key_block_value,
                (((2,), (2,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            return online_softmax_update(current_carry, scores_qk, value_block)

        carry = jax.lax.fori_loop(0, query_block, earlier_key_block, carry)
        diagonal_slice = pl.ds(query_block * block_size, block_size)
        diagonal_key = key_ref.at[:, diagonal_slice, :][...]
        diagonal_value = value_ref.at[:, diagonal_slice, :][...]
        diagonal_scores_qk = jax.lax.dot_general(
            query_block_value,
            diagonal_key,
            (((2,), (2,)), ((0,), (0,))),
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )
        query_positions = jax.lax.broadcasted_iota(
            jnp.int32,
            (block_size, block_size),
            0,
        )
        key_positions = jax.lax.broadcasted_iota(
            jnp.int32,
            (block_size, block_size),
            1,
        )
        diagonal_scores_qk = jnp.where(
            (query_positions >= key_positions)[None, :, :],
            diagonal_scores_qk,
            jnp.float32(-1e30),
        )
        accumulator, row_max, row_sum = online_softmax_update(
            carry,
            diagonal_scores_qk,
            diagonal_value,
        )
        output_ref[...] = (accumulator / row_sum[..., None]).astype(
            output_ref.dtype
        )
        logsumexp_ref[...] = row_max + jnp.log(row_sum)

    def tail_forward_body(
        query_ref,
        key_ref,
        value_ref,
        output_ref,
        logsumexp_ref,
    ):
        query_block_value = query_ref[:, :tail_query_rows, :]
        carry = (
            jnp.zeros(
                (group_size, tail_query_rows, head_dim),
                dtype=jnp.float32,
            ),
            jnp.full(
                (group_size, tail_query_rows),
                -jnp.inf,
                dtype=jnp.float32,
            ),
            jnp.zeros((group_size, tail_query_rows), dtype=jnp.float32),
        )

        def earlier_key_block(key_block, current_carry):
            key_slice = pl.ds(key_block * block_size, block_size)
            key_block_value = key_ref.at[:, key_slice, :][...]
            value_block = value_ref.at[:, key_slice, :][...]
            scores_qk = jax.lax.dot_general(
                query_block_value,
                key_block_value,
                (((2,), (2,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            return online_softmax_update(current_carry, scores_qk, value_block)

        carry = jax.lax.fori_loop(
            0,
            last_query_block,
            earlier_key_block,
            carry,
        )
        diagonal_slice = pl.ds(last_query_block * block_size, block_size)
        diagonal_key = key_ref.at[:, diagonal_slice, :][...]
        diagonal_value = value_ref.at[:, diagonal_slice, :][...]
        key_rows = jax.lax.broadcasted_iota(
            jnp.int32,
            diagonal_key.shape,
            1,
        )
        valid_key = key_rows < tail_query_rows
        diagonal_key = jnp.where(
            valid_key,
            diagonal_key,
            jnp.zeros((), diagonal_key.dtype),
        )
        diagonal_value = jnp.where(
            valid_key,
            diagonal_value,
            jnp.zeros((), diagonal_value.dtype),
        )
        diagonal_scores_qk = jax.lax.dot_general(
            query_block_value,
            diagonal_key,
            (((2,), (2,)), ((0,), (0,))),
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )
        query_positions = jax.lax.broadcasted_iota(
            jnp.int32,
            (tail_query_rows, block_size),
            0,
        )
        key_positions = jax.lax.broadcasted_iota(
            jnp.int32,
            (tail_query_rows, block_size),
            1,
        )
        tail_mask = (
            (query_positions >= key_positions)
            & (key_positions < tail_query_rows)
        )
        diagonal_scores_qk = jnp.where(
            tail_mask[None, :, :],
            diagonal_scores_qk,
            jnp.float32(-1e30),
        )
        accumulator, row_max, row_sum = online_softmax_update(
            carry,
            diagonal_scores_qk,
            diagonal_value,
        )
        output_ref[:, :tail_query_rows, :] = (
            accumulator / row_sum[..., None]
        ).astype(output_ref.dtype)
        logsumexp_ref[:, :tail_query_rows] = row_max + jnp.log(row_sum)

    def forward_kernel(
        query_ref,
        key_ref,
        value_ref,
        output_ref,
        logsumexp_ref,
    ):
        query_block = pl.program_id(1)

        @pl.when(query_block < num_full_query_blocks)
        def full_block():
            full_forward_body(
                query_block,
                query_ref,
                key_ref,
                value_ref,
                output_ref,
                logsumexp_ref,
            )

        if tail_query_rows:
            @pl.when(query_block == last_query_block)
            def tail_block():
                tail_forward_body(
                    query_ref,
                    key_ref,
                    value_ref,
                    output_ref,
                    logsumexp_ref,
                )

    query_spec = pl.BlockSpec(
        (group_size, block_size, head_dim),
        lambda group, query_block: (group, query_block, 0),
    )
    full_sequence_spec = pl.BlockSpec(
        (group_size, padded_sequence_length, head_dim),
        lambda group, query_block: (group, 0, 0),
    )
    output_spec = pl.BlockSpec(
        (group_size, block_size, head_dim),
        lambda group, query_block: (group, query_block, 0),
    )
    logsumexp_spec = pl.BlockSpec(
        (group_size, block_size),
        lambda group, query_block: (group, query_block),
    )

    def forward_call(query, key, value):
        query_flat = query.reshape(num_batch_heads, sequence_length, head_dim)
        key_flat = key.reshape(num_batch_heads, sequence_length, head_dim)
        value_flat = value.reshape(num_batch_heads, sequence_length, head_dim)
        output, logsumexp = pl.pallas_call(
            forward_kernel,
            out_shape=(
                jax.ShapeDtypeStruct(query_flat.shape, query.dtype),
                jax.ShapeDtypeStruct(
                    (num_batch_heads, sequence_length),
                    jnp.float32,
                ),
            ),
            in_specs=(
                query_spec,
                full_sequence_spec,
                full_sequence_spec,
            ),
            out_specs=(output_spec, logsumexp_spec),
            grid=(num_batch_head_groups, num_query_blocks),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
            ),
        )(query_flat, key_flat, value_flat)
        return (
            output.reshape(batch_size, num_heads, sequence_length, head_dim),
            logsumexp.reshape(batch_size, num_heads, sequence_length),
        )

    def full_backward_dq_body(
        query_block,
        query_ref,
        key_ref,
        value_ref,
        logsumexp_ref,
        output_gradient_ref,
        output_dot_gradient_ref,
        query_gradient_ref,
    ):
        query_block_value = query_ref[...]
        output_gradient = output_gradient_ref[...]
        logsumexp = logsumexp_ref[...]
        output_dot_gradient = output_dot_gradient_ref[...]
        query_gradient = jnp.zeros(
            (group_size, block_size, head_dim),
            dtype=jnp.float32,
        )

        def key_block_body(key_block, accumulator):
            key_slice = pl.ds(key_block * block_size, block_size)
            key_block_value = key_ref.at[:, key_slice, :][...]
            value_block = value_ref.at[:, key_slice, :][...]
            scores_qk = jax.lax.dot_general(
                query_block_value,
                key_block_value,
                (((2,), (2,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            query_positions = jax.lax.broadcasted_iota(
                jnp.int32,
                (block_size, block_size),
                0,
            )
            key_positions = jax.lax.broadcasted_iota(
                jnp.int32,
                (block_size, block_size),
                1,
            )
            scores_qk = jnp.where(
                (
                    (key_block < query_block)
                    | (query_positions >= key_positions)
                )[None, :, :],
                scores_qk,
                jnp.float32(-1e30),
            )
            probabilities = jnp.exp(scores_qk - logsumexp[..., None])
            probability_gradient = jax.lax.dot_general(
                output_gradient,
                value_block,
                (((2,), (2,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            score_gradient = probabilities * (
                probability_gradient - output_dot_gradient[..., None]
            )
            return accumulator + jax.lax.dot_general(
                score_gradient.astype(jnp.bfloat16),
                key_block_value.astype(jnp.bfloat16),
                (((2,), (1,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )

        query_gradient = jax.lax.fori_loop(
            0,
            query_block + 1,
            key_block_body,
            query_gradient,
        )
        query_gradient_ref[...] = query_gradient.astype(
            query_gradient_ref.dtype
        )

    def tail_backward_dq_body(
        query_ref,
        key_ref,
        value_ref,
        logsumexp_ref,
        output_gradient_ref,
        output_dot_gradient_ref,
        query_gradient_ref,
    ):
        query_block_value = query_ref[:, :tail_query_rows, :]
        output_gradient = output_gradient_ref[:, :tail_query_rows, :]
        logsumexp = logsumexp_ref[:, :tail_query_rows]
        output_dot_gradient = output_dot_gradient_ref[:, :tail_query_rows]
        query_gradient = jnp.zeros(
            (group_size, tail_query_rows, head_dim),
            dtype=jnp.float32,
        )

        def key_block_body(key_block, accumulator):
            key_slice = pl.ds(key_block * block_size, block_size)
            key_block_value = key_ref.at[:, key_slice, :][...]
            value_block = value_ref.at[:, key_slice, :][...]
            key_rows = jax.lax.broadcasted_iota(
                jnp.int32,
                key_block_value.shape,
                1,
            )
            valid_key = (
                (key_block < last_query_block)
                | (key_rows < tail_query_rows)
            )
            key_block_value = jnp.where(
                valid_key,
                key_block_value,
                jnp.zeros((), key_block_value.dtype),
            )
            value_block = jnp.where(
                valid_key,
                value_block,
                jnp.zeros((), value_block.dtype),
            )
            scores_qk = jax.lax.dot_general(
                query_block_value,
                key_block_value,
                (((2,), (2,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            query_positions = jax.lax.broadcasted_iota(
                jnp.int32,
                (tail_query_rows, block_size),
                0,
            )
            key_positions = jax.lax.broadcasted_iota(
                jnp.int32,
                (tail_query_rows, block_size),
                1,
            )
            scores_qk = jnp.where(
                (
                    (key_block < last_query_block)
                    | (
                        (query_positions >= key_positions)
                        & (key_positions < tail_query_rows)
                    )
                )[None, :, :],
                scores_qk,
                jnp.float32(-1e30),
            )
            probabilities = jnp.exp(scores_qk - logsumexp[..., None])
            probability_gradient = jax.lax.dot_general(
                output_gradient,
                value_block,
                (((2,), (2,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            score_gradient = probabilities * (
                probability_gradient - output_dot_gradient[..., None]
            )
            return accumulator + jax.lax.dot_general(
                score_gradient.astype(jnp.bfloat16),
                key_block_value.astype(jnp.bfloat16),
                (((2,), (1,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )

        query_gradient = jax.lax.fori_loop(
            0,
            num_query_blocks,
            key_block_body,
            query_gradient,
        )
        query_gradient_ref[:, :tail_query_rows, :] = query_gradient.astype(
            query_gradient_ref.dtype
        )

    def backward_dq_kernel(
        query_ref,
        key_ref,
        value_ref,
        logsumexp_ref,
        output_gradient_ref,
        output_dot_gradient_ref,
        query_gradient_ref,
    ):
        query_block = pl.program_id(1)

        @pl.when(query_block < num_full_query_blocks)
        def full_block():
            full_backward_dq_body(
                query_block,
                query_ref,
                key_ref,
                value_ref,
                logsumexp_ref,
                output_gradient_ref,
                output_dot_gradient_ref,
                query_gradient_ref,
            )

        if tail_query_rows:
            @pl.when(query_block == last_query_block)
            def tail_block():
                tail_backward_dq_body(
                    query_ref,
                    key_ref,
                    value_ref,
                    logsumexp_ref,
                    output_gradient_ref,
                    output_dot_gradient_ref,
                    query_gradient_ref,
                )

    backward_query_spec = pl.BlockSpec(
        (group_size, block_size, head_dim),
        lambda group, query_block: (group, query_block, 0),
    )
    backward_row_spec = pl.BlockSpec(
        (group_size, block_size),
        lambda group, query_block: (group, query_block),
    )

    def backward_dkv_kernel(
        query_ref,
        key_ref,
        value_ref,
        logsumexp_ref,
        output_gradient_ref,
        output_dot_gradient_ref,
        key_gradient_ref,
        value_gradient_ref,
    ):
        key_block = pl.program_id(1)
        key_block_value = key_ref[...]
        value_block = value_ref[...]
        if tail_query_rows:
            key_rows = jax.lax.broadcasted_iota(
                jnp.int32,
                key_block_value.shape,
                1,
            )
            valid_key = (
                (key_block < last_query_block)
                | (key_rows < tail_query_rows)
            )
            key_block_value = jnp.where(
                valid_key,
                key_block_value,
                jnp.zeros((), key_block_value.dtype),
            )
            value_block = jnp.where(
                valid_key,
                value_block,
                jnp.zeros((), value_block.dtype),
            )
        key_gradient = jnp.zeros(
            (group_size, block_size, head_dim),
            dtype=jnp.float32,
        )
        value_gradient = jnp.zeros(
            (group_size, block_size, head_dim),
            dtype=jnp.float32,
        )

        def query_block_body(query_block, carry):
            current_key_gradient, current_value_gradient = carry
            query_slice = pl.ds(query_block * block_size, block_size)
            query_block_value = query_ref.at[:, query_slice, :][...]
            output_gradient = output_gradient_ref.at[:, query_slice, :][...]
            logsumexp = logsumexp_ref.at[:, query_slice][...]
            output_dot_gradient = output_dot_gradient_ref.at[:, query_slice][...]
            scores_qk = jax.lax.dot_general(
                query_block_value,
                key_block_value,
                (((2,), (2,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            query_positions = jax.lax.broadcasted_iota(
                jnp.int32,
                (block_size, block_size),
                0,
            )
            key_positions = jax.lax.broadcasted_iota(
                jnp.int32,
                (block_size, block_size),
                1,
            )
            scores_qk = jnp.where(
                (
                    (query_block > key_block)
                    | (query_positions >= key_positions)
                )[None, :, :],
                scores_qk,
                jnp.float32(-1e30),
            )
            probabilities = jnp.exp(scores_qk - logsumexp[..., None])
            probability_gradient = jax.lax.dot_general(
                output_gradient,
                value_block,
                (((2,), (2,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            score_gradient = probabilities * (
                probability_gradient - output_dot_gradient[..., None]
            )
            current_key_gradient += jax.lax.dot_general(
                score_gradient.astype(jnp.bfloat16),
                query_block_value.astype(jnp.bfloat16),
                (((1,), (1,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            current_value_gradient += jax.lax.dot_general(
                probabilities.astype(jnp.bfloat16),
                output_gradient.astype(jnp.bfloat16),
                (((1,), (1,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            return current_key_gradient, current_value_gradient

        key_gradient, value_gradient = jax.lax.fori_loop(
            key_block,
            num_full_query_blocks,
            query_block_body,
            (key_gradient, value_gradient),
        )
        if tail_query_rows:
            tail_query_slice = pl.ds(
                last_query_block * block_size,
                block_size,
            )
            query_block_value = query_ref.at[:, tail_query_slice, :][...]
            query_block_value = query_block_value[:, :tail_query_rows, :]
            output_gradient = output_gradient_ref.at[:, tail_query_slice, :][...]
            output_gradient = output_gradient[:, :tail_query_rows, :]
            logsumexp = logsumexp_ref.at[:, tail_query_slice][...]
            logsumexp = logsumexp[:, :tail_query_rows]
            output_dot_gradient = output_dot_gradient_ref.at[
                :, tail_query_slice
            ][...]
            output_dot_gradient = output_dot_gradient[:, :tail_query_rows]
            scores_qk = jax.lax.dot_general(
                query_block_value,
                key_block_value,
                (((2,), (2,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            query_positions = jax.lax.broadcasted_iota(
                jnp.int32,
                (tail_query_rows, block_size),
                0,
            )
            key_positions = jax.lax.broadcasted_iota(
                jnp.int32,
                (tail_query_rows, block_size),
                1,
            )
            scores_qk = jnp.where(
                (
                    (key_block < last_query_block)
                    | (
                        (query_positions >= key_positions)
                        & (key_positions < tail_query_rows)
                    )
                )[None, :, :],
                scores_qk,
                jnp.float32(-1e30),
            )
            probabilities = jnp.exp(scores_qk - logsumexp[..., None])
            probability_gradient = jax.lax.dot_general(
                output_gradient,
                value_block,
                (((2,), (2,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            score_gradient = probabilities * (
                probability_gradient - output_dot_gradient[..., None]
            )
            key_gradient += jax.lax.dot_general(
                score_gradient.astype(jnp.bfloat16),
                query_block_value.astype(jnp.bfloat16),
                (((1,), (1,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
            value_gradient += jax.lax.dot_general(
                probabilities.astype(jnp.bfloat16),
                output_gradient.astype(jnp.bfloat16),
                (((1,), (1,)), ((0,), (0,))),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
        key_gradient_ref[...] = key_gradient.astype(key_gradient_ref.dtype)
        value_gradient_ref[...] = value_gradient.astype(
            value_gradient_ref.dtype
        )

    backward_full_sequence_spec = pl.BlockSpec(
        (group_size, padded_sequence_length, head_dim),
        lambda group, key_block: (group, 0, 0),
    )
    backward_full_rows_spec = pl.BlockSpec(
        (group_size, padded_sequence_length),
        lambda group, key_block: (group, 0),
    )
    backward_key_block_spec = pl.BlockSpec(
        (group_size, block_size, head_dim),
        lambda group, key_block: (group, key_block, 0),
    )

    def backward_call(
        query,
        key,
        value,
        output,
        logsumexp,
        output_gradient,
    ):
        query_flat = query.reshape(num_batch_heads, sequence_length, head_dim)
        key_flat = key.reshape(num_batch_heads, sequence_length, head_dim)
        value_flat = value.reshape(num_batch_heads, sequence_length, head_dim)
        output_flat = output.reshape(num_batch_heads, sequence_length, head_dim)
        logsumexp_flat = logsumexp.reshape(num_batch_heads, sequence_length)
        output_gradient_flat = output_gradient.reshape(
            num_batch_heads,
            sequence_length,
            head_dim,
        )
        output_dot_gradient = jnp.sum(
            output_flat.astype(jnp.float32)
            * output_gradient_flat.astype(jnp.float32),
            axis=-1,
        )
        (query_gradient,) = pl.pallas_call(
            backward_dq_kernel,
            out_shape=(jax.ShapeDtypeStruct(query_flat.shape, query.dtype),),
            in_specs=(
                backward_query_spec,
                full_sequence_spec,
                full_sequence_spec,
                backward_row_spec,
                backward_query_spec,
                backward_row_spec,
            ),
            out_specs=(backward_query_spec,),
            grid=(num_batch_head_groups, num_query_blocks),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
            ),
        )(
            query_flat,
            key_flat,
            value_flat,
            logsumexp_flat,
            output_gradient_flat,
            output_dot_gradient,
        )
        key_gradient, value_gradient = pl.pallas_call(
            backward_dkv_kernel,
            out_shape=(
                jax.ShapeDtypeStruct(key_flat.shape, key.dtype),
                jax.ShapeDtypeStruct(value_flat.shape, value.dtype),
            ),
            in_specs=(
                backward_full_sequence_spec,
                backward_key_block_spec,
                backward_key_block_spec,
                backward_full_rows_spec,
                backward_full_sequence_spec,
                backward_full_rows_spec,
            ),
            out_specs=(backward_key_block_spec, backward_key_block_spec),
            grid=(num_batch_head_groups, num_query_blocks),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
            ),
        )(
            query_flat,
            key_flat,
            value_flat,
            logsumexp_flat,
            output_gradient_flat,
            output_dot_gradient,
        )
        gradient_shape = (batch_size, num_heads, sequence_length, head_dim)
        return (
            query_gradient.reshape(gradient_shape),
            key_gradient.reshape(gradient_shape),
            value_gradient.reshape(gradient_shape),
        )

    @jax.custom_vjp
    def attention(query, key, value):
        scaled_query = query
        if not query_is_scaled:
            scaled_query = (
                query.astype(jnp.float32) * jnp.float32(attention_scale)
            ).astype(query.dtype)
        output, _ = forward_call(scaled_query, key, value)
        return output

    def attention_fwd(query, key, value):
        scaled_query = query
        if not query_is_scaled:
            scaled_query = (
                query.astype(jnp.float32) * jnp.float32(attention_scale)
            ).astype(query.dtype)
        output, logsumexp = forward_call(scaled_query, key, value)
        return output, (
            scaled_query,
            key,
            value,
            output,
            logsumexp,
        )

    def attention_bwd(residuals, output_gradient):
        scaled_query, key, value, output, logsumexp = residuals
        query_gradient, key_gradient, value_gradient = backward_call(
            scaled_query,
            key,
            value,
            output,
            logsumexp,
            output_gradient,
        )
        if not query_is_scaled:
            query_gradient = (
                query_gradient.astype(jnp.float32) * jnp.float32(attention_scale)
            ).astype(query_gradient.dtype)
        return query_gradient, key_gradient, value_gradient

    attention.defvjp(attention_fwd, attention_bwd)
    return attention
