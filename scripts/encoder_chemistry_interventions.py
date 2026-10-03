"""Native JAX causal interventions for the single-stream peak encoder.

The forward pass preserves the production attention kernel, RoPE, Q/K RMSNorm,
AdaLN conditioning, attention output gate, post norms and preferred-accumulation
MLP. It never reads chemical labels. Only single-stream encoders are supported;
silently treating a pair-stream model as a transformer would change its function.

``patch_layer`` is dynamic: 0 patches embedded tokens, 1..L patches the raw
residual after that block, and -1 disables patching. ``patch_mask`` uses original
token slots [batch, peaks + CLS]; it is intersected with visibility. CLS is the
last slot and must be selected deliberately. Neither patching nor ablation moves
tokens, changes RoPE positions, or unmasks missing peaks. Clean/corrupted inputs
must keep the same physical slots for token-specific activation patching.

Branch scales act on complete, metadata-gated residual updates. Head scales act
on attention values before the per-channel sigmoid gate, output projection and
post norm. Consequently head effects are causal interventions, not an additive
decomposition after normalization. Zero is an ablation; 1 is identity. All layer,
token and head selectors are arrays so sweeps reuse a compiled forward function.
``block_condition_scale`` multiplies the metadata embedding entering each
block's modulation linear; zero retains that linear's learned bias but removes
sample-dependent conditioning. ``final_modulation_scale=0`` removes the final
metadata shift/scale while retaining final RMSNorm. These are explicit causal
interventions, not alternative names for the original encoder output.

The raw residual cache and finalized output are distinct: final RMSNorm and the
final metadata affine are applied only to ``final``. Invalid-query outputs are
not zeroed by the production dense encoder and must be excluded by token_mask.
Optional attention patterns are float32 QK/softmax diagnostics; they are never
substituted for the production attention kernel or interpreted as explanations.

Eager identity forwards are bitwise equivalent to production. Under JIT, dynamic
controls can change XLA fusion and bfloat16 rounding even when set to identity;
bitwise compiled equivalence is not claimed. Before interpreting an experiment,
measure native-versus-instrumented identity and same-coordinate identity-patch
noise on the actual device/checkpoint and compare it with intervention effects.
The tiny-model tests bound bf16 relative L2 by the existing export tolerance
(0.005); this is an instrumentation gate, not a substitute for device validation.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from spectra_learning.models.common_jax import Array, scaled_dot_product_attention
from spectra_learning.models.encoder_jax import PeakSetEncoder
from spectra_learning.models.pairmixer_jax import (
    AttentionPairBias,
    _apply_rope,
    _transition_with_preferred_acc,
)


class EncoderInterventions(NamedTuple):
    patch_layer: Array
    patch_values: Array
    patch_mask: Array
    attention_scale: Array
    mlp_scale: Array
    head_scale: Array
    block_condition_scale: Array
    final_modulation_scale: Array


class EncoderTrace(NamedTuple):
    final: Array
    final_normalized: Array
    raw: Array
    attention_residual: Array
    attention_update: Array
    mlp_update: Array
    token_mask: Array
    attention_patterns: Array | None


def identity_interventions(
    encoder: PeakSetEncoder, batch_size: int, num_peak_tokens: int
) -> EncoderInterventions:
    """Create dynamic identity controls; all returned arrays are JAX pytrees."""
    if encoder.use_pair_path:
        raise ValueError("Chemistry interventions require a single-stream encoder")
    attention = encoder.blocks[0].single_attention
    tokens = num_peak_tokens + int(encoder.use_cls_token)
    width = attention.qkv.weight.shape[1]
    depth = len(encoder.blocks)
    return EncoderInterventions(
        patch_layer=jnp.asarray(-1, dtype=jnp.int32),
        patch_values=jnp.zeros(
            (batch_size, tokens, width), dtype=attention.qkv.compute_dtype
        ),
        patch_mask=jnp.zeros((batch_size, tokens), dtype=jnp.bool_),
        attention_scale=jnp.ones((depth,), dtype=jnp.float32),
        mlp_scale=jnp.ones((depth,), dtype=jnp.float32),
        head_scale=jnp.ones((depth, attention.num_heads), dtype=jnp.float32),
        block_condition_scale=jnp.ones((depth,), dtype=jnp.float32),
        final_modulation_scale=jnp.asarray(1.0, dtype=jnp.float32),
    )


def _attention(
    module: AttentionPairBias,
    normalized: Array,
    token_mask: Array,
    head_scale: Array,
    capture_attention: bool,
) -> tuple[Array, Array | None]:
    batch_size, num_tokens, width = normalized.shape
    qkv = module.qkv(normalized).reshape(
        batch_size, num_tokens, 3, module.num_heads, module.head_dim
    )
    q, k, v = jnp.moveaxis(qkv, 2, 0)
    q = module.q_norm(jnp.swapaxes(q, 1, 2))
    k = module.k_norm(jnp.swapaxes(k, 1, 2))
    v = jnp.swapaxes(v, 1, 2)
    if module.use_rope:
        positions = jnp.broadcast_to(jnp.arange(num_tokens), (batch_size, num_tokens))
        q = _apply_rope(q, positions)
        k = _apply_rope(k, positions)
    bias = jnp.where(
        token_mask[:, None, None, :],
        jnp.zeros((batch_size, 1, 1, num_tokens), dtype=jnp.float32),
        jnp.asarray(-jnp.inf, dtype=jnp.float32),
    )
    heads = scaled_dot_product_attention(q, k, v, attn_mask=bias)
    heads = heads * head_scale[None, :, None, None].astype(heads.dtype)
    combined = jnp.swapaxes(heads, 1, 2).reshape(batch_size, num_tokens, width)
    combined = combined * jax.nn.sigmoid(module.g(normalized))
    output = module.o(combined)
    patterns = None
    if capture_attention:
        scores = jnp.einsum(
            "bhqd,bhkd->bhqk",
            q.astype(jnp.float32),
            k.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        ) / math.sqrt(module.head_dim)
        patterns = jax.nn.softmax(scores + bias, axis=-1)
    return output, patterns


def trace_encoder(
    encoder: PeakSetEncoder,
    peak_mz: Array,
    peak_intensity: Array,
    *,
    valid_mask: Array,
    spectrum_metadata: Array | None = None,
    visible_mask: Array | None = None,
    interventions: EncoderInterventions | None = None,
    capture_attention: bool = False,
) -> EncoderTrace:
    """Run a dense native forward with traces and optional dynamic interventions.

    Wrap this function with ``nnx.jit`` in the caller. ``capture_attention`` is a
    static choice; the arrays inside ``interventions`` must remain dynamic.
    Preserve their dtypes when replacing controls (e.g. use ``jnp.int32`` for
    patch_layer); changing a scalar's JAX weak-type status triggers recompilation.
    Masks with no visible peak need a CLS token, as in the production encoder.
    The return caches have leading layer axes: raw has L+1 entries,
    and attention_residual/attention_update/mlp_update have L entries.
    """
    if encoder.use_pair_path:
        raise ValueError("Chemistry interventions require a single-stream encoder")
    if interventions is None:
        interventions = identity_interventions(encoder, *peak_mz.shape)
    x, visible, condition = encoder._embed_peaks(
        peak_mz, peak_intensity, valid_mask, visible_mask, spectrum_metadata
    )
    x = encoder._append_cls_token(
        x, condition if encoder.metadata_conditioning != "adaln_zero" else None
    )
    token_mask = encoder._append_cls_mask(visible)
    depth = len(encoder.blocks)
    heads = encoder.blocks[0].single_attention.num_heads
    expected_shapes = {
        "patch_layer": (),
        "patch_values": x.shape,
        "patch_mask": token_mask.shape,
        "attention_scale": (depth,),
        "mlp_scale": (depth,),
        "head_scale": (depth, heads),
        "block_condition_scale": (depth,),
        "final_modulation_scale": (),
    }
    for name, shape in expected_shapes.items():
        if getattr(interventions, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if interventions.patch_mask.dtype != jnp.bool_:
        raise ValueError("patch_mask must be boolean")

    def patch(value: Array, layer: int) -> Array:
        selected = (
            (interventions.patch_layer == layer)
            & interventions.patch_mask
            & token_mask
        )
        return jnp.where(
            selected[..., None], interventions.patch_values.astype(value.dtype), value
        )

    x = patch(x, 0)
    raw_residuals = [x]
    attention_residuals, attention_updates, mlp_updates, patterns = [], [], [], []
    for index, block in enumerate(encoder.blocks):
        attention_input = block.single_attention.single_norm(x)
        if block.adaLN_modulation is not None:
            block_condition = condition * interventions.block_condition_scale[index].astype(
                condition.dtype
            )
            attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate = (
                jnp.split(block.adaLN_modulation(block_condition), 6, axis=-1)
            )
            attention_input = (
                attention_input * (1 + attn_scale[:, None]) + attn_shift[:, None]
            )
        attention_update, attention_pattern = _attention(
            block.single_attention,
            attention_input,
            token_mask,
            interventions.head_scale[index],
            capture_attention,
        )
        attention_update = block.single_attention_post_norm(attention_update)
        if block.adaLN_modulation is not None:
            attention_update = attn_gate[:, None] * attention_update
        attention_update = attention_update * interventions.attention_scale[index].astype(
            attention_update.dtype
        )
        x = x + attention_update
        attention_residuals.append(x)
        attention_updates.append(attention_update)
        transition_input = block.single_transition_norm(x)
        if block.adaLN_modulation is not None:
            transition_input = (
                transition_input * (1 + mlp_scale[:, None]) + mlp_shift[:, None]
            )
        transition_update = block.single_transition_post_norm(
            _transition_with_preferred_acc(block.single_transition, transition_input)
        )
        if block.adaLN_modulation is not None:
            transition_update = mlp_gate[:, None] * transition_update
        transition_update = transition_update * interventions.mlp_scale[index].astype(
            transition_update.dtype
        )
        x = patch(x + transition_update, index + 1)
        raw_residuals.append(x)
        mlp_updates.append(transition_update)
        if capture_attention:
            patterns.append(attention_pattern)
    normalized = encoder.final_norm(x) if encoder.final_norm is not None else x
    final = normalized
    if encoder.final_adaLN_modulation is not None:
        shift, scale = jnp.split(encoder.final_adaLN_modulation(condition), 2, axis=-1)
        modulation_scale = interventions.final_modulation_scale.astype(scale.dtype)
        final = final * (1 + scale[:, None] * modulation_scale) + shift[:, None] * modulation_scale
    return EncoderTrace(
        final=final,
        final_normalized=normalized,
        raw=jnp.stack(raw_residuals),
        attention_residual=jnp.stack(attention_residuals),
        attention_update=jnp.stack(attention_updates),
        mlp_update=jnp.stack(mlp_updates),
        token_mask=token_mask,
        attention_patterns=jnp.stack(patterns) if capture_attention else None,
    )
