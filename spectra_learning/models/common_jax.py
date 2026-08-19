from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

jax.config.update("jax_default_matmul_precision", "highest")

Array = jax.Array


def resolve_jax_compute_dtype(name: str):
    value = name.lower()
    if value in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    return jnp.float32


def torch_to_jax(value: torch.Tensor | np.ndarray | Any) -> Array:
    if isinstance(value, torch.Tensor):
        return jnp.asarray(value.detach().cpu().numpy())
    return jnp.asarray(value)


def assign_param(param: nnx.Param, value: torch.Tensor | np.ndarray | Any) -> None:
    param[...] = torch_to_jax(value)


def load_param(param: nnx.Param, state_dict: dict[str, torch.Tensor], key: str) -> None:
    assign_param(param, state_dict[key])


def silu(x: Array) -> Array:
    return x * jax.nn.sigmoid(x)


def gelu(x: Array) -> Array:
    return jax.nn.gelu(x, approximate=False)


def activation_checkpoint_policy(mode: str):
    if mode == "selective":
        return jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    return None


def should_activation_checkpoint(
    *,
    mode: str,
    modules: tuple[str, ...],
    module: str,
    block_idx: int,
    every_n: int,
) -> bool:
    return mode != "none" and module in modules and block_idx % every_n == 0


class Linear(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        compute_dtype: Any = jnp.float32,
        init: str = "xavier_normal",
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.compute_dtype = compute_dtype
        self.matmul_precision = (
            jax.lax.Precision.DEFAULT
            if compute_dtype == jnp.bfloat16
            else None
        )
        weight_shape = (out_features, in_features)
        if init in {"gate", "zeros"}:
            weight = jnp.zeros(weight_shape, dtype=jnp.float32)
        elif init == "trunc_normal_fan_in":
            weight = (
                jax.random.truncated_normal(
                    rngs.params(),
                    -2.0,
                    2.0,
                    weight_shape,
                    dtype=jnp.float32,
                )
                / math.sqrt(in_features)
            )
        else:
            weight = rngs.params.normal(weight_shape, dtype=jnp.float32) * math.sqrt(
                2.0 / (in_features + out_features)
            )
        self.weight = nnx.Param(weight)
        self.bias = (
            nnx.Param(
                jnp.ones((out_features,), dtype=jnp.float32)
                if init == "gate"
                else jnp.zeros((out_features,), dtype=jnp.float32)
            )
            if bias
            else None
        )

    def __call__(self, x: Array) -> Array:
        y = jnp.matmul(
            x.astype(self.compute_dtype),
            jnp.swapaxes(self.weight[...].astype(self.compute_dtype), -1, -2),
            precision=self.matmul_precision,
        )
        if self.bias is not None:
            y = y + self.bias[...].astype(y.dtype)
        return y

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        load_param(self.weight, state_dict, f"{prefix}.weight")
        if self.bias is not None:
            load_param(self.bias, state_dict, f"{prefix}.bias")


class LayerNorm(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        self.eps = eps
        self.weight = (
            nnx.Param(jnp.ones((dim,), dtype=jnp.float32)) if affine else None
        )
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32)) if affine else None

    def __call__(self, x: Array) -> Array:
        x_float = x.astype(jnp.float32)
        mean = jnp.mean(x_float, axis=-1, keepdims=True)
        mean_square = jnp.mean(jnp.square(x_float), axis=-1, keepdims=True)
        var = jnp.maximum(mean_square - jnp.square(mean), 0.0)
        y = (x_float - mean) * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight[...] + self.bias[...]
        return y.astype(x.dtype)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        if self.weight is not None:
            load_param(self.weight, state_dict, f"{prefix}.weight")
            load_param(self.bias, state_dict, f"{prefix}.bias")


class RMSNorm(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        self.eps = eps
        self.weight = (
            nnx.Param(jnp.ones((dim,), dtype=jnp.float32)) if affine else None
        )

    def __call__(self, x: Array) -> Array:
        x_float = x.astype(jnp.float32)
        y = x_float * jax.lax.rsqrt(
            jnp.mean(jnp.square(x_float), axis=-1, keepdims=True) + self.eps
        )
        if self.weight is not None:
            y = y * self.weight[...]
        return y.astype(x.dtype)

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        if f"{prefix}.bias" in state_dict:
            raise ValueError(
                f"{prefix} is a LayerNorm checkpoint; RMSNorm requires a fresh run"
            )
        if self.weight is not None:
            load_param(self.weight, state_dict, f"{prefix}.weight")


class Embedding(nnx.Module):
    def __init__(self, num_embeddings: int, features: int) -> None:
        self.weight = nnx.Param(
            jnp.zeros((num_embeddings, features), dtype=jnp.float32)
        )

    def __call__(self, indices: Array) -> Array:
        return self.weight[...][indices]

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        load_param(self.weight, state_dict, f"{prefix}.weight")


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        *,
        compute_dtype: Any = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        rngs = nnx.Rngs(0) if rngs is None else rngs
        assert num_layers >= 2
        dims = [input_dim, *([hidden_dim] * (num_layers - 1)), output_dim]
        self.layers = nnx.List(
            [
                Linear(
                    dims[idx],
                    dims[idx + 1],
                    compute_dtype=compute_dtype,
                    rngs=rngs,
                )
                for idx in range(num_layers)
            ]
        )

    def __call__(self, x: Array) -> Array:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1:
                x = silu(x)
        return x

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        for idx, layer in enumerate(self.layers):
            layer.load_torch_state_dict(state_dict, f"{prefix}.{2 * idx}")


class Identity(nnx.Module):
    def __call__(self, x: Array) -> Array:
        return x

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        del state_dict, prefix


class Sequential(nnx.Module):
    def __init__(self, layers: list[Any]) -> None:
        self.layers = nnx.List(layers)

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x

    def load_torch_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        linear_idx = 0
        for layer in self.layers:
            if hasattr(layer, "load_torch_state_dict"):
                layer.load_torch_state_dict(state_dict, f"{prefix}.{linear_idx}")
                linear_idx += 2


def merge_visible_mask(valid_mask: Array | None, visible_mask: Array | None) -> Array | None:
    if visible_mask is not None and valid_mask is not None:
        return visible_mask & valid_mask
    return visible_mask if visible_mask is not None else valid_mask


def pair_mask(peak_mask: Array) -> Array:
    return peak_mask[:, :, None] & peak_mask[:, None, :]


def masked_fill(values: Array, mask: Array, fill_value: float) -> Array:
    return jnp.where(mask, values, jnp.asarray(fill_value, dtype=values.dtype))


def scaled_dot_product_attention(
    q: Array,
    k: Array,
    v: Array,
    *,
    attn_mask: Array | None = None,
    is_causal: bool = False,
    implementation: str | None = "xla",
) -> Array:
    query = jnp.swapaxes(q, -3, -2)
    key = jnp.swapaxes(k, -3, -2)
    value = jnp.swapaxes(v, -3, -2)
    kwargs: dict[str, Any] = {
        "is_causal": is_causal,
        "implementation": implementation,
    }
    if attn_mask is not None:
        if attn_mask.dtype == jnp.bool_:
            kwargs["mask"] = attn_mask
        else:
            kwargs["bias"] = attn_mask
    out = jax.nn.dot_product_attention(query, key, value, **kwargs)
    return jnp.swapaxes(out, -3, -2)
