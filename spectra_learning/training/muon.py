from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from spectra_learning.models.common_jax import Array


_JAX_MUON_NS_COEFFICIENT_PRESETS = {
    "standard": (3.4445, -4.7750, 2.0315),
    "dion": (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ),
    "polar_express": tuple(
        (a / 1.05, b / 1.05**3, c / 1.05**5)
        for a, b, c in (
            (8.28721201814563, -23.595886519098837, 17.300387312530933),
            (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
            (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
            (3.3184196573706015, -2.488488024314874, 0.51004894012372),
            (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        )
    ),
}


def build_muon_transform(
    config: Any,
    *,
    learning_rate: Any,
    adam_learning_rate: Any,
    weight_decay_mask: Any,
) -> optax.GradientTransformation:
    if adam_learning_rate is None:
        adam_learning_rate = learning_rate
    eps = float(config.get("muon_eps", 1e-8))
    mu_dtype = config.get("muon_mu_dtype", None)
    nesterov = bool(config.get("muon_nesterov", True))
    configured_coeffs = config.get("muon_ns_coeffs", "standard")
    state_ns_coeffs = np.asarray(
        _JAX_MUON_NS_COEFFICIENT_PRESETS[configured_coeffs]
        if isinstance(configured_coeffs, str)
        else configured_coeffs,
        dtype=np.float32,
    )
    if state_ns_coeffs.ndim == 2:
        state_ns_coeffs = state_ns_coeffs[-int(config.get("muon_ns_steps", 5)) :]
    return _jax_split_qkv_transform(
        optax.partition(
            {
                "muon": optax.chain(
                    _jax_scale_by_gram_muon(
                        ns_coeffs=_jax_muon_ns_coefficients(config),
                        state_ns_coeffs=state_ns_coeffs,
                        beta=float(config.get("muon_beta", 0.95)),
                        eps=eps,
                        mu_dtype=mu_dtype,
                        nesterov=nesterov,
                        adaptive=bool(config.get("muon_adaptive", False)),
                        gram_min_aspect_ratio=float(
                            config.get("muon_gram_min_aspect_ratio", 2.0)
                        ),
                        gram_min_dimension=int(
                            config.get("muon_gram_min_dimension", 128)
                        ),
                        polar_express=(
                            config.get("muon_ns_coeffs", "standard")
                            == "polar_express"
                        ),
                        consistent_rms=_jax_muon_consistent_rms(config),
                    ),
                    optax.identity(),
                    optax.add_decayed_weights(
                        float(config.get("weight_decay", 0.0)),
                        mask=weight_decay_mask,
                    ),
                    optax.scale_by_learning_rate(learning_rate),
                ),
                "adam": optax.adamw(
                    learning_rate=adam_learning_rate,
                    b1=float(config.get("muon_adam_b1", 0.9)),
                    b2=float(
                        config.get("muon_adam_b2", config.get("b2", 0.999))
                    ),
                    eps=eps,
                    eps_root=float(config.get("muon_adam_eps_root", 0.0)),
                    weight_decay=float(
                        config.get("muon_adam_weight_decay", 0.0)
                    ),
                    mu_dtype=mu_dtype,
                    nesterov=nesterov,
                ),
            },
            _jax_muon_param_labels,
        )
    )


def _jax_muon_ns_coefficients(
    config: Any,
) -> tuple[tuple[float, float, float], ...]:
    configured_coeffs = config.get("muon_ns_coeffs", "standard")
    coeffs = np.asarray(
        _JAX_MUON_NS_COEFFICIENT_PRESETS[configured_coeffs]
        if isinstance(configured_coeffs, str)
        else configured_coeffs,
        dtype=np.float32,
    )
    ns_steps = int(config.get("muon_ns_steps", 5))
    if coeffs.ndim == 1:
        coeffs = np.repeat(coeffs[None], ns_steps, axis=0)
    return tuple(tuple(float(value) for value in row) for row in coeffs)


def _jax_scale_by_gram_muon(
    *,
    ns_coeffs: tuple[tuple[float, float, float], ...],
    state_ns_coeffs: Any,
    beta: float,
    eps: float,
    mu_dtype: Any,
    nesterov: bool,
    adaptive: bool,
    gram_min_aspect_ratio: float,
    gram_min_dimension: int,
    polar_express: bool,
    consistent_rms: float | None,
) -> optax.GradientTransformation:
    def init_fn(params):
        return optax.contrib.MuonState(
            count=jnp.zeros([], dtype=jnp.int32),
            mu=optax.tree.zeros_like(params, dtype=mu_dtype),
            ns_coeffs=jnp.asarray(state_ns_coeffs),
        )

    def update_fn(updates, state, params=None):
        del params
        mu = optax.tree.update_moment(updates, state.mu, beta, 1)
        count = optax.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree.map(
                lambda moment, grad: beta * moment + (1.0 - beta) * grad,
                optax.tree.bias_correction(
                    mu,
                    beta,
                    optax.safe_increment(count),
                ),
                optax.tree.bias_correction(updates, beta, count),
            )
        else:
            mu_hat = optax.tree.bias_correction(mu, beta, count)
        dimension_numbers = _jax_muon_weight_dimension_numbers(updates)

        def orthogonalize(momentum, dim_nums):
            orthogonalized = _jax_gram_muon_orthogonalize(
                momentum,
                ns_coeffs=ns_coeffs,
                eps=eps,
                gram_min_aspect_ratio=gram_min_aspect_ratio,
                gram_min_dimension=gram_min_dimension,
                polar_express=polar_express,
                dimension_numbers=dim_nums,
            )
            if adaptive:
                orthogonalized *= jnp.sum(
                    momentum.conj() * orthogonalized
                )
            fan_in, fan_out = _jax_muon_fan_in_out(momentum, dim_nums)
            scale = (
                math.sqrt(max(fan_in, fan_out)) * consistent_rms
                if consistent_rms is not None
                else math.sqrt(max(1.0, fan_out / fan_in))
            )
            return scale * orthogonalized

        updates = jax.tree.map(
            orthogonalize,
            mu_hat,
            dimension_numbers,
            is_leaf=_jax_is_muon_dimension_numbers,
        )
        if mu_dtype is not None:
            mu = optax.tree.cast(mu, mu_dtype)
        return updates, optax.contrib.MuonState(
            count=count,
            mu=mu,
            ns_coeffs=state.ns_coeffs,
        )

    return optax.GradientTransformation(init_fn, update_fn)


def _jax_gram_muon_orthogonalize(
    value: Array,
    *,
    ns_coeffs: tuple[tuple[float, float, float], ...],
    eps: float,
    gram_min_aspect_ratio: float,
    gram_min_dimension: int,
    polar_express: bool,
    dimension_numbers: optax.contrib.MuonDimensionNumbers,
) -> Array:
    reduction_axes = _jax_muon_axes(dimension_numbers.reduction_axis, value.ndim)
    output_axes = _jax_muon_axes(dimension_numbers.output_axis, value.ndim)
    batch_axes = tuple(
        axis
        for axis in range(value.ndim)
        if axis not in reduction_axes + output_axes
    )
    permutation = batch_axes + reduction_axes + output_axes
    inverse_permutation = tuple(
        sorted(range(value.ndim), key=permutation.__getitem__)
    )
    batch_shape = tuple(value.shape[axis] for axis in batch_axes)
    reduction_shape = tuple(value.shape[axis] for axis in reduction_axes)
    output_shape = tuple(value.shape[axis] for axis in output_axes)
    fan_in = math.prod(reduction_shape)
    fan_out = math.prod(output_shape)
    matrices = value.transpose(permutation).reshape(-1, fan_in, fan_out)
    matrices = matrices.astype(jnp.float32)
    matrices /= (
        jnp.linalg.norm(matrices, axis=(-2, -1), keepdims=True) + eps
    )
    transposed = fan_in > fan_out
    if transposed:
        matrices = jnp.swapaxes(matrices, -1, -2)
    aspect_ratio = max(fan_in, fan_out) / min(fan_in, fan_out)
    # Without symmetric GEMM kernels, standard NS is faster near square.
    if (
        aspect_ratio > gram_min_aspect_ratio
        and min(fan_in, fan_out) >= gram_min_dimension
    ):
        if polar_express:
            matrices = _jax_standard_newton_schulz(
                matrices,
                ns_coeffs[:3],
                ns_dtype=jnp.bfloat16,
            )
            matrices = _jax_gram_newton_schulz(
                matrices,
                ns_coeffs[3:],
                ns_dtype=jnp.bfloat16,
            )
        else:
            matrices = _jax_gram_newton_schulz(
                matrices,
                ns_coeffs,
                ns_dtype=jnp.bfloat16,
            )
    else:
        matrices = _jax_standard_newton_schulz(
            matrices,
            ns_coeffs,
            ns_dtype=jnp.bfloat16,
        )
    if transposed:
        matrices = jnp.swapaxes(matrices, -1, -2)
    return (
        matrices.astype(value.dtype)
        .reshape(batch_shape + reduction_shape + output_shape)
        .transpose(inverse_permutation)
    )


def _jax_standard_newton_schulz(
    matrices: Array,
    ns_coeffs: tuple[tuple[float, float, float], ...],
    *,
    ns_dtype: Any = jnp.float32,
) -> Array:
    for a, b, c in ns_coeffs:
        gram = _jax_muon_matmul(
            matrices,
            jnp.swapaxes(matrices, -1, -2),
            ns_dtype,
        )
        polynomial = b * gram + c * _jax_muon_matmul(
            gram,
            gram,
            ns_dtype,
        )
        matrices = a * matrices + _jax_muon_matmul(
            polynomial,
            matrices,
            ns_dtype,
        )
    return matrices


def _jax_gram_newton_schulz(
    matrices: Array,
    ns_coeffs: tuple[tuple[float, float, float], ...],
    *,
    ns_dtype: Any = jnp.float32,
) -> Array:
    apply_accumulated = (
        _jax_muon_matmul
        if len(ns_coeffs) <= 2
        else _jax_muon_apply_accumulated
    )
    gram = _jax_muon_matmul(
        matrices,
        jnp.swapaxes(matrices, -1, -2),
        ns_dtype,
    )
    identity = jnp.eye(gram.shape[-1], dtype=gram.dtype)
    accumulated = None
    for step, (a, b, c) in enumerate(ns_coeffs):
        # A three-step block limits BF16 drift in the accumulated transform.
        if step == 3:
            matrices = apply_accumulated(
                accumulated,
                matrices,
                ns_dtype,
            )
            gram = _jax_muon_matmul(
                matrices,
                jnp.swapaxes(matrices, -1, -2),
                ns_dtype,
            )
            accumulated = None
        polynomial = b * gram + c * _jax_muon_matmul(
            gram,
            gram,
            ns_dtype,
        )
        accumulated = (
            polynomial + a * identity
            if accumulated is None
            else _jax_muon_matmul(accumulated, polynomial, ns_dtype)
            + a * accumulated
        )
        if step < len(ns_coeffs) - 1 and step + 1 != 3:
            gram_polynomial = (
                _jax_muon_matmul(gram, polynomial, ns_dtype) + a * gram
            )
            gram = (
                _jax_muon_matmul(polynomial, gram_polynomial, ns_dtype)
                + a * gram_polynomial
            )
    return apply_accumulated(accumulated, matrices, ns_dtype)


def _jax_muon_matmul(lhs: Array, rhs: Array, dtype: Any) -> Array:
    return jnp.matmul(
        lhs.astype(dtype),
        rhs.astype(dtype),
        preferred_element_type=jnp.float32,
    )


def _jax_muon_apply_accumulated(
    accumulated: Array,
    matrices: Array,
    dtype: Any,
) -> Array:
    diagonal = jnp.diagonal(accumulated, axis1=-2, axis2=-1)
    off_diagonal = accumulated - (
        jnp.eye(accumulated.shape[-1], dtype=accumulated.dtype)
        * diagonal[..., None, :]
    )
    return diagonal[..., :, None] * matrices + _jax_muon_matmul(
        off_diagonal,
        matrices,
        dtype,
    )


def _jax_muon_axes(axes: Any, ndim: int) -> tuple[int, ...]:
    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    return tuple(axis % ndim for axis in axes)


def _jax_muon_fan_in_out(
    value: Array,
    dimension_numbers: optax.contrib.MuonDimensionNumbers,
) -> tuple[int, int]:
    reduction_axes = _jax_muon_axes(
        dimension_numbers.reduction_axis,
        value.ndim,
    )
    output_axes = _jax_muon_axes(
        dimension_numbers.output_axis,
        value.ndim,
    )
    return (
        math.prod(value.shape[axis] for axis in reduction_axes),
        math.prod(value.shape[axis] for axis in output_axes),
    )


def _jax_is_muon_dimension_numbers(value: Any) -> bool:
    return isinstance(value, optax.contrib.MuonDimensionNumbers)


def _jax_muon_consistent_rms(config: Any) -> float | None:
    adjust_lr_fn = str(config.get("muon_adjust_lr_fn", "") or "").lower()
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2
    return config.get("muon_consistent_rms", None)


def _jax_muon_param_labels(params: Any) -> Any:
    return jax.tree.map_with_path(
        lambda path, value: (
            "muon"
            if _tree_path_key(path[-1]) == "weight" and value.ndim >= 2
            else "adam"
        ),
        params,
    )


def _jax_muon_weight_dimension_numbers(params: Any) -> Any:
    return jax.tree.map_with_path(
        lambda path, value: (
            _jax_muon_weight_dimension_number(path, value)
            if _tree_path_key(path[-1]) == "weight" and value.ndim >= 2
            else None
        ),
        params,
    )


def _jax_muon_weight_dimension_number(
    path: tuple[Any, ...],
    value: Any,
) -> optax.contrib.MuonDimensionNumbers:
    if _jax_qkv_weight_path(path) and value.ndim == 3:
        return optax.contrib.MuonDimensionNumbers(reduction_axis=2, output_axis=1)
    return optax.contrib.MuonDimensionNumbers(reduction_axis=1, output_axis=0)


def _jax_split_qkv_transform(
    transform: optax.GradientTransformation,
) -> optax.GradientTransformation:
    def init_fn(params):
        return transform.init(_jax_split_qkv_tree(params))

    def update_fn(updates, state, params=None):
        split_updates = _jax_split_qkv_tree(updates)
        split_params = None if params is None else _jax_split_qkv_tree(params)
        split_updates, state = transform.update(split_updates, state, split_params)
        return _jax_unsplit_qkv_tree(split_updates, updates), state

    return optax.GradientTransformation(init_fn, update_fn)


def _jax_split_qkv_tree(tree: Any) -> Any:
    return jax.tree.map_with_path(_jax_split_qkv_leaf, tree)


def _jax_unsplit_qkv_tree(tree: Any, reference: Any) -> Any:
    return jax.tree.map_with_path(
        lambda path, value: _jax_unsplit_qkv_leaf(path, value, reference),
        tree,
    )


def _jax_split_qkv_leaf(path: tuple[Any, ...], value: Any) -> Any:
    if _jax_split_qkv_leaf_path(path, value):
        return value.reshape(3, value.shape[0] // 3, value.shape[1])
    return value


def _jax_unsplit_qkv_leaf(path: tuple[Any, ...], value: Any, reference: Any) -> Any:
    if _jax_split_qkv_leaf_path(path, _jax_tree_get_path(reference, path)):
        return value.reshape(value.shape[0] * value.shape[1], value.shape[2])
    return value


def _jax_split_qkv_leaf_path(path: tuple[Any, ...], value: Any) -> bool:
    return (
        _jax_qkv_weight_path(path)
        and value.ndim == 2
        and value.shape[0] % 3 == 0
    )


def _jax_qkv_weight_path(path: tuple[Any, ...]) -> bool:
    parts = tuple(str(_tree_path_key(path_entry)) for path_entry in path)
    return len(parts) >= 2 and parts[-1] == "weight" and parts[-2] in {"qkv", "wqkv"}


def _jax_tree_get_path(tree: Any, path: tuple[Any, ...]) -> Any:
    value = tree
    for path_entry in path:
        value = value[_tree_path_key(path_entry)]
    return value


def _tree_path_key(path_entry: Any) -> Any:
    return getattr(path_entry, "key", path_entry)
