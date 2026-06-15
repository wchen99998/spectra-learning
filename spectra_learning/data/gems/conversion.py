from typing import Any

import numpy as np
import torch


def _numpy_to_torch(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_numpy_to_torch(item) for item in value.tolist()]
        if value.dtype.kind in {"U", "S"}:
            return value.tolist()
        if not value.flags.c_contiguous or not value.flags.writeable:
            value = value.copy()
        return torch.from_numpy(value)
    if isinstance(value, list):
        return [_numpy_to_torch(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _torch_to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _torch_to_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_torch_to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_torch_to_numpy(item) for item in value)
    return value


def _to_jax(value: Any) -> Any:
    import jax.numpy as jnp

    if isinstance(value, torch.Tensor):
        return jnp.asarray(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "U", "S"}:
            return value.tolist()
        return jnp.asarray(value)
    if isinstance(value, np.generic):
        return jnp.asarray(value)
    if isinstance(value, dict):
        return {key: _to_jax(item) for key, item in value.items()}
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return value
        return [_to_jax(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_jax(item) for item in value)
    return value


def numpy_batch_to_torch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: _numpy_to_torch(value) for key, value in batch.items()}


def batch_to_numpy(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: _torch_to_numpy(value) for key, value in batch.items()}


def batch_to_jax(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: _to_jax(value) for key, value in batch.items()}


def format_batch(batch: dict[str, Any], output_format: str) -> dict[str, Any]:
    if output_format == "torch":
        return batch
    if output_format == "numpy":
        return batch_to_numpy(batch)
    if output_format == "jax":
        return batch_to_jax(batch)
    raise ValueError(f"Unknown dataloader output format: {output_format}")
