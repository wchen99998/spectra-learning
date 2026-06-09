from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


TORCHAX_DEVICE_BACKENDS = {"jax", "torchax", "tpu"}


@dataclass(frozen=True)
class TorchAXMesh:
    mesh: Any
    axis_name: str
    device_count: int

    @property
    def enabled(self) -> bool:
        return self.device_count > 1


def device_backend_name(value: Any) -> str:
    return str(value).lower()


def use_torchax_backend(config: Any) -> bool:
    backend = device_backend_name(_config_get(config, "device_backend", "auto"))
    return backend in TORCHAX_DEVICE_BACKENDS or bool(
        _config_get(config, "use_torchax", False)
    )


def enable_torchax() -> None:
    import torchax

    torchax.enable_globally()


def torchax_device() -> torch.device:
    enable_torchax()
    return torch.device("jax")


def is_torchax_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and callable(getattr(value, "jax", None))


def tensor_to_portable_cpu(value: torch.Tensor) -> torch.Tensor:
    if is_torchax_tensor(value):
        import jax

        array = jax.device_get(value.jax())
        return torch.as_tensor(np.array(array, copy=True)).detach()
    return value.detach().to("cpu", copy=True)


def initialize_torchax_distributed(config: Any) -> None:
    import jax

    if jax.distributed.is_initialized():
        return
    enabled = bool(_config_get(config, "torchax_distributed_initialize", False)) or any(
        os.environ.get(key)
        for key in (
            "JAX_DISTRIBUTED_INITIALIZE",
            "JAX_COORDINATOR_ADDRESS",
            "JAX_COORDINATOR_ADDR",
            "JAX_NUM_PROCESSES",
            "JAX_PROCESS_COUNT",
        )
    )
    if not enabled:
        return
    coordinator_address = _config_or_env(
        config,
        "torchax_coordinator_address",
        ("JAX_COORDINATOR_ADDRESS", "JAX_COORDINATOR_ADDR"),
    )
    num_processes = _optional_int(
        _config_or_env(
            config,
            "torchax_num_processes",
            ("JAX_NUM_PROCESSES", "JAX_PROCESS_COUNT", "WORLD_SIZE"),
        )
    )
    process_id = _optional_int(
        _config_or_env(
            config,
            "torchax_process_id",
            ("JAX_PROCESS_ID", "JAX_PROCESS_INDEX", "RANK"),
        )
    )
    local_device_ids = _local_device_ids(
        _config_or_env(
            config,
            "torchax_local_device_ids",
            ("JAX_LOCAL_DEVICE_IDS", "LOCAL_DEVICE_IDS"),
        )
    )
    cluster_detection_method = _config_or_env(
        config,
        "torchax_cluster_detection_method",
        ("JAX_CLUSTER_DETECTION_METHOD",),
    )
    initialization_timeout = int(
        _config_or_env(
            config,
            "torchax_initialization_timeout",
            ("JAX_INITIALIZATION_TIMEOUT",),
            300,
        )
    )
    kwargs = {
        key: value
        for key, value in {
            "coordinator_address": coordinator_address or None,
            "num_processes": num_processes,
            "process_id": process_id,
            "local_device_ids": local_device_ids,
            "cluster_detection_method": cluster_detection_method or None,
            "initialization_timeout": initialization_timeout,
        }.items()
        if value is not None
    }
    jax.distributed.initialize(**kwargs)


def build_torchax_mesh(config: Any) -> TorchAXMesh:
    import jax

    axis_name = str(_config_get(config, "torchax_mesh_axis", "data"))
    requested = _config_get(config, "torchax_mesh_devices", 1)
    devices = jax.devices()
    if str(requested).lower() == "all":
        mesh_devices = devices
    else:
        mesh_devices = devices[: int(requested)]
    mesh = jax.make_mesh((len(mesh_devices),), (axis_name,), devices=mesh_devices)
    return TorchAXMesh(mesh=mesh, axis_name=axis_name, device_count=len(mesh_devices))


def replicate_torchax_tree(tree: Any, mesh: TorchAXMesh) -> Any:
    if not mesh.enabled:
        return tree
    import jax
    import torchax.interop
    from jax.sharding import NamedSharding, PartitionSpec

    def replicate(value: Any) -> Any:
        if not is_torchax_tensor(value):
            return value
        sharding = NamedSharding(mesh.mesh, PartitionSpec())
        return torchax.interop.call_jax(jax.device_put, value, sharding)

    return jax.tree.map(replicate, tree)


def shard_torchax_batch(batch: dict[str, Any], mesh: TorchAXMesh) -> dict[str, Any]:
    if not mesh.enabled:
        return batch
    import jax
    import torchax.interop
    from jax.sharding import NamedSharding, PartitionSpec

    def shard(value: Any) -> Any:
        if not is_torchax_tensor(value) or value.ndim == 0:
            return value
        spec = PartitionSpec(mesh.axis_name, *([None] * (value.ndim - 1)))
        sharding = NamedSharding(mesh.mesh, spec)
        if jax.process_count() > 1:
            assert (value.shape[0] * jax.process_count()) % mesh.device_count == 0
            return torchax.interop.call_jax(
                jax.make_array_from_process_local_data,
                sharding,
                value,
            )
        assert value.shape[0] % mesh.device_count == 0
        return torchax.interop.call_jax(
            jax.device_put,
            value,
            sharding,
        )

    return {key: shard(value) for key, value in batch.items()}


def place_torchax_tensor_like(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not is_torchax_tensor(reference):
        return value.to(reference.device)
    import jax
    import torchax.interop

    return torchax.interop.call_jax(
        jax.device_put,
        value.to("jax"),
        reference.jax().sharding,
    )


def synchronize_torchax() -> None:
    import jax

    jax.effects_barrier()


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_or_env(
    config: Any,
    key: str,
    env_keys: tuple[str, ...],
    default: Any = "",
) -> Any:
    value = _config_get(config, key, None)
    if value not in (None, ""):
        return value
    for env_key in env_keys:
        value = os.environ.get(env_key)
        if value not in (None, ""):
            return value
    return default


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "none", "None"):
        return None
    return int(value)


def _local_device_ids(value: Any) -> tuple[int, ...] | int | None:
    if value in (None, "", "none", "None"):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return tuple(int(part) for part in value.split(",") if part)
    return tuple(int(part) for part in value)
