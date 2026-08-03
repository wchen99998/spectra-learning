import argparse
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
from flax.traverse_util import flatten_dict, unflatten_dict

from spectra_learning.config import load_config
from spectra_learning.models.factory import build_model_from_config
from spectra_learning.models.settings import PeakSetJEPASettings


def torch_key(path: tuple[str, ...]) -> str:
    parts = list(path)
    index = 0
    while index < len(parts):
        if parts[index] == "layers":
            parts.pop(index)
            parts[index] = str(2 * int(parts[index]))
        elif (
            parts[index]
            in {"raw_ffn", "raw_proj", "single_pair_proj", "target_projector"}
            and index + 1 < len(parts)
            and parts[index + 1].isdigit()
        ):
            parts[index + 1] = str(2 * int(parts[index + 1]))
        index += 1
    return ".".join(parts)


def export_generator(
    checkpoint_path: str,
    config_path: str,
    output_path: Path,
) -> None:
    config = load_config(config_path)
    with torch.device("meta"):
        torch_model = build_model_from_config(config)
    torch_shapes = {
        key: (tuple(value.shape), value.dtype)
        for key, value in torch_model.state_dict().items()
    }

    state_path = f"{checkpoint_path.rstrip('/')}/state"
    with ocp.StandardCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(state_path).item_metadata
    source_metadata = {
        root: flatten_dict(metadata[root])
        for root in ("trainable_params", "static_state")
    }
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    target: dict[str, dict] = {}
    restore_args: dict[str, dict] = {}
    source_to_torch: dict[tuple[str, tuple[str, ...]], str] = {}
    skipped_source_paths: set[tuple[str, tuple[str, ...]]] = set()
    for root, leaves in source_metadata.items():
        target_leaves = {}
        restore_leaves = {}
        for source_path, leaf in leaves.items():
            key = torch_key(tuple(str(part) for part in source_path))
            if key not in torch_shapes:
                skipped_source_paths.add((root, source_path))
                continue
            shape, dtype = torch_shapes[key]
            if (
                shape != tuple(leaf.shape)
                or dtype != torch.float32
                or leaf.dtype != jnp.float32
            ):
                raise ValueError(f"Checkpoint tensor mismatch at {key}.")
            target_leaves[source_path] = jax.ShapeDtypeStruct(
                shape,
                jnp.float32,
                sharding=sharding,
            )
            restore_leaves[source_path] = ocp.ArrayRestoreArgs(
                sharding=sharding,
                global_shape=shape,
                dtype=jnp.float32,
            )
            source_to_torch[(root, source_path)] = key
        target[root] = unflatten_dict(target_leaves)
        restore_args[root] = unflatten_dict(restore_leaves)
    if set(source_to_torch.values()) != torch_shapes.keys():
        raise ValueError("Checkpoint does not contain the complete PyTorch model.")
    expected_skipped_paths = {
        (
            "static_state",
            ("masked_latent_predictor", str(index), "attention", name),
        )
        for index in range(int(config.masked_latent_predictor_num_layers))
        for name in ("rope_cos", "rope_sin")
    }
    if skipped_source_paths != expected_skipped_paths:
        raise ValueError("Checkpoint contains unexpected model tensors.")

    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            state_path,
            args=ocp.args.PyTreeRestore(
                item=target,
                restore_args=restore_args,
                partial_restore=True,
            ),
        )
    state_dict = {}
    for root in ("trainable_params", "static_state"):
        for source_path, value in flatten_dict(restored[root]).items():
            key = source_to_torch[(root, source_path)]
            state_dict[key] = torch.tensor(
                np.asarray(value),
                dtype=torch.float32,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    torch.save(
        {
            "format_version": 1,
            "model": state_dict,
            "settings": asdict(PeakSetJEPASettings.from_config(config)),
            "source_checkpoint": checkpoint_path.rstrip("/"),
        },
        temporary_path,
    )
    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_generator(args.checkpoint, args.config, args.output)


if __name__ == "__main__":
    main()
