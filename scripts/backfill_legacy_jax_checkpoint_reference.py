"""Backfill references for the pre-65f0314 1B metadata-AdaLN checkpoint.

That launch used normalized peak intensities and a 28-field acquisition vector.
The two precursor-intensity fields were removed while the run was still active,
before its launch worktree was committed.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from ml_collections import config_dict

from spectra_learning.models.common_jax import Linear
from spectra_learning.models.factory_jax import build_model_from_config
from spectra_learning.models.fastmixer_capacity import (
    pairmixer_fast_full_visible_tokens,
)
from spectra_learning.training.checkpointing_jax import (
    load_jax_checkpoint_reference,
    restore_jax_encoder_state,
)
from spectra_learning.training.pretrain_jax import (
    JAX_CHECKPOINT_REFERENCE_VERSION,
    _jax_checkpoint_reference_inputs,
)
from spectra_learning.training.storage import read_text, storage_join


LEGACY_METADATA_DIM = 28
PRECURSOR_INTENSITY_ZSCORE = jnp.asarray([0.25, -0.5], dtype=jnp.float32)
PRECURSOR_INTENSITY_PRESENT = jnp.asarray([1.0, 0.0], dtype=jnp.float32)


def legacy_reference_inputs(
    config: config_dict.ConfigDict,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    raw_input, encoder_input = _jax_checkpoint_reference_inputs(config)
    raw_input = {
        **raw_input,
        "precursor_intensity_zscore": PRECURSOR_INTENSITY_ZSCORE,
        "precursor_intensity_present": PRECURSOR_INTENSITY_PRESENT,
    }

    peak_intensity = encoder_input["peak_intensity"]
    peak_intensity /= jnp.maximum(
        peak_intensity.max(axis=1, keepdims=True),
        1e-8,
    )
    metadata = encoder_input["spectrum_metadata"]
    legacy_fields = jnp.stack(
        [PRECURSOR_INTENSITY_ZSCORE, PRECURSOR_INTENSITY_PRESENT],
        axis=-1,
    )
    encoder_input = {
        **encoder_input,
        "peak_intensity": peak_intensity,
        "spectrum_metadata": jnp.concatenate(
            [metadata[:, :10], legacy_fields, metadata[:, 10:]],
            axis=-1,
        ),
    }
    return raw_input, encoder_input


def build_legacy_model(config: config_dict.ConfigDict):
    model = build_model_from_config(config)
    model.encoder.metadata_embedder.linear0 = Linear(
        LEGACY_METADATA_DIM,
        int(config.encoder_metadata_condition_dim),
        compute_dtype=model.compute_dtype,
        rngs=nnx.Rngs(int(config.seed)),
    )
    if model.use_fastmixer:
        full_visible_tokens = pairmixer_fast_full_visible_tokens(config)
        model.set_fastmixer_capacities(
            full_visible_tokens,
            full_visible_tokens,
            model.num_peak_tokens,
        )
    return model


@nnx.jit
def encode_reference(
    encoder: nnx.Module,
    encoder_input: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    kwargs = {
        "valid_mask": encoder_input["peak_valid_mask"],
        "visible_mask": encoder_input["peak_valid_mask"],
        "precursor_mz": encoder_input["precursor_mz"],
        "spectrum_metadata": encoder_input["spectrum_metadata"],
    }
    if encoder.use_pair_path:
        single, pair = encoder.forward_with_pair(
            encoder_input["peak_mz"],
            encoder_input["peak_intensity"],
            **kwargs,
        )
        return {"single": single, "pair": pair}
    return {
        "single": encoder(
            encoder_input["peak_mz"],
            encoder_input["peak_intensity"],
            **kwargs,
        )
    }


def checkpoint_config(checkpoint: str) -> config_dict.ConfigDict:
    metadata = json.loads(
        read_text(storage_join(checkpoint, "metadata", "metadata"))
    )
    return config_dict.ConfigDict(metadata["task_contract"]["config"])


def run(checkpoint: str, *, save_reference: bool) -> dict[str, Any]:
    started = time.time()
    config = checkpoint_config(checkpoint)
    model = build_legacy_model(config)
    encoder_state = nnx.as_pure(nnx.state(model.encoder))
    restored = restore_jax_encoder_state(checkpoint, encoder_state)
    nnx.update(model.encoder, restored)

    raw_input, encoder_input = legacy_reference_inputs(config)
    encoder_output = jax.block_until_ready(
        encode_reference(model.encoder, encoder_input)
    )
    if save_reference:
        reference = {
            "version": jnp.asarray(
                JAX_CHECKPOINT_REFERENCE_VERSION,
                dtype=jnp.int32,
            ),
            "raw_input": raw_input,
            "encoder_input": encoder_input,
            "encoder_output": encoder_output,
        }
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(storage_join(checkpoint, "reference"), reference)

    stored = load_jax_checkpoint_reference(checkpoint)
    actual = jax.tree.map(
        np.asarray,
        encode_reference(
            model.encoder,
            {
                key: jnp.asarray(value)
                for key, value in stored["encoder_input"].items()
            },
        ),
    )
    for key, expected in stored["encoder_output"].items():
        np.testing.assert_allclose(
            expected,
            actual[key],
            rtol=1e-5,
            atol=1e-6,
        )

    return {
        "checkpoint": checkpoint,
        "reference": str(storage_join(checkpoint, "reference")),
        "reference_version": int(stored["version"]),
        "encoder_input_shapes": jax.tree.map(
            lambda value: list(value.shape),
            stored["encoder_input"],
        ),
        "encoder_output_shapes": jax.tree.map(
            lambda value: list(value.shape),
            stored["encoder_output"],
        ),
        "encoder_output_dtypes": jax.tree.map(
            lambda value: str(value.dtype),
            stored["encoder_output"],
        ),
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-reference", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.checkpoint, save_reference=args.save_reference),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
