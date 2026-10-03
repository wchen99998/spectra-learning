"""Layerwise semantic probes of a pair-only encoder's pair representations.

Companion to `modal_layerwise_semantic_probe.py` (same targets, evaluator, and
binned-spectrum controls, taken from the 127-peak single-stream inputs). Each
layer's [32, 32, D] pair map (31 m/z-ordered peaks + CLS) is pooled as:

- pair_mean: mean over valid off-diagonal peak-peak pairs;
- diag_mean: mean over peak self-pairs;
- cls_row / cls_col: mean over peaks of CLS->peak / peak->CLS pairs;
- cls_cls: the CLS-CLS pair.

    JAX_PLATFORMS=cpu .venv/bin/modal run scripts/modal_pair_semantic_probe.py::stage
    .venv/bin/modal run scripts/modal_pair_semantic_probe.py::main --output-dir DIR
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import modal
import numpy as np

from scripts.modal_layerwise_semantic_probe import (
    BUCKET,
    LOCAL_TSV,
    PROJECT_ROOT,
    VOLUME_ROOT,
    _binned_spectra,
    _checkpoint_config,
    _Evaluator,
    _molecule_targets,
    image as base_image,
    volume,
)

RUN_ID = "pair279m-attn-n32-b2048-v6e32-2d-20260910-rootcause"
STEP = 730_000
CHECKPOINT = f"{BUCKET}/skypilot/{RUN_ID}/checkpoints/orbax/{STEP}"
NAME = "pair279m_730k"
# Single-stream staged config whose 127-peak inputs define the binned controls.
CONTROL_CONFIG = "ftjepa_1190k"
POOLS = ("pair_mean", "diag_mean", "cls_row", "cls_col", "cls_cls")
MAX_VISIBLE_TOKENS = 32

image = base_image.uv_pip_install(
    "jax[cuda13]==0.11.0", "flax==0.12.8", "orbax-checkpoint==0.12.4"
).env({"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
app = modal.App("spectra-pair-semantic-probe")


def _build_encoder(config: Any) -> Any:
    import jax.numpy as jnp
    from flax import nnx

    from spectra_learning.models.encoder_jax import PairOnlyEncoder
    from spectra_learning.models.settings import PeakSetJEPASettings

    config.activation_checkpoint_mode = "none"
    return PairOnlyEncoder(
        PeakSetJEPASettings.from_config(config),
        compute_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(config.seed),
    )


@app.local_entrypoint()
def stage() -> None:
    from flax import nnx
    from ml_collections import config_dict

    from spectra_learning.training.checkpointing_jax import restore_jax_encoder_state

    config = _checkpoint_config(CHECKPOINT)
    encoder = _build_encoder(config_dict.ConfigDict(config))
    template = nnx.as_pure(nnx.state(encoder))
    restored = restore_jax_encoder_state(CHECKPOINT, template)
    directory = Path(tempfile.mkdtemp())
    (directory / "config.json").write_text(json.dumps(config))
    np.savez(
        directory / "encoder.npz",
        **{
            "/".join(str(part) for part in path): np.asarray(value)
            for path, value in nnx.to_flat_state(restored)
        },
    )
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(directory), f"/{NAME}")
    print(f"staged {NAME}")


def _pooled_pair_features(encoder: Any, data: dict, config: Any, batch_size: int):
    """bf16 CPU tensors [layers, N, D] per pooling; layer 0 is the pair embedder
    output, 1..L raw block outputs, L+1 the final pair-normed output."""
    import jax
    import jax.numpy as jnp
    import torch
    from flax import nnx

    from spectra_learning.models import encoder_jax
    from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch

    # Record every block's pair tensor by wrapping the block loop the encoder calls.
    captured: list[Any] = []
    run_blocks = encoder_jax.run_pair_only_blocks

    def recording_run_blocks(blocks, pair, token_mask, condition, **kwargs):
        captured.append(pair)
        for block in blocks:
            pair = block(pair, token_mask, condition)
            captured.append(pair)
        return pair

    def pool(pair, token_mask):
        pair = pair.astype(jnp.float32)
        tokens = pair.shape[1]
        peak = token_mask.at[:, -1].set(False)
        off_diagonal = (
            peak[:, :, None] & peak[:, None, :] & ~jnp.eye(tokens, dtype=bool)[None]
        )

        def masked_mean(values, mask):
            weights = mask[..., None].astype(jnp.float32)
            axes = tuple(range(1, mask.ndim))
            return (values * weights).sum(axes) / jnp.maximum(weights.sum(axes), 1.0)

        diagonal = pair[:, jnp.arange(tokens), jnp.arange(tokens)]
        return jnp.stack(
            [
                masked_mean(pair, off_diagonal),
                masked_mean(diagonal, peak),
                masked_mean(pair[:, -1], peak),
                masked_mean(pair[:, :, -1], peak),
                pair[:, -1, -1],
            ]
        )

    @nnx.jit
    def encode(model, batch):
        captured.clear()
        encoder_jax.run_pair_only_blocks = recording_run_blocks
        try:
            final, _idx, token_mask = model(
                batch["peak_mz"],
                batch["peak_intensity"],
                valid_mask=batch["peak_valid_mask"],
                visible_mask=batch["peak_valid_mask"],
                max_visible_tokens=MAX_VISIBLE_TOKENS,
                precursor_mz=batch["precursor_mz"],
                spectrum_metadata=jax_spectrum_metadata_from_batch(
                    batch, schema=config.encoder_metadata_schema
                ),
            )
        finally:
            encoder_jax.run_pair_only_blocks = run_blocks
        return jnp.stack([pool(p, token_mask) for p in [*captured, final]]).astype(
            jnp.bfloat16
        )

    keys = (
        "peak_mz",
        "peak_intensity",
        "peak_valid_mask",
        "precursor_mz",
        "collision_energy",
        "collision_energy_present",
        "charge",
        "charge_present",
        "polarity_id",
        "instrument_family_id",
    )
    count = len(data["peak_mz"])
    pooled = None
    for start in range(0, count, batch_size):
        take = min(batch_size, count - start)
        batch = {
            key: jnp.asarray(
                np.pad(
                    data[key][start : start + take],
                    ((0, batch_size - take),) + ((0, 0),) * (data[key].ndim - 1),
                )
            )
            for key in keys
        }
        # [layers, pools, batch, D]
        values = np.asarray(jax.device_get(encode(encoder, batch)).astype(np.float32))
        if pooled is None:
            layers, _, _, dim = values.shape
            pooled = {
                name: torch.empty((layers, count, dim), dtype=torch.bfloat16)
                for name in POOLS
            }
        for index, name in enumerate(POOLS):
            pooled[name][:, start : start + take] = torch.from_numpy(
                values[:, index, :take]
            ).bfloat16()
        if (start // batch_size) % 100 == 0:
            print(f"embedded {start + take}/{count}", flush=True)
    return pooled


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=128 * 1024,
    timeout=4 * 60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def probe_pairs(targets: dict[str, np.ndarray], batch_size: int = 128) -> dict:
    os.chdir(PROJECT_ROOT)

    from flax import nnx
    from ml_collections import config_dict

    from scripts.modal_export_jepa_fps_embeddings import _prepare_massspecgym_inputs

    started = time.time()
    tsv = VOLUME_ROOT / LOCAL_TSV.name
    config = config_dict.ConfigDict(
        json.loads((VOLUME_ROOT / NAME / "config.json").read_text())
    )
    control_config = config_dict.ConfigDict(
        json.loads((VOLUME_ROOT / CONTROL_CONFIG / "config.json").read_text())
    )
    control = _prepare_massspecgym_inputs(tsv, control_config)
    data = _prepare_massspecgym_inputs(tsv, config)
    assert (data["fold"] == control["fold"]).all()
    evaluator = _Evaluator(
        targets, control["fold"], _binned_spectra(control), control["precursor_mz"]
    )

    encoder = _build_encoder(config)
    template = nnx.state(encoder)
    with np.load(VOLUME_ROOT / NAME / "encoder.npz") as archive:
        nnx.update(
            encoder,
            nnx.from_flat_state(
                (path, archive["/".join(str(part) for part in path)])
                for path, _value in nnx.to_flat_state(template)
            ),
        )
    pooled = _pooled_pair_features(encoder, data, config, batch_size)

    results: dict[str, Any] = {
        "encoder": NAME,
        "checkpoint": CHECKPOINT,
        "baselines": evaluator.baselines(),
        "layers": {},
    }
    num_layers = pooled[POOLS[0]].shape[0]
    for layer in range(num_layers):
        label = "embed" if layer == 0 else "final" if layer == num_layers - 1 else str(layer)
        results["layers"][label] = {
            name: evaluator.evaluate(pooled[name][layer]) for name in POOLS
        }
        print(
            NAME,
            label,
            {
                name: round(values["auc_maccs_macro"], 4)
                for name, values in results["layers"][label].items()
            },
            flush=True,
        )
    results["seconds"] = time.time() - started
    return results


@app.local_entrypoint()
def main(output_dir: str = "") -> None:
    result = probe_pairs.remote(_molecule_targets(LOCAL_TSV))
    print(json.dumps({"encoder": result["encoder"], "seconds": result["seconds"]}))
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{NAME}.json").write_text(json.dumps(result, indent=2))
