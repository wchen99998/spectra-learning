"""Layerwise semantic probes of a joint single+pair fastmixer encoder.

The run (1B, single width 1536, pair width 640, 25 blocks, 47 intensity-ordered
peaks + CLS, MAE 0.7 + distogram 0.3) predates encoder changes current code
cannot load, so the GPU container runs the repository at 0736a5f, the commit
that landed its RMS post-norm blocks the day after launch, from a git worktree:

    git worktree add --detach artifacts/code_0736a5f 0736a5f
    PYTHONPATH=artifacts/code_0736a5f JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/modal_joint_fastmixer_probe.py stage
    .venv/bin/modal run scripts/modal_joint_fastmixer_probe.py --output-dir DIR

Every layer's single tokens are pooled (peak mean, CLS) and its pair map is
pooled like `modal_pair_semantic_probe.py`. Targets, evaluator, and binned
127-peak spectral controls are those of `modal_layerwise_semantic_probe.py`,
so numbers are directly comparable. All peaks are visible at inference
(48 positions), whereas training exposed at most 17 encoder positions.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal
import numpy as np

BUCKET = "gs://metal-repeater-411410-spectra-checkpoints"
RUN_ID = (
    "1b-pairmixer-rms-postnorm-v6e8x8-b2048-ga8-16-16-mae07-disto03-optfix-"
    "fp32-lr3e4-wd01-95h-noflags-longwait-20260729-014634"
)
STEP = 150_000
CHECKPOINT = f"{BUCKET}/skypilot/{RUN_ID}/checkpoints/orbax/{STEP}"
NAME = "joint_fastmixer_1b_150k"
OLD_TREE = Path("artifacts/code_0736a5f")
OLD_ROOT = Path("/root/old")
LIB = Path("/root/probe_lib/layerwise_probe.py")
VOLUME_ROOT = Path("/vol")
TSV_NAME = "MassSpecGym1.5.tsv"
PAIR_POOLS = ("pair_mean", "diag_mean", "cls_row", "cls_col", "cls_cls")
SINGLE_POOLS = ("single_mean", "single_cls")

volume = modal.Volume.from_name("spectra-layerwise-semantic-probe", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("libgomp1")
    .pip_install_from_pyproject("pyproject.toml")
    .uv_pip_install(
        "torch==2.13.0",
        index_url="https://download.pytorch.org/whl/cu130",
        extra_index_url="https://pypi.org/simple",
    )
    .uv_pip_install("jax[cuda13]==0.11.0", "flax==0.12.8", "orbax-checkpoint==0.12.4")
    .add_local_dir(
        str(OLD_TREE / "spectra_learning"), str(OLD_ROOT / "spectra_learning"), copy=True
    )
    .add_local_file(str(OLD_TREE / "pyproject.toml"), str(OLD_ROOT / "pyproject.toml"), copy=True)
    .add_local_file("scripts/modal_layerwise_semantic_probe.py", str(LIB), copy=True)
    .env(
        {
            "PYTHONPATH": str(OLD_ROOT),
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
)
app = modal.App("spectra-joint-fastmixer-probe")


def _old_config(raw: dict[str, Any]) -> Any:
    from ml_collections import config_dict

    return config_dict.ConfigDict(raw)


def _build_old_encoder(config: Any) -> Any:
    """Old-code encoder with every peak + CLS visible to the fastmixer."""
    from spectra_learning.models.factory_jax import build_model_from_config

    model = build_model_from_config(config)
    positions = int(config.num_peaks) + 1
    model.set_fastmixer_capacities(
        positions,
        model.pairmixer_fast_max_visible_tokens,
        model.pairmixer_fast_target_max_visible_tokens,
    )
    return model.encoder


def stage() -> None:
    """Run with PYTHONPATH pointing at the old worktree (see module docstring)."""
    import tempfile

    import etils.epath as epath
    from flax import nnx

    from spectra_learning.training.checkpointing_jax import restore_frozen_teacher_encoder

    raw = json.loads(
        epath.Path(f"{CHECKPOINT}/metadata/metadata").read_text()
    )["task_contract"]["config"]
    encoder = _build_old_encoder(_old_config(raw))
    restored = restore_frozen_teacher_encoder(CHECKPOINT, nnx.as_pure(nnx.state(encoder)))
    directory = Path(tempfile.mkdtemp())
    (directory / "config.json").write_text(json.dumps(raw))
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


def _prepare_inputs(tsv_path: Path, config: Any) -> dict[str, np.ndarray]:
    import pyarrow.csv as csv

    from spectra_learning.data.spectra import preprocess_peak_batch_numpy

    rows = csv.read_csv(
        tsv_path,
        parse_options=csv.ParseOptions(delimiter="\t"),
        convert_options=csv.ConvertOptions(
            include_columns=["mzs", "intensities", "precursor_mz"]
        ),
    ).to_pydict()
    count = len(rows["mzs"])
    num_peaks = int(config.num_peaks)
    out = {
        "peak_mz": np.empty((count, num_peaks), np.float32),
        "peak_intensity": np.empty((count, num_peaks), np.float32),
        "peak_valid_mask": np.empty((count, num_peaks), bool),
        "precursor_mz": np.asarray(rows["precursor_mz"], np.float32),
    }
    for start in range(0, count, 4096):
        end = min(start + 4096, count)
        mzs = [np.fromstring(v, sep=",", dtype=np.float32) for v in rows["mzs"][start:end]]
        ints = [
            np.fromstring(v, sep=",", dtype=np.float32)
            for v in rows["intensities"][start:end]
        ]
        spectra = np.zeros((end - start, 2, max(map(len, mzs))), np.float32)
        for i, (mz, inten) in enumerate(zip(mzs, ints, strict=True)):
            spectra[i, 0, : len(mz)] = mz
            spectra[i, 1, : len(inten)] = inten
        batch = preprocess_peak_batch_numpy(
            spectra,
            out["precursor_mz"][start:end],
            num_peaks=num_peaks,
            peak_drop_min_intensity=float(config.peak_drop_min_intensity),
            peak_ordering=str(config.peak_ordering),
            max_precursor_mz=float(config.max_precursor_mz),
            precursor_peak_exclusion_window_da=float(
                config.precursor_peak_exclusion_window_da
            ),
            min_peak_intensity=float(config.min_peak_intensity),
            peak_filtering=str(config.peak_filtering),
            grouped_peak_shoulder_da=float(config.grouped_peak_shoulder_da),
            grouped_peak_isotope_charges=tuple(config.grouped_peak_isotope_charges),
        )
        for key in out:
            out[key][start:end] = batch[key]
    return out


def _layer_features(encoder: Any, data: dict, batch_size: int):
    """bf16 CPU [layers, N, D] per pool; layer 0 = embeddings entering block 1,
    1..L = raw block outputs, L+1 = final (normed) encoder outputs."""
    import jax
    import jax.numpy as jnp
    import torch
    from flax import nnx

    from spectra_learning.models.pairmixer_jax import (
        _active_indices,
        _gather_pair,
        _scatter_pair,
    )

    def pool_pair(pair, token_mask):
        pair = pair.astype(jnp.float32)
        tokens = pair.shape[1]
        peak = token_mask.at[:, -1].set(False)
        off_diagonal = peak[:, :, None] & peak[:, None, :] & ~jnp.eye(tokens, dtype=bool)[None]

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

    def pool_single(x, token_mask):
        x = x.astype(jnp.float32)
        peak = token_mask.at[:, -1].set(False)[..., None].astype(jnp.float32)
        return jnp.stack([(x * peak).sum(1) / jnp.maximum(peak.sum(1), 1.0), x[:, -1]])

    @nnx.jit
    def encode(enc, mz, intensity, valid, precursor):
        # Mirrors old PeakSetEncoder.forward_with_pair (fastmixer branch,
        # no metadata conditioning), recording every block.
        x = enc._add_positions(enc.embedder(mz, intensity))
        z = enc.pair_embedder(mz, intensity, x, valid, precursor_mz=precursor)
        x = enc._append_cls_token(x, None)
        z = enc._append_cls_pair_tokens(z)
        token_mask = enc._append_cls_mask(valid)
        pair_mask = (token_mask[:, :, None] & token_mask[:, None, :])[..., None]
        idx, compact_mask = _active_indices(token_mask, enc.pairmixer_fast_max_visible_tokens)
        dense_shape = z.shape
        singles, pairs = [x], [z]
        z = _gather_pair(z, idx)
        for block in enc.blocks:
            x, z = block.fastmixer_compact_call(x, z, idx, compact_mask, token_mask)
            singles.append(x)
            pairs.append(_scatter_pair(z, idx, dense_shape))
        final_x = enc.final_norm(x) if enc.final_norm is not None else x
        final_z = enc.final_pair_norm(z) if enc.final_pair_norm is not None else z
        final_z = _scatter_pair(final_z, idx, dense_shape) * pair_mask.astype(final_z.dtype)
        singles.append(final_x)
        pairs.append(final_z)
        single_pools = jnp.stack([pool_single(s, token_mask) for s in singles])
        pair_pools = jnp.stack(
            [pool_pair(p * pair_mask.astype(p.dtype), token_mask) for p in pairs]
        )
        return (
            single_pools.astype(jnp.bfloat16),
            pair_pools.astype(jnp.bfloat16),
            final_x,
            final_z,
        )

    @nnx.jit
    def reference(enc, mz, intensity, valid, precursor):
        return enc.forward_with_pair(
            mz, intensity, valid_mask=valid, visible_mask=valid, precursor_mz=precursor
        )

    count = len(data["peak_mz"])
    pooled = None
    for start in range(0, count, batch_size):
        take = min(batch_size, count - start)
        args = [
            jnp.asarray(
                np.pad(
                    data[key][start : start + take],
                    ((0, batch_size - take),) + ((0, 0),) * (data[key].ndim - 1),
                )
            )
            for key in ("peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz")
        ]
        single_pools, pair_pools, final_x, final_z = encode(encoder, *args)
        if start == 0:
            ref_x, ref_z = reference(encoder, *args)
            # Separately compiled bf16 graphs fuse differently; compare mean error.
            report = {}
            for name, ours, ref in (("single", final_x, ref_x), ("pair", final_z, ref_z)):
                scale = jnp.abs(ref).mean()
                report[name] = (
                    float(jnp.abs(ours - ref).mean() / scale),
                    float(jnp.abs(ours - ref).max() / scale),
                )
            print(f"capture vs forward_with_pair (mean, max) relative diff: {report}")
            assert all(mean < 1e-2 for mean, _ in report.values()), report
        # [layers, pools, batch, D] per stream.
        streams = [
            (SINGLE_POOLS, np.asarray(jax.device_get(single_pools).astype(np.float32))),
            (PAIR_POOLS, np.asarray(jax.device_get(pair_pools).astype(np.float32))),
        ]
        if pooled is None:
            pooled = {
                name: torch.empty((values.shape[0], count, values.shape[-1]), dtype=torch.bfloat16)
                for pools, values in streams
                for name in pools
            }
        for pools, values in streams:
            for index, name in enumerate(pools):
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
    memory=256 * 1024,
    timeout=5 * 60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def probe_joint(targets: dict[str, np.ndarray], batch_size: int = 128) -> dict:
    os.chdir(OLD_ROOT)
    from flax import nnx

    spec = importlib.util.spec_from_file_location("layerwise_probe", LIB)
    lib = importlib.util.module_from_spec(spec)
    sys.modules["layerwise_probe"] = lib
    spec.loader.exec_module(lib)

    started = time.time()
    controls = np.load(VOLUME_ROOT / "controls_127.npz")
    evaluator = lib._Evaluator(
        targets, controls["fold"], controls["binned"].astype(np.float32), controls["precursor_mz"]
    )
    config = _old_config(json.loads((VOLUME_ROOT / NAME / "config.json").read_text()))
    data = _prepare_inputs(VOLUME_ROOT / TSV_NAME, config)
    encoder = _build_old_encoder(config)
    template = nnx.state(encoder)
    with np.load(VOLUME_ROOT / NAME / "encoder.npz") as archive:
        nnx.update(
            encoder,
            nnx.from_flat_state(
                (path, archive["/".join(str(part) for part in path)])
                for path, _value in nnx.to_flat_state(template)
            ),
        )
    pooled = _layer_features(encoder, data, batch_size)
    results: dict[str, Any] = {
        "encoder": NAME,
        "checkpoint": CHECKPOINT,
        "baselines": evaluator.baselines(),
        "layers": {},
    }
    num_layers = pooled[SINGLE_POOLS[0]].shape[0]
    for layer in range(num_layers):
        label = "embed" if layer == 0 else "final" if layer == num_layers - 1 else str(layer)
        results["layers"][label] = {
            name: evaluator.evaluate(values[layer]) for name, values in pooled.items()
        }
        print(
            NAME,
            label,
            {k: round(v["auc_maccs_macro"], 4) for k, v in results["layers"][label].items()},
            flush=True,
        )
    results["seconds"] = time.time() - started
    return results


@app.local_entrypoint()
def main(output_dir: str = "") -> None:
    from scripts.modal_export_jepa_fps_embeddings import _prepare_massspecgym_inputs
    from scripts.modal_layerwise_semantic_probe import (
        LOCAL_TSV,
        _binned_spectra,
        _molecule_targets,
    )
    from spectra_learning.config import load_config

    # 127-peak single-stream inputs define the shared spectral controls.
    control = _prepare_massspecgym_inputs(
        LOCAL_TSV, load_config("configs/1p8b_periodic_teacher_jepa.py")
    )
    local = Path(output_dir or ".") / "controls_127.npz"
    local.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        local,
        binned=_binned_spectra(control).astype(np.float16),
        precursor_mz=control["precursor_mz"],
        fold=control["fold"],
    )
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(local), "/controls_127.npz")
    result = probe_joint.remote(_molecule_targets(LOCAL_TSV))
    print(json.dumps({"encoder": result["encoder"], "seconds": result["seconds"]}))
    if output_dir:
        (Path(output_dir) / f"{NAME}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__" and sys.argv[1:] == ["stage"]:
    stage()
