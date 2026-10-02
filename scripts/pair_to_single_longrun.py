"""Longer matched pair-to-single pilots with resumable training snapshots.

Use the preserved pilot backbone snapshot. Intermediate exports contain only
probe train/validation spectra; the prespecified final step also exports test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from scripts.pair_to_single_pilot import build_model, encode, mask_batch, step, validation


def save_state(module: nnx.Module, path: Path) -> None:
    arrays = {"/".join(map(str, key)): np.asarray(value)
              for key, value in nnx.to_flat_state(nnx.state(module))}
    np.savez(path, **arrays)


def load_state(module: nnx.Module, path: Path) -> None:
    with np.load(path) as saved:
        nnx.update(module, nnx.from_flat_state(
            (key, jnp.asarray(saved["/".join(map(str, key))]))
            for key, _ in nnx.to_flat_state(nnx.state(module))))


def make_optimizer(model: nnx.Module, total_steps: int) -> nnx.Optimizer:
    schedule = optax.warmup_cosine_decay_schedule(
        0, 3e-4, 100, total_steps, end_value=3e-5)
    return nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0),
                         optax.adamw(schedule, b2=.95, weight_decay=.01)), wrt=nnx.Param)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def snapshot(model, optimizer, data, rng, destination, metadata, *, final):
    destination.mkdir(parents=True, exist_ok=True)
    save_state(model, destination / "model.npz")
    save_state(optimizer, destination / "optimizer.npz")
    output = {**metadata, "rng_state": rng.bit_generator.state, "validation": {}}
    for condition in ("intact", "zero", "shuffle"):
        if metadata["arm"] == "bias" and condition != "intact":
            continue
        for block in model.encoder.blocks:
            block.intervention = condition
        val_rng = np.random.default_rng(42)
        values = []
        for start in range(0, len(data["probe_val"]), metadata["batch_size"]):
            idx = data["probe_val"][start:start + metadata["batch_size"]]
            metrics = validation(model, mask_batch(data, idx, val_rng))
            values.append({key: float(metrics[key]) for key in
                           ("loss", "mae_mz_loss", "distogram_loss")})
        output["validation"][condition] = {
            key: float(np.mean([row[key] for row in values])) for key in values[0]}
    for block in model.encoder.blocks:
        block.intervention = "intact"
    folds = ("train", "val", "test") if final else ("train", "val")
    indices = np.concatenate([data[f"probe_{fold}"] for fold in folds])
    features = []
    for start in range(0, len(indices), metadata["batch_size"]):
        idx = indices[start:start + metadata["batch_size"]]
        batch = {key: jnp.asarray(data[key][idx]) for key in
                 ("peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz")}
        features.append(np.asarray(encode(model, batch)))
    np.savez(destination / "embeddings.npz", indices=indices, features=np.concatenate(features))
    output["exported_folds"] = list(folds)
    write_json(destination / "training.json", output)
    print(json.dumps({"checkpoint": str(destination), "step": metadata["steps"],
                      "validation": output["validation"]}), flush=True)


def train(args):
    if args.steps <= 100 or not args.checkpoints or args.checkpoints[-1] != args.steps:
        raise ValueError("Steps must exceed warmup; checkpoints must end at the final step")
    if sorted(set(args.checkpoints)) != args.checkpoints or args.checkpoints[0] <= 0:
        raise ValueError("Checkpoints must be positive, increasing, and unique")
    data = dict(np.load(args.inputs))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=args.resume is not None)
    model = build_model(args.arm, args.seed)
    optimizer = make_optimizer(model, args.steps)
    rng = np.random.default_rng(args.seed + 77)
    train_idx = np.flatnonzero((data["fold"] == "train") & (data["peak_valid_mask"].sum(1) >= 4))
    history, start_step, training_seconds = [], 0, 0.0
    if args.resume is not None:
        previous = json.loads((args.resume / "training.json").read_text())
        for key in ("arm", "seed", "batch_size"):
            if previous[key] != getattr(args, key):
                raise ValueError(f"Resume checkpoint has a different {key}")
        if previous["schedule_steps"] != args.steps:
            raise ValueError("Resume must preserve the original learning-rate schedule")
        load_state(model, args.resume / "model.npz")
        load_state(optimizer, args.resume / "optimizer.npz")
        rng.bit_generator.state = previous["rng_state"]
        start_step, history = previous["steps"], previous["history"]
        training_seconds = previous["train_seconds"]
    total_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    chunk_started = time.perf_counter()
    for i in range(start_step, args.steps):
        idx = rng.choice(train_idx, args.batch_size, replace=False)
        metrics = step(model, optimizer, mask_batch(data, idx, rng))
        if i == start_step or (i + 1) % 250 == 0:
            row = {key: float(metrics[key]) for key in ("loss", "mae_mz_loss", "distogram_loss")}
            if not all(np.isfinite(value) for value in row.values()):
                raise FloatingPointError(f"Nonfinite training metrics at step {i + 1}")
            row.update(step=i + 1, train_seconds=training_seconds + time.perf_counter() - chunk_started)
            history.append(row)
            write_json(destination / "progress.json", row)
            print(json.dumps(row), flush=True)
        if i + 1 in args.checkpoints:
            jax.block_until_ready(metrics)
            training_seconds += time.perf_counter() - chunk_started
            metadata = {"arm": args.arm, "seed": args.seed, "steps": i + 1,
                        "schedule_steps": args.steps, "batch_size": args.batch_size,
                        "use_cls_token": True, "params": total_params,
                        "train_seconds": training_seconds, "train_examples": len(train_idx),
                        "sampled_epoch_equivalents": (i + 1) * args.batch_size / len(train_idx),
                        "jax_version": jax.__version__, "device": str(jax.devices()[0]),
                        "history": history}
            snapshot(model, optimizer, data, rng, destination / f"step_{i + 1:06d}",
                     metadata, final=i + 1 == args.steps)
            chunk_started = time.perf_counter()
    write_json(destination / "completed.json", {"steps": args.steps, "arm": args.arm, "seed": args.seed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=("bias", "bias_poolnorm"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[3000, 7500, 15000])
    parser.add_argument("--resume", type=Path)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
