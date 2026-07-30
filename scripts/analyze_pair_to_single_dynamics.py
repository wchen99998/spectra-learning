from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch
from flax import nnx

from spectra_learning.config import load_config
from spectra_learning.data.gems.datamodule import GemsDataModule
from spectra_learning.data.gems.mask_schedule import jepa_mask_stage
from spectra_learning.models.common_jax import merge_visible_mask
from spectra_learning.models.factory_jax import build_model_from_config
from spectra_learning.models.pairmixer_jax import (
    _active_indices,
    _dot_precision,
    _gather_pair,
    _gather_single,
    _linear_with_preferred_acc,
    _preferred_acc_dtype,
    _transition_with_preferred_acc,
)
from spectra_learning.models.spectrum_metadata import (
    jax_spectrum_metadata_from_batch,
)
from spectra_learning.training.checkpointing_jax import (
    build_jax_checkpoint_manager,
    restore_frozen_teacher_encoder,
)
from spectra_learning.training.storage import (
    normalize_storage_path,
    storage_join,
)


def _masked_rms(x: jax.Array, mask: jax.Array) -> jax.Array:
    weights = mask[..., None].astype(jnp.float32)
    return jnp.sqrt(
        (jnp.square(x.astype(jnp.float32)) * weights).sum()
        / (weights.sum() * x.shape[-1])
    )


def _masked_cosine(a: jax.Array, b: jax.Array, mask: jax.Array) -> jax.Array:
    weights = mask[..., None].astype(jnp.float32)
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    return (a * b * weights).sum() / jnp.sqrt(
        (jnp.square(a) * weights).sum() * (jnp.square(b) * weights).sum()
    )


def _analyze_encoder(encoder, batch: dict[str, jax.Array], capacity: int):
    peak_valid_mask = batch["peak_valid_mask"]
    peak_visible_mask = merge_visible_mask(peak_valid_mask, batch["context_mask"])
    spectrum_metadata = jax_spectrum_metadata_from_batch(batch)

    single = encoder.embedder(batch["peak_mz"], batch["peak_intensity"])
    metadata_embedding = encoder._metadata_embedding(spectrum_metadata, single.dtype)
    single = single + metadata_embedding[:, None, :].astype(single.dtype)
    single = encoder._add_positions(single)
    pair = encoder.pair_embedder(
        batch["peak_mz"],
        batch["peak_intensity"],
        single,
        peak_visible_mask,
        precursor_mz=batch["precursor_mz"],
    )
    single = encoder._append_cls_token(single, metadata_embedding)
    pair = encoder._append_cls_pair_tokens(pair)
    token_mask = encoder._append_cls_mask(peak_visible_mask)
    idx, compact_mask = _active_indices(token_mask, capacity)
    single = _gather_single(single, idx)
    pair = _gather_pair(pair, idx)
    single = single * compact_mask[..., None].astype(single.dtype)

    no_pair_single = single
    cls_mask = compact_mask & (idx == batch["peak_mz"].shape[1])
    pair_mask = compact_mask[:, :, None] & compact_mask[:, None, :]
    rows: dict[str, list[jax.Array]] = {
        key: []
        for key in (
            "pair_rms",
            "pair_delta_over_input",
            "pair_bias_centered_rms",
            "qk_logit_rms",
            "qk_norm_logit_rms",
            "pair_to_qk_logit_ratio",
            "attention_pair_total_variation",
            "attention_entropy_frac",
            "qk_norm_attention_pair_total_variation",
            "qk_norm_attention_entropy_frac",
            "qk_norm_no_pair_attention_entropy_frac",
            "qk_norm_attention_max_prob",
            "qk_norm_no_pair_attention_max_prob",
            "qk_norm_attention_fraction_max_prob_gt_0_9",
            "q_rms",
            "k_rms",
            "single_attention_norm_gain_rms",
            "single_attention_norm_gain_max",
            "single_rms",
            "attention_rms",
            "ffn_rms",
            "single_transition_norm_gain_rms",
            "single_transition_norm_gain_max",
            "attention_over_single",
            "ffn_over_single",
            "total_update_over_single",
            "local_pair_effect_over_single",
            "local_pair_effect_fraction_of_attention",
            "local_pair_effect_cls_over_single",
            "local_attention_no_pair_cosine",
            "cumulative_no_pair_cosine",
            "cumulative_no_pair_cls_cosine",
            "cumulative_pair_effect_over_single",
        )
    }

    for block in encoder.blocks:
        single_in = single
        pair_in = pair
        single, pair = block.fastmixer_compact_only_call(
            single_in,
            pair_in,
            compact_mask,
        )

        attention = block.single_attention_post_norm(
            block._fast_attention_pair_bias_compact(
                single_in,
                pair,
                compact_mask,
            )
        )
        no_pair_attention_local = block.single_attention_post_norm(
            block._fast_attention_pair_bias_compact(
                single_in,
                jnp.zeros_like(pair),
                compact_mask,
            )
        )
        pair_effect = attention - no_pair_attention_local
        ffn = single - single_in - attention

        no_pair_attention = block.single_attention_post_norm(
            block._fast_attention_pair_bias_compact(
                no_pair_single,
                jnp.zeros_like(pair),
                compact_mask,
            )
        )
        no_pair_single = no_pair_single + no_pair_attention
        no_pair_single = no_pair_single + block.single_transition_post_norm(
            _transition_with_preferred_acc(
                block.single_transition,
                block.single_transition_norm(no_pair_single),
            )
        )
        no_pair_single = no_pair_single * compact_mask[..., None].astype(
            no_pair_single.dtype
        )

        pair_bias = _linear_with_preferred_acc(
            block.single_attention.pair_bias,
            block.single_attention.pair_norm(pair),
        )
        attention_module = block.single_attention
        single_norm = attention_module.single_norm(single_in)
        qkv = _linear_with_preferred_acc(
            attention_module.qkv,
            single_norm,
        ).reshape(
            single_in.shape[0],
            single_in.shape[1],
            3,
            attention_module.num_heads,
            attention_module.head_dim,
        )
        q, k, _v = jnp.moveaxis(qkv, 2, 0)
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        qk_logits = jnp.einsum(
            "...qd,...kd->...qk",
            q,
            k,
            precision=_dot_precision(q.dtype),
            preferred_element_type=_preferred_acc_dtype(q.dtype),
        ).astype(jnp.float32) / math.sqrt(attention_module.head_dim)
        pair_logits = jnp.transpose(pair_bias.astype(jnp.float32), (0, 3, 1, 2))
        attention_mask = compact_mask[:, None, None, :]
        full_attention = jax.nn.softmax(
            jnp.where(attention_mask, qk_logits + pair_logits, -jnp.inf),
            axis=-1,
        )
        no_pair_attention_weights = jax.nn.softmax(
            jnp.where(attention_mask, qk_logits, -jnp.inf),
            axis=-1,
        )
        q_normalized = q.astype(jnp.float32) * jax.lax.rsqrt(
            jnp.mean(jnp.square(q.astype(jnp.float32)), axis=-1, keepdims=True)
            + attention_module.q_norm.eps
        )
        k_normalized = k.astype(jnp.float32) * jax.lax.rsqrt(
            jnp.mean(jnp.square(k.astype(jnp.float32)), axis=-1, keepdims=True)
            + attention_module.k_norm.eps
        )
        normalized_qk_logits = jnp.einsum(
            "...qd,...kd->...qk",
            q_normalized,
            k_normalized,
            precision=jax.lax.Precision.HIGHEST,
        ) / math.sqrt(attention_module.head_dim)
        qk_norm_attention = jax.nn.softmax(
            jnp.where(
                attention_mask,
                normalized_qk_logits + pair_logits,
                -jnp.inf,
            ),
            axis=-1,
        )
        qk_norm_no_pair_attention = jax.nn.softmax(
            jnp.where(attention_mask, normalized_qk_logits, -jnp.inf),
            axis=-1,
        )
        query_weights = compact_mask[:, None, :, None].astype(jnp.float32)
        attention_entropy = -jnp.sum(
            full_attention * jnp.log(jnp.maximum(full_attention, 1e-12)),
            axis=-1,
        )
        max_entropy = jnp.log(
            jnp.maximum(compact_mask.astype(jnp.float32).sum(axis=-1), 2.0)
        )[:, None, None]
        key_weights = compact_mask[:, None, :, None].astype(jnp.float32)
        pair_bias_centered = pair_bias.astype(jnp.float32) - (
            (pair_bias.astype(jnp.float32) * key_weights).sum(axis=2, keepdims=True)
            / key_weights.sum(axis=2, keepdims=True)
        )

        single_rms = _masked_rms(single_in, compact_mask)
        attention_rms = _masked_rms(attention, compact_mask)
        rows["pair_rms"].append(_masked_rms(pair, pair_mask))
        rows["pair_delta_over_input"].append(
            _masked_rms(pair - pair_in, pair_mask)
            / _masked_rms(pair_in, pair_mask)
        )
        rows["pair_bias_centered_rms"].append(
            _masked_rms(pair_bias_centered, pair_mask)
        )
        qk_rms = _masked_rms(
            jnp.transpose(qk_logits, (0, 2, 3, 1)),
            pair_mask,
        )
        pair_logit_rms = _masked_rms(pair_bias_centered, pair_mask)
        rows["qk_logit_rms"].append(qk_rms)
        rows["pair_to_qk_logit_ratio"].append(pair_logit_rms / qk_rms)
        rows["attention_pair_total_variation"].append(
            (
                jnp.abs(full_attention - no_pair_attention_weights)
                * query_weights
            ).sum()
            / (2.0 * query_weights.sum() * attention_module.num_heads)
        )
        rows["attention_entropy_frac"].append(
            (attention_entropy * query_weights[..., 0]).sum()
            / (
                (max_entropy * query_weights[..., 0]).sum()
                * attention_module.num_heads
            )
        )
        qk_norm_entropy = -jnp.sum(
            qk_norm_attention * jnp.log(jnp.maximum(qk_norm_attention, 1e-12)),
            axis=-1,
        )
        qk_norm_no_pair_entropy = -jnp.sum(
            qk_norm_no_pair_attention
            * jnp.log(jnp.maximum(qk_norm_no_pair_attention, 1e-12)),
            axis=-1,
        )
        query_head_weights = compact_mask[:, None, :].astype(jnp.float32)
        query_head_count = query_head_weights.sum() * attention_module.num_heads
        qk_norm_max_prob = qk_norm_attention.max(axis=-1)
        qk_norm_no_pair_max_prob = qk_norm_no_pair_attention.max(axis=-1)
        rows["qk_norm_attention_pair_total_variation"].append(
            (
                jnp.abs(qk_norm_attention - qk_norm_no_pair_attention)
                * query_weights
            ).sum()
            / (2.0 * query_weights.sum() * attention_module.num_heads)
        )
        rows["qk_norm_attention_entropy_frac"].append(
            (qk_norm_entropy * query_weights[..., 0]).sum()
            / (
                (max_entropy * query_weights[..., 0]).sum()
                * attention_module.num_heads
            )
        )
        rows["qk_norm_no_pair_attention_entropy_frac"].append(
            (qk_norm_no_pair_entropy * query_head_weights).sum()
            / (
                (max_entropy * query_weights[..., 0]).sum()
                * attention_module.num_heads
            )
        )
        rows["qk_norm_attention_max_prob"].append(
            (qk_norm_max_prob * query_head_weights).sum() / query_head_count
        )
        rows["qk_norm_no_pair_attention_max_prob"].append(
            (qk_norm_no_pair_max_prob * query_head_weights).sum()
            / query_head_count
        )
        rows["qk_norm_attention_fraction_max_prob_gt_0_9"].append(
            ((qk_norm_max_prob > 0.9) * query_head_weights).sum()
            / query_head_count
        )
        rows["q_rms"].append(jnp.sqrt(jnp.mean(jnp.square(q.astype(jnp.float32)))))
        rows["k_rms"].append(jnp.sqrt(jnp.mean(jnp.square(k.astype(jnp.float32)))))
        rows["qk_norm_logit_rms"].append(
            _masked_rms(
                jnp.transpose(normalized_qk_logits, (0, 2, 3, 1)),
                pair_mask,
            )
        )
        rows["single_attention_norm_gain_rms"].append(
            jnp.sqrt(
                jnp.mean(
                    jnp.square(
                        attention_module.single_norm.weight[...].astype(jnp.float32)
                    )
                )
            )
        )
        rows["single_attention_norm_gain_max"].append(
            jnp.max(
                jnp.abs(
                    attention_module.single_norm.weight[...].astype(jnp.float32)
                )
            )
        )
        rows["single_rms"].append(single_rms)
        rows["attention_rms"].append(attention_rms)
        rows["ffn_rms"].append(_masked_rms(ffn, compact_mask))
        rows["single_transition_norm_gain_rms"].append(
            jnp.sqrt(
                jnp.mean(
                    jnp.square(
                        block.single_transition_norm.weight[...].astype(jnp.float32)
                    )
                )
            )
        )
        rows["single_transition_norm_gain_max"].append(
            jnp.max(
                jnp.abs(
                    block.single_transition_norm.weight[...].astype(jnp.float32)
                )
            )
        )
        rows["attention_over_single"].append(
            attention_rms / single_rms
        )
        rows["ffn_over_single"].append(_masked_rms(ffn, compact_mask) / single_rms)
        rows["total_update_over_single"].append(
            _masked_rms(single - single_in, compact_mask) / single_rms
        )
        rows["local_pair_effect_over_single"].append(
            _masked_rms(pair_effect, compact_mask) / single_rms
        )
        rows["local_pair_effect_fraction_of_attention"].append(
            _masked_rms(pair_effect, compact_mask) / attention_rms
        )
        rows["local_pair_effect_cls_over_single"].append(
            _masked_rms(pair_effect, cls_mask) / _masked_rms(single_in, cls_mask)
        )
        rows["local_attention_no_pair_cosine"].append(
            _masked_cosine(attention, no_pair_attention_local, compact_mask)
        )
        rows["cumulative_no_pair_cosine"].append(
            _masked_cosine(single, no_pair_single, compact_mask)
        )
        rows["cumulative_no_pair_cls_cosine"].append(
            _masked_cosine(single, no_pair_single, cls_mask)
        )
        rows["cumulative_pair_effect_over_single"].append(
            _masked_rms(single - no_pair_single, compact_mask)
            / _masked_rms(single, compact_mask)
        )

    if encoder.final_norm is not None:
        single = encoder.final_norm(single)
        no_pair_single = encoder.final_norm(no_pair_single)
    summary = {
        "final_no_pair_cosine": _masked_cosine(
            single,
            no_pair_single,
            compact_mask,
        ),
        "final_no_pair_cls_cosine": _masked_cosine(
            single,
            no_pair_single,
            cls_mask,
        ),
        "final_pair_effect_over_single": (
            _masked_rms(single - no_pair_single, compact_mask)
            / _masked_rms(single, compact_mask)
        ),
        "mean_visible_peaks": peak_visible_mask.astype(jnp.float32).sum(axis=1).mean(),
        "mean_valid_peaks": peak_valid_mask.astype(jnp.float32).sum(axis=1).mean(),
    }
    return {key: jnp.stack(values) for key, values in rows.items()}, summary


def _plot(rows: list[dict[str, float]], output: Path) -> None:
    layers = [row["layer"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    pair_delta_axis = axes[0].twinx()
    axes[0].plot(layers, [row["pair_rms"] for row in rows], label="pair RMS")
    pair_delta_axis.plot(
        layers,
        [100 * row["pair_delta_over_input"] for row in rows],
        color="tab:orange",
        label="pair Δ / input (%)",
    )
    axes[0].set_title("Pair stream")
    axes[0].set_ylabel("pair RMS")
    pair_delta_axis.set_ylabel("pair Δ / input (%)")
    axes[0].legend(loc="upper left")
    pair_delta_axis.legend(loc="upper right")

    axes[1].plot(
        layers,
        [row["pair_bias_centered_rms"] for row in rows],
        color="tab:purple",
    )
    axes[1].set_title("Pair-derived attention-logit RMS")

    axes[2].plot(
        layers,
        [100 * row["attention_over_single"] for row in rows],
        label="attention / residual",
    )
    axes[2].plot(
        layers,
        [100 * row["local_pair_effect_over_single"] for row in rows],
        label="local pair effect / residual",
    )
    axes[2].plot(
        layers,
        [100 * row["cumulative_pair_effect_over_single"] for row in rows],
        label="cumulative pair effect / residual",
    )
    axes[2].set_title("Single-stream updates (%)")
    axes[2].set_yscale("log")
    axes[2].legend()

    max_prob_axis = axes[3].twinx()
    axes[3].plot(
        layers,
        [row["qk_norm_attention_entropy_frac"] for row in rows],
        label="entropy with pair bias",
    )
    axes[3].plot(
        layers,
        [row["qk_norm_no_pair_attention_entropy_frac"] for row in rows],
        label="entropy without pair bias",
    )
    max_prob_axis.plot(
        layers,
        [row["qk_norm_attention_max_prob"] for row in rows],
        color="tab:red",
        label="mean max probability",
    )
    axes[3].set_title("Post-QKNorm attention saturation")
    axes[3].set_ylabel("entropy / maximum entropy")
    max_prob_axis.set_ylabel("max probability")
    axes[3].legend(loc="lower right")
    max_prob_axis.legend(loc="upper right")
    for axis in axes:
        axis.set_xlabel("Encoder layer")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--full-visible", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/pair_to_single_dynamics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workdir = normalize_storage_path(args.workdir)
    checkpoint_dir = storage_join(workdir, "checkpoints")
    manager = build_jax_checkpoint_manager(
        checkpoint_dir,
        enable_async_checkpointing=False,
    )
    step = args.checkpoint_step or manager.latest_step()
    manager.close()
    assert step is not None

    stage = jepa_mask_stage(config, step, int(config.training_max_steps))
    data_config = copy.deepcopy(config)
    data_config.batch_size = args.batch_size
    data_config.gradient_accumulation_steps = 1
    data_config.jepa_context_fraction = stage.context_fraction
    data_config.jepa_target_fraction = stage.target_fraction
    data_config.dataloader_num_workers = 0
    data_config.dataloader_output_format = "numpy"
    torch.manual_seed(int(config.seed))
    datamodule = GemsDataModule(data_config, seed=int(config.seed))
    batch = next(iter(datamodule.val_loader_for_eval(augment=True)))
    batch = {
        key: jnp.asarray(value)
        for key, value in batch.items()
        if key
        in {
            "peak_mz",
            "peak_intensity",
            "peak_valid_mask",
            "context_mask",
            "precursor_mz",
            "collision_energy",
            "charge",
        }
    }
    if args.full_visible:
        batch["context_mask"] = batch["peak_valid_mask"]

    model_config = copy.deepcopy(config)
    model_config.training_mode = "mae"
    model = build_model_from_config(model_config)
    encoder = model.encoder
    del model
    gc.collect()

    checkpoint_path = storage_join(
        checkpoint_dir,
        "orbax",
        str(step),
    )
    restored = restore_frozen_teacher_encoder(
        checkpoint_path,
        nnx.state(encoder),
        path_renames={
            "mz_fourier": "mz_features",
            "fourier_ffn": "mz_ffn",
        },
    )
    nnx.update(encoder, restored)
    capacity = max(
        int(batch["context_mask"].sum(axis=1).max()) + 1,
        1,
    )
    metrics, summary = nnx.jit(
        lambda module, values: _analyze_encoder(module, values, capacity)
    )(encoder, batch)
    metrics, summary = jax.device_get((metrics, summary))

    rows = [
        {
            "layer": layer + 1,
            **{key: float(values[layer]) for key, values in metrics.items()},
        }
        for layer in range(len(encoder.blocks))
    ]
    summary = {key: float(value) for key, value in summary.items()}
    result = {
        "checkpoint_step": step,
        "batch_size": args.batch_size,
        "context_fraction": 1.0 if args.full_visible else stage.context_fraction,
        "target_fraction": stage.target_fraction,
        "summary": summary,
        "layers": rows,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"step_{step}"
    (output_dir / f"{stem}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with (output_dir / f"{stem}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, output_dir / f"{stem}.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
