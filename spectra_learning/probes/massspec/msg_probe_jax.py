from __future__ import annotations

import logging
import math
import pickle
import time
from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ml_collections import config_dict

from spectra_learning.config.msg_probe import validate_msg_probe_config
from spectra_learning.data.loading import local_batch_size
from spectra_learning.data.massspec_probe import MassSpecProbeData
from spectra_learning.models.common_jax import Array
from spectra_learning.models.induced_pair_jax import InducedPairState
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.probes.massspec.msg_settings import (
    BINARY_PROBE_TASKS,
    MACCS_TASK,
    PROBE_FINGERPRINT_BITS,
    REGRESSION_PROBE_TASKS,
    MsgProbeSplitTargets,
    MsgProbeTaskSpec,
    msg_probe_variants_from_config,
    resolve_msg_probe_fingerprint,
    resolve_msg_probe_num_repeats,
)
from spectra_learning.probes.massspec.pr_curves import build_precision_recall_curve


log = logging.getLogger(__name__)

JaxBatch = dict[str, Any]
JaxFeatures = Array | tuple[Array, Array]
JaxProbeParams = dict[str, Any]
EpochState = dict[str, Any]
PendingPrediction = tuple[dict[str, Array], np.ndarray, dict[str, np.ndarray]]
JAX_PROBE_DATA_AXIS = "data"


def run_msg_probe_jax(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetJEPAJax,
    data_mesh: Mesh | None = None,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    def run_once(
        repeat_index: int,
        repeat_on_epoch_end: Callable[[dict[str, float]], None] | None,
    ) -> dict[str, Any]:
        return _run_msg_probe_once_jax(
            config=config,
            model=model,
            data_mesh=data_mesh,
            on_epoch_end=repeat_on_epoch_end,
            repeat_index=repeat_index,
        )

    return _run_repeated_probe_jax(
        repeat_count=resolve_msg_probe_num_repeats(config),
        metric_prefix="msg_probe",
        run_once=run_once,
        on_epoch_end=on_epoch_end,
    )


def precompile_msg_probe_jax(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetJEPAJax,
    data_mesh: Mesh | None = None,
) -> dict[str, float]:
    fingerprint_task = resolve_msg_probe_fingerprint(config)
    probe_data = MassSpecProbeData.from_config(
        config,
        distributed_world_size=_distributed_world_size_jax(),
        distributed_rank=_distributed_rank_jax(),
        distributed_local_rank=0,
    )
    variants = msg_probe_variants_from_config(config)
    use_pair_features = any(_uses_pair_features(variant) for variant in variants)
    peak_ordering = str(_config_get(config, "peak_ordering", "intensity"))
    max_samples = int(_config_get(config, "msg_probe_batch_size", probe_data.batch_size))
    train_seed = int(config.seed) + 1_100_000
    val_seed = int(config.seed) + 1_110_000
    train_targets = _collect_split_targets_jax(
        probe_data=probe_data,
        split="massspec_train",
        peak_ordering=peak_ordering,
        seed=train_seed,
        fingerprint_task=fingerprint_task,
        max_samples=max_samples,
    )
    val_targets = _collect_split_targets_jax(
        probe_data=probe_data,
        split="massspec_val",
        peak_ordering=peak_ordering,
        seed=val_seed,
        fingerprint_task=fingerprint_task,
        max_samples=max_samples,
    )
    task_spec = _build_task_spec_jax(
        train_targets=train_targets,
        test_targets=val_targets,
        fingerprint_task=fingerprint_task,
        single_pair_covariance_include_diagonal=bool(
            _config_get(
                config,
                "msg_probe_single_pair_covariance_include_diagonal",
                False,
            )
        ),
    )
    steps_per_epoch = probe_steps_per_epoch_jax(
        probe_data,
        split="massspec_train",
        drop_remainder=False,
        max_samples=max_samples,
        distributed_world_size=_distributed_world_size_jax(),
    )
    schedule = _probe_lr_schedule(
        base_lr=float(_config_get(config, "msg_probe_learning_rate", 1e-3)),
        total_steps=max(1, steps_per_epoch),
        warmup_steps=_resolve_probe_warmup_steps(config, steps_per_epoch),
    )
    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=float(_config_get(config, "msg_probe_weight_decay", 1e-2)),
    )
    key = jax.random.PRNGKey(int(config.seed) + 300_000)
    params_by_variant: dict[str, JaxProbeParams] = {}
    opt_state_by_variant: dict[str, optax.OptState] = {}
    for variant in variants:
        key, init_key = jax.random.split(key)
        params_by_variant[variant] = _init_probe_params(
            init_key,
            variant=variant,
            config=config,
            task_spec=task_spec,
        )
        opt_state_by_variant[variant] = optimizer.init(params_by_variant[variant])
    train_step_by_variant = {
        variant: _make_jitted_probe_train_step(
            optimizer=optimizer,
            variant=variant,
            task_spec=task_spec,
            distributed=_is_distributed_jax(),
        )
        for variant in variants
    }
    predict_step_by_variant = {
        variant: _make_jitted_probe_predict_step(
            variant=variant,
            task_spec=task_spec,
        )
        for variant in variants
    }
    train_batch = next(
        iter_massspec_probe_jax(
            probe_data=probe_data,
            split="massspec_train",
            seed=train_seed,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_samples,
            distributed_world_size=_distributed_world_size_jax(),
            distributed_rank=_distributed_rank_jax(),
            pad_distributed=True,
            data_mesh=data_mesh,
        )
    )
    train_features = _extract_features(
        model,
        train_batch,
        use_pair_features=use_pair_features,
    )
    train_step_batch = _probe_step_batch(train_batch, task_spec)
    compiled_train_steps = 0
    for variant in variants:
        params, opt_state, logits = train_step_by_variant[variant](
            params_by_variant[variant],
            opt_state_by_variant[variant],
            batch=train_step_batch,
            features=train_features,
        )
        jax.block_until_ready((params, opt_state, logits))
        params_by_variant[variant] = params
        opt_state_by_variant[variant] = opt_state
        compiled_train_steps += 1
    eval_batch = next(
        iter_massspec_probe_jax(
            probe_data=probe_data,
            split="massspec_val",
            seed=val_seed,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_samples,
            distributed_world_size=_distributed_world_size_jax(),
            distributed_rank=_distributed_rank_jax(),
            data_mesh=data_mesh,
        )
    )
    eval_features = _extract_features(
        model,
        eval_batch,
        use_pair_features=use_pair_features,
    )
    eval_step_batch = _probe_step_batch(eval_batch, task_spec)
    compiled_predict_steps = 0
    for variant in variants:
        logits = predict_step_by_variant[variant](
            params_by_variant[variant],
            eval_step_batch,
            eval_features,
        )
        jax.block_until_ready(logits)
        compiled_predict_steps += 1
    return {
        "features": 2.0,
        "train_steps": float(compiled_train_steps),
        "predict_steps": float(compiled_predict_steps),
    }


def _run_msg_probe_once_jax(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetJEPAJax,
    data_mesh: Mesh | None,
    on_epoch_end: Callable[[dict[str, float]], None] | None,
    repeat_index: int,
) -> dict[str, Any]:
    num_probe_epochs = int(_config_get(config, "msg_probe_num_epochs", 5))
    probe_lr = float(_config_get(config, "msg_probe_learning_rate", 1e-3))
    probe_weight_decay = float(_config_get(config, "msg_probe_weight_decay", 1e-2))
    peak_ordering = str(_config_get(config, "peak_ordering", "intensity"))
    early_stopping = bool(_config_get(config, "msg_probe_early_stopping", False))
    early_stopping_patience = int(
        _config_get(config, "msg_probe_early_stopping_patience", 10)
    )
    early_stopping_min_delta = float(
        _config_get(config, "msg_probe_early_stopping_min_delta", 0.0)
    )
    early_stopping_min_epochs = int(
        _config_get(config, "msg_probe_early_stopping_min_epochs", 1)
    )
    fingerprint_task = resolve_msg_probe_fingerprint(config)
    probe_data = MassSpecProbeData.from_config(
        config,
        distributed_world_size=_distributed_world_size_jax(),
        distributed_rank=_distributed_rank_jax(),
        distributed_local_rank=0,
    )
    variants = msg_probe_variants_from_config(config)
    use_pair_features = any(_uses_pair_features(variant) for variant in variants)
    max_train_samples = _optional_positive_int(
        config,
        "jax_msg_probe_max_train_samples",
    )
    max_val_samples = _optional_positive_int(
        config,
        "jax_msg_probe_max_val_samples",
    )
    max_test_samples = _optional_positive_int(
        config,
        "jax_msg_probe_max_test_samples",
    )
    max_mcebio_test_samples = _optional_positive_int(
        config,
        "jax_msg_probe_max_mcebio_test_samples",
    )
    seed_offset = 100_000 * repeat_index
    train_seed_base = int(config.seed) + 1_100_000 + seed_offset
    test_seed_base = int(config.seed) + 1_200_000 + seed_offset

    phase_timing = {
        "target_collection_seconds": 0.0,
        "train_epoch_seconds": 0.0,
        "eval_epoch_seconds": 0.0,
        "score_epoch_seconds": 0.0,
        "final_test_seconds": 0.0,
        "mcebio_seconds": 0.0,
        "final_score_seconds": 0.0,
    }
    probe_start = time.perf_counter()
    phase_start = time.perf_counter()
    train_targets = _collect_split_targets_jax(
        probe_data=probe_data,
        split="massspec_train",
        peak_ordering=peak_ordering,
        seed=train_seed_base,
        fingerprint_task=fingerprint_task,
        max_samples=max_train_samples,
    )
    val_targets = _collect_split_targets_jax(
        probe_data=probe_data,
        split="massspec_val",
        peak_ordering=peak_ordering,
        seed=train_seed_base + 10_000,
        fingerprint_task=fingerprint_task,
        max_samples=max_val_samples,
    )
    selection_targets = (
        val_targets
        if early_stopping
        else _collect_split_targets_jax(
            probe_data=probe_data,
            split="massspec_test",
            peak_ordering=peak_ordering,
            seed=test_seed_base,
            fingerprint_task=fingerprint_task,
            max_samples=max_test_samples,
        )
    )
    phase_timing["target_collection_seconds"] += time.perf_counter() - phase_start
    task_spec = _build_task_spec_jax(
        train_targets=train_targets,
        test_targets=selection_targets,
        fingerprint_task=fingerprint_task,
        single_pair_covariance_include_diagonal=bool(
            _config_get(
                config,
                "msg_probe_single_pair_covariance_include_diagonal",
                False,
            )
        ),
    )
    steps_per_epoch = probe_steps_per_epoch_jax(
        probe_data,
        split="massspec_train",
        drop_remainder=False,
        max_samples=max_train_samples,
        distributed_world_size=_distributed_world_size_jax(),
    )
    warmup_steps = _resolve_probe_warmup_steps(config, steps_per_epoch)
    schedule = _probe_lr_schedule(
        base_lr=probe_lr,
        total_steps=num_probe_epochs * steps_per_epoch,
        warmup_steps=warmup_steps,
    )
    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=probe_weight_decay,
    )
    key = jax.random.PRNGKey(int(config.seed) + 300_000 + seed_offset)
    params_by_variant: dict[str, JaxProbeParams] = {}
    opt_state_by_variant: dict[str, optax.OptState] = {}
    for variant in variants:
        key, init_key = jax.random.split(key)
        params_by_variant[variant] = _init_probe_params(
            init_key,
            variant=variant,
            config=config,
            task_spec=task_spec,
        )
        opt_state_by_variant[variant] = optimizer.init(params_by_variant[variant])

    select_metric = resolve_msg_probe_select_metric_jax(config)
    higher_is_better = msg_probe_metric_higher_is_better(select_metric)
    best_metrics_by_variant: dict[str, dict[str, Any]] = {}
    best_params_by_variant: dict[str, JaxProbeParams] = {}
    best_test_state_by_variant: dict[str, EpochState] = {}
    best_metric_values = {
        variant: -float("inf") if higher_is_better else float("inf")
        for variant in variants
    }
    epochs_without_improvement = {variant: 0 for variant in variants}
    train_step_by_variant = {
        variant: _make_jitted_probe_train_step(
            optimizer=optimizer,
            variant=variant,
            task_spec=task_spec,
            distributed=_is_distributed_jax(),
        )
        for variant in variants
    }
    predict_step_by_variant = {
        variant: _make_jitted_probe_predict_step(
            variant=variant,
            task_spec=task_spec,
        )
        for variant in variants
    }

    for epoch_idx in range(num_probe_epochs):
        train_states = {variant: _new_epoch_state(task_spec) for variant in variants}
        train_pending = {variant: [] for variant in variants}
        phase_start = time.perf_counter()
        for batch in iter_massspec_probe_jax(
            probe_data=probe_data,
            split="massspec_train",
            seed=train_seed_base + epoch_idx,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_train_samples,
            distributed_world_size=_distributed_world_size_jax(),
            distributed_rank=_distributed_rank_jax(),
            pad_distributed=True,
            data_mesh=data_mesh,
        ):
            features = _extract_features(model, batch, use_pair_features=use_pair_features)
            step_batch = _probe_step_batch(batch, task_spec)
            for variant in variants:
                params, opt_state, logits = train_step_by_variant[variant](
                    params_by_variant[variant],
                    opt_state_by_variant[variant],
                    batch=step_batch,
                    features=features,
                )
                params_by_variant[variant] = params
                opt_state_by_variant[variant] = opt_state
                _append_pending_prediction(
                    train_pending[variant],
                    logits=logits,
                    batch=batch,
                    task_spec=task_spec,
                )
        _flush_variant_prediction_queues(train_pending, train_states, task_spec)
        phase_timing["train_epoch_seconds"] += time.perf_counter() - phase_start
        train_states = _gather_variant_states_jax(train_states, task_spec)

        eval_states = {
            variant: _new_epoch_state(task_spec)
            for variant in variants
        }
        eval_pending = {variant: [] for variant in variants}
        eval_split = "massspec_val" if early_stopping else "massspec_test"
        eval_seed = train_seed_base + 10_000 if early_stopping else test_seed_base
        phase_start = time.perf_counter()
        for batch in iter_massspec_probe_jax(
            probe_data=probe_data,
            split=eval_split,
            seed=eval_seed,
            peak_ordering=peak_ordering,
            drop_remainder=False,
            max_samples=max_val_samples if early_stopping else max_test_samples,
            distributed_world_size=_distributed_world_size_jax(),
            distributed_rank=_distributed_rank_jax(),
            data_mesh=data_mesh,
        ):
            features = _extract_features(model, batch, use_pair_features=use_pair_features)
            step_batch = _probe_step_batch(batch, task_spec)
            for variant in variants:
                _append_pending_prediction(
                    eval_pending[variant],
                    logits=predict_step_by_variant[variant](
                        params_by_variant[variant],
                        step_batch,
                        features,
                    ),
                    batch=batch,
                    task_spec=task_spec,
                )
        _flush_variant_prediction_queues(eval_pending, eval_states, task_spec)
        phase_timing["eval_epoch_seconds"] += time.perf_counter() - phase_start
        eval_states = _gather_variant_states_jax(eval_states, task_spec)

        phase_start = time.perf_counter()
        epoch_metrics: dict[str, float] = {}
        for variant in variants:
            variant_prefix = f"msg_probe/{variant}"
            variant_metrics = {
                **_score_epoch_state(
                    prefix=f"{variant_prefix}/train",
                    epoch_state=train_states[variant],
                    task_spec=task_spec,
                ),
                **_score_epoch_state(
                    prefix=f"{variant_prefix}/{'val' if early_stopping else 'test'}",
                    epoch_state=eval_states[variant],
                    task_spec=task_spec,
                ),
                f"{variant_prefix}/num_{fingerprint_task}_bits": float(
                    task_spec.maccs_bits
                ),
                f"{variant_prefix}/epoch": float(epoch_idx + 1),
            }
            epoch_metrics.update(variant_metrics)
            variant_select_metric = _msg_probe_variant_metric_key(variant, select_metric)
            if early_stopping:
                variant_select_metric = variant_select_metric.replace("/test/", "/val/")
            current_value = variant_metrics[variant_select_metric]
            previous_best = best_metric_values[variant]
            is_better = (
                current_value > previous_best + early_stopping_min_delta
                if higher_is_better
                else current_value < previous_best - early_stopping_min_delta
            )
            if is_better:
                best_metric_values[variant] = current_value
                best_metrics_by_variant[variant] = dict(variant_metrics)
                best_params_by_variant[variant] = _clone_tree(params_by_variant[variant])
                if not early_stopping:
                    best_test_state_by_variant[variant] = eval_states[variant]
                epochs_without_improvement[variant] = 0
            else:
                epochs_without_improvement[variant] += 1
            if _is_main_process_jax():
                _log_epoch_metrics(
                    variant=variant,
                    epoch_idx=epoch_idx,
                    num_probe_epochs=num_probe_epochs,
                    fingerprint_task=fingerprint_task,
                    metrics=variant_metrics,
                    early_stopping=early_stopping,
                )
        if on_epoch_end is not None and _is_main_process_jax():
            on_epoch_end(epoch_metrics)
        phase_timing["score_epoch_seconds"] += time.perf_counter() - phase_start
        if (
            early_stopping
            and epoch_idx + 1 >= early_stopping_min_epochs
            and all(
                epochs_without_improvement[variant] >= early_stopping_patience
                for variant in variants
            )
        ):
            if _is_main_process_jax():
                log.info(
                    "JAX MSG probe early stopping at epoch %d/%d after %d epochs without validation improvement",
                    epoch_idx + 1,
                    num_probe_epochs,
                    early_stopping_patience,
                )
            break

    final_test_metrics_by_variant: dict[str, dict[str, Any]] = {}
    if early_stopping:
        final_states = {variant: _new_epoch_state(task_spec) for variant in variants}
        final_pending = {variant: [] for variant in variants}
        phase_start = time.perf_counter()
        for batch in iter_massspec_probe_jax(
            probe_data=probe_data,
            split="massspec_test",
            seed=test_seed_base,
            peak_ordering=peak_ordering,
            drop_remainder=False,
                max_samples=max_test_samples,
                distributed_world_size=_distributed_world_size_jax(),
                distributed_rank=_distributed_rank_jax(),
                data_mesh=data_mesh,
            ):
            features = _extract_features(model, batch, use_pair_features=use_pair_features)
            step_batch = _probe_step_batch(batch, task_spec)
            for variant in variants:
                params = best_params_by_variant.get(variant, params_by_variant[variant])
                _append_pending_prediction(
                    final_pending[variant],
                    logits=predict_step_by_variant[variant](
                        params,
                        step_batch,
                        features,
                    ),
                    batch=batch,
                    task_spec=task_spec,
                )
        _flush_variant_prediction_queues(final_pending, final_states, task_spec)
        phase_timing["final_test_seconds"] += time.perf_counter() - phase_start
        final_states = _gather_variant_states_jax(final_states, task_spec)
        phase_start = time.perf_counter()
        final_test_metrics_by_variant = {
            variant: _score_epoch_state(
                prefix=f"msg_probe/{variant}/test",
                epoch_state=state,
                task_spec=task_spec,
                include_pr_curves=True,
            )
            for variant, state in final_states.items()
        }
        phase_timing["final_score_seconds"] += time.perf_counter() - phase_start
    else:
        phase_start = time.perf_counter()
        final_test_metrics_by_variant = {
            variant: _score_epoch_state(
                prefix=f"msg_probe/{variant}/test",
                epoch_state=state,
                task_spec=task_spec,
                include_pr_curves=True,
            )
            for variant, state in best_test_state_by_variant.items()
        }
        phase_timing["final_score_seconds"] += time.perf_counter() - phase_start

    mcebio_states = {variant: _new_epoch_state(task_spec) for variant in variants}
    mcebio_pending = {variant: [] for variant in variants}
    phase_start = time.perf_counter()
    for batch in iter_massspec_probe_jax(
        probe_data=probe_data,
        split="massspec_mcebio_test",
        seed=test_seed_base + 75_000,
        peak_ordering=peak_ordering,
        drop_remainder=False,
        max_samples=max_mcebio_test_samples,
        distributed_world_size=_distributed_world_size_jax(),
        distributed_rank=_distributed_rank_jax(),
        data_mesh=data_mesh,
    ):
        features = _extract_features(model, batch, use_pair_features=use_pair_features)
        step_batch = _probe_step_batch(batch, task_spec)
        for variant in variants:
            params = best_params_by_variant.get(variant, params_by_variant[variant])
            _append_pending_prediction(
                mcebio_pending[variant],
                logits=predict_step_by_variant[variant](
                    params,
                    step_batch,
                    features,
                ),
                batch=batch,
                task_spec=task_spec,
            )
    _flush_variant_prediction_queues(mcebio_pending, mcebio_states, task_spec)
    phase_timing["mcebio_seconds"] += time.perf_counter() - phase_start
    mcebio_states = _gather_variant_states_jax(mcebio_states, task_spec)
    phase_start = time.perf_counter()
    mcebio_sulfur_metrics_by_variant = {
        variant: _sulfur_metric_subset(
            _score_epoch_state(
                prefix=f"msg_probe/{variant}/mcebio_sulfur_test",
                epoch_state=state,
                task_spec=task_spec,
                include_pr_curves=True,
            )
        )
        for variant, state in mcebio_states.items()
    }
    phase_timing["final_score_seconds"] += time.perf_counter() - phase_start

    best_metrics: dict[str, Any] = {}
    for variant in variants:
        variant_metrics = dict(best_metrics_by_variant.get(variant, {}))
        if not variant_metrics:
            continue
        variant_metrics.update(final_test_metrics_by_variant.get(variant, {}))
        variant_metrics.update(mcebio_sulfur_metrics_by_variant.get(variant, {}))
        best_metrics.update(variant_metrics)
    best_metrics["msg_probe/jax_wall_seconds"] = time.perf_counter() - probe_start
    for name, value in phase_timing.items():
        best_metrics[f"msg_probe/jax_{name}"] = value
    return best_metrics


def iter_massspec_probe_jax(
    *,
    probe_data: MassSpecProbeData,
    split: str,
    seed: int,
    peak_ordering: str,
    drop_remainder: bool,
    max_samples: int | None = None,
    sample_randomly: bool = False,
    pad_distributed: bool = False,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
    data_mesh: Mesh | None = None,
) -> Iterator[JaxBatch]:
    dataset = probe_data.build_dataset(
        split,
        seed=seed,
        peak_ordering=peak_ordering,
        shuffle=(split == "massspec_train") or sample_randomly,
        drop_remainder=drop_remainder,
        max_samples=max_samples,
        pad_distributed=pad_distributed or distributed_world_size > 1,
        distributed_world_size=distributed_world_size,
        distributed_rank=distributed_rank,
        output_format="numpy" if data_mesh is not None else "jax",
    )
    size = int(probe_data.info[f"{split}_size"])
    if max_samples is not None:
        size = min(size, max_samples)
    sampler_local_size = size
    valid_local_size = size
    if distributed_world_size > 1:
        sampler_local_size = (
            size // distributed_world_size
            if drop_remainder
            else math.ceil(size / distributed_world_size)
        )
        valid_local_size = (
            sampler_local_size
            if drop_remainder
            else _distributed_probe_shard_size(
                size,
                world_size=distributed_world_size,
                rank=distributed_rank,
            )
        )
    seen = 0
    valid_seen = 0
    for batch in dataset:
        if seen >= sampler_local_size:
            break
        take = min(int(batch["peak_mz"].shape[0]), sampler_local_size - seen)
        if take != int(batch["peak_mz"].shape[0]):
            batch = _slice_batch(batch, take)
        valid_take = min(take, max(valid_local_size - valid_seen, 0))
        if valid_take < take:
            batch = _mask_probe_padding_rows(batch, valid_take)
        seen += take
        valid_seen += valid_take
        yield _probe_batch_to_jax(batch, data_mesh=data_mesh)


def _distributed_probe_shard_size(
    size: int,
    *,
    world_size: int,
    rank: int,
) -> int:
    if rank >= size:
        return 0
    return (size - 1 - rank) // world_size + 1


def _mask_probe_padding_rows(batch: JaxBatch, valid_take: int) -> JaxBatch:
    masked = dict(batch)
    valid_mol = batch["probe_valid_mol"]
    if isinstance(valid_mol, np.ndarray):
        valid_mol = valid_mol.copy()
        valid_mol[valid_take:] = False
    else:
        valid_mol = valid_mol.at[valid_take:].set(False)
    masked["probe_valid_mol"] = valid_mol
    return masked


def _probe_batch_to_jax(
    batch: dict[str, Any],
    *,
    data_mesh: Mesh | None,
) -> dict[str, Any]:
    return {
        key: _probe_value_to_jax(value, data_mesh=data_mesh)
        for key, value in batch.items()
    }


def _probe_value_to_jax(value: Any, *, data_mesh: Mesh | None) -> Any:
    if isinstance(value, np.ndarray | np.generic):
        return _probe_array_to_jax(value, data_mesh=data_mesh)
    if isinstance(value, dict):
        return {
            key: _probe_value_to_jax(item, data_mesh=data_mesh)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return value
        return [_probe_value_to_jax(item, data_mesh=data_mesh) for item in value]
    if isinstance(value, tuple):
        return tuple(_probe_value_to_jax(item, data_mesh=data_mesh) for item in value)
    return value


def _probe_array_to_jax(value: np.ndarray | np.generic, *, data_mesh: Mesh | None) -> Array:
    host_value = np.asarray(value)
    if data_mesh is None or host_value.ndim == 0:
        return jnp.asarray(host_value)
    local_shard_count = min(jax.local_device_count(), int(np.asarray(data_mesh.devices).size))
    if local_shard_count <= 1 or int(host_value.shape[0]) % local_shard_count != 0:
        return jnp.asarray(host_value)
    sharding = NamedSharding(
        data_mesh,
        P(JAX_PROBE_DATA_AXIS, *((None,) * (host_value.ndim - 1))),
    )
    if jax.process_count() > 1:
        return jax.make_array_from_process_local_data(sharding, host_value)
    return jax.device_put(host_value, sharding)


def probe_steps_per_epoch_jax(
    probe_data: MassSpecProbeData,
    *,
    split: str,
    drop_remainder: bool,
    max_samples: int | None = None,
    distributed_world_size: int = 1,
) -> int:
    size = int(probe_data.info[f"{split}_size"])
    if max_samples is not None:
        size = min(size, max_samples)
    if distributed_world_size > 1:
        size = (
            size // distributed_world_size
            if drop_remainder
            else math.ceil(size / distributed_world_size)
        )
    batch_size = local_batch_size(int(probe_data.batch_size), distributed_world_size)
    return size // batch_size if drop_remainder else math.ceil(size / batch_size)


def _collect_split_targets_jax(
    *,
    probe_data: MassSpecProbeData,
    split: str,
    peak_ordering: str,
    seed: int,
    max_samples: int | None = None,
    sample_randomly: bool = False,
    fingerprint_task: str = MACCS_TASK,
) -> MsgProbeSplitTargets:
    regression = {name: [] for name in REGRESSION_PROBE_TASKS}
    binary = {name: [] for name in BINARY_PROBE_TASKS}
    fingerprints = []
    fingerprint_key = f"probe_{fingerprint_task}"
    for batch in iter_massspec_probe_jax(
        probe_data=probe_data,
        split=split,
        seed=seed,
        peak_ordering=peak_ordering,
        drop_remainder=False,
        max_samples=max_samples,
        sample_randomly=sample_randomly,
        distributed_world_size=_distributed_world_size_jax(),
        distributed_rank=_distributed_rank_jax(),
    ):
        valid_mask = np.asarray(batch["probe_valid_mol"]).astype(bool, copy=False)
        if not valid_mask.any():
            continue
        for name in REGRESSION_PROBE_TASKS:
            regression[name].append(np.asarray(batch[f"probe_{name}"])[valid_mask])
        for name in BINARY_PROBE_TASKS:
            binary[name].append(np.asarray(batch[f"probe_{name}"])[valid_mask])
        fingerprints.append(np.asarray(batch[fingerprint_key])[valid_mask])

    def cat(values: dict[str, list[np.ndarray]], dtype: Any) -> dict[str, np.ndarray]:
        return {
            name: np.concatenate(chunks).astype(dtype, copy=False)
            if chunks
            else np.empty(0, dtype=dtype)
            for name, chunks in values.items()
        }

    fingerprint_bits = (
        int(fingerprints[0].shape[1])
        if fingerprints
        else int(probe_data.info.get(f"probe_{fingerprint_task}_bits", 0))
    )
    targets = MsgProbeSplitTargets(
        regression=cat(regression, np.float32),
        binary=cat(binary, np.float32),
        maccs=(
            np.concatenate(fingerprints, axis=0).astype(np.int32, copy=False)
            if fingerprints
            else np.empty((0, fingerprint_bits), dtype=np.int32)
        ),
    )
    if _is_distributed_jax():
        targets = _merge_split_targets_jax(
            _all_gather_object_jax(targets),
            fingerprint_task=fingerprint_task,
        )
    return targets


def _is_distributed_jax() -> bool:
    return _distributed_world_size_jax() > 1


def _is_main_process_jax() -> bool:
    return _distributed_rank_jax() == 0


def _distributed_world_size_jax() -> int:
    return int(jax.process_count())


def _distributed_rank_jax() -> int:
    return int(jax.process_index())


def _all_gather_object_jax(value: object) -> list[object]:
    if not _is_distributed_jax():
        return [value]
    payload = pickle.dumps(value)
    local_length = np.asarray(len(payload), dtype=np.int32)
    lengths = np.asarray(multihost_utils.process_allgather(local_length)).reshape(-1)
    max_length = int(lengths.max()) if lengths.size else 0
    local_buffer = np.zeros(max_length, dtype=np.uint8)
    if payload:
        local_buffer[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(local_buffer)).reshape(
        -1,
        max_length,
    )
    return [
        pickle.loads(bytes(buffer[: int(length)]))
        for buffer, length in zip(gathered, lengths, strict=True)
    ]


def _merge_split_targets_jax(
    targets_by_rank: list[object],
    *,
    fingerprint_task: str,
) -> MsgProbeSplitTargets:
    targets = [
        target
        for target in targets_by_rank
        if isinstance(target, MsgProbeSplitTargets)
    ]

    def merge_named_arrays(
        attr: str,
        names: tuple[str, ...],
        dtype: Any,
    ) -> dict[str, np.ndarray]:
        return {
            name: (
                np.concatenate(
                    [getattr(target, attr)[name] for target in targets],
                    axis=0,
                ).astype(dtype, copy=False)
                if targets
                else np.empty(0, dtype=dtype)
            )
            for name in names
        }

    fingerprint_bits = (
        int(targets[0].maccs.shape[1])
        if targets
        else int(PROBE_FINGERPRINT_BITS[fingerprint_task])
    )
    return MsgProbeSplitTargets(
        regression=merge_named_arrays("regression", REGRESSION_PROBE_TASKS, np.float32),
        binary=merge_named_arrays("binary", BINARY_PROBE_TASKS, np.float32),
        maccs=(
            np.concatenate([target.maccs for target in targets], axis=0).astype(
                np.int32,
                copy=False,
            )
            if targets
            else np.empty((0, fingerprint_bits), dtype=np.int32)
        ),
    )


def _merge_epoch_states_jax(
    states: list[EpochState],
    task_spec: MsgProbeTaskSpec,
) -> EpochState:
    merged = _new_epoch_state(task_spec)
    merged["count"] = sum(int(state["count"]) for state in states)
    merged_predictions = merged["predictions"]
    merged_targets = merged["targets"]
    for state in states:
        predictions = state["predictions"]
        targets = state["targets"]
        for name in _probe_prediction_names(task_spec):
            merged_predictions[name].extend(predictions[name])
            merged_targets[name].extend(targets[name])
    return merged


def _gather_variant_states_jax(
    states: dict[str, EpochState],
    task_spec: MsgProbeTaskSpec,
) -> dict[str, EpochState]:
    if not _is_distributed_jax():
        return states
    gathered = [
        rank_states
        for rank_states in _all_gather_object_jax(states)
        if isinstance(rank_states, dict)
    ]
    return {
        variant: _merge_epoch_states_jax(
            [
                rank_states[variant]
                for rank_states in gathered
                if variant in rank_states
            ],
            task_spec,
        )
        for variant in states
    }


def _build_task_spec_jax(
    *,
    train_targets: MsgProbeSplitTargets,
    test_targets: MsgProbeSplitTargets,
    fingerprint_task: str,
    single_pair_covariance_include_diagonal: bool = False,
) -> MsgProbeTaskSpec:
    del test_targets
    regression_means, regression_stds = {}, {}
    for name in REGRESSION_PROBE_TASKS:
        values = train_targets.regression[name].astype(np.float32)
        regression_means[name] = float(values.mean())
        regression_stds[name] = float(np.clip(values.std(), 1e-8, None))
    return MsgProbeTaskSpec(
        regression_tasks=REGRESSION_PROBE_TASKS,
        binary_tasks=BINARY_PROBE_TASKS,
        maccs_bits=int(train_targets.maccs.shape[1]),
        regression_means=regression_means,
        regression_stds=regression_stds,
        fingerprint_task=fingerprint_task,
        single_pair_covariance_include_diagonal=(
            single_pair_covariance_include_diagonal
        ),
    )


def _make_jitted_probe_train_step(
    *,
    optimizer: optax.GradientTransformation,
    variant: str,
    task_spec: MsgProbeTaskSpec,
    distributed: bool,
):
    if distributed:

        @jax.jit
        def grad_step(
            params: JaxProbeParams,
            batch: dict[str, Array],
            features: JaxFeatures,
        ) -> JaxProbeParams:
            loss, grads = jax.value_and_grad(_probe_loss)(
                params,
                variant=variant,
                task_spec=task_spec,
                batch=batch,
                features=features,
            )
            del loss
            return grads

        @jax.jit
        def apply_step(
            params: JaxProbeParams,
            opt_state: optax.OptState,
            grads: JaxProbeParams,
            batch: dict[str, Array],
            features: JaxFeatures,
        ) -> tuple[JaxProbeParams, optax.OptState, dict[str, Array]]:
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            logits = _probe_logits(
                params,
                variant=variant,
                task_spec=task_spec,
                features=features,
                valid_mask=batch["peak_valid_mask"],
            )
            return params, opt_state, logits

        def train_step(
            params: JaxProbeParams,
            opt_state: optax.OptState,
            *,
            batch: dict[str, Array],
            features: JaxFeatures,
        ) -> tuple[JaxProbeParams, optax.OptState, dict[str, Array]]:
            grads = grad_step(params, batch, features)
            grads = _mean_tree_across_processes(grads)
            return apply_step(params, opt_state, grads, batch, features)

        train_step._jitted_compile_fns = (  # type: ignore[attr-defined]
            ("probe_grad", grad_step),
            ("probe_apply", apply_step),
        )
        return train_step

    @jax.jit
    def local_train_step(
        params: JaxProbeParams,
        opt_state: optax.OptState,
        batch: dict[str, Array],
        features: JaxFeatures,
    ) -> tuple[JaxProbeParams, optax.OptState, dict[str, Array]]:
        loss, grads = jax.value_and_grad(_probe_loss)(
            params,
            variant=variant,
            task_spec=task_spec,
            batch=batch,
            features=features,
        )
        del loss
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        logits = _probe_logits(
            params,
            variant=variant,
            task_spec=task_spec,
            features=features,
            valid_mask=batch["peak_valid_mask"],
        )
        return params, opt_state, logits

    def train_step(
        params: JaxProbeParams,
        opt_state: optax.OptState,
        *,
        batch: dict[str, Array],
        features: JaxFeatures,
    ) -> tuple[JaxProbeParams, optax.OptState, dict[str, Array]]:
        return local_train_step(params, opt_state, batch, features)

    train_step._jitted_compile_fns = (("probe_train", local_train_step),)  # type: ignore[attr-defined]
    return train_step


def jitted_probe_train_step_compile_fns(train_step: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(getattr(train_step, "_jitted_compile_fns", ()))


def _make_jitted_probe_predict_step(
    *,
    variant: str,
    task_spec: MsgProbeTaskSpec,
):
    @jax.jit
    def predict_step(
        params: JaxProbeParams,
        batch: dict[str, Array],
        features: JaxFeatures,
    ) -> dict[str, Array]:
        return _probe_logits(
            params,
            variant=variant,
            task_spec=task_spec,
            features=features,
            valid_mask=batch["peak_valid_mask"],
        )

    return predict_step


def _probe_step_batch(batch: JaxBatch, task_spec: MsgProbeTaskSpec) -> dict[str, Array]:
    keys = ["peak_valid_mask", "probe_valid_mol"]
    keys.extend(f"probe_{name}" for name in task_spec.regression_tasks)
    keys.extend(f"probe_{name}" for name in task_spec.binary_tasks)
    if task_spec.maccs_bits > 0:
        keys.append(f"probe_{task_spec.fingerprint_task}")
    return {key: batch[key] for key in keys}


def _mean_tree_across_processes(tree: Any) -> Any:
    if not _is_distributed_jax():
        return tree

    def gather_mean(value: Any) -> Array:
        if _is_global_non_fully_addressable_array(value):
            gathered = multihost_utils.process_allgather(value, tiled=True)
            return jax.device_put(jnp.asarray(gathered), value.sharding)
        gathered = multihost_utils.process_allgather(value)
        return jnp.mean(jnp.asarray(gathered), axis=0)

    return jax.tree.map(gather_mean, tree)


def _is_global_non_fully_addressable_array(value: Any) -> bool:
    return isinstance(value, jax.Array) and not value.is_fully_addressable


def _probe_loss(
    params: JaxProbeParams,
    *,
    variant: str,
    task_spec: MsgProbeTaskSpec,
    batch: JaxBatch,
    features: JaxFeatures,
) -> Array:
    logits = _probe_logits(
        params,
        variant=variant,
        task_spec=task_spec,
        features=features,
        valid_mask=batch["peak_valid_mask"],
    )
    valid_weight = batch["probe_valid_mol"].astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(valid_weight), 1.0)
    losses = []
    joint_logits = (
        logits[task_spec.fingerprint_task] if task_spec.maccs_bits > 0 else None
    )
    for regression_idx, name in enumerate(task_spec.regression_tasks):
        target = batch[f"probe_{name}"].astype(jnp.float32)
        mean = task_spec.regression_means[name]
        std = task_spec.regression_stds[name]
        pred = (
            logits[name].squeeze(-1)
            if joint_logits is None
            else joint_logits[:, regression_idx]
        )
        losses.append(jnp.sum(jnp.square(pred - (target - mean) / std) * valid_weight) / denom)
    for name in task_spec.binary_tasks:
        target = batch[f"probe_{name}"].astype(jnp.float32)
        pred = logits[name].squeeze(-1)
        losses.append(
            jnp.sum(optax.sigmoid_binary_cross_entropy(pred, target) * valid_weight)
            / denom
        )
    if task_spec.maccs_bits > 0:
        target = batch[f"probe_{task_spec.fingerprint_task}"].astype(jnp.float32)
        pred = joint_logits[:, len(task_spec.regression_tasks):]
        fingerprint_loss = optax.sigmoid_binary_cross_entropy(pred, target)
        losses.append(
            jnp.sum(fingerprint_loss * valid_weight[:, None])
            / jnp.maximum(denom * float(task_spec.maccs_bits), 1.0)
        )
    return jnp.stack(losses).mean()


def _probe_predictions(
    params: JaxProbeParams,
    *,
    variant: str,
    task_spec: MsgProbeTaskSpec,
    batch: JaxBatch,
    features: JaxFeatures,
) -> dict[str, Any]:
    logits = _probe_logits(
        params,
        variant=variant,
        task_spec=task_spec,
        features=features,
        valid_mask=batch["peak_valid_mask"],
    )
    return _probe_predictions_from_logits(
        logits,
        task_spec=task_spec,
        batch=batch,
    )


def _append_pending_prediction(
    pending: list[PendingPrediction],
    *,
    logits: dict[str, Array],
    batch: JaxBatch,
    task_spec: MsgProbeTaskSpec,
) -> None:
    valid_mask = _host_local_array(batch["probe_valid_mol"]).astype(bool, copy=False)
    if not valid_mask.any():
        jax.block_until_ready(logits)
        return
    pending.append(
        (
            logits,
            valid_mask,
            _probe_targets_from_batch(
                batch,
                valid_mask=valid_mask,
                task_spec=task_spec,
            ),
        )
    )


def _flush_variant_prediction_queues(
    queues: dict[str, list[PendingPrediction]],
    states: dict[str, EpochState],
    task_spec: MsgProbeTaskSpec,
) -> None:
    for variant, pending in queues.items():
        _flush_pending_predictions(states[variant], pending, task_spec)


def _flush_pending_predictions(
    epoch_state: EpochState,
    pending: list[PendingPrediction],
    task_spec: MsgProbeTaskSpec,
) -> None:
    if not pending:
        return
    host_logits = [_host_local_tree(logits) for logits, _, _ in pending]
    for (logits, valid_mask, targets), host_logit in zip(
        pending,
        host_logits,
        strict=True,
    ):
        del logits
        _update_epoch_state_from_predictions(
            epoch_state,
            _probe_predictions_from_host_logits(
                host_logit,
                valid_mask=valid_mask,
                target_values=targets,
                task_spec=task_spec,
            ),
            task_spec,
        )
    pending.clear()


def _probe_predictions_from_logits(
    logits: dict[str, Array],
    *,
    task_spec: MsgProbeTaskSpec,
    batch: JaxBatch,
) -> dict[str, Any]:
    valid_mask = _host_local_array(batch["probe_valid_mol"]).astype(bool, copy=False)
    if not valid_mask.any():
        return {"batch_size": 0, "predictions": {}, "targets": {}}
    targets = _probe_targets_from_batch(
        batch,
        valid_mask=valid_mask,
        task_spec=task_spec,
    )
    return _probe_predictions_from_host_logits(
        _host_local_tree(logits),
        valid_mask=valid_mask,
        target_values=targets,
        task_spec=task_spec,
    )


def _probe_predictions_from_host_logits(
    logits: dict[str, np.ndarray],
    *,
    valid_mask: np.ndarray,
    target_values: dict[str, np.ndarray],
    task_spec: MsgProbeTaskSpec,
) -> dict[str, Any]:
    batch_size = int(np.count_nonzero(valid_mask))
    if batch_size == 0:
        return {"batch_size": 0, "predictions": {}, "targets": {}}
    predictions, targets = {}, {}
    joint_logits = (
        logits[task_spec.fingerprint_task] if task_spec.maccs_bits > 0 else None
    )
    for regression_idx, name in enumerate(task_spec.regression_tasks):
        mean = task_spec.regression_means[name]
        std = task_spec.regression_stds[name]
        pred = (
            np.asarray(logits[name]).squeeze(-1)
            if joint_logits is None
            else np.asarray(joint_logits)[:, regression_idx]
        )
        predictions[name] = pred[valid_mask] * std + mean
    for name in task_spec.regression_tasks:
        targets[name] = target_values[name]
    for name in task_spec.binary_tasks:
        pred = _sigmoid_numpy(np.asarray(logits[name]).squeeze(-1))
        predictions[name] = pred[valid_mask]
        targets[name] = target_values[name]
    if task_spec.maccs_bits > 0:
        assert joint_logits is not None
        pred = _sigmoid_numpy(
            np.asarray(joint_logits)[:, len(task_spec.regression_tasks):]
        )
        fingerprint_task = task_spec.fingerprint_task
        predictions[fingerprint_task] = pred[valid_mask]
        targets[fingerprint_task] = target_values[fingerprint_task]
    return {
        "batch_size": batch_size,
        "predictions": predictions,
        "targets": targets,
    }


def _probe_targets_from_batch(
    batch: JaxBatch,
    *,
    valid_mask: np.ndarray,
    task_spec: MsgProbeTaskSpec,
) -> dict[str, np.ndarray]:
    targets = {
        name: _host_local_array(batch[f"probe_{name}"]).astype(np.float32, copy=False)[
            valid_mask
        ]
        for name in task_spec.regression_tasks
    }
    targets.update(
        {
            name: _host_local_array(batch[f"probe_{name}"]).astype(
                np.float32,
                copy=False,
            )[valid_mask]
            for name in task_spec.binary_tasks
        }
    )
    if task_spec.maccs_bits > 0:
        fingerprint_task = task_spec.fingerprint_task
        targets[fingerprint_task] = _host_local_array(
            batch[f"probe_{fingerprint_task}"]
        ).astype(np.float32, copy=False)[valid_mask]
    return targets


def _host_local_tree(tree: Any) -> Any:
    return jax.tree.map(_host_local_array, tree)


def _host_local_array(value: Any) -> np.ndarray:
    if _is_global_non_fully_addressable_array(value):
        shards = sorted(
            value.addressable_shards,
            key=lambda shard: shard.index[0].start or 0,
        )
        return np.concatenate([np.asarray(shard.data) for shard in shards], axis=0)
    return np.asarray(jax.device_get(value))


def _sigmoid_numpy(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _probe_logits(
    params: JaxProbeParams,
    *,
    variant: str,
    task_spec: MsgProbeTaskSpec,
    features: JaxFeatures,
    valid_mask: Array,
) -> dict[str, Array]:
    pooled = _pool_features(
        params.get("pooler", {}),
        variant=variant,
        task_spec=task_spec,
        features=features,
        valid_mask=valid_mask,
    )
    return {
        name: _mlp_apply(params["heads"][name], pooled)
        for name in _probe_task_names(task_spec)
    }


def _pool_features(
    params: dict[str, Array],
    *,
    variant: str,
    task_spec: MsgProbeTaskSpec,
    features: JaxFeatures,
    valid_mask: Array,
) -> Array:
    peak_embeddings = _feature_single(features).astype(jnp.float32)
    if variant == "mean":
        return _mean_pool(peak_embeddings, valid_mask)
    if variant == "covariance":
        return _covariance_pool(
            peak_embeddings,
            valid_mask,
            left=params["left"],
            right=params["right"],
        )
    if variant == "cls":
        pair_embeddings = _feature_pair(features).astype(jnp.float32)
        cls_idx = valid_mask.shape[1]
        return jnp.concatenate(
            [peak_embeddings[:, cls_idx], pair_embeddings[:, cls_idx, cls_idx]],
            axis=-1,
        )
    if _is_single_pair_covariance_variant(variant):
        pair_embeddings = _feature_pair(features).astype(jnp.float32)
        return _single_pair_covariance_pool(
            peak_embeddings,
            valid_mask,
            pair_embeddings,
            params=params,
            include_diagonal=task_spec.single_pair_covariance_include_diagonal,
        )
    raise ValueError(f"Unsupported JAX MSG probe variant: {variant!r}")


def _mean_pool(peak_embeddings: Array, valid_mask: Array) -> Array:
    peak_embeddings = peak_embeddings[:, : valid_mask.shape[1]]
    mask = valid_mask[..., None].astype(jnp.float32)
    return jnp.sum(peak_embeddings * mask, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1.0)


def _covariance_pool(
    peak_embeddings: Array,
    valid_mask: Array,
    *,
    left: Array,
    right: Array,
) -> Array:
    peak_embeddings = peak_embeddings[:, : valid_mask.shape[1]]
    mask = valid_mask[..., None].astype(jnp.float32)
    left_proj = jnp.matmul(peak_embeddings, left) * mask
    right_proj = jnp.matmul(peak_embeddings, right) * mask
    denom = jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    covariance = jnp.einsum("btc,btd->bcd", left_proj, right_proj)
    return (covariance / denom[:, None]).reshape(peak_embeddings.shape[0], -1)


def _single_pair_covariance_pool(
    peak_embeddings: Array,
    valid_mask: Array,
    pair_embeddings: Array,
    *,
    params: dict[str, Array],
    include_diagonal: bool,
) -> Array:
    token_mask = valid_mask
    num_tokens = token_mask.shape[1]
    single = _covariance_pool(
        peak_embeddings[:, :num_tokens],
        token_mask,
        left=params["single_left"],
        right=params["single_right"],
    )
    if pair_embeddings.shape[1] >= num_tokens and pair_embeddings.shape[2] >= num_tokens:
        pair_embeddings = pair_embeddings[:, :num_tokens, :num_tokens]
        pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    else:
        num_tokens = pair_embeddings.shape[1]
        pair_mask = jnp.ones(
            (pair_embeddings.shape[0], num_tokens, num_tokens),
            dtype=jnp.bool_,
        )
    if not include_diagonal:
        diagonal = jnp.eye(num_tokens, dtype=bool)[None]
        pair_mask = pair_mask & ~diagonal
    pair_mask_f = pair_mask[..., None].astype(jnp.float32)
    left = jnp.matmul(pair_embeddings, params["pair_left"]) * pair_mask_f
    right = jnp.matmul(pair_embeddings, params["pair_right"]) * pair_mask_f
    batch_size = pair_embeddings.shape[0]
    compressed_dim = left.shape[-1]
    left = left.reshape(batch_size, num_tokens * num_tokens, compressed_dim)
    right = right.reshape(batch_size, num_tokens * num_tokens, compressed_dim)
    denom = jnp.maximum(jnp.sum(pair_mask_f, axis=(1, 2)), 1.0)
    pair_covariance = jnp.einsum("btc,btd->bcd", left, right)
    pair = (pair_covariance / denom[:, None]).reshape(batch_size, -1)
    pooled = jnp.concatenate([_layer_norm(single), _layer_norm(pair)], axis=-1)
    return jnp.matmul(pooled, params["output_w"]) + params["output_b"]


@nnx.jit
def _extract_pair_features_jitted(
    model: PeakSetJEPAJax,
    peak_mz: Array,
    peak_intensity: Array,
    peak_valid_mask: Array,
    precursor_mz: Array | None,
) -> tuple[Array, Array]:
    single, pair = model.encoder.forward_with_pair(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        precursor_mz=precursor_mz,
    )
    if isinstance(pair, InducedPairState):
        pair = pair.pair
    return single, pair


@nnx.jit
def _extract_single_features_jitted(
    model: PeakSetJEPAJax,
    peak_mz: Array,
    peak_intensity: Array,
    peak_valid_mask: Array,
    precursor_mz: Array | None,
) -> Array:
    return model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        precursor_mz=precursor_mz,
    )


def _extract_features(
    model: PeakSetJEPAJax,
    batch: JaxBatch,
    *,
    use_pair_features: bool,
) -> JaxFeatures:
    precursor_mz = batch.get("precursor_mz", None)
    if use_pair_features:
        return _extract_pair_features_jitted(
            model,
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
            precursor_mz,
        )
    return _extract_single_features_jitted(
        model,
        batch["peak_mz"],
        batch["peak_intensity"],
        batch["peak_valid_mask"],
        precursor_mz,
    )


def _init_probe_params(
    key: Array,
    *,
    variant: str,
    config: config_dict.ConfigDict,
    task_spec: MsgProbeTaskSpec,
) -> JaxProbeParams:
    model_dim = int(config.model_dim)
    hidden_dim = int(_config_get(config, "msg_probe_mlp_hidden_dim", model_dim))
    num_layers = int(_config_get(config, "msg_probe_mlp_num_layers", 2))
    key, pool_key, heads_key = jax.random.split(key, 3)
    pooler, pooled_dim = _init_pooler(
        pool_key,
        variant=variant,
        config=config,
        model_dim=model_dim,
    )
    task_names = _probe_task_names(task_spec)
    output_dims = _probe_task_output_dims(task_spec)
    head_keys = jax.random.split(heads_key, len(task_names))
    heads = {
        name: _init_mlp(
            head_key,
            input_dim=pooled_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dims.get(name, 1),
            num_layers=num_layers,
        )
        for name, head_key in zip(task_names, head_keys, strict=True)
    }
    return {"pooler": pooler, "heads": heads}


def _init_pooler(
    key: Array,
    *,
    variant: str,
    config: config_dict.ConfigDict,
    model_dim: int,
) -> tuple[dict[str, Array], int]:
    if variant == "mean":
        return {}, model_dim
    if variant == "covariance":
        compressed_dim = int(_config_get(config, "covariance_pooling_dim", 32))
        left_key, right_key = jax.random.split(key)
        return {
            "left": _xavier(left_key, model_dim, compressed_dim),
            "right": _xavier(right_key, model_dim, compressed_dim),
        }, compressed_dim * compressed_dim
    if variant == "cls":
        pair_dim = int(_config_get(config, "pairmixer_pair_dim", model_dim))
        return {}, model_dim + pair_dim
    if _is_single_pair_covariance_variant(variant):
        compressed_dim = int(_config_get(config, "covariance_pooling_dim", 32))
        pair_dim = int(_config_get(config, "pairmixer_pair_dim", model_dim))
        keys = jax.random.split(key, 5)
        output_dim = compressed_dim * compressed_dim
        return {
            "single_left": _xavier(keys[0], model_dim, compressed_dim),
            "single_right": _xavier(keys[1], model_dim, compressed_dim),
            "pair_left": _xavier(keys[2], pair_dim, compressed_dim),
            "pair_right": _xavier(keys[3], pair_dim, compressed_dim),
            "output_w": _xavier(keys[4], 2 * output_dim, output_dim),
            "output_b": jnp.zeros((output_dim,), dtype=jnp.float32),
        }, output_dim
    raise ValueError(f"Unsupported JAX MSG probe variant: {variant!r}")


def _init_mlp(
    key: Array,
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> list[dict[str, Array]]:
    if num_layers == 1:
        return [_init_output_layer(key, input_dim, output_dim)]
    keys = jax.random.split(key, num_layers)
    layers = [_init_hidden_layer(keys[0], input_dim, hidden_dim)]
    for idx in range(1, num_layers - 1):
        layers.append(_init_hidden_layer(keys[idx], hidden_dim, hidden_dim))
    layers.append(_init_output_layer(keys[-1], hidden_dim, output_dim))
    return layers


def _mlp_apply(layers: list[dict[str, Array]], x: Array) -> Array:
    for idx, layer in enumerate(layers):
        x = jnp.matmul(x, layer["w"]) + layer["b"]
        if idx != len(layers) - 1:
            x = jax.nn.silu(x)
    return x


def _init_hidden_layer(key: Array, input_dim: int, output_dim: int) -> dict[str, Array]:
    return {
        "w": _xavier(key, input_dim, output_dim),
        "b": jnp.zeros((output_dim,), dtype=jnp.float32),
    }


def _init_output_layer(key: Array, input_dim: int, output_dim: int) -> dict[str, Array]:
    return {
        "w": jax.random.normal(key, (input_dim, output_dim), dtype=jnp.float32) * 1e-3,
        "b": jnp.zeros((output_dim,), dtype=jnp.float32),
    }


def _xavier(key: Array, input_dim: int, output_dim: int) -> Array:
    limit = math.sqrt(6.0 / float(input_dim + output_dim))
    return jax.random.uniform(
        key,
        (input_dim, output_dim),
        minval=-limit,
        maxval=limit,
        dtype=jnp.float32,
    )


def _new_epoch_state(task_spec: MsgProbeTaskSpec) -> EpochState:
    task_names = _probe_prediction_names(task_spec)
    return {
        "count": 0,
        "predictions": {name: [] for name in task_names},
        "targets": {name: [] for name in task_names},
    }


def _update_epoch_state_from_predictions(
    epoch_state: EpochState,
    result: dict[str, Any],
    task_spec: MsgProbeTaskSpec,
) -> None:
    batch_size = int(result["batch_size"])
    if batch_size == 0:
        return
    epoch_state["count"] += batch_size
    for name in _probe_prediction_names(task_spec):
        epoch_state["predictions"][name].append(result["predictions"][name])
        epoch_state["targets"][name].append(result["targets"][name])


def _score_epoch_state(
    *,
    prefix: str,
    epoch_state: EpochState,
    task_spec: MsgProbeTaskSpec,
    include_pr_curves: bool = False,
) -> dict[str, Any]:
    count = int(epoch_state["count"])
    metrics: dict[str, Any] = {f"{prefix}/samples": float(count)}
    regression_r2_values, regression_mae_values = [], []
    predictions = epoch_state["predictions"]
    targets = epoch_state["targets"]
    for name in task_spec.regression_tasks:
        pred = np.concatenate(predictions[name], axis=0)
        target = np.concatenate(targets[name], axis=0)
        metrics[f"{prefix}/r2_{name}"] = _r2_score(target, pred)
        metrics[f"{prefix}/mae_{name}"] = float(np.mean(np.abs(target - pred)))
        regression_r2_values.append(metrics[f"{prefix}/r2_{name}"])
        regression_mae_values.append(metrics[f"{prefix}/mae_{name}"])
    for name in task_spec.binary_tasks:
        pred = np.concatenate(predictions[name], axis=0).astype(np.float64)
        target = np.concatenate(targets[name], axis=0).astype(np.float64)
        metrics.update(_binary_metrics(prefix, name, pred, target))
        if include_pr_curves and name in ("fluorine", "sulfur"):
            metrics[f"{prefix}/pr_curve_{name}"] = build_precision_recall_curve(
                prefix=prefix,
                name=name,
                pred=pred,
                target=target,
            )
    if task_spec.maccs_bits > 0:
        fingerprint_task = task_spec.fingerprint_task
        pred = np.concatenate(predictions[fingerprint_task], axis=0)
        target = np.concatenate(targets[fingerprint_task], axis=0)
        metrics.update(_fingerprint_metrics(prefix, fingerprint_task, pred, target))
    if regression_r2_values:
        metrics[f"{prefix}/r2_mean"] = float(np.mean(regression_r2_values))
        metrics[f"{prefix}/mae_mean"] = float(np.mean(regression_mae_values))
    return metrics


def _sulfur_metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key.endswith("/samples") or key.rsplit("/", 1)[-1].endswith("_sulfur")
    }


def _binary_metrics(
    prefix: str,
    name: str,
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    positives = float(target.sum())
    negatives = float(target.shape[0]) - positives
    if positives > 0 and negatives > 0:
        order = np.argsort(pred)
        target_ordered = target[order]
        negatives_before = np.cumsum(1.0 - target_ordered)
        auc = float((target_ordered * negatives_before).sum() / (positives * negatives))
        target_descending = target_ordered[::-1]
        true_positives_at_rank = np.cumsum(target_descending)
        ranks = np.arange(1, target_descending.shape[0] + 1, dtype=np.float64)
        average_precision = float(
            (target_descending * true_positives_at_rank / ranks).sum() / positives
        )
    else:
        auc = float("nan")
        average_precision = float("nan")
    predicted = pred >= 0.5
    target_bits = target > 0
    true_positives = float(np.count_nonzero(predicted & target_bits))
    predicted_positives = float(np.count_nonzero(predicted))
    return {
        f"{prefix}/positive_{name}": positives,
        f"{prefix}/auc_{name}": auc,
        f"{prefix}/average_precision_{name}": average_precision,
        f"{prefix}/recall_{name}": (
            true_positives / positives if positives > 0 else float("nan")
        ),
        f"{prefix}/precision_{name}": (
            true_positives / predicted_positives
            if predicted_positives > 0
            else float("nan")
        ),
    }


def _fingerprint_metrics(
    prefix: str,
    fingerprint_task: str,
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    positives = target.sum(axis=0)
    valid_metric_mask = (positives > 0) & (positives < target.shape[0])
    if np.count_nonzero(valid_metric_mask) > 0:
        valid_target = target[:, valid_metric_mask].astype(np.float64)
        valid_pred = pred[:, valid_metric_mask].astype(np.float64)
        valid_positives = positives[valid_metric_mask].astype(np.float64)
        valid_negatives = float(target.shape[0]) - valid_positives
        ascending = np.argsort(valid_pred, axis=0)
        target_ascending = np.take_along_axis(valid_target, ascending, axis=0)
        negatives_before = np.cumsum(1.0 - target_ascending, axis=0)
        auc_values = (
            (target_ascending * negatives_before).sum(axis=0)
            / (valid_positives * valid_negatives)
        )
        target_descending = target_ascending[::-1]
        true_positives_at_rank = np.cumsum(target_descending, axis=0)
        ranks = np.arange(1, target_descending.shape[0] + 1, dtype=np.float64)[:, None]
        average_precision_values = (
            (target_descending * true_positives_at_rank / ranks).sum(axis=0)
            / valid_positives
        )
    else:
        auc_values = np.asarray([], dtype=np.float64)
        average_precision_values = np.asarray([], dtype=np.float64)
    positive_mask = positives > 0
    bit_pred = pred >= 0.5
    target_bits = target > 0
    true_positives = (bit_pred & target_bits).sum(axis=0)
    predicted_positives = bit_pred.sum(axis=0)
    recall_values = true_positives[positive_mask] / positives[positive_mask]
    precision_values = np.divide(
        true_positives[positive_mask],
        predicted_positives[positive_mask],
        out=np.zeros_like(true_positives[positive_mask], dtype=np.float64),
        where=predicted_positives[positive_mask] > 0,
    )
    intersection = np.count_nonzero(bit_pred & target_bits, axis=1)
    union = np.count_nonzero(bit_pred | target_bits, axis=1)
    tanimoto_values = intersection / np.maximum(union, 1)
    dot = np.sum(pred * target, axis=1)
    cosine_values = dot / np.maximum(
        np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1),
        1e-12,
    )
    return {
        f"{prefix}/num_{fingerprint_task}_auc_bits": float(len(auc_values)),
        f"{prefix}/num_{fingerprint_task}_average_precision_bits": float(
            len(average_precision_values)
        ),
        f"{prefix}/num_{fingerprint_task}_recall_bits": float(len(recall_values)),
        f"{prefix}/num_{fingerprint_task}_precision_bits": float(len(precision_values)),
        f"{prefix}/auc_{fingerprint_task}_mean": (
            float(np.mean(auc_values)) if len(auc_values) else float("nan")
        ),
        f"{prefix}/average_precision_{fingerprint_task}_mean": (
            float(np.mean(average_precision_values))
            if len(average_precision_values)
            else float("nan")
        ),
        f"{prefix}/recall_{fingerprint_task}_mean": (
            float(np.mean(recall_values)) if len(recall_values) else float("nan")
        ),
        f"{prefix}/precision_{fingerprint_task}_mean": (
            float(np.mean(precision_values)) if len(precision_values) else float("nan")
        ),
        f"{prefix}/tanimoto_{fingerprint_task}_mean": float(np.mean(tanimoto_values)),
        f"{prefix}/cosine_{fingerprint_task}_mean": float(np.mean(cosine_values)),
    }


def _run_repeated_probe_jax(
    *,
    repeat_count: int,
    metric_prefix: str,
    run_once: Callable[
        [int, Callable[[dict[str, float]], None] | None],
        dict[str, Any],
    ],
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    if repeat_count == 1:
        metrics = dict(run_once(0, on_epoch_end))
        if metrics:
            metrics[f"{metric_prefix}/repeats"] = 1.0
        return metrics

    repeat_metrics: list[dict[str, Any]] = []
    repeat_curves: list[list[dict[str, float]]] = []
    for repeat_idx in range(repeat_count):
        repeat_curve: list[dict[str, float]] = []
        metrics = run_once(repeat_idx, repeat_curve.append)
        if metrics:
            repeat_metrics.append(metrics)
        if repeat_curve:
            repeat_curves.append(repeat_curve)
    averaged_metrics = _average_metric_dicts(repeat_metrics)
    if averaged_metrics:
        averaged_metrics[f"{metric_prefix}/repeats"] = float(repeat_count)
    if on_epoch_end is not None and repeat_curves:
        num_epochs = max(len(curve) for curve in repeat_curves)
        for epoch_idx in range(num_epochs):
            epoch_metrics = _average_metric_dicts(
                [curve[epoch_idx] for curve in repeat_curves if epoch_idx < len(curve)]
            )
            if epoch_metrics:
                on_epoch_end(epoch_metrics)
    return averaged_metrics


def resolve_msg_probe_select_metric_jax(config: Any) -> str:
    validate_msg_probe_config(config)
    return str(_config_get(config, "msg_probe_select_metric", "msg_probe/test/auc_fluorine"))


def msg_probe_metric_higher_is_better(metric_key: str) -> bool:
    return "/mae_" not in metric_key


def _probe_task_names(task_spec: MsgProbeTaskSpec) -> tuple[str, ...]:
    task_names = task_spec.regression_tasks + task_spec.binary_tasks
    if task_spec.maccs_bits > 0:
        task_names += (task_spec.fingerprint_task,)
    return task_names


def _probe_prediction_names(task_spec: MsgProbeTaskSpec) -> tuple[str, ...]:
    return _probe_task_names(task_spec)


def _probe_task_output_dims(task_spec: MsgProbeTaskSpec) -> dict[str, int]:
    if task_spec.maccs_bits <= 0:
        return {}
    return {
        task_spec.fingerprint_task: (
            len(task_spec.regression_tasks) + task_spec.maccs_bits
        )
    }


def _msg_probe_variant_metric_key(variant: str, metric_key: str) -> str:
    variant_prefix = f"msg_probe/{variant}/"
    if metric_key.startswith(variant_prefix):
        return metric_key
    if metric_key.startswith("msg_probe/"):
        return variant_prefix + metric_key[len("msg_probe/"):]
    return metric_key


def _uses_pair_features(variant: str) -> bool:
    return variant == "cls" or _is_single_pair_covariance_variant(variant)


def _is_single_pair_covariance_variant(variant: str) -> bool:
    return variant in ("single_pair_covariance", "pair_covariance")


def _feature_single(features: JaxFeatures) -> Array:
    return features[0] if isinstance(features, tuple) else features


def _feature_pair(features: JaxFeatures) -> Array:
    assert isinstance(features, tuple)
    return features[1]


def _layer_norm(value: Array, eps: float = 1e-5) -> Array:
    mean = jnp.mean(value, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(value - mean), axis=-1, keepdims=True)
    return (value - mean) * jax.lax.rsqrt(var + eps)


def _probe_lr_schedule(
    *,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
) -> Callable[[Array], Array]:
    min_lr = 0.1 * base_lr

    def schedule(step: Array) -> Array:
        step = step.astype(jnp.float32) + 1.0
        warmup = base_lr * (1e-8 + (1.0 - 1e-8) * step / max(1, warmup_steps))
        ratio = jnp.clip(
            (step - float(warmup_steps)) / float(max(1, total_steps - warmup_steps)),
            0.0,
            1.0,
        )
        decay = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + jnp.cos(jnp.pi * ratio))
        if warmup_steps <= 0:
            return decay
        return jnp.where(step < warmup_steps, warmup, decay)

    return schedule


def _resolve_probe_warmup_steps(config: Any, steps_per_epoch: int) -> int:
    warmup_epochs = _config_get(config, "msg_probe_warmup_epochs", None)
    if warmup_epochs is not None:
        return int(round(float(warmup_epochs) * steps_per_epoch))
    return int(_config_get(config, "msg_probe_warmup_steps", 100))


def _average_metric_dicts(metric_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    artifacts: dict[str, Any] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            if not isinstance(value, (int, float, np.number)):
                artifacts.setdefault(key, value)
                continue
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {
        **{key: totals[key] / counts[key] for key in totals},
        **artifacts,
    }


def _r2_score(target: np.ndarray, pred: np.ndarray) -> float:
    residual = float(np.sum(np.square(target - pred)))
    centered = target - float(np.mean(target))
    total = float(np.sum(np.square(centered)))
    return 1.0 - residual / total if total > 0 else float("nan")


def _log_epoch_metrics(
    *,
    variant: str,
    epoch_idx: int,
    num_probe_epochs: int,
    fingerprint_task: str,
    metrics: dict[str, float],
    early_stopping: bool,
) -> None:
    split = "val" if early_stopping else "test"
    prefix = f"msg_probe/{variant}"
    log.info(
        "JAX MSG probe [%s] epoch %d/%d train_samples=%d %s_auc_%s_mean=%.4f",
        variant,
        epoch_idx + 1,
        num_probe_epochs,
        int(metrics[f"{prefix}/train/samples"]),
        split,
        fingerprint_task,
        metrics[f"{prefix}/{split}/auc_{fingerprint_task}_mean"],
    )


def _slice_batch(batch: JaxBatch, take: int) -> JaxBatch:
    sliced = {}
    for key, value in batch.items():
        if isinstance(value, list):
            sliced[key] = value[:take]
        elif hasattr(value, "shape") and value.shape and value.shape[0] >= take:
            sliced[key] = value[:take]
        else:
            sliced[key] = value
    return sliced


def _clone_tree(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: jnp.array(value, copy=True)
        if isinstance(value, jax.Array)
        else value,
        tree,
    )


def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _optional_positive_int(config: Any, key: str) -> int | None:
    raw = _config_get(config, key, None)
    if raw is None:
        return None
    value = int(raw)
    return value if value > 0 else None
