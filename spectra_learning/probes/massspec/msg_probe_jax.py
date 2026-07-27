from __future__ import annotations

import logging
import math
import pickle
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ml_collections import config_dict

from spectra_learning.data.contracts import peak_preprocessing_contract
from spectra_learning.data.loading import local_batch_size
from spectra_learning.data.massspec_probe import MassSpecProbeData
from spectra_learning.models.common_jax import Array
from spectra_learning.models.fastmixer_capacity import (
    pairmixer_fast_full_visible_tokens,
)
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch
from spectra_learning.probes.massspec.msg_probe_common import (
    EpochState,
    merge_epoch_states as _merge_epoch_states,
    msg_probe_metric_higher_is_better,
    msg_probe_variant_metric_key as _msg_probe_variant_metric_key,
    new_epoch_state as _new_epoch_state,
    probe_prediction_names as _probe_prediction_names,
    probe_task_names as _probe_task_names,
    probe_task_output_dims as _probe_task_output_dims,
    resolve_msg_probe_select_metric as resolve_msg_probe_select_metric_jax,
    resolve_probe_warmup_steps as _resolve_probe_warmup_steps,
    run_repeated_probe as _run_repeated_probe_jax,
    score_epoch_state as _score_epoch_state,
)
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


log = logging.getLogger(__name__)

JaxBatch = dict[str, Any]
JaxFeatures = Array | tuple[Array, Array]
JaxProbeParams = dict[str, Any]
JaxProbeTrainStep = Callable[..., tuple[JaxProbeParams, optax.OptState, dict[str, Array]]]
JaxProbePredictStep = Callable[..., dict[str, Array]]
PendingPrediction = tuple[dict[str, Array], np.ndarray, dict[str, np.ndarray]]
JAX_PROBE_DATA_AXIS = "data"
MAX_PENDING_PREDICTIONS = 4


@dataclass(frozen=True)
class _JaxMsgProbeSetup:
    probe_data: MassSpecProbeData
    feature_model: PeakSetJEPAJax
    task_spec: MsgProbeTaskSpec
    variants: tuple[str, ...]
    probe_lr: float
    probe_weight_decay: float
    num_probe_epochs: int
    peak_ordering: str
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_min_epochs: int
    fingerprint_task: str
    use_pair_features: bool
    max_train_samples: int | None
    max_val_samples: int | None
    max_test_samples: int | None
    train_seed_base: int
    test_seed_base: int
    probe_seed: int
    online_maccs_only: bool
    data_mesh: Mesh | None


@dataclass
class _JaxMsgProbeTrainingState:
    params_by_variant: dict[str, JaxProbeParams]
    opt_state_by_variant: dict[str, optax.OptState]
    train_step_by_variant: dict[str, JaxProbeTrainStep]
    predict_step_by_variant: dict[str, JaxProbePredictStep]
    select_metric: str
    higher_is_better: bool
    best_metrics_by_variant: dict[str, dict[str, Any]]
    best_params_by_variant: dict[str, JaxProbeParams]
    best_metric_values: dict[str, float]
    epochs_without_improvement: dict[str, int]


def run_msg_probe_jax(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetJEPAJax,
    data_mesh: Mesh | None = None,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
    online_maccs_only: bool = False,
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
            online_maccs_only=online_maccs_only,
        )

    return _run_repeated_probe_jax(
        repeat_count=resolve_msg_probe_num_repeats(config),
        metric_prefix="msg_probe",
        run_once=run_once,
        on_epoch_end=on_epoch_end,
    )


def _evaluate_probe_split_jax(
    *,
    probe_data: MassSpecProbeData,
    feature_model: PeakSetJEPAJax,
    task_spec: MsgProbeTaskSpec,
    params_by_variant: dict[str, JaxProbeParams],
    predict_step_by_variant: dict[str, JaxProbePredictStep],
    split: str,
    seed: int,
    peak_ordering: str,
    max_samples: int | None,
    use_pair_features: bool,
    data_mesh: Mesh | None,
) -> tuple[dict[str, EpochState], float]:
    states = {
        variant: _new_epoch_state(task_spec)
        for variant in params_by_variant
    }
    pending_by_variant = {variant: [] for variant in params_by_variant}
    phase_start = time.perf_counter()
    for batch in iter_massspec_probe_jax(
        probe_data=probe_data,
        split=split,
        seed=seed,
        peak_ordering=peak_ordering,
        drop_remainder=False,
        max_samples=max_samples,
        distributed_world_size=_distributed_world_size_jax(),
        distributed_rank=_distributed_rank_jax(),
        data_mesh=data_mesh,
    ):
        features = _extract_features(
            feature_model,
            batch,
            use_pair_features=use_pair_features,
        )
        step_batch = _probe_step_batch(batch, task_spec)
        for variant, params in params_by_variant.items():
            _append_pending_prediction(
                pending_by_variant[variant],
                logits=predict_step_by_variant[variant](
                    params,
                    step_batch,
                    features,
                ),
                batch=batch,
                task_spec=task_spec,
            )
            _flush_pending_predictions_if_full(
                states[variant],
                pending_by_variant[variant],
                task_spec,
            )
    _flush_variant_prediction_queues(pending_by_variant, states, task_spec)
    return states, time.perf_counter() - phase_start


def _collect_msg_probe_task_spec_jax(
    *,
    config: config_dict.ConfigDict,
    probe_data: MassSpecProbeData,
    peak_ordering: str,
    train_seed_base: int,
    fingerprint_task: str,
    regression_tasks: tuple[str, ...],
    binary_tasks: tuple[str, ...],
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> tuple[MsgProbeTaskSpec, float]:
    del max_val_samples
    phase_start = time.perf_counter()
    train_targets = _collect_split_targets_jax(
        probe_data=probe_data,
        split="massspec_train",
        peak_ordering=peak_ordering,
        seed=train_seed_base,
        fingerprint_task=fingerprint_task,
        regression_tasks=regression_tasks,
        binary_tasks=binary_tasks,
        max_samples=max_train_samples,
    )
    elapsed = time.perf_counter() - phase_start
    task_spec = _build_task_spec_jax(
        train_targets=train_targets,
        test_targets=train_targets,
        fingerprint_task=fingerprint_task,
        regression_tasks=regression_tasks,
        binary_tasks=binary_tasks,
        single_pair_covariance_include_diagonal=bool(
            config.get(
                "msg_probe_single_pair_covariance_include_diagonal",
                False,
            )
        ),
    )
    return task_spec, elapsed


def _setup_msg_probe_jax(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetJEPAJax,
    data_mesh: Mesh | None,
    repeat_index: int,
    online_maccs_only: bool,
) -> tuple[_JaxMsgProbeSetup, dict[str, float], float]:
    num_probe_epochs = int(config.get("msg_probe_num_epochs", 5))
    probe_lr = float(config.get("msg_probe_learning_rate", 1e-3))
    probe_weight_decay = float(config.get("msg_probe_weight_decay", 1e-2))
    early_stopping = bool(config.get("msg_probe_early_stopping", False))
    early_stopping_patience = int(
        config.get("msg_probe_early_stopping_patience", 10)
    )
    early_stopping_min_delta = float(
        config.get("msg_probe_early_stopping_min_delta", 0.0)
    )
    early_stopping_min_epochs = int(
        config.get("msg_probe_early_stopping_min_epochs", 1)
    )
    fingerprint_task = (
        MACCS_TASK if online_maccs_only else resolve_msg_probe_fingerprint(config)
    )
    regression_tasks = () if online_maccs_only else REGRESSION_PROBE_TASKS
    binary_tasks = () if online_maccs_only else BINARY_PROBE_TASKS
    probe_data_kwargs: dict[str, Any] = {
        "distributed_world_size": _distributed_world_size_jax(),
        "distributed_rank": _distributed_rank_jax(),
        "distributed_local_rank": 0,
    }
    if online_maccs_only:
        probe_data_kwargs["maccs_only"] = True
    probe_data = MassSpecProbeData.from_config(config, **probe_data_kwargs)
    peak_ordering = peak_preprocessing_contract(config)["peak_ordering"]
    variants = msg_probe_variants_from_config(config)
    use_pair_features = any(_uses_pair_features(variant) for variant in variants)
    feature_model = _full_visible_fastmixer_probe_model(config, model)
    max_train_samples = _optional_positive_int(
        config, "jax_msg_probe_max_train_samples"
    )
    max_val_samples = _optional_positive_int(config, "jax_msg_probe_max_val_samples")
    max_test_samples = _optional_positive_int(config, "jax_msg_probe_max_test_samples")
    seed_offset = 100_000 * repeat_index
    train_seed_base = int(config.seed) + 1_100_000 + seed_offset
    test_seed_base = int(config.seed) + 1_200_000 + seed_offset
    phase_timing = {
        "target_collection_seconds": 0.0,
        "train_epoch_seconds": 0.0,
        "eval_epoch_seconds": 0.0,
        "score_epoch_seconds": 0.0,
        "final_test_seconds": 0.0,
        "final_score_seconds": 0.0,
    }
    probe_start = time.perf_counter()
    task_spec, target_elapsed = _collect_msg_probe_task_spec_jax(
        config=config,
        probe_data=probe_data,
        peak_ordering=peak_ordering,
        train_seed_base=train_seed_base,
        fingerprint_task=fingerprint_task,
        regression_tasks=regression_tasks,
        binary_tasks=binary_tasks,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    phase_timing["target_collection_seconds"] += target_elapsed
    setup = _JaxMsgProbeSetup(
        probe_data=probe_data,
        feature_model=feature_model,
        task_spec=task_spec,
        variants=variants,
        probe_lr=probe_lr,
        probe_weight_decay=probe_weight_decay,
        num_probe_epochs=num_probe_epochs,
        peak_ordering=peak_ordering,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_min_epochs=early_stopping_min_epochs,
        fingerprint_task=fingerprint_task,
        use_pair_features=use_pair_features,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        train_seed_base=train_seed_base,
        test_seed_base=test_seed_base,
        probe_seed=int(config.seed) + 300_000 + seed_offset,
        online_maccs_only=online_maccs_only,
        data_mesh=data_mesh,
    )
    return setup, phase_timing, probe_start


def _initialize_msg_probe_training_jax(
    config: config_dict.ConfigDict,
    setup: _JaxMsgProbeSetup,
) -> _JaxMsgProbeTrainingState:
    steps_per_epoch = probe_steps_per_epoch_jax(
        setup.probe_data,
        split="massspec_train",
        drop_remainder=False,
        max_samples=setup.max_train_samples,
        distributed_world_size=_distributed_world_size_jax(),
    )
    schedule = _probe_lr_schedule(
        base_lr=setup.probe_lr,
        total_steps=setup.num_probe_epochs * steps_per_epoch,
        warmup_steps=_resolve_probe_warmup_steps(config, steps_per_epoch),
    )
    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=setup.probe_weight_decay,
    )
    key = jax.random.PRNGKey(setup.probe_seed)
    params_by_variant: dict[str, JaxProbeParams] = {}
    opt_state_by_variant: dict[str, optax.OptState] = {}
    for variant in setup.variants:
        key, init_key = jax.random.split(key)
        params_by_variant[variant] = _init_probe_params(
            init_key,
            variant=variant,
            config=config,
            task_spec=setup.task_spec,
        )
        opt_state_by_variant[variant] = optimizer.init(params_by_variant[variant])
    select_metric = (
        "msg_probe/test/auc_maccs_mean"
        if setup.online_maccs_only
        else resolve_msg_probe_select_metric_jax(config)
    )
    higher_is_better = msg_probe_metric_higher_is_better(select_metric)
    return _JaxMsgProbeTrainingState(
        params_by_variant=params_by_variant,
        opt_state_by_variant=opt_state_by_variant,
        train_step_by_variant={
            variant: _make_jitted_probe_train_step(
                optimizer=optimizer,
                variant=variant,
                task_spec=setup.task_spec,
                distributed=_is_distributed_jax(),
            )
            for variant in setup.variants
        },
        predict_step_by_variant={
            variant: _make_jitted_probe_predict_step(
                variant=variant,
                task_spec=setup.task_spec,
            )
            for variant in setup.variants
        },
        select_metric=select_metric,
        higher_is_better=higher_is_better,
        best_metrics_by_variant={},
        best_params_by_variant={},
        best_metric_values={
            variant: -float("inf") if higher_is_better else float("inf")
            for variant in setup.variants
        },
        epochs_without_improvement={variant: 0 for variant in setup.variants},
    )


def _train_and_evaluate_msg_probe_epoch_jax(
    setup: _JaxMsgProbeSetup,
    state: _JaxMsgProbeTrainingState,
    epoch_idx: int,
) -> tuple[dict[str, EpochState], dict[str, EpochState], float, float]:
    train_states = {
        variant: _new_epoch_state(setup.task_spec)
        for variant in setup.variants
    }
    train_pending = {variant: [] for variant in setup.variants}
    phase_start = time.perf_counter()
    for batch in iter_massspec_probe_jax(
        probe_data=setup.probe_data,
        split="massspec_train",
        seed=setup.train_seed_base + epoch_idx,
        peak_ordering=setup.peak_ordering,
        drop_remainder=False,
        max_samples=setup.max_train_samples,
        distributed_world_size=_distributed_world_size_jax(),
        distributed_rank=_distributed_rank_jax(),
        pad_distributed=True,
        data_mesh=setup.data_mesh,
    ):
        features = _extract_features(
            setup.feature_model,
            batch,
            use_pair_features=setup.use_pair_features,
        )
        step_batch = _probe_step_batch(batch, setup.task_spec)
        for variant in setup.variants:
            params, opt_state, logits = state.train_step_by_variant[variant](
                state.params_by_variant[variant],
                state.opt_state_by_variant[variant],
                batch=step_batch,
                features=features,
            )
            state.params_by_variant[variant] = params
            state.opt_state_by_variant[variant] = opt_state
            _append_pending_prediction(
                train_pending[variant],
                logits=logits,
                batch=batch,
                task_spec=setup.task_spec,
            )
            _flush_pending_predictions_if_full(
                train_states[variant],
                train_pending[variant],
                setup.task_spec,
            )
    _flush_variant_prediction_queues(train_pending, train_states, setup.task_spec)
    train_elapsed = time.perf_counter() - phase_start
    train_states = _gather_variant_states_jax(train_states, setup.task_spec)

    eval_states, eval_elapsed = _evaluate_probe_split_jax(
        probe_data=setup.probe_data,
        feature_model=setup.feature_model,
        task_spec=setup.task_spec,
        params_by_variant=state.params_by_variant,
        predict_step_by_variant=state.predict_step_by_variant,
        split="massspec_val",
        seed=setup.train_seed_base + 10_000,
        peak_ordering=setup.peak_ordering,
        max_samples=setup.max_val_samples,
        use_pair_features=setup.use_pair_features,
        data_mesh=setup.data_mesh,
    )
    eval_states = _gather_variant_states_jax(eval_states, setup.task_spec)
    return train_states, eval_states, train_elapsed, eval_elapsed


def _score_msg_probe_epoch_jax(
    *,
    setup: _JaxMsgProbeSetup,
    state: _JaxMsgProbeTrainingState,
    train_states: dict[str, EpochState],
    eval_states: dict[str, EpochState],
    epoch_idx: int,
    on_epoch_end: Callable[[dict[str, float]], None] | None,
) -> tuple[float, bool]:
    phase_start = time.perf_counter()
    epoch_metrics: dict[str, float] = {}
    for variant in setup.variants:
        variant_prefix = f"msg_probe/{variant}"
        variant_metrics = {
            **_score_epoch_state(
                prefix=f"{variant_prefix}/train",
                epoch_state=train_states[variant],
                task_spec=setup.task_spec,
            ),
            **_score_epoch_state(
                prefix=f"{variant_prefix}/val",
                epoch_state=eval_states[variant],
                task_spec=setup.task_spec,
            ),
            f"{variant_prefix}/num_{setup.fingerprint_task}_bits": float(
                setup.task_spec.maccs_bits
            ),
            f"{variant_prefix}/epoch": float(epoch_idx + 1),
        }
        epoch_metrics.update(variant_metrics)
        variant_select_metric = _msg_probe_variant_metric_key(
            variant, state.select_metric
        ).replace("/test/", "/val/")
        current_value = variant_metrics[variant_select_metric]
        previous_best = state.best_metric_values[variant]
        is_better = (
            current_value > previous_best + setup.early_stopping_min_delta
            if state.higher_is_better
            else current_value < previous_best - setup.early_stopping_min_delta
        )
        if is_better:
            state.best_metric_values[variant] = current_value
            state.best_metrics_by_variant[variant] = dict(variant_metrics)
            state.best_params_by_variant[variant] = _clone_tree(
                state.params_by_variant[variant]
            )
            state.epochs_without_improvement[variant] = 0
        else:
            state.epochs_without_improvement[variant] += 1
        if _is_main_process_jax():
            _log_epoch_metrics(
                variant=variant,
                epoch_idx=epoch_idx,
                num_probe_epochs=setup.num_probe_epochs,
                fingerprint_task=setup.fingerprint_task,
                metrics=variant_metrics,
            )
    if on_epoch_end is not None and _is_main_process_jax():
        on_epoch_end(epoch_metrics)
    elapsed = time.perf_counter() - phase_start
    should_stop = (
        setup.early_stopping
        and epoch_idx + 1 >= setup.early_stopping_min_epochs
        and all(
            state.epochs_without_improvement[variant]
            >= setup.early_stopping_patience
            for variant in setup.variants
        )
    )
    return elapsed, should_stop


def _finalize_msg_probe_jax(
    *,
    setup: _JaxMsgProbeSetup,
    state: _JaxMsgProbeTrainingState,
    phase_timing: dict[str, float],
    probe_start: float,
) -> dict[str, Any]:
    phase_start = time.perf_counter()
    selected_params_by_variant = {
        variant: state.best_params_by_variant.get(
            variant, state.params_by_variant[variant]
        )
        for variant in setup.variants
    }
    final_states, _ = _evaluate_probe_split_jax(
        probe_data=setup.probe_data,
        feature_model=setup.feature_model,
        task_spec=setup.task_spec,
        params_by_variant=selected_params_by_variant,
        predict_step_by_variant=state.predict_step_by_variant,
        split="massspec_test",
        seed=setup.test_seed_base,
        peak_ordering=setup.peak_ordering,
        max_samples=setup.max_test_samples,
        use_pair_features=setup.use_pair_features,
        data_mesh=setup.data_mesh,
    )
    phase_timing["final_test_seconds"] += time.perf_counter() - phase_start
    final_states = _gather_variant_states_jax(final_states, setup.task_spec)
    phase_start = time.perf_counter()
    final_test_metrics_by_variant = {
        variant: _score_epoch_state(
            prefix=f"msg_probe/{variant}/test",
            epoch_state=epoch_state,
            task_spec=setup.task_spec,
            include_pr_curves=True,
        )
        for variant, epoch_state in final_states.items()
    }
    phase_timing["final_score_seconds"] += time.perf_counter() - phase_start

    best_metrics: dict[str, Any] = {}
    for variant in setup.variants:
        variant_metrics = dict(state.best_metrics_by_variant.get(variant, {}))
        if not variant_metrics:
            continue
        variant_metrics.update(final_test_metrics_by_variant.get(variant, {}))
        best_metrics.update(variant_metrics)
    best_metrics["msg_probe/jax_wall_seconds"] = time.perf_counter() - probe_start
    for name, value in phase_timing.items():
        best_metrics[f"msg_probe/jax_{name}"] = value
    return best_metrics


def _run_msg_probe_once_jax(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetJEPAJax,
    data_mesh: Mesh | None,
    on_epoch_end: Callable[[dict[str, float]], None] | None,
    repeat_index: int,
    online_maccs_only: bool = False,
) -> dict[str, Any]:
    setup, phase_timing, probe_start = _setup_msg_probe_jax(
        config=config,
        model=model,
        data_mesh=data_mesh,
        repeat_index=repeat_index,
        online_maccs_only=online_maccs_only,
    )
    state = _initialize_msg_probe_training_jax(config, setup)

    for epoch_idx in range(setup.num_probe_epochs):
        train_states, eval_states, train_elapsed, eval_elapsed = (
            _train_and_evaluate_msg_probe_epoch_jax(setup, state, epoch_idx)
        )
        phase_timing["train_epoch_seconds"] += train_elapsed
        phase_timing["eval_epoch_seconds"] += eval_elapsed

        score_elapsed, should_stop = _score_msg_probe_epoch_jax(
            setup=setup,
            state=state,
            train_states=train_states,
            eval_states=eval_states,
            epoch_idx=epoch_idx,
            on_epoch_end=on_epoch_end,
        )
        phase_timing["score_epoch_seconds"] += score_elapsed
        if should_stop:
            if _is_main_process_jax():
                log.info(
                    "JAX MSG probe early stopping at epoch %d/%d after %d epochs without validation improvement",
                    epoch_idx + 1,
                    setup.num_probe_epochs,
                    setup.early_stopping_patience,
                )
            break

    return _finalize_msg_probe_jax(
        setup=setup,
        state=state,
        phase_timing=phase_timing,
        probe_start=probe_start,
    )


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
    regression_tasks: tuple[str, ...] = REGRESSION_PROBE_TASKS,
    binary_tasks: tuple[str, ...] = BINARY_PROBE_TASKS,
) -> MsgProbeSplitTargets:
    regression = {name: [] for name in regression_tasks}
    binary = {name: [] for name in binary_tasks}
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
        for name in regression_tasks:
            regression[name].append(np.asarray(batch[f"probe_{name}"])[valid_mask])
        for name in binary_tasks:
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
            regression_tasks=regression_tasks,
            binary_tasks=binary_tasks,
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
    regression_tasks: tuple[str, ...] = REGRESSION_PROBE_TASKS,
    binary_tasks: tuple[str, ...] = BINARY_PROBE_TASKS,
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
        regression=merge_named_arrays("regression", regression_tasks, np.float32),
        binary=merge_named_arrays("binary", binary_tasks, np.float32),
        maccs=(
            np.concatenate([target.maccs for target in targets], axis=0).astype(
                np.int32,
                copy=False,
            )
            if targets
            else np.empty((0, fingerprint_bits), dtype=np.int32)
        ),
    )


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
        variant: _merge_epoch_states(
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
    regression_tasks: tuple[str, ...] = REGRESSION_PROBE_TASKS,
    binary_tasks: tuple[str, ...] = BINARY_PROBE_TASKS,
    single_pair_covariance_include_diagonal: bool = False,
) -> MsgProbeTaskSpec:
    del test_targets
    regression_means, regression_stds = {}, {}
    for name in regression_tasks:
        values = train_targets.regression[name].astype(np.float32)
        regression_means[name] = float(values.mean())
        regression_stds[name] = float(np.clip(values.std(), 1e-8, None))
    return MsgProbeTaskSpec(
        regression_tasks=regression_tasks,
        binary_tasks=binary_tasks,
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

    return train_step


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
    _copy_tree_to_host_async(logits)
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


def _flush_pending_predictions_if_full(
    epoch_state: EpochState,
    pending: list[PendingPrediction],
    task_spec: MsgProbeTaskSpec,
) -> None:
    if len(pending) >= MAX_PENDING_PREDICTIONS:
        _flush_oldest_pending_prediction(epoch_state, pending, task_spec)


def _flush_oldest_pending_prediction(
    epoch_state: EpochState,
    pending: list[PendingPrediction],
    task_spec: MsgProbeTaskSpec,
) -> None:
    if not pending:
        return
    logits, valid_mask, targets = pending.pop(0)
    host_logit = _host_local_tree(logits)
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


def _flush_pending_predictions(
    epoch_state: EpochState,
    pending: list[PendingPrediction],
    task_spec: MsgProbeTaskSpec,
) -> None:
    while pending:
        _flush_oldest_pending_prediction(epoch_state, pending, task_spec)


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


def _copy_tree_to_host_async(tree: Any) -> None:
    jax.tree.map(_copy_array_to_host_async, tree)


def _copy_array_to_host_async(value: Any) -> None:
    if _is_global_non_fully_addressable_array(value):
        for shard in value.addressable_shards:
            shard.data.copy_to_host_async()
        return
    if isinstance(value, jax.Array):
        value.copy_to_host_async()


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


def _full_visible_fastmixer_probe_model(
    config: config_dict.ConfigDict,
    model: PeakSetJEPAJax,
) -> PeakSetJEPAJax:
    if not model.use_fastmixer:
        return model
    full_visible_tokens = pairmixer_fast_full_visible_tokens(config)
    if model.pairmixer_fast_max_visible_tokens == full_visible_tokens:
        return model
    probe_model = PeakSetJEPAJax(
        model.settings,
        pairmixer_fast_max_visible_tokens=full_visible_tokens,
        rngs=nnx.Rngs(int(config.get("seed", 0))),
    )
    nnx.update(probe_model, nnx.as_pure(nnx.state(model, nnx.Param)))
    return probe_model


@nnx.jit
def _extract_pair_features_jitted(
    model: PeakSetJEPAJax,
    peak_mz: Array,
    peak_intensity: Array,
    peak_valid_mask: Array,
    precursor_mz: Array | None,
    spectrum_metadata: Array | None,
) -> tuple[Array, Array]:
    single, pair = model.encoder.forward_with_pair(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        precursor_mz=precursor_mz,
        spectrum_metadata=spectrum_metadata,
    )
    return single, pair


@nnx.jit
def _extract_single_features_jitted(
    model: PeakSetJEPAJax,
    peak_mz: Array,
    peak_intensity: Array,
    peak_valid_mask: Array,
    precursor_mz: Array | None,
    spectrum_metadata: Array | None,
) -> Array:
    return model.encoder(
        peak_mz,
        peak_intensity,
        valid_mask=peak_valid_mask,
        precursor_mz=precursor_mz,
        spectrum_metadata=spectrum_metadata,
    )


def _extract_features(
    model: PeakSetJEPAJax,
    batch: JaxBatch,
    *,
    use_pair_features: bool,
) -> JaxFeatures:
    precursor_mz = batch.get("precursor_mz", None)
    spectrum_metadata = jax_spectrum_metadata_from_batch(batch)
    if use_pair_features:
        return _extract_pair_features_jitted(
            model,
            batch["peak_mz"],
            batch["peak_intensity"],
            batch["peak_valid_mask"],
            precursor_mz,
            spectrum_metadata,
        )
    return _extract_single_features_jitted(
        model,
        batch["peak_mz"],
        batch["peak_intensity"],
        batch["peak_valid_mask"],
        precursor_mz,
        spectrum_metadata,
    )


def _init_probe_params(
    key: Array,
    *,
    variant: str,
    config: config_dict.ConfigDict,
    task_spec: MsgProbeTaskSpec,
) -> JaxProbeParams:
    model_dim = int(config.model_dim)
    hidden_dim = int(config.get("msg_probe_mlp_hidden_dim", model_dim))
    num_layers = int(config.get("msg_probe_mlp_num_layers", 2))
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
        compressed_dim = int(config.get("covariance_pooling_dim", 32))
        left_key, right_key = jax.random.split(key)
        return {
            "left": _xavier(left_key, model_dim, compressed_dim),
            "right": _xavier(right_key, model_dim, compressed_dim),
        }, compressed_dim * compressed_dim
    if variant == "cls":
        pair_dim = int(config.get("pairmixer_pair_dim", model_dim))
        return {}, model_dim + pair_dim
    if _is_single_pair_covariance_variant(variant):
        compressed_dim = int(config.get("covariance_pooling_dim", 32))
        pair_dim = int(config.get("pairmixer_pair_dim", model_dim))
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


def _log_epoch_metrics(
    *,
    variant: str,
    epoch_idx: int,
    num_probe_epochs: int,
    fingerprint_task: str,
    metrics: dict[str, float],
) -> None:
    split = "val"
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


def _optional_positive_int(config: Any, key: str) -> int | None:
    raw = config.get(key, None)
    if raw is None:
        return None
    value = int(raw)
    return value if value > 0 else None
