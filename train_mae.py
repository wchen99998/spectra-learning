import functools
import threading
from queue import Queue
from collections.abc import Mapping
from typing import Any

from flax import nnx

import jax
import ml_collections
import numpy as np
from absl import logging
from clu import metric_writers, metrics as clu_metrics, periodic_actions
from etils import epath
from jax.sharding import NamedSharding

from train import build_mesh_and_sharding, get_data_sharding_for_rank
import input_pipeline
from models import bert as bert_models
from utils import (
    checkpoint_utils,
    learning_rate,
    state_utils,
    utils,
    wandb_writer,
)

def prefix_metrics(metrics: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Return a copy of `metrics` with every key prefixed."""
    normalized_prefix = prefix.rstrip("/") + "/"
    metrics_dict = dict(metrics)
    return {f"{normalized_prefix}{key}": value for key, value in metrics_dict.items()}

def select_model_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("token_ids", "segment_ids", "precursor_mz", "rt")
    return {key: batch[key] for key in keys}


def shard_batch_element(
    element,
    data_sharding: NamedSharding | None,
):
    """Put `element` on device while always partitioning the batch axis."""
    if data_sharding is None:
        return element

    rank = getattr(element, "ndim", np.ndim(element))
    sharding = get_data_sharding_for_rank(data_sharding, rank)
    if sharding is None:
        return element

    return jax.device_put(element, sharding)


def prefetch_sharded_batches(
    iterator,
    data_sharding: NamedSharding | None,
    prefetch_size: int = 2,
):
    """Prefetch and shard batches to devices in a background thread."""
    def maybe_shard(batch):
        batch = select_model_batch(batch)
        return jax.tree.map(lambda x: shard_batch_element(x, data_sharding), batch)

    if prefetch_size is None or prefetch_size <= 0:
        for batch in iterator:
            yield maybe_shard(batch)
        return

    queue: Queue[Any] = Queue(maxsize=prefetch_size)

    def _worker():
        try:
            for batch in iterator:
                queue.put(maybe_shard(batch))
        finally:
            queue.put(None)

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        batch = queue.get()
        if batch is None:
            break
        yield batch


def train_step(
    graphdef,
    state,
    batch,
    metrics_class: type[clu_metrics.Collection],
):
    """Single training step."""
    model, optimizer = nnx.merge(graphdef, state)

    def compute_loss(mdl, batch):
        return mdl.compute_loss(batch, train=True)

    # NNX-friendly value_and_grad (grads structure matches mdl Params)
    vgrad = nnx.value_and_grad(compute_loss, has_aux=True)

    (loss, metrics_dict), grads = vgrad(model, batch)
    del loss

    optimizer.update(model, grads)

    metrics_update = metrics_class.single_from_model_output(
        **metrics_dict,
    )

    new_state = nnx.state((model, optimizer))
    return new_state, metrics_update


def eval_step(
    model: nnx.Module,
    batch: Mapping[str, jax.Array],
    *,
    metrics_class: type[clu_metrics.Collection],
) -> clu_metrics.Collection:
    """Compute the metrics for the given model in inference mode."""
    metrics_dict = model(
        batch,
        train=False,
        apply_mask=True,
    )
    metrics_update = metrics_class.single_from_model_output(
        **metrics_dict,
    )

    return metrics_update


# Note: Metrics are now handled by CLU metrics in state_utils.create_train_metrics()


def make_global_array_data_only(
    data,  # numpy/jax array (global or host-local)
    global_shape: tuple[int, ...],
    data_sharding: NamedSharding | None,
) -> jax.Array:
    sharding = get_data_sharding_for_rank(data_sharding, len(global_shape))
    if sharding is None:
        return data

    return jax.make_array_from_process_local_data(sharding, data, global_shape)


def evaluate(
    model: nnx.Module,
    jit_eval_step: Any,
    eval_loader: Any,
    eval_loader_key: str = "validation",
    num_eval_steps: int = -1,
    writer: Any = None,
    step: int = 0,
    *,
    create_train_metrics_fn,
) -> None:
    """Evaluate the model on the given dataset using sharded execution.

    All metrics are written directly to the writer and not returned.

    Args:
        model: NNX model to evaluate.
        jit_eval_step: JIT-compiled evaluation step function.
        eval_loader: Data loader for evaluation.
        eval_loader_key: Name of the eval loader (used for metric prefixes).
        num_eval_steps: Number of evaluation steps (-1 for all).
        writer: Metric writer for logging.
        step: Current training step for logging.
        create_train_metrics_fn: Factory for creating a metrics collection.
    """
    logging.info("Starting evaluation (%s).", eval_loader_key)

    # Default prefix matches previous behavior; otherwise namespace by loader key.
    metrics_prefix = "val" if eval_loader_key == "validation" else eval_loader_key
    
    # Create fresh eval metrics
    eval_metrics = create_train_metrics_fn()
    
    with utils.StepTraceContextHelper("eval", 0) as trace_context:
        for eval_step, batch_raw in enumerate(iter(eval_loader)):
            batch = select_model_batch(batch_raw)
            eval_metrics_update = jit_eval_step(model, batch)
            eval_metrics = eval_metrics.merge(eval_metrics_update)

            if num_eval_steps > 0 and eval_step + 1 == num_eval_steps:
                break
            trace_context.next_step()

    # Compute and write eval metrics
    eval_metrics_dict = eval_metrics.compute()
    eval_metrics_cpu = jax.tree.map(np.array, eval_metrics_dict)
    eval_metrics_prefixed = prefix_metrics(eval_metrics_cpu, metrics_prefix)
    if writer is not None:
        writer.write_scalars(step, eval_metrics_prefixed)
    logging.info(
        "Evaluation metrics at step %d (%s): %s",
        step,
        eval_loader_key,
        eval_metrics_dict,
    )

def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: epath.PathLike,
    olddir: epath.PathLike | None = None,
):
    """Runs a training and evaluation loop.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
      olddir: Optional directory to load old checkpoints from for partial loading.
        If provided, checkpoints will be loaded from olddir but saved to workdir.
    """
    # Initialize multi-host JAX if enabled
    if config.get("initialize_multihost", False):
        jax.distributed.initialize()
        logging.info(
            f"JAX distributed initialized. Process {jax.process_index()}/{jax.process_count()}"
        )
        logging.info(f"Local devices: {jax.local_devices()}")
        logging.info(f"Global devices: {len(jax.devices())}")

    workdir = epath.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    rng = utils.get_rng(config.seed)
    logging.info("Using random seed %s.", rng)
    logging.info("Training BERT.")
    num_transformer_blocks = config.get("num_transformer_blocks", None)
    logging.info(
        "Transformer blocks (encoder, decoder)=%s.",
        num_transformer_blocks,
    )
    writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() > 0
    )

    # Start the profiler if requested
    if config.get("start_profiler", False) and jax.process_index() == 0:
        logging.info("Starting profiler.")
        jax.profiler.start_server(9999)

    # Add wandb writer to the multi-writer if we're on the main process
    if (
        jax.process_index() == 0
        and hasattr(writer, "_writers")
        and config.get("enable_wandb", False)
    ):
        wandb_kwargs = wandb_writer.build_wandb_init_kwargs(config)
        wandb_w = wandb_writer.WandBWriter(
            project=config.get("wandb_project", "md4"), **wandb_kwargs
        )
        wandb_w.log_config(config)
        writer._writers = tuple([wandb_w] + list(writer._writers))
        run_name = wandb_kwargs.get("name")
        if run_name:
            logging.info("wandb run name set to %s", run_name)
        logging.info("Added WandB writer to metric writers.")

    # Learning rate schedule.
    assert config.batch_size % jax.device_count() == 0
    num_train_steps = input_pipeline.get_num_train_steps(config)
    if config.num_epochs > 0:
        logging.info(
            "num_train_steps=%d, steps_per_epoch=%d",
            num_train_steps,
            num_train_steps // config.num_epochs,
        )
    else:
        logging.info("num_train_steps=%d (num_epochs unset)", num_train_steps)
    schedule_fn = functools.partial(
        learning_rate.get_learning_rate,
        base_learning_rate=config.learning_rate,
        num_steps=num_train_steps,
        warmup_steps=config.warmup_steps,
        schedule_type=config.learning_rate_schedule,
        min_learning_rate=config.get("min_learning_rate", None),
    )

    # Build input pipeline.
    rng, data_seed = jax.random.split(rng)
    data_seed = int(
        jax.random.randint(data_seed, [], minval=0, maxval=np.iinfo(np.int32).max)
    )
    # The input pipeline runs on each process and loads data for local TPUs.
    train_loader, eval_loaders, _ = input_pipeline.create_datasets(
        config, data_seed
    )

    # Train loader is already an iterator from the input pipeline
    logging.info("Created data loaders.")
    # Initialize sharding
    mesh, replicated_sharding, data_sharding = build_mesh_and_sharding(config)
    jax.set_mesh(mesh)

    # Prefetch train batches onto device to overlap host <-> device transfer.
    train_prefetch_size = int(config.get("device_prefetch_size", 2))
    train_iter = prefetch_sharded_batches(
        iter(train_loader),
        data_sharding,
        prefetch_size=train_prefetch_size,
    )

    # Initialize model.
    rng, model_rng = jax.random.split(rng)

    logging.info("Creating train state.")
    model, optimizer, _, _ = (
        state_utils.create_nnx_model(
            config,
            mesh,
            schedule_fn=schedule_fn,
        )
    )

    # Set up checkpointing with preemption tolerance
    checkpoint_manager, load_checkpoint_manager = (
        checkpoint_utils.get_checkpoint_managers(config, workdir, olddir)
    )

    # Restore from checkpoint if available
    start_step = 0
    if load_checkpoint_manager.latest_step() is not None:
        # If olddir is provided and differs from workdir, reset to step 0
        if olddir is not None and workdir != epath.Path(olddir):
            start_step = 0
            logging.info(f"Loading from different directory (olddir={olddir}), resetting start_step to 0")
        else:
            start_step = load_checkpoint_manager.latest_step()
            logging.info(f"Restoring from step {start_step}")

        # Standard checkpoint loading
        model, optimizer, _ = checkpoint_utils.restore_nnx_checkpoint(
            checkpoint_manager=load_checkpoint_manager,
            model=model,
            optimizer=optimizer,
            step=start_step,
        )

    logging.info("Batch Size: %s", config.batch_size)

    train_metrics_class = bert_models.get_train_metrics_class()
    train_metrics = bert_models.create_train_metrics()
    # Keep a persistent graph/state pair so we don't split/merge every step.
    graphdef, train_state = nnx.split((model, optimizer))

    # JIT compile training and eval functions
    train_step_func = functools.partial(
        train_step,
        metrics_class=train_metrics_class,
    )

    # JIT train step
    jit_train_step = jax.jit(
        train_step_func,
    )

    # JIT eval and sampling helpers
    jit_eval_step = nnx.jit(
        functools.partial(
            eval_step,
            metrics_class=bert_models.EvalMetrics,
        )
    )

    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=num_train_steps, writer=writer
    )
    if jax.process_index() == 0:
        hooks += [
            report_progress,
            periodic_actions.Profile(num_profile_steps=5, logdir=workdir),
        ]
    # Use the restored step or start from 0
    initial_step = start_step

    # Run training within mesh context
    with mesh:
        with metric_writers.ensure_flushes(writer):
            # Steps are in interval [1, num_train_steps], not [0, num_train_steps - 1].
            for step in range(initial_step + 1, num_train_steps + 1):
                is_last_step = step == num_train_steps

                with jax.profiler.StepTraceAnnotation("train", step_num=step):
                    batch = next(train_iter)

                    train_state, train_metrics_update = jit_train_step(
                        graphdef, train_state, batch
                    )

                    train_metrics = train_metrics.merge(train_metrics_update)

                # Quick indication that training is happening.
                logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
                for h in hooks:
                    h(step)

                if step % config.log_loss_every_steps == 0 or is_last_step:
                    # Compute and write metrics
                    metrics_dict = train_metrics.compute()
                    metrics_cpu = jax.tree.map(np.array, metrics_dict)
                    train_metrics_prefixed = prefix_metrics(metrics_cpu, "train")
                    learning_rate_value = float(np.asarray(schedule_fn(step)))
                    train_metrics_prefixed["train/learning_rate"] = learning_rate_value
                    writer.write_scalars(step, train_metrics_prefixed)
                    # Reset metrics for next interval
                    train_metrics = bert_models.create_train_metrics()

                if eval_loaders and (
                    step == 1
                    or step % config.eval_every_steps == 0
                    or is_last_step
                ):
                    model, optimizer = nnx.merge(graphdef, train_state)
                    for eval_key, eval_loader in eval_loaders.items():
                        with report_progress.timed("eval"):
                            evaluate(
                                model,
                                jit_eval_step,
                                eval_loader,
                                eval_loader_key=eval_key,
                                num_eval_steps=config.num_eval_steps,
                                writer=writer,
                                step=step,
                                create_train_metrics_fn=bert_models.create_eval_metrics,
                            )

                # Save checkpoint with preemption tolerance
                with report_progress.timed("checkpoint"):
                    checkpoint_utils.save_nnx_checkpoint(
                        checkpoint_manager,
                        step=step,
                        state=train_state,
                    )

                # Check for preemption and handle gracefully
                if checkpoint_manager.reached_preemption(step):
                    logging.info(
                        f"Preemption detected at step {step}. Waiting for checkpointing to finish."
                    )
                    checkpoint_manager.wait_until_finished()
                    logging.info("Checkpointing completed. Exiting gracefully.")
                    return

    logging.info("Finishing training at step %d", num_train_steps)
    checkpoint_manager.wait_until_finished()
