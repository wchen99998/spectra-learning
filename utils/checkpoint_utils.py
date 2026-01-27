import logging
from collections.abc import Mapping
from typing import Any

import ml_collections
from etils import epath
from flax import nnx
from orbax import checkpoint as orbax_checkpoint


def get_checkpoint_manager(
    config: ml_collections.ConfigDict,
    workdir: epath.PathLike,
    create: bool = True,
) -> orbax_checkpoint.CheckpointManager:
    """Create a checkpoint manager with preemption tolerance.
    
    Args:
        config: Configuration dict
        workdir: Working directory path
        create: Whether to create the checkpoint directory
    
    Returns:
        CheckpointManager instance
    """
    # Use checkpoint_dir from config if specified, otherwise default to workdir/checkpoints
    if (
        hasattr(config, "checkpoint_dir")
        and config.checkpoint_dir
        and isinstance(config.checkpoint_dir, str)
    ):
        checkpoint_dir = epath.Path(config.checkpoint_dir).expanduser().resolve()
    else:
        checkpoint_dir = epath.Path(workdir).expanduser().resolve() / "checkpoints"

    # Ensure checkpoint_every_steps is an integer
    checkpoint_every_steps = config.get("checkpoint_every_steps", 10000)
    if not isinstance(checkpoint_every_steps, int):
        checkpoint_every_steps = 10000

    return orbax_checkpoint.CheckpointManager(
        checkpoint_dir,
        options=orbax_checkpoint.CheckpointManagerOptions(
            create=create,
            save_interval_steps=checkpoint_every_steps,
        ),
        item_names=("model", "optimizer", "step"),
    )


def get_checkpoint_managers(
    config: ml_collections.ConfigDict,
    workdir: epath.PathLike,
    olddir: epath.PathLike | None = None,
) -> tuple[orbax_checkpoint.CheckpointManager, orbax_checkpoint.CheckpointManager]:
    """Get save and load checkpoint managers with preemption tolerance.

    Args:
        config: Configuration to use.
        workdir: Working directory for saving checkpoints.
        olddir: Optional directory to load old checkpoints from. If provided,
            load manager will point to olddir, otherwise same as save manager.

    Returns:
        A tuple of (save_manager, load_manager).
    """
    # Create checkpoint manager for saving (new checkpoints)
    save_checkpoint_manager = get_checkpoint_manager(
        config, workdir, create=True
    )

    # Create checkpoint manager for loading (old checkpoints if olddir is provided)
    if olddir is not None:
        load_checkpoint_manager = get_checkpoint_manager(
            config, olddir, create=False
        )
    else:
        load_checkpoint_manager = save_checkpoint_manager

    return save_checkpoint_manager, load_checkpoint_manager


def save_nnx_checkpoint(
    checkpoint_manager: orbax_checkpoint.CheckpointManager,
    step: int,
    model: nnx.Module | None = None,
    optimizer: nnx.Optimizer | None = None,
    state: Any | None = None,
    force: bool = False,
) -> None:
    """Save NNX model and optimizer checkpoint.
    
    Args:
        checkpoint_manager: Orbax checkpoint manager
        step: Current training step
        model: NNX model to save (required if state is None)
        optimizer: NNX optimizer to save (required if state is None)
        state: Optional combined NNX state (e.g. from nnx.split). If provided,
            the model/optimizer arguments are ignored.
        force: If True, force save even if step doesn't match save interval.
    """
    model_state = None
    optimizer_state = None

    if state is not None:
        # State may be a tuple-style nnx.State (indices) or a dict with explicit keys.
        try:
            model_state = state[0]
            optimizer_state = state[1]
        except (KeyError, TypeError):
            if isinstance(state, Mapping):
                model_state = state.get("model")
                optimizer_state = state.get("optimizer")

        if model_state is None or optimizer_state is None:
            raise ValueError(
                "Combined state must contain both model and optimizer entries."
            )
    else:
        if model is None or optimizer is None:
            raise ValueError(
                "Either provide `state` or both `model` and `optimizer`."
            )
        # Extract state from NNX objects - these are plain PyTrees
        model_state = nnx.state(model)
        optimizer_state = nnx.state(optimizer)
    
    checkpoint_manager.save(
        step,
        args=orbax_checkpoint.args.Composite(
            model=orbax_checkpoint.args.StandardSave(model_state),
            optimizer=orbax_checkpoint.args.StandardSave(optimizer_state),
            step=orbax_checkpoint.args.JsonSave(step),
        ),
        force=force,
    )


def restore_nnx_checkpoint(
    checkpoint_manager: orbax_checkpoint.CheckpointManager,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    step: int | None = None,
) -> tuple[nnx.Module, nnx.Optimizer, dict[str, Any]]:
    """Restore NNX model and optimizer from checkpoint.
    
    Args:
        checkpoint_manager: Orbax checkpoint manager
        model: NNX model to restore into (provides structure)
        optimizer: NNX optimizer to restore into (provides structure)
        step: Specific step to restore. If None, restores latest checkpoint.
    
    Returns:
        Tuple of (restored model, restored optimizer, extra_data dict with step and any other saved items)
    """
    # Determine which step to load
    if step is None:
        step = checkpoint_manager.latest_step()
    
    if step is None:
        logging.info("No checkpoint found, returning fresh model and optimizer")
        return model, optimizer, {'step': 0}
    
    logging.info(f"Restoring checkpoint from step: {step}")
    
    # Create target structure for restoration.
    # This is critical: Orbax needs the target pytree structure to correctly
    # restore namedtuples and other complex structures used by optax optimizers.
    # Without this, tuples become dicts with string keys, breaking optimizer state.
    target = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
    }

    restored = checkpoint_manager.restore(
        step,
        args=orbax_checkpoint.args.Composite(
            model=orbax_checkpoint.args.StandardRestore(target["model"]),
            optimizer=orbax_checkpoint.args.StandardRestore(target["optimizer"]),
            step=orbax_checkpoint.args.JsonRestore(),
        ),
    )

    model_graphdef, _ = nnx.split(model)
    optimizer_graphdef, _ = nnx.split(optimizer)
    model = nnx.merge(model_graphdef, restored.model)
    optimizer = nnx.merge(optimizer_graphdef, restored.optimizer)
    extra_data = {"step": restored.step}

    logging.info(f"Successfully restored checkpoint at step: {restored.step}")

    return model, optimizer, extra_data
