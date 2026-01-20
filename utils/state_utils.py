"""Utilities for creating and managing training state."""

import functools
from collections.abc import Callable
from typing import Any

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import ml_collections
import optax
from absl import logging
from clu import metrics as clu_metrics
from flax import nnx

import numpy as np
from models import utils as model_utils


def get_default_logical_axis_rules():
    """Get default logical axis rules for model sharding.

    Returns:
        List of tuples mapping logical axis names to physical mesh axis names.
    """
    return [
        ("batch", "data"),
        ("hidden", "model"),
        ("attn_qkv", "model"),
        ("attn_o", "model"),
        ("ff_mlp", "model"),
        ("embed_vocab", "model"),
        ("input_embed", "model"),
        ("cross_attn", "model"),
        ("cond", "model"),
        ("cond_input", "model"),
        ("cond_hidden", "model"),
        ("cond_output", "model"),
        ("vocab", "model"),
        # leave sequence/time unsharded
    ]


def get_conditioning_from_batch(batch, dtype=jnp.float32):
    """Extract conditioning information from batch data.

    Handles all combinations of fingerprint, true_fingerprint, and atom_types.

    Args:
        batch: Batch data dictionary
        dtype: Data type to use for floating point conditioning data
    """
    # Build conditioning dict based on available fields
    conditioning = {}

    # Handle fingerprint (prioritize true_fingerprint if available)
    if "true_fingerprint" in batch:
        conditioning["true_fingerprint"] = batch["true_fingerprint"].astype(dtype)

    if "fingerprint" in batch:
        conditioning["cross_conditioning"] = batch["fingerprint"].astype(dtype)

    # Return None if no conditioning information is available
    return conditioning if conditioning else None

@flax.struct.dataclass
class TrainMetrics(clu_metrics.Collection):
    """CLU metric collection tracking training objectives."""

    loss: clu_metrics.Average
    token_accuracy: clu_metrics.Average


def create_train_metrics() -> TrainMetrics:
    """Create CLU-based train metrics collection."""
    return TrainMetrics.empty()


def build_optimizer(
    config: ml_collections.ConfigDict,
    schedule_fn: Callable[[Any], Any],
    model: nnx.Module,
    mesh: jax.sharding.Mesh,
):
    """Build and shard an optimizer for the given model.
    The optimizer state is sharded to match the model's parameter sharding.
    Args:
        config: Configuration dict
        schedule_fn: Learning rate schedule function
        model: NNX model with sharded parameters
        mesh: JAX mesh used for sharding the model
    Returns:
        Sharded NNX optimizer
    """
    optimizer_name = config.get("optimizer", "adamw")

    if optimizer_name == "muon":
        base_optimizer = optax.contrib.muon(
            schedule_fn,
            adam_b1=0.9,
            adam_b2=config.b2,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "adamw":
        base_optimizer = optax.adamw(
            schedule_fn,
            b1=0.9,
            b2=config.b2,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "adam":
        # Note: optax.adam does not support decoupled weight decay.
        # config.weight_decay will be ignored.
        base_optimizer = optax.adam(
            schedule_fn,
            b1=0.9,
            b2=config.b2,
        )
    else:
        raise ValueError(f"Unsupported optimizer: '{optimizer_name}'")

    chains = [
        optax.clip(config.clip) if config.clip > 0.0 else optax.identity(),
        base_optimizer,
    ]

    optimizer_tx = optax.chain(*chains)
    optimizer = nnx.Optimizer(model, optimizer_tx, wrt=nnx.Param)

    # Shard optimizer state to match model parameter sharding
    logical_axis_rules = config.get(
        "logical_axis_rules", get_default_logical_axis_rules()
    )

    with nn.logical_axis_rules(logical_axis_rules):
        optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
        optimizer_shardings = nnx.get_named_sharding(optimizer_state, mesh)

        optimizer_sharded_state = jax.lax.with_sharding_constraint(
            optimizer_state, optimizer_shardings
        )
        nnx.update(optimizer, optimizer_sharded_state)

    return optimizer


def create_nnx_model(
    config: ml_collections.ConfigDict,
    mesh: jax.sharding.Mesh,
    schedule_fn: Callable[[Any], Any],
) -> tuple[nnx.Module, nnx.Optimizer, TrainMetrics, jax.sharding.Mesh]:
    """Create a sharded NNX model with optimizer and metrics.

    Args:
        config: Configuration dict
        mesh: JAX mesh for sharding
        schedule_fn: Learning rate schedule function

    Returns:
        Tuple of (sharded model, sharded optimizer, CLU metrics collection, mesh)
    """

    logical_axis_rules = config.get("logical_axis_rules", get_default_logical_axis_rules())
    
    with mesh, nn.logical_axis_rules(logical_axis_rules):
        def _create_model():
            init_rng = jax.random.PRNGKey(config.get("init_seed", 0))
            return model_utils.get_model(config=config, rngs=nnx.Rngs(init_rng))
    
        abstract_model = nnx.eval_shape(_create_model)
        graphdef, abstract_state = nnx.split(abstract_model)
        specs = nnx.get_partition_spec(abstract_state)
        out_shardings = nn.logical_to_mesh_sharding(specs, mesh)

        @functools.partial(jax.jit, out_shardings=out_shardings)
        def create_sharded_state():
            # This will be JIT-compiled. JAX knows the output sharding and can
            # initialize the parameters directly on the target devices in a sharded way.
            model = _create_model()
            return nnx.state(model)

        # Create the model with sharded parameters.
        sharded_state = create_sharded_state()
        model = nnx.merge(graphdef, sharded_state)

        total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(sharded_state))
        logging.info(f"Total model parameters: {total_params}")

        print_parameter_counts(model, module_name="Model", max_depth=3)

        # Create optimizer with matching sharding
        optimizer = build_optimizer(config, schedule_fn, model, mesh)

    # Create CLU metrics
    train_metrics = create_train_metrics()

    return model, optimizer, train_metrics, mesh


def print_parameter_counts(module: nnx.Module, module_name: str = "Model", max_depth: int = 3):
    """
    Print parameter counts for each layer in an nnx module with hierarchical formatting.
    
    Args:
        module: The nnx.Module to analyze
        module_name: Name to display for the root module
        max_depth: Maximum depth to traverse (default: 3)
    """
    
    def format_number(n: int) -> str:
        """Format number with thousands separator."""
        return f"{n:,}"
    
    def format_percentage(part: int, total: int) -> str:
        """Format percentage with 1 decimal place."""
        if total == 0:
            return "0.0%"
        return f"{100.0 * part / total:.1f}%"
    
    def get_param_count(state_dict) -> int:
        """Count parameters in a state dictionary."""
        count = 0
        for value in jax.tree.leaves(state_dict):
            if hasattr(value, 'size'):
                count += value.size
        return count
    
    def traverse_module(mod, name: str, depth: int, prefix: str = ""):
        """Recursively traverse module and collect parameter info."""
        # Get state for this module
        try:
            state = nnx.state(mod, nnx.Param)
        except:
            return []
        
        # Count parameters at this level
        param_count = get_param_count(state)
        
        results = []
        if param_count > 0 or depth == 0:
            results.append({
                'name': name,
                'depth': depth,
                'prefix': prefix,
                'params': param_count,
            })
        
        # Don't traverse deeper than max_depth
        if depth >= max_depth:
            return results
        
        # Get all attributes that are modules or lists of modules
        module_attrs = []
        
        # Use vars() instead of dir() to avoid sorting issues
        try:
            obj_dict = vars(mod)
        except TypeError:
            obj_dict = {}
        
        for attr_name, attr_value in obj_dict.items():
            # Skip non-string keys and private attributes
            if not isinstance(attr_name, str) or attr_name.startswith('_'):
                continue
            try:
                if isinstance(attr_value, nnx.Module):
                    module_attrs.append((attr_name, attr_value))
                elif isinstance(attr_value, (list, tuple, nnx.List)):
                    for idx, item in enumerate(attr_value):
                        if isinstance(item, nnx.Module):
                            module_attrs.append((f"{attr_name}[{idx}]", item))
            except:
                continue
        
        # Sort by name for consistent ordering
        module_attrs.sort(key=lambda x: x[0])
        
        # Traverse submodules
        for i, (attr_name, submod) in enumerate(module_attrs):
            # Create proper prefix for tree structure
            is_last = (i == len(module_attrs) - 1)
            branch = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")
            
            sub_results = traverse_module(submod, attr_name, depth + 1, new_prefix)
            for result in sub_results:
                if result['depth'] == depth + 1:
                    result['prefix'] = prefix + branch
                results.append(result)
        
        return results
    
    # Collect all parameter info
    total_params = get_param_count(nnx.state(module, nnx.Param))
    results = traverse_module(module, module_name, 0)
    
    # Print header
    print("=" * 80)
    print(f"Parameter Count Analysis: {module_name}")
    print("=" * 80)
    print()
    
    # Print results
    max_name_len = max(len(r['prefix'] + r['name']) for r in results)
    
    for result in results:
        full_name = result['prefix'] + result['name']
        params = result['params']
        pct = format_percentage(params, total_params)
        
        # Format with aligned columns
        name_str = full_name.ljust(max_name_len + 2)
        param_str = format_number(params).rjust(12)
        pct_str = pct.rjust(8)
        
        # Add visual indicator for size
        bar_width = 30
        filled = int(bar_width * params / total_params) if total_params > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        print(f"{name_str} {param_str} params  {pct_str}  [{bar}]")
    
    print()
    print("─" * 80)
    print(f"{'TOTAL'.ljust(max_name_len + 2)} {format_number(total_params).rjust(12)} params  100.0%")
    print("=" * 80)
    print()
    
    # Summary statistics
    if len(results) > 1:
        param_counts = [r['params'] for r in results if r['depth'] == 1]
        if param_counts:
            print("Summary Statistics (depth=1 modules):")
            print(f"  - Number of modules: {len(param_counts)}")
            print(f"  - Largest module: {format_number(max(param_counts))} params")
            print(f"  - Smallest module: {format_number(min(param_counts))} params")
            print(f"  - Average module: {format_number(int(np.mean(param_counts)))} params")
            print()
