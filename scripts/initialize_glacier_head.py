"""Restore the selected frozen-token cross-attention head beneath online JEPA."""

import hashlib
import json
from pathlib import Path

from flax import serialization
import jax
import numpy as np


def restore_head(checkpoint: Path, params, stats, args, peak_input, graph_metadata, fragment_metadata):
    """Require the same architecture and inputs, preserving the online encoder."""
    manifest_bytes = checkpoint.with_name("manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    source_args = manifest["args"]
    for name in ("hidden", "d", "dropout"):
        if source_args[name] != getattr(args, name):
            raise ValueError(f"Head initialization architecture differs: {name}")
    fixed = {"molecule_encoder": "gine_global_glacier", "similarity": "qk", "peak_input": "raw",
             "fragment_readout": "ion_logmixture", "fragment_cross_attention": True}
    if any(source_args.get(name) != value for name, value in fixed.items()):
        raise ValueError("Head initialization requires the selected cross-attention architecture")
    if source_args.get("cross_attention_dim", 32) != 32 or source_args.get("gine_layers", 3) != 3:
        raise ValueError("Head initialization architecture differs in attention width or GINE depth")
    unsupported = ("train_jepa", "gine_pseudonodes", "qk_normalize", "imolclr_context",
                   "imolclr_atom_weights", "spectrum_fragment_prior", "learned_peak_weights",
                   "pseudonode_retain_global", "peak_messages", "fragment_relations",
                   "fragment_induced_refinement", "joint_pair_relations", "fragment_coupling")
    if any(source_args.get(name) for name in unsupported) or manifest["peak_input"]["kind"] != "raw":
        raise ValueError("Head initialization requires the selected frozen-token Global + GLACIER model")
    if not args.train_jepa:
        raise ValueError("Head initialization is the transition to online JEPA training")
    if manifest["peak_input"]["embedding_metadata"] != peak_input["embedding_metadata"]:
        raise ValueError("Head initialization spectrum-encoder provenance differs")
    if manifest["graphs"] != graph_metadata or manifest["fragments"] != fragment_metadata:
        raise ValueError("Head initialization graph or fragment inputs differ")
    checkpoint_bytes = checkpoint.read_bytes()
    source = serialization.msgpack_restore(checkpoint_bytes)
    template_params = {**params, "peak": params["peak"]["readout"]}
    template = {"params": template_params, "stats": stats}
    if jax.tree.structure(source) != jax.tree.structure(template):
        raise ValueError("Head checkpoint parameter/statistic structure differs")
    for expected, actual in zip(jax.tree.leaves(template), jax.tree.leaves(source), strict=True):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError("Head checkpoint parameter/statistic shape or dtype differs")
        if not np.isfinite(actual).all():
            raise ValueError("Head checkpoint contains nonfinite parameters or statistics")
    restored = jax.device_put(source)
    restored["params"]["peak"] = {"encoder": params["peak"]["encoder"],
                                  "readout": restored["params"]["peak"]}
    provenance = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_fragment_readout": "ion_logmixture", "target_fragment_readout": "ion_logmixture",
        "peak_parameter_mapping": "peak -> peak/readout",
        "restored_groups": ["peak_readout", "gnn", "compatibility", "agreement", "gnn_batch_stats"],
        "optimizer": "fresh", "initial_validation_eligible_for_selection": True,
        "restored_cross_attention": True,
    }
    return restored["params"], restored["stats"], provenance
