"""End-to-end gradients, frozen-token equivalence, and checkpoint replay."""

import argparse
import json
from pathlib import Path
import sys

from flax import nnx, serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import trainable_jepa_peak as bridge
import train_global_glacier_jax as training
import run_retrieval_jax as runner
from atom_flare_data import build_graph_store
from glacier_fragment_data import build_fragment_store
from spectra_learning.models.encoder_jax import PeakSetEncoder
from spectra_learning.models.peak_features_jax import PeakFeatureEmbedder


@pytest.mark.parametrize("adaptation", ["full", "lora", "frozen"])
def test_real_encoder_flare_gradient_checkpoint_and_frozen_equivalence(tmp_path, monkeypatch, adaptation):
    encoder = PeakSetEncoder(
        model_dim=16, embedder=PeakFeatureEmbedder(model_dim=16, hidden_dim=16, fourier_num_freqs=4),
        num_layers=2, num_heads=2, num_peaks=3, use_pair_path=False,
        activation_checkpoint_mode="full", rngs=nnx.Rngs(4))
    if adaptation == "lora":
        from retrieval_lora import install_attention_lora
        install_attention_lora(encoder, rank=4, alpha=4., seed=0)
    peak = bridge.TrainableJEPAPeak(encoder, 8, 0, 16, adaptation)
    frozen_before = [np.array(x) for x in jax.tree.leaves(peak.static)]
    monkeypatch.setattr(bridge, "restore_peak", lambda args: peak)
    args = argparse.Namespace(train_jepa=True, d=8, dropout=0, hidden=8, lr=5e-4, tau=.05,
        score_chunk=2, molecule_encoder="gine_global_glacier", batch=4, negatives=1,
        eval_batch=4, atom_cap=8, node_capacity=64, edge_capacity=128,
        fragment_readout="ion_logmixture", jepa_lr=3e-6)
    store = build_graph_store(["CCO", "CCN", "CCC", "COC"], tmp_path / "graphs")
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    np.savez(proposals / "part.npz", graph_ids=np.arange(4), frag_off=np.zeros(5, np.int64),
             mem_off=np.zeros(1, np.int64), atoms=np.zeros(0, np.int16), inten=np.zeros(0, np.float32))
    (proposals / "metadata.json").write_text(json.dumps({
        "smiles_sha256": store.metadata["smiles_sha256"], "completed_shards": ["part.npz"]}))
    store.fragments = build_fragment_store(store, proposals, tmp_path / "fragments", f_max=4)
    graph = runner.graph_batch(store, [0, 1, 2, 3], args, train=True)
    peak, gnn, agreement, optimizer, step, _, encode_peaks, _ = training.functions(args)
    x = np.random.default_rng(8).uniform(.1, 1, (4, 3, 6)).astype(np.float32)
    x[..., 2] = 1
    mask = np.ones((4, 3), bool)
    from spectra_learning.probes.massspec.glacier_ion_logmixture_jax import PROTON_MASS

    mass = store.fragments.total_mass[:4]
    adduct_mass = np.array([PROTON_MASS, 22.989769282 - .000548579909065,
                           PROTON_MASS, PROTON_MASS], np.float32)
    fractions = np.tile(np.array([.25, .5, 1], np.float32), (4, 1))
    mz = mass[:, None] * fractions + adduct_mass[:, None]
    x[..., 0] = mz / 1000
    x[:, 0, 3] = (mass + adduct_mass) / 1000
    fractions = {"fraction": fractions, "mz": mz, "eligible": np.ones(4, bool),
                     "adduct_mass": adduct_mass}
    params = {"peak": peak.init(jax.random.key(0), x)["params"],
              "agreement": agreement.init(jax.random.key(1), jnp.zeros((1, 1)),
                  jnp.ones((1, 1)), jnp.ones((1, 1), bool))["params"]}
    gp = gnn.init(jax.random.key(2), graph, train=False)
    params["gnn"] = gp["params"]
    params["compatibility"] = runner.model.QueryKeyProjection(8).init(
        jax.random.key(3), jnp.zeros((1, 8)), jnp.zeros((1, 8)))["params"]
    before = jax.tree.map(lambda a: np.array(a), params)
    tokens = peak.encode(params["peak"]["encoder"], x)
    direct = encoder(x[..., 0], x[..., 1], valid_mask=mask, visible_mask=mask,
                     precursor_mz=x[:, 0, 3], spectrum_metadata=x[:, 0, 4:])[:, :-1]
    np.testing.assert_array_equal(tokens, direct.astype(jnp.float16).astype(jnp.float32))
    frozen = peak.readout.apply({"params": params["peak"]["readout"]}, tokens, train=False)
    np.testing.assert_array_equal(peak.apply({"params": params["peak"]}, x), frozen)
    params, stats, opt_state, loss = step(params, gp["batch_stats"], optimizer.init(params),
        graph, x, mask, fractions, np.arange(4, dtype=np.int32),
        np.array([[1], [2], [3], [0]], np.int32), np.ones((4, 1), bool), jax.random.key(5))
    assert np.isfinite(float(loss))
    if adaptation == "frozen":
        assert not jax.tree.leaves(params["peak"]["encoder"])
    for block in params["peak"]["encoder"].get("blocks", {}):
        old, new = (t["peak"]["encoder"]["blocks"][block] for t in (before, params))
        assert any(np.any(a != np.asarray(b)) for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new)))
    moments = opt_state[0].mu
    bad = [jax.tree_util.keystr(path) for path, a in jax.tree_util.tree_flatten_with_path(moments)[0]
           if not np.isfinite(np.asarray(a)).all()]
    assert not bad, bad
    for tree in (moments["peak"]["readout"], moments["gnn"]):
        assert any(np.any(np.asarray(a) != 0) for a in jax.tree.leaves(tree))
    if adaptation != "frozen":
        assert any(np.any(np.asarray(a) != 0) for a in jax.tree.leaves(moments["peak"]["encoder"]))
    for old, new in zip(frozen_before, jax.tree.leaves(peak.static), strict=True):
        np.testing.assert_array_equal(old, np.asarray(new))
    state = {"params": params, "stats": stats, "opt_state": opt_state}
    restored = serialization.from_bytes(state, serialization.to_bytes(state))
    actual = encode_peaks(params["peak"], x, params["compatibility"])
    replay = encode_peaks(restored["params"]["peak"], x, restored["params"]["compatibility"])
    np.testing.assert_array_equal(actual, replay)
    # Resuming must retain Adam moments as well as the trained encoder weights.
    step_inputs = (graph, x, mask, fractions, np.arange(4, dtype=np.int32),
                   np.array([[1], [2], [3], [0]], np.int32), np.ones((4, 1), bool), jax.random.key(9))
    uninterrupted = step(params, stats, opt_state, *step_inputs)
    resumed = step(restored["params"], restored["stats"], restored["opt_state"], *step_inputs)
    for a, b in zip(jax.tree.leaves(uninterrupted), jax.tree.leaves(resumed), strict=True):
        np.testing.assert_array_equal(a, b)


def test_group_learning_rate_scales_encoder_and_head_independently():
    updates = {"peak": {"encoder": {"w": jnp.ones((2,)), "blocks": {"0": jnp.ones((3,))}},
                        "readout": {"w": jnp.ones((2,))}},
               "gnn": {"w": jnp.ones((2,))}, "agreement": {"log_sigma": jnp.ones(())}}
    transform = training.scale_by_group_learning_rate(1e-5, 3e-4)
    scaled, _ = transform.update(updates, transform.init(updates))
    for leaf in jax.tree.leaves(scaled["peak"]["encoder"]):
        np.testing.assert_allclose(np.asarray(leaf), -1e-5, rtol=1e-6)
    for tree in (scaled["peak"]["readout"], scaled["gnn"], scaled["agreement"]):
        for leaf in jax.tree.leaves(tree):
            np.testing.assert_allclose(np.asarray(leaf), -3e-4, rtol=1e-6)


def test_equal_group_rates_reproduce_plain_adam():
    args = argparse.Namespace(train_jepa=True, lr=5e-4, jepa_lr=5e-4)
    params = {"peak": {"encoder": {"w": jnp.ones((2,))}, "readout": {"w": jnp.ones((2,))}}}
    grads = jax.tree.map(lambda x: x * 0.5, params)
    grouped = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0),
                          training.scale_by_group_learning_rate(args.jepa_lr, args.lr))
    plain = optax.adam(args.lr, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)
    a, _ = grouped.update(grads, grouped.init(params), params)
    b, _ = plain.update(grads, plain.init(params), params)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6)
