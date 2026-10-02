"""Masking and equivariance contracts of the edge-gated pair->single bridge."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
pytest.importorskip("torch")

from scripts.pair_single_dual_pilot import EdgeGatedMessage


def _inputs(seed=1):
    single = jax.random.normal(jax.random.key(seed), (2, 5, 16))
    pair = jax.random.normal(jax.random.key(seed + 1), (2, 5, 5, 8))
    mask = jnp.array([[True, False, True, False, True], [False] * 5])
    return single, pair, mask


def test_padding_is_ignored_and_receives_no_gradient():
    bridge = EdgeGatedMessage(16, 8, rank=4, rngs=nnx.Rngs(3))
    single, pair, mask = _inputs()
    valid = mask[:, :, None] & mask[:, None, :]
    actual = bridge(single, pair, mask)
    np.testing.assert_allclose(
        bridge(jnp.where(mask[..., None], single, 99.0),
               jnp.where(valid[..., None], pair, pair * 30 + 50), mask),
        actual, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual)[~np.asarray(mask)], 0)
    gradient = jax.grad(lambda z: bridge(single, z, mask)[..., 0].sum())(pair)
    assert np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_array_equal(np.asarray(gradient)[~np.asarray(valid)], 0)
    assert float(jnp.linalg.norm(gradient)) > 0


def test_compact_equivariance():
    bridge = EdgeGatedMessage(16, 8, rank=4, rngs=nnx.Rngs(4))
    single, pair, mask = _inputs(7)
    idx = jnp.array([4, 0, 2])
    compact = bridge(single[:1, idx], pair[:1, idx][:, :, idx], jnp.ones((1, 3), bool))
    np.testing.assert_allclose(compact, bridge(single, pair, mask)[:1, idx], atol=2e-6)


def test_pair_content_reaches_singles_when_neighbor_singles_are_equal():
    bridge = EdgeGatedMessage(16, 8, rank=4, rngs=nnx.Rngs(5))
    _, pair, _ = _inputs(11)
    single = jnp.ones((2, 5, 16))
    mask = jnp.ones((2, 5), bool)
    jacobian = jax.jacfwd(lambda z: bridge(single, z, mask))(pair)
    assert float(jnp.abs(jacobian).max()) > 1e-3


def test_thompson_span_targets_close_isotopes_and_respect_budget():
    from scripts.pair_single_dual_pilot import isotope_adjacency, thompson_span_targets

    # Peaks (Da): an M/M+1/M+2 cluster at 150, and an H-spaced pair at 200
    # (1.0078 Da) that must not be treated as isotopes.
    mz_da = np.array([100.0, 150.0, 151.003355, 152.00671, 200.0, 201.007825,
                      250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 0, 0])
    valid = mz_da > 0
    intensity = np.where(valid, 0.1, 0.0)
    intensity[1] = 1.0
    isotopes = isotope_adjacency((mz_da / 1000)[None], valid[None])[0]
    assert isotopes[1, 2] and isotopes[1, 3] and not isotopes[4, 5]
    for seed in range(50):
        target = thompson_span_targets(intensity, valid, isotopes, np.random.default_rng(seed))
        assert not target[~valid].any() and 1 <= target.sum() <= min(8, valid.sum() - 1)
        cluster = target[[1, 2, 3]]
        assert cluster.all() or not cluster.any()


def _tiny_args(**overrides):
    from argparse import Namespace

    args = dict(single_dim=32, pair_dim=16, layers=2, heads=2, predictor_layers=1,
                bridge_rank=8, mz_bin_size=2.0, meta_fields="precursor+ce+instrument+adduct",
                meta_sites="", meta_dim=16, meta_dropout=0.0, pair_precursor="on",
                offset_heads="", offset_weight=1.0, registers=0)
    args.update(overrides)
    return Namespace(**args)


def _tiny_batch(dim):
    rng = np.random.default_rng(0)
    valid = np.arange(31)[None] < np.array([[12], [20]])
    mz = np.where(valid, np.sort(rng.uniform(0.05, 0.4, (2, 31)), axis=1), 0.0).astype(np.float32)
    intensity = np.where(valid, rng.uniform(0.01, 1, (2, 31)), 0.0).astype(np.float32)
    return dict(peak_mz=jnp.asarray(mz), peak_intensity=jnp.asarray(intensity),
                peak_valid_mask=jnp.asarray(valid), precursor_mz=jnp.asarray([0.45, 0.5]),
                spectrum_metadata=jnp.asarray(rng.uniform(0, 1, (2, dim)), jnp.float32))


def _perturb(module, seed=0):
    for i, (path, value) in enumerate(nnx.to_flat_state(nnx.state(module, nnx.Param))):
        value[...] = value[...] + 0.05 * jax.random.normal(jax.random.key(seed + i), value.shape)


def test_metadata_wrapper_matches_encoder_and_starts_as_identity():
    from scripts.pair_single_dual_pilot import META_SITES, PilotModel, encode_joint

    sites = "+".join(META_SITES)
    model = PilotModel("dual_bias_mean_gated", 0, _tiny_args(meta_sites=sites))
    batch = _tiny_batch(31)
    common = dict(valid_mask=batch["peak_valid_mask"], visible_mask=batch["peak_valid_mask"],
                  max_visible_tokens=32, precursor_mz=batch["precursor_mz"])
    native = model.joint.encoder.forward_with_pair_compact(
        batch["peak_mz"], batch["peak_intensity"], **common)
    plain = encode_joint(model.joint.encoder, None, batch["peak_mz"], batch["peak_intensity"], **common)
    zero_init = encode_joint(model.joint.encoder, model.meta, batch["peak_mz"], batch["peak_intensity"],
                             **common, metadata=batch["spectrum_metadata"])
    for a, b, c in zip(native, plain, zero_init):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(c))


@pytest.mark.parametrize("site", ["token", "cls", "pair_in", "film_single", "film_pair",
                                  "cls_edge", "final"])
def test_each_metadata_site_reaches_the_encoder_and_ignores_padding(site):
    from scripts.pair_single_dual_pilot import PilotModel, encode_joint

    model = PilotModel("dual_bias_mean_gated", 0, _tiny_args(meta_sites=site))
    _perturb(model.meta)
    batch = _tiny_batch(31)
    valid = batch["peak_valid_mask"]
    common = dict(valid_mask=valid, visible_mask=valid, max_visible_tokens=32,
                  precursor_mz=batch["precursor_mz"])

    def run(metadata, mz=batch["peak_mz"]):
        single, pair, _, _ = encode_joint(model.joint.encoder, model.meta, mz,
                                          batch["peak_intensity"], **common, metadata=metadata)
        return np.asarray(single, np.float32), np.asarray(pair, np.float32)

    base = run(batch["spectrum_metadata"])
    other = run(batch["spectrum_metadata"][::-1])
    changed = max(np.abs(base[0] - other[0]).max(), np.abs(base[1] - other[1]).max())
    assert changed > 1e-3
    padded = run(batch["spectrum_metadata"], jnp.where(valid, batch["peak_mz"], 0.9))
    np.testing.assert_allclose(padded[0], base[0], atol=1e-6)
    np.testing.assert_allclose(padded[1], base[1], atol=1e-6)


def test_metadata_vectors_dropout_and_blind():
    from scripts import pair_single_dual_metadata as lib

    meta = dict(collision_energy=np.array([30.0, 0.0, 250.0], np.float32),
                collision_energy_present=np.array([True, False, True]),
                instrument_family_id=np.array([1, 2, 0]), adduct_id=np.array([0, 1, 0]))
    precursor = np.array([0.2, 0.3, 0.4], np.float32)
    full = lib.condition_vectors(meta, precursor, lib.FIELDS)
    assert full.shape == (3, 31)
    np.testing.assert_allclose(full[:, 2], [0.3, 0.0, 1.0])
    assert full[0, 22] == 1 and full[1, 23] == 1 and full[2, 21] == 1
    assert full[1, 30] == 1 and full[0, 29] == 1
    blind = lib.blind(full)
    np.testing.assert_array_equal(lib.drop_fields(full, 1.0, np.random.default_rng(0)), blind)
    np.testing.assert_array_equal(lib.drop_fields(full, 0.0, np.random.default_rng(0)), full)
    precursor_only = lib.condition_vectors(meta, precursor, ("precursor",))
    assert precursor_only.shape == (3, 28) and precursor_only[:, 3].sum() == 0
    assert (precursor_only[:, 21] == 1).all()


def _masked_batch(dim=31):
    from scripts.pair_to_single_pilot import mask_batch

    batch = _tiny_batch(dim)
    data = {k: np.asarray(v) for k, v in batch.items() if k != "spectrum_metadata"}
    masked = mask_batch(data, np.arange(2), np.random.default_rng(3))
    return {**masked, "spectrum_metadata": batch["spectrum_metadata"]}


def test_bin_offsets_and_head_start_at_bin_centre():
    from scripts.pair_single_dual_pilot import BinOffsetHead, bin_offsets

    bins, position, in_range = bin_offsets(jnp.array([0.05, 0.25, 12.0]), 0.1, 100)
    np.testing.assert_array_equal(np.asarray(bins), [0, 2, 99])
    np.testing.assert_allclose(np.asarray(position)[:2], [0.5, 0.5], atol=1e-5)
    np.testing.assert_array_equal(np.asarray(in_range), [True, True, False])
    head = BinOffsetHead(8, 100, compute_dtype=jnp.float32, rngs=nnx.Rngs(0))
    features = jax.random.normal(jax.random.key(0), (3, 8))
    np.testing.assert_allclose(np.asarray(head(features, jnp.array([1, 2, 3]))), 0.5)
    grad = nnx.grad(lambda h: h(features, jnp.array([1, 2, 2])).sum())(head)
    rows = np.flatnonzero(np.abs(np.asarray(grad.weight[...])).sum(1))
    np.testing.assert_array_equal(rows, [1, 2])


def test_offset_paths_match_library_losses_without_heads():
    from scripts.pair_single_dual_pilot import PilotModel, pair_forward, pair_distogram_with_offsets

    batch = _masked_batch(28)
    model = PilotModel("pair_only", 0, _tiny_args(meta_fields="precursor+ce+instrument"))
    native = model.pair.forward_pair_distogram(batch)
    loss, metrics, offset = pair_forward(model.pair, batch)
    assert offset is None
    np.testing.assert_allclose(float(loss), float(native["loss"]), rtol=1e-6)
    np.testing.assert_allclose(float(metrics["distogram_loss"]), float(native["distogram_loss"]),
                               rtol=1e-6)
    dual = PilotModel("dual_bias_mean_gated", 0, _tiny_args(offset_heads="mz+dist"))
    valid = batch["peak_valid_mask"]
    union = jnp.any(batch["target_masks"], axis=1)
    from scripts.pair_single_dual_pilot import encode_joint

    _, pair, idx, mask = encode_joint(
        dual.joint.encoder, None, batch["peak_mz"], batch["peak_intensity"], valid_mask=valid,
        visible_mask=batch["context_mask"] & ~union, max_visible_tokens=None,
        precursor_mz=batch["precursor_mz"])
    library = dual.pair._distogram_loss_pair_branch(pair, idx, mask, union, batch["peak_mz"])
    ours = pair_distogram_with_offsets(dual.pair, pair, idx, mask, union, batch["peak_mz"],
                                       dual.offsets.dist)
    np.testing.assert_array_equal(np.asarray(ours["distogram_loss"]),
                                  np.asarray(library["distogram_loss"]))
    assert 0.0 < float(ours["distogram_offset_loss"]) < 0.5
    metrics = dual.loss(batch)
    for key in ("mz_offset_loss", "distogram_offset_loss", "mz_abs_error_mda",
                "mz_center_error_mda"):
        assert np.isfinite(float(metrics[key])), key
    expected = (float(metrics["mae_mz_loss"]) * 0.7 + 0.3 * float(metrics["distogram_loss"])
                + float(metrics["mz_offset_loss"]) + float(metrics["distogram_offset_loss"]))
    np.testing.assert_allclose(float(metrics["loss"]), expected, rtol=2e-3)


@pytest.mark.parametrize("count", [1, 4])
def test_registers_join_the_encoder_and_are_dropped_from_outputs(count):
    from scripts.pair_single_dual_pilot import PilotModel, encode_joint

    plain = PilotModel("dual_bias_mean_gated", 0, _tiny_args())
    model = PilotModel("dual_bias_mean_gated", 0, _tiny_args(registers=count))
    batch = _tiny_batch(31)
    valid = batch["peak_valid_mask"]
    common = dict(valid_mask=valid, visible_mask=valid, max_visible_tokens=24,
                  precursor_mz=batch["precursor_mz"])

    def run(m, mz=batch["peak_mz"]):
        return [np.asarray(t, np.float32) for t in encode_joint(
            m.joint.encoder, None, mz, batch["peak_intensity"], **common, registers=m.registers)]

    with_registers, without = run(model), run(plain)
    for a, b in zip(with_registers, without):
        assert a.shape == b.shape
    np.testing.assert_array_equal(with_registers[2], without[2])
    assert np.abs(with_registers[0] - without[0]).max() > 1e-3
    padded = run(model, jnp.where(valid, batch["peak_mz"], 0.9))
    for a, b in zip(padded, with_registers):
        np.testing.assert_allclose(a, b, atol=1e-6)
    loss = model.loss(_masked_batch(31))
    assert np.isfinite(float(loss["loss"]))
