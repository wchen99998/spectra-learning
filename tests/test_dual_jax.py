"""Joint single+pair ``dual`` training mode and peak-count stages."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from ml_collections import config_dict

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.mask_schedule import jepa_mask_stages
from spectra_learning.models.fastmixer_capacity import pairmixer_fast_stage_capacities
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.models.pair_to_single_jax import PairToSingleBridge
from spectra_learning.models.pairmixer_jax import _acc_dot_general, _triangle_contract
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.models.spectrum_metadata import (
    MASSIVE_V2_ACQUISITION_SCHEMA,
    MASSIVE_V2_CONDITION_DIM,
)
from spectra_learning.training.checkpointing_jax import (
    build_jax_checkpoint_manager,
    jax_training_checkpoint_metadata,
)
from spectra_learning.training.logging import MetricLogger
from spectra_learning.training.pretrain_jax import _run_jax_training_loop


def _dual_kwargs(num_peaks: int = 7) -> dict[str, object]:
    return {
        "training_mode": "dual",
        "model_dim": 8,
        "predictor_dim": 8,
        "encoder_num_layers": 2,
        "encoder_num_heads": 2,
        "attention_mlp_multiple": 1.0,
        "predictor_mlp_multiple": 1.0,
        "feature_mlp_hidden_dim": 8,
        "encoder_fourier_num_freqs": 2,
        "pairmixer_fourier_num_freqs": 2,
        "pairmixer_pair_dim": 8,
        "pairmixer_pair_feature_hidden_dim": 8,
        "pairmixer_block_type": "fastmixer",
        "encoder_apply_final_pair_norm": True,
        "masked_latent_predictor_num_layers": 1,
        "masked_latent_predictor_num_heads": 2,
        "pair_predictor_num_layers": 2,
        "masked_token_input_mode": "latent_token",
        "num_peaks": num_peaks,
        "jepa_num_target_blocks": 1,
        "jepa_mae_mz_bin_size": 50.0,
        "mae_loss_weight": 0.7,
        "mae_intensity_loss_weight": 0.0,
        "distogram_loss_weight": 0.3,
        "predictor_apply_final_norm": True,
        "target_projector_dim": -1,
        "pair_to_single_rank": 4,
        "encoder_metadata_schema": MASSIVE_V2_ACQUISITION_SCHEMA,
        "encoder_metadata_conditioning": "precursor_edge",
        "encoder_metadata_condition_dim": 8,
    }


def _metadata(precursor_mz: np.ndarray) -> np.ndarray:
    metadata = np.zeros((len(precursor_mz), MASSIVE_V2_CONDITION_DIM), np.float32)
    metadata[:, 0] = precursor_mz
    metadata[:, 1] = 1.0
    metadata[:, [12, 15, 21]] = 1.0  # unknown polarity, acquisition, instrument
    return metadata


def _batch(num_peaks: int, *, batch_size: int = 3, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    valid = np.ones((batch_size, num_peaks), bool)
    valid[0, num_peaks - 2:] = False
    mz = np.sort(rng.uniform(0.05, 0.5, (batch_size, num_peaks)), 1) * valid
    context = np.zeros_like(valid)
    targets = np.zeros((batch_size, 1, num_peaks), bool)
    for row in range(batch_size):
        order = rng.permutation(np.flatnonzero(valid[row]))
        targets[row, 0, order[:2]] = True
        context[row, order[2:5]] = True
    precursor = np.full(batch_size, 0.6, np.float32)
    metadata = _metadata(precursor)
    metadata[-1, :2] = 0.0  # dropped precursor
    return {
        "peak_mz": mz.astype(np.float32),
        "peak_intensity": (rng.uniform(0.1, 1.0, mz.shape) * valid).astype(np.float32),
        "peak_valid_mask": valid,
        "context_mask": context,
        "target_masks": targets,
        "precursor_mz": precursor,
        "spectrum_metadata": metadata,
    }


def _perturb(model: nnx.Module, scale: float = 0.2) -> None:
    """Move zero-initialized heads off zero so every path carries signal."""
    state = nnx.state(model, nnx.Param)
    for index, (_, value) in enumerate(nnx.to_flat_state(state)):
        value[...] = value[...] + scale * jax.random.normal(
            jax.random.key(index), value.shape
        )
    nnx.update(model, state)


def test_bridge_ignores_padding_and_reads_pair_content():
    bridge = PairToSingleBridge(6, 4, rank=3, rngs=nnx.Rngs(0))
    single = jax.random.normal(jax.random.key(1), (2, 5, 6))
    pair = jax.random.normal(jax.random.key(2), (2, 5, 5, 4))
    mask = jnp.array([[True, True, False, True, False], [True] * 5])
    valid = mask[:, :, None] & mask[:, None, :]
    update = bridge(single, pair, mask)

    np.testing.assert_array_equal(np.asarray(update)[~np.asarray(mask)], 0)
    np.testing.assert_allclose(
        bridge(jnp.where(mask[..., None], single, 50.0),
               jnp.where(valid[..., None], pair, -30.0), mask),
        update,
        atol=1e-6,
    )
    gradient = jax.grad(lambda z: (bridge(single, z, mask) ** 2).sum())(pair)
    np.testing.assert_array_equal(np.asarray(gradient)[~np.asarray(valid)], 0)
    assert np.linalg.norm(np.asarray(gradient)) > 1e-3


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_triangle_contraction_gemm_path_matches_broadcast(monkeypatch, direction):
    a = jax.random.normal(jax.random.key(3), (2, 5, 5, 4))
    b = jax.random.normal(jax.random.key(4), (2, 5, 5, 4))
    broadcast = _triangle_contract(a, b, direction)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    np.testing.assert_allclose(
        _triangle_contract(a, b, direction), broadcast, rtol=1e-5, atol=1e-5
    )


def test_gpu_dots_keep_f32_outputs_and_take_bf16_cotangents(monkeypatch):
    lhs = jax.random.normal(jax.random.key(5), (3, 6, 8)).astype(jnp.bfloat16)
    rhs = jax.random.normal(jax.random.key(6), (5, 8)).astype(jnp.bfloat16)
    dims = (((2,), (1,)), ((), ()))
    weights = jax.random.normal(jax.random.key(7), (3, 6, 5))

    def loss(dot, a, b):
        return jnp.sum(dot(a, b, dims, preferred_element_type=jnp.float32) * weights)

    reference = jax.grad(partial(loss, jax.lax.dot_general), argnums=(0, 1))(lhs, rhs)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    np.testing.assert_array_equal(
        _acc_dot_general(lhs, rhs, dims, preferred_element_type=jnp.float32),
        jax.lax.dot_general(lhs, rhs, dims, preferred_element_type=jnp.float32),
    )
    grad_fn = jax.grad(partial(loss, _acc_dot_general), argnums=(0, 1))
    for actual, expected in zip(grad_fn(lhs, rhs), reference, strict=True):
        assert actual.dtype == expected.dtype == jnp.bfloat16
        np.testing.assert_allclose(
            np.asarray(actual, np.float32), np.asarray(expected, np.float32),
            rtol=2e-2, atol=2e-2,
        )
    # The gradient GEMMs sit outside the custom-VJP forward; all take bf16.
    backward_dots = [
        eqn for eqn in jax.make_jaxpr(grad_fn)(lhs, rhs).jaxpr.eqns
        if eqn.primitive.name == "dot_general"
    ]
    assert backward_dots
    for eqn in backward_dots:
        assert all(var.aval.dtype == jnp.bfloat16 for var in eqn.invars)


def test_precursor_edges_touch_only_cls_edges_and_start_at_zero():
    model = PeakSetJEPAJax(rngs=nnx.Rngs(0), **_dual_kwargs())
    encoder = model.encoder
    batch = {k: jnp.asarray(v) for k, v in _batch(7).items()}
    args = (batch["peak_mz"], batch["peak_intensity"])
    x, visible, _ = encoder._embed_peaks(*args, batch["peak_valid_mask"], None, None)

    def pairs(metadata):
        return encoder._embed_pairs(*args, x, visible, batch["precursor_mz"], metadata)

    without = pairs(None)
    np.testing.assert_array_equal(pairs(batch["spectrum_metadata"]), without)

    _perturb(encoder.precursor_edges)
    delta = np.asarray(pairs(batch["spectrum_metadata"]) - without)
    cls = 7
    changed = np.zeros(delta.shape[:3], bool)
    changed[:, :cls, cls] = changed[:, cls, :cls] = True
    changed &= np.concatenate([np.asarray(visible), np.zeros((3, 1), bool)], 1)[:, :, None] | (
        np.concatenate([np.asarray(visible), np.zeros((3, 1), bool)], 1)[:, None, :]
    )
    assert np.abs(delta[changed]).min() > 0
    np.testing.assert_array_equal(delta[~changed], 0)
    # A dropped precursor leaves only the condition term: equal for every peak.
    dropped = delta[-1, :cls, cls][np.asarray(visible)[-1]]
    np.testing.assert_allclose(dropped, np.broadcast_to(dropped[:1], dropped.shape), atol=1e-6)


def test_dual_trains_every_path():
    model = PeakSetJEPAJax(rngs=nnx.Rngs(0), **_dual_kwargs())
    model.set_fastmixer_capacities(6, 8, 3)
    _perturb(model.encoder.precursor_edges)
    batch = {k: jnp.asarray(v) for k, v in _batch(7).items()}
    metrics = model(batch)

    assert model.encoder.blocks[0].pair_to_single is not None
    assert model.latent_mask_token is not None and model.pair_mask_token is not None
    for name in ("loss", "mae_mz_loss", "distogram_loss"):
        assert np.isfinite(float(metrics[name]))
    np.testing.assert_allclose(
        float(metrics["loss"]),
        float(metrics["mae_term"] + metrics["distogram_term"]),
        rtol=1e-6,
    )
    gradient = nnx.grad(lambda m: m(batch, loss_only=True)["loss"])(model)
    flat = {
        "/".join(map(str, path)): float(jnp.abs(value[...]).sum())
        for path, value in nnx.to_flat_state(gradient)
    }
    for name in (
        "pair_to_single", "precursor_edges", "pair_predictor", "pair_mask_token",
        "distogram_head", "masked_latent_predictor", "jepa_mae_mz_head", "pair_embedder",
    ):
        assert sum(value for key, value in flat.items() if name in key) > 0, name


@pytest.mark.parametrize("ordered_mode", ["ordered", "ordered_saved"])
def test_ordered_checkpointing_matches_plain_gradients(ordered_mode):
    """Ordered recomputation (the GPU remat mode) changes memory, not values."""
    batch = {k: jnp.asarray(v) for k, v in _batch(7).items()}

    def loss_and_grads(mode: str):
        model = PeakSetJEPAJax(
            rngs=nnx.Rngs(0),
            **_dual_kwargs(),
            activation_checkpoint_mode=mode,
            activation_checkpoint_modules=("encoder", "predictor", "distogram"),
        )
        model.set_fastmixer_capacities(6, 8, 3)
        _perturb(model)
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def loss(p):
            return nnx.merge(graphdef, p, rest)(batch, loss_only=True)["loss"]

        # Each checkpointed block holds its recomputation behind a barrier.
        jaxpr = str(jax.make_jaxpr(jax.grad(loss))(params))
        assert (jaxpr.count("optimization_barrier") > 0) == (mode != "none")
        value, grads = jax.jit(jax.value_and_grad(loss))(params)
        return value, dict(nnx.to_flat_state(grads))

    plain_loss, plain = loss_and_grads("none")
    ordered_loss, ordered = loss_and_grads(ordered_mode)
    np.testing.assert_allclose(ordered_loss, plain_loss, rtol=1e-6)
    assert ordered.keys() == plain.keys()
    for path in plain:
        np.testing.assert_allclose(
            ordered[path][...], plain[path][...], rtol=1e-4, atol=1e-6, err_msg=str(path)
        )


def test_dual_encoder_paths_agree():
    """Probes use the scattered-single path; training uses the compact one."""
    model = PeakSetJEPAJax(rngs=nnx.Rngs(0), **_dual_kwargs())
    _perturb(model.encoder)
    batch = {k: jnp.asarray(v) for k, v in _batch(7).items()}
    common = dict(
        valid_mask=batch["peak_valid_mask"],
        precursor_mz=batch["precursor_mz"],
        spectrum_metadata=batch["spectrum_metadata"],
    )
    full_single, _ = model.encoder.forward_with_pair(
        batch["peak_mz"], batch["peak_intensity"], **common
    )
    compact_single, _, idx, mask = model.encoder.forward_with_pair_compact(
        batch["peak_mz"], batch["peak_intensity"], visible_mask=batch["peak_valid_mask"],
        max_visible_tokens=8, **common,
    )
    gathered = jnp.take_along_axis(full_single, idx[..., None], axis=1)
    np.testing.assert_allclose(
        np.asarray(gathered)[np.asarray(mask)],
        np.asarray(compact_single)[np.asarray(mask)],
        atol=2e-5,
    )


def test_peak_count_stages_set_capacities():
    config = config_dict.ConfigDict({
        **_dual_kwargs(num_peaks=63),
        "jepa_context_fraction": 0.6,
        "jepa_target_fraction": 0.25,
        "jepa_mask_strategy": ["random"],
        "num_peaks_schedule": (31, 63),
        "gradient_accumulation_steps_schedule": (1, 2),
    })
    stages = jepa_mask_stages(config)
    assert [(stage.num_peaks, stage.gradient_accumulation_steps) for stage in stages] == [
        (31, 1), (63, 2),
    ]
    assert [stage.context_fraction for stage in stages] == [0.6, 0.6]
    # Encoder: context + CLS; predictor: context + target union + CLS.
    assert pairmixer_fast_stage_capacities(config) == ((20, 28, 8), (39, 55, 16))
    config.num_peaks = 47
    with pytest.raises(ValueError, match="largest num_peaks_schedule"):
        jepa_mask_stages(config)


class _PeakStageDataModule:
    def __init__(self, num_peaks: int, train_steps: int) -> None:
        self.num_peaks = num_peaks
        self.train_steps = train_steps
        self.global_batch_size = self.batch_size = 2
        self.gradient_accumulation_steps = 1
        self.num_peaks_calls: list[int] = []

    def _batch(self) -> dict[str, np.ndarray]:
        def sample(index):
            mz = 100.0 + 15.0 * np.arange(self.num_peaks) + index
            spectra = np.zeros((2, 128), np.float32)
            spectra[0, :self.num_peaks] = mz
            spectra[1, :self.num_peaks] = np.linspace(1.0, 0.2, self.num_peaks)
            return {
                "spectra": spectra,
                "precursor_mz_raw": np.float32(900.0),
                "collision_energy": np.float32(20.0),
                "charge": np.float32(1.0),
            }

        batch = GemsBatchCollator(
            augment=True, num_target_blocks=1, context_fraction=0.6,
            target_fraction=0.25, block_min_len=1, num_peaks=self.num_peaks,
            max_precursor_mz=1000.0, min_peak_intensity=0.0,
            peak_drop_min_intensity=0.0, peak_ordering="mz",
            precursor_peak_exclusion_window_da=0.0, mask_strategy="random",
            mask_lengths=(1, 2, 3), mask_round_from=2, output_format="numpy",
        )([sample(0), sample(1)])
        batch["spectrum_metadata"] = _metadata(batch["precursor_mz"])
        return batch

    def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
        return [self._batch()] * (self.train_steps - start_batch) * self.gradient_accumulation_steps

    @property
    def val_loader(self):
        return [self._batch()]

    def val_loader_for_eval(self, *, augment: bool):
        return [self._batch()]

    def set_mask_fractions(self, context_fraction: float, target_fraction: float) -> None:
        pass

    def set_gradient_accumulation_steps(self, steps: int) -> None:
        self.gradient_accumulation_steps = steps

    def set_num_peaks(self, num_peaks: int) -> None:
        self.num_peaks = num_peaks
        self.num_peaks_calls.append(num_peaks)


def test_training_loop_expands_peak_count(tmp_path):
    config = config_dict.ConfigDict({
        **_dual_kwargs(num_peaks=7),
        "seed": 5,
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "jax_mesh_devices": "1",
        "gradient_accumulation_steps": 1,
        "checkpoint_every_steps": 0,
        "log_every_n_steps": 0,
        "msg_probe_every_n_steps": -1,
        "jepa_mask_strategy": ["random"],
        "jepa_context_fraction": 0.6,
        "jepa_target_fraction": 0.25,
        "num_peaks_schedule": (4, 7),
        "jepa_mask_schedule_steps": (2,),
    })
    model = PeakSetJEPAJax(PeakSetJEPASettings.from_config(config))
    datamodule = _PeakStageDataModule(num_peaks=7, train_steps=4)
    manager = build_jax_checkpoint_manager(tmp_path / "checkpoints")
    metrics = _run_jax_training_loop(
        config=config,
        datamodule=datamodule,
        model=model,
        logger=MetricLogger(),
        total_steps=4,
        checkpoint_manager=manager,
        resume_step=None,
        checkpoint_metadata=jax_training_checkpoint_metadata("test", {}),
        metric_reduction="mean",
        enable_msg_probe=False,
    )
    manager.close()

    assert metrics["run/final_global_step"] == 4.0
    assert datamodule.num_peaks_calls == [4, 7]
    first, last = pairmixer_fast_stage_capacities(config)
    assert first < last
    assert model.pairmixer_fast_encoder_max_visible_tokens == last[0]
    assert np.isfinite(metrics["train/loss"])
