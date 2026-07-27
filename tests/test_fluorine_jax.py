from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np

from spectra_learning.models import model_jax, settings
from spectra_learning.probes.massspec import fluorine, msg_probe_jax
from spectra_learning.training import checkpointing_jax, pretrain_jax


class _TinyJaxEncoder:
    def __call__(
        self,
        peak_mz,
        peak_intensity,
        *,
        valid_mask,
        precursor_mz=None,
        spectrum_metadata=None,
    ):
        return jnp.stack((peak_mz, peak_intensity), axis=-1)


class _TinyJaxFeatureModel:
    def __init__(self) -> None:
        self.encoder = _TinyJaxEncoder()


class _TinyJaxPairEncoder(_TinyJaxEncoder):
    def forward_with_pair(
        self,
        peak_mz,
        peak_intensity,
        *,
        valid_mask,
        precursor_mz=None,
        spectrum_metadata=None,
    ):
        single = super().__call__(
            peak_mz,
            peak_intensity,
            valid_mask=valid_mask,
            precursor_mz=precursor_mz,
            spectrum_metadata=spectrum_metadata,
        )
        pair = jnp.stack(
            (
                jnp.broadcast_to(peak_mz[:, :, None], (*peak_mz.shape, peak_mz.shape[1])),
                jnp.broadcast_to(
                    peak_intensity[:, None, :],
                    (*peak_intensity.shape, peak_intensity.shape[1]),
                ),
            ),
            axis=-1,
        )
        single = jnp.pad(single, ((0, 0), (0, 1), (0, 0)), constant_values=jnp.nan)
        pair = jnp.pad(
            pair,
            ((0, 0), (0, 1), (0, 1), (0, 0)),
            constant_values=jnp.nan,
        )
        return single, pair


class _TinyJaxPairFeatureModel:
    def __init__(self) -> None:
        self.encoder = _TinyJaxPairEncoder()


class _CheckpointManager:
    def latest_step(self) -> int:
        return 7


def _tiny_probe_batch() -> dict[str, np.ndarray]:
    return {
        "peak_mz": np.asarray(
            [[0.1, 0.2], [0.2, 0.4], [0.6, 0.8], [0.8, 0.9]],
            dtype=np.float32,
        ),
        "peak_intensity": np.asarray(
            [[0.2, 0.1], [0.3, 0.2], [0.7, 0.6], [0.9, 0.8]],
            dtype=np.float32,
        ),
        "peak_valid_mask": np.ones((4, 2), dtype=bool),
        "label": np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        "row_idx": np.arange(4, dtype=np.int64),
    }


def _probe_config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict(
        {
            "seed": 13,
            "training_max_steps": 1,
            "model_dim": 2,
            "covariance_pooling_dim": 1,
            "pairmixer_pair_dim": 2,
            "num_peaks": 2,
        }
    )


def _probe_data() -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "nist_repo_id": "owner/nist",
            "nist_revision": "nist-sha",
            "nist_subdir": "nist",
            "nist_source_dir": "/cache/nist",
            "peak_preprocessing": {"version": 1},
            "data_provenance": {"massspec_nist_revision": "nist-sha"},
            "train_size": 4,
            "train_positive": 2,
            "val_size": 4,
            "val_positive": 2,
            "test_size": 4,
            "test_positive": 2,
        }
    )


def _patch_probe_runtime(monkeypatch, config, data, feature_model) -> None:
    monkeypatch.setattr(fluorine, "load_config", lambda _path: config)
    monkeypatch.setattr(
        settings,
        "PeakSetJEPASettings",
        SimpleNamespace(from_config=lambda _config: object()),
    )
    monkeypatch.setattr(model_jax, "PeakSetJEPAJax", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(nnx, "jit", lambda function: function)
    monkeypatch.setattr(nnx, "merge", lambda *_args: feature_model)
    monkeypatch.setattr(pretrain_jax, "prepare_jax_training_config", lambda _config: None)
    monkeypatch.setattr(
        pretrain_jax,
        "_build_pretrain_jax_datamodule",
        lambda *_args, **_kwargs: SimpleNamespace(info={"source": "gems"}),
    )
    monkeypatch.setattr(
        pretrain_jax,
        "_pretrain_jax_checkpoint_contract",
        lambda *_args, **_kwargs: {
            "peak_preprocessing": {"version": 1},
            "data_provenance": {"source": "gems"},
        },
    )
    monkeypatch.setattr(
        pretrain_jax,
        "init_pure_optax_train_state",
        lambda *_args, **_kwargs: (object(), {}, {}, {}, None),
    )
    monkeypatch.setattr(
        pretrain_jax,
        "_jax_data_mesh_for_device_count",
        lambda _device_count: None,
    )
    monkeypatch.setattr(
        pretrain_jax,
        "_replicate_tree_on_data_mesh",
        lambda tree, _mesh: tree,
    )
    monkeypatch.setattr(
        checkpointing_jax,
        "jax_training_checkpoint_metadata",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        checkpointing_jax,
        "build_jax_checkpoint_manager",
        lambda *_args, **_kwargs: _CheckpointManager(),
    )
    monkeypatch.setattr(
        checkpointing_jax,
        "restore_jax_training_state",
        lambda _manager, _step, target, **_kwargs: target,
    )
    monkeypatch.setattr(
        msg_probe_jax,
        "_full_visible_fastmixer_probe_model",
        lambda _config, _model: feature_model,
    )
    monkeypatch.setattr(
        fluorine,
        "build_fluorine_data",
        lambda **_kwargs: data,
    )


def _probe_args(tmp_path: Path, **overrides) -> SimpleNamespace:
    values = {
        "config": tmp_path / "config.py",
        "checkpoint": tmp_path / "checkpoints",
        "workdir": None,
        "jax_checkpoint_step": None,
        "cache_dir": tmp_path / "cache",
        "output_prefix": tmp_path / "fluorine-jax",
        "output_state": tmp_path / "fluorine-jax.pt",
        "batch_size": 4,
        "num_workers": 0,
        "pooling": "covariance",
        "focal_alpha": "auto",
        "focal_gamma": 2.0,
        "hidden_dims": "2",
        "learning_rates": "0.01",
        "weight_decays": "0.0",
        "dropouts": "0.0",
        "select_metric": "average_precision",
        "seed": 13,
        "epochs": 1,
        "patience": 1,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
        "comparison_dir": None,
        "output_json": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_jax_restore_uses_full_pretraining_checkpoint_contract(
    monkeypatch,
    tmp_path: Path,
):
    config = _probe_config()
    _patch_probe_runtime(
        monkeypatch,
        config,
        _probe_data(),
        _TinyJaxFeatureModel(),
    )
    restored_metadata = {}

    monkeypatch.setattr(
        checkpointing_jax,
        "jax_training_checkpoint_metadata",
        lambda task, contract: {
            "training_task": task,
            "task_contract": contract,
        },
    )

    def restore(_manager, _step, target, *, expected_metadata):
        restored_metadata.update(expected_metadata)
        return target

    monkeypatch.setattr(
        checkpointing_jax,
        "restore_jax_training_state",
        restore,
    )

    checkpoint = fluorine._restore_jax_fluorine_checkpoint(
        _probe_args(tmp_path)
    )

    assert restored_metadata == {
        "training_task": "pretrain",
        "task_contract": {
            "peak_preprocessing": {"version": 1},
            "data_provenance": {"source": "gems"},
        },
    }
    assert (
        checkpoint.source_checkpoint_contract["checkpoint_metadata"]
        == restored_metadata
    )


def test_run_probe_jax_trains_tiny_trial_and_builds_artifacts(
    monkeypatch,
    tmp_path: Path,
):
    config = _probe_config()
    data = _probe_data()
    loader_calls: list[tuple[str, bool, int, str]] = []
    saved_states: list[tuple[dict, object]] = []
    standard_output_calls: list[dict] = []
    feature_model = _TinyJaxFeatureModel()

    _patch_probe_runtime(monkeypatch, config, data, feature_model)

    def build_loader(_data, split, *, shuffle, seed, output_format, **_kwargs):
        loader_calls.append((split, shuffle, seed, output_format))
        return [_tiny_probe_batch()]

    def save_state(state, path):
        saved_states.append((state, path))

    def write_standard_outputs(**kwargs):
        standard_output_calls.append(kwargs)
        return {"mode": "probe", "metrics": {"average_precision": 1.0}}

    monkeypatch.setattr(fluorine, "build_murcko_fluorine_loader", build_loader)
    monkeypatch.setattr(fluorine, "save_torch_checkpoint", save_state)
    monkeypatch.setattr(
        fluorine,
        "write_standard_fluorine_outputs",
        write_standard_outputs,
    )
    monkeypatch.setattr(
        fluorine,
        "write_all_pr_curve_comparison",
        lambda **_kwargs: {"curves": []},
    )

    checkpoint_dir = tmp_path / "checkpoints"
    output_prefix = tmp_path / "fluorine-jax"
    state_path = tmp_path / "fluorine-jax.pt"
    args = _probe_args(tmp_path)

    payload = fluorine.run_probe_jax(args)

    assert loader_calls == [
        ("train", False, 13, "numpy"),
        ("val", False, 10_013, "numpy"),
        ("test", False, 20_013, "numpy"),
    ]
    assert payload["backend"] == "jax"
    assert payload["best_epoch"] == 1
    assert payload["best_hparams"] == {
        "hidden_dim": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "dropout": 0.0,
    }
    assert payload["focal_alpha"] == 0.5
    assert payload["train_size"] == 4
    assert np.isfinite(payload["test"]["test/average_precision"])
    assert len(payload["trials"]) == 1
    assert payload["standard_outputs"] == {
        "summary": str(output_prefix.with_suffix(".summary.json")),
        "state": str(state_path),
        "output_prefix": str(output_prefix),
    }

    assert len(saved_states) == 1
    head_state, saved_path = saved_states[0]
    assert saved_path == state_path
    assert head_state["backend"] == "jax"
    assert head_state["complete"] is True
    assert head_state["checkpoint_path"] == str(checkpoint_dir / "orbax" / "7")
    assert head_state["state_contract"] == {
        "version": fluorine.FLUORINE_STATE_CONTRACT_VERSION,
        "source_checkpoint": {
            "backend": "jax",
            "checkpoint_path": str(checkpoint_dir / "orbax" / "7"),
            "restore_step": 7,
            "checkpoint_metadata": {},
        },
        "evaluation_data_provenance": data.metadata["data_provenance"],
        "peak_preprocessing": data.metadata["peak_preprocessing"],
    }
    assert head_state["best_epoch"] == 1
    assert len(head_state["history"]) == 1
    assert np.isfinite(head_state["history"][0]["train_loss"])
    assert head_state["jax_params"]["pooler"]["left"].shape == (2, 1)
    assert head_state["jax_params"]["head"][-1]["w"].shape == (2, 1)

    assert len(standard_output_calls) == 1
    standard_output = standard_output_calls[0]
    assert standard_output["head_state"] is head_state
    assert standard_output["targets"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert standard_output["row_indices"].tolist() == [0, 1, 2, 3]
    summary = json.loads(output_prefix.with_suffix(".summary.json").read_text())
    assert summary["backend"] == "jax"
    assert summary["all_pr_curves"] == {"curves": []}


def test_run_probe_jax_streams_pair_features_and_keeps_first_strict_best(
    monkeypatch,
    tmp_path: Path,
):
    config = _probe_config()
    data = _probe_data()
    feature_model = _TinyJaxPairFeatureModel()
    loader_calls: list[tuple[str, bool, int, str]] = []
    metric_calls: list[str] = []
    saved_states: list[dict] = []

    _patch_probe_runtime(monkeypatch, config, data, feature_model)

    def build_loader(_data, split, *, shuffle, seed, output_format, **_kwargs):
        loader_calls.append((split, shuffle, seed, output_format))
        return [_tiny_probe_batch()]

    def tied_metrics(_targets, logits, prefix):
        assert np.isfinite(logits).all()
        metric_calls.append(prefix)
        return {
            f"{prefix}/average_precision": 0.5,
            f"{prefix}/roc_auc": 0.5,
        }

    monkeypatch.setattr(fluorine, "build_murcko_fluorine_loader", build_loader)
    monkeypatch.setattr(fluorine, "_metric_dict", tied_metrics)
    monkeypatch.setattr(
        fluorine,
        "save_torch_checkpoint",
        lambda state, _path: saved_states.append(state),
    )
    monkeypatch.setattr(
        fluorine,
        "write_standard_fluorine_outputs",
        lambda **_kwargs: {"mode": "probe", "metrics": {"average_precision": 0.5}},
    )
    monkeypatch.setattr(
        fluorine,
        "write_all_pr_curve_comparison",
        lambda **_kwargs: {"curves": []},
    )

    baseline_args = _probe_args(
        tmp_path,
        pooling="single_pair_covariance",
        output_prefix=tmp_path / "pair-baseline",
        output_state=tmp_path / "pair-baseline.pt",
    )
    fluorine.run_probe_jax(baseline_args)
    baseline_state = saved_states[-1]

    loader_calls.clear()
    metric_calls.clear()
    tied_args = _probe_args(
        tmp_path,
        pooling="single_pair_covariance",
        epochs=8,
        patience=1,
        output_prefix=tmp_path / "pair-tied",
        output_state=tmp_path / "pair-tied.pt",
    )
    payload = fluorine.run_probe_jax(tied_args)
    tied_state = saved_states[-1]

    assert loader_calls == [
        ("train", True, 13, "numpy"),
        ("val", False, 10_013, "numpy"),
        ("train", True, 14, "numpy"),
        ("val", False, 10_013, "numpy"),
        ("test", False, 20_013, "numpy"),
    ]
    assert metric_calls == ["val", "val", "test"]
    assert payload["best_epoch"] == 1
    assert len(tied_state["history"]) == 2
    assert all(np.isfinite(epoch["train_loss"]) for epoch in tied_state["history"])
    assert set(tied_state["jax_params"]["pooler"]) == {
        "single_left",
        "single_right",
        "pair_left",
        "pair_right",
        "output_w",
        "output_b",
    }
    for baseline_leaf, tied_leaf in zip(
        jax.tree.leaves(baseline_state["jax_params"]),
        jax.tree.leaves(tied_state["jax_params"]),
        strict=True,
    ):
        np.testing.assert_array_equal(tied_leaf, baseline_leaf)
        assert np.isfinite(tied_leaf).all()
