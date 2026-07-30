from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from flax import nnx
from ml_collections import config_dict

import train
from spectra_learning.data.ar_spectra import SpectraARTokenizer, SpectraARTokenizerConfig
from spectra_learning.data.spectra import DEFAULT_MAX_PRECURSOR_MZ, PEAK_MZ_MAX
import spectra_learning.models.ar_spectra_jax as ar_spectra_jax
from spectra_learning.models.ar_spectra_jax import (
    RotaryEmbeddingJax,
    SpectraARCausalSelfAttentionJax,
    SpectraARTransformerJax,
    SpectraARTransformerJaxConfig,
    pack_qk_projection_rows,
)
from spectra_learning.models.causal_attention_pallas import pallas_causal_attention
from spectra_learning.models.common_jax import scaled_dot_product_attention
from spectra_learning.training.checkpointing_jax import (
    build_jax_checkpoint_manager,
    jax_training_checkpoint_metadata,
)
from spectra_learning.training.logging import MetricLogger
from spectra_learning.training.pretrain_jax import (
    _evaluate_jax_validation_loss,
    _run_jax_training_loop,
    _validate_jax_task_probe_config,
    JaxTrainingTask,
    make_pure_accumulated_train_step,
    make_pure_eval_step,
    trainable_param_filter,
)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "peak_mz": torch.tensor(
            [[138.75, 938.50, 466.25, 0.0]],
            dtype=torch.float32,
        )
        / PEAK_MZ_MAX,
        "peak_intensity": torch.tensor([[0.10, 1.00, 0.50, 0.0]], dtype=torch.float32),
        "peak_valid_mask": torch.tensor([[True, True, True, False]]),
        "precursor_mz": torch.tensor([512.50], dtype=torch.float32)
        / DEFAULT_MAX_PRECURSOR_MZ,
        "collision_energy": torch.tensor([0.35], dtype=torch.float32),
        "charge": torch.tensor([2.0], dtype=torch.float32),
    }


def _jax_batch(batch: dict[str, torch.Tensor]) -> dict[str, jax.Array]:
    return {
        key: jnp.asarray(value.detach().cpu().numpy())
        for key, value in batch.items()
    }


class _TokenWeightedToyModel(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.asarray(0.0))

    def __call__(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        target = batch["target"]
        mask = batch["mask"].astype(jnp.float32)
        target_tokens = jnp.sum(mask)
        squared_error = jnp.square(self.weight[...] - target)
        loss = jnp.sum(squared_error * mask) / jnp.maximum(target_tokens, 1.0)
        return {
            "loss": loss,
            "loss/value": loss,
            "target_tokens": target_tokens,
            "target_tokens/value": target_tokens,
        }


class _TokenWeightedValidationData:
    def val_loader_for_eval(self, *, augment: bool):
        assert augment is True
        return [
            {
                "target": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                "mask": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            },
            {
                "target": np.asarray([3.0, 3.0, 3.0], dtype=np.float32),
                "mask": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            },
        ]


class _ARLoopData:
    train_steps = 1
    global_batch_size = 2
    batch_size = 1

    def __init__(self, batches: list[dict[str, np.ndarray]]) -> None:
        self.batches = batches
        self.loader_calls = []

    def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
        self.loader_calls.append((epoch, start_batch))
        return self.batches[start_batch * 2 :]

    def val_loader_for_eval(self, *, augment: bool):
        assert augment is True
        return self.batches


class _RecordingLogger(MetricLogger):
    def __init__(self) -> None:
        self.logs = []

    def log_metrics(self, metrics, step=None) -> None:
        self.logs.append((dict(metrics), step))


def test_jax_scaled_dot_product_attention_matches_manual_causal_attention() -> None:
    key = jax.random.PRNGKey(0)
    q, k, v = (
        jax.random.normal(subkey, (2, 4, 7, 8), dtype=jnp.float32)
        for subkey in jax.random.split(key, 3)
    )

    actual = scaled_dot_product_attention(q, k, v, is_causal=True)
    scores = jnp.einsum("...qd,...kd->...qk", q, k) / np.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones((q.shape[-2], k.shape[-2]), dtype=jnp.bool_))
    scores = jnp.where(causal, scores, -jnp.inf)
    expected = jnp.einsum("...qk,...kd->...qd", jax.nn.softmax(scores, axis=-1), v)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5)


def test_packed_qk_projection_matches_interleaved_rope_coordinates() -> None:
    model_dim = 16
    num_heads = 2
    head_dim = model_dim // num_heads
    hidden = jax.random.normal(jax.random.key(1), (2, 5, model_dim))
    weight = jax.random.normal(jax.random.key(2), (3 * model_dim, model_dim))
    query_scale = 0.5
    projected = jnp.matmul(hidden, weight.T).reshape(
        2,
        5,
        3,
        num_heads,
        head_dim,
    )
    packed_projected = jnp.matmul(
        hidden,
        pack_qk_projection_rows(
            weight,
            num_heads=num_heads,
            query_scale=query_scale,
        ).T,
    ).reshape(2, 5, 3, num_heads, head_dim)
    rope = RotaryEmbeddingJax(head_dim, 5, base=10_000.0)

    for index, scale in ((0, query_scale), (1, 1.0)):
        interleaved = rope(projected[:, :, index] * scale)
        expected_split = jnp.concatenate(
            (interleaved[..., 0::2], interleaved[..., 1::2]),
            axis=-1,
        )
        actual_split = rope.split_half(packed_projected[:, :, index])
        np.testing.assert_allclose(
            np.asarray(actual_split),
            np.asarray(expected_split),
            atol=2e-5,
            rtol=2e-5,
        )
    np.testing.assert_allclose(
        np.asarray(packed_projected[:, :, 2]),
        np.asarray(projected[:, :, 2]),
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")
@pytest.mark.parametrize(
    "shape",
    (
        (1, 8, 128, 128),
        (1, 8, 136, 128),
        (1, 14, 136, 128),
        (4, 14, 776, 128),
    ),
    ids=("full-block", "tail", "padded-head-group", "300m"),
)
def test_pallas_causal_attention_forward_and_backward_match_xla(
    shape: tuple[int, int, int, int],
) -> None:
    query, key, value, output_gradient = (
        jax.random.normal(subkey, shape, dtype=jnp.bfloat16)
        for subkey in jax.random.split(jax.random.key(3), 4)
    )

    def loss(attention, query, key, value):
        output = attention(query, key, value)
        objective = jnp.sum(
            output.astype(jnp.float32) * output_gradient.astype(jnp.float32)
        )
        return objective, output

    pallas_fn = jax.jit(
        jax.value_and_grad(
            lambda q, k, v: loss(pallas_causal_attention, q, k, v),
            argnums=(0, 1, 2),
            has_aux=True,
        )
    )
    xla_fn = jax.jit(
        jax.value_and_grad(
            lambda q, k, v: loss(
                lambda query, key, value: scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=True,
                ),
                q,
                k,
                v,
            ),
            argnums=(0, 1, 2),
            has_aux=True,
        )
    )
    (_pallas_loss, pallas_output), pallas_grads = pallas_fn(query, key, value)
    (_xla_loss, xla_output), xla_grads = xla_fn(query, key, value)

    for actual, expected in zip(
        (pallas_output, *pallas_grads),
        (xla_output, *xla_grads),
        strict=True,
    ):
        actual_np = np.asarray(actual, dtype=np.float32)
        assert np.isfinite(actual_np).all()
        np.testing.assert_allclose(
            actual_np,
            np.asarray(expected, dtype=np.float32),
            atol=4e-2,
            rtol=3e-2,
        )


def test_jax_ar_transformer_forward_returns_finite_loss() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    tokenized = tokenizer.tokenize_batch(_batch())
    model = SpectraARTransformerJax(
        SpectraARTransformerJaxConfig(
            vocab_size=tokenizer.vocab_size,
            num_token_kinds=tokenizer.num_token_kinds,
            max_sequence_length=tokenizer.sequence_length,
            pad_token_id=tokenizer.pad_token_id,
            model_dim=32,
            num_layers=1,
            num_heads=4,
            mlp_multiple=2.0,
            compute_dtype=jnp.float32,
        )
    )

    train_output = model(_jax_batch(tokenized))
    inference_batch = {
        key: value
        for key, value in tokenized.items()
        if not key.startswith("target_")
    }
    inference_output = model(_jax_batch(inference_batch))

    assert inference_output["logits"].shape == (
        1,
        tokenizer.sequence_length - 1,
        tokenizer.vocab_size,
    )
    assert "logits" not in train_output
    assert np.isfinite(np.asarray(train_output["loss"]))
    assert float(np.asarray(train_output["target_tokens"])) == 19.0
    assert "token_accuracy/fragment_mz_level_0" in train_output


@pytest.mark.parametrize(
    "removed_setting",
    (
        {"ar_splash_block_size": 128},
        {"ar_attention_kernel": "splash"},
    ),
)
def test_jax_ar_config_rejects_removed_splash_path(removed_setting) -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    config = config_dict.ConfigDict(removed_setting)

    with pytest.raises(ValueError, match="has been removed"):
        SpectraARTransformerJaxConfig.from_config(config, tokenizer)


def test_jax_token_weighted_train_and_validation_reduce_by_target_tokens() -> None:
    model = _TokenWeightedToyModel()
    graphdef, trainable_params, static_state = nnx.split(
        model,
        trainable_param_filter,
        ...,
    )
    trainable_params = nnx.as_pure(trainable_params)
    static_state = nnx.as_pure(static_state)
    optimizer = optax.sgd(1.0)
    opt_state = optimizer.init(trainable_params)
    train_step = make_pure_accumulated_train_step(
        graphdef,
        optimizer,
        sharded=False,
        metric_reduction="token_weighted",
    )
    accumulated_batch = {
        "target": jnp.asarray(
            [[1.0, 0.0, 0.0], [3.0, 3.0, 3.0]],
            dtype=jnp.float32,
        ),
        "mask": jnp.asarray(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=jnp.float32,
        ),
    }

    trainable_params, _opt_state, train_metrics = train_step(
        trainable_params,
        static_state,
        opt_state,
        accumulated_batch,
    )
    nnx.update(model, trainable_params)

    assert float(np.asarray(train_metrics["loss"])) == pytest.approx(7.0)
    assert float(np.asarray(train_metrics["loss/value"])) == pytest.approx(7.0)
    assert float(np.asarray(train_metrics["target_tokens"])) == 4.0
    assert float(np.asarray(model.weight[...])) == pytest.approx(5.0)

    validation_model = _TokenWeightedToyModel()
    validation_graphdef, validation_params, validation_static = nnx.split(
        validation_model,
        trainable_param_filter,
        ...,
    )
    eval_step = make_pure_eval_step(
        validation_graphdef,
        sharded=False,
        metric_reduction="token_weighted",
    )
    validation_metrics = _evaluate_jax_validation_loss(
        datamodule=_TokenWeightedValidationData(),
        trainable_params=nnx.as_pure(validation_params),
        static_state=nnx.as_pure(validation_static),
        eval_step=eval_step,
        max_steps=2,
        use_sharded_step=False,
        data_mesh=None,
        metric_reduction="token_weighted",
    )

    assert validation_metrics["val/loss"] == pytest.approx(7.0)
    assert validation_metrics["val/loss/value"] == pytest.approx(7.0)
    assert validation_metrics["val/target_tokens"] == 4.0
    assert validation_metrics["val/target_tokens/value"] == 4.0


def test_jax_ar_training_loop_logs_validates_checkpoints_and_resumes(tmp_path) -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    full_batch = {
        key: value.detach().cpu().numpy()
        for key, value in tokenizer.tokenize_batch(_batch()).items()
    }
    short_batch = {key: value.copy() for key, value in full_batch.items()}
    short_batch["target_loss_mask"][:] = False
    short_batch["target_loss_mask"][:, :5] = True
    batches = [full_batch, short_batch]
    expected_target_tokens = float(
        full_batch["target_loss_mask"].sum() + short_batch["target_loss_mask"].sum()
    )

    def build_model() -> SpectraARTransformerJax:
        return SpectraARTransformerJax(
            SpectraARTransformerJaxConfig(
                vocab_size=tokenizer.vocab_size,
                num_token_kinds=tokenizer.num_token_kinds,
                max_sequence_length=tokenizer.sequence_length,
                pad_token_id=tokenizer.pad_token_id,
                model_dim=16,
                num_layers=1,
                num_heads=2,
                mlp_multiple=2.0,
                attention_kernel="xla",
                compute_dtype=jnp.float32,
            )
        )

    cfg = config_dict.ConfigDict(
        {
            "num_epochs": 2,
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-4,
            "warmup_steps": 0,
            "optimizer": "adamw",
            "b1": 0.9,
            "b2": 0.95,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "gradient_accumulation_steps": 2,
            "jax_mesh_devices": "1",
            "checkpoint_every_steps": 1,
            "log_every_n_steps": 1,
            "val_every_n_steps": 1,
            "val_num_steps": 2,
            "msg_probe_every_n_steps": -1,
            "throughput_warmup_steps": 0,
        }
    )
    metadata = jax_training_checkpoint_metadata(
        "ar_spectra",
        {"tokenizer": {"mz_bin_widths": list(tokenizer.config.mz_bin_widths)}},
    )
    model = build_model()
    datamodule = _ARLoopData(batches)
    logger = _RecordingLogger()
    manager = build_jax_checkpoint_manager(
        tmp_path / "checkpoints",
        enable_async_checkpointing=False,
    )

    first_metrics = _run_jax_training_loop(
        config=cfg,
        datamodule=datamodule,
        model=model,
        logger=logger,
        total_steps=1,
        checkpoint_manager=manager,
        resume_step=None,
        checkpoint_metadata=metadata,
        metric_reduction="token_weighted",
        enable_msg_probe=False,
    )
    manager.close()

    assert first_metrics["run/final_global_step"] == 1.0
    assert first_metrics["train/target_tokens"] == expected_target_tokens
    assert first_metrics["val/target_tokens"] == expected_target_tokens
    assert any(
        step == 1 and payload.get("train/target_tokens") == expected_target_tokens
        for payload, step in logger.logs
    )
    assert any(
        step == 1 and payload.get("val/target_tokens") == expected_target_tokens
        for payload, step in logger.logs
    )

    resumed_model = build_model()
    resumed_datamodule = _ARLoopData(batches)
    resumed_manager = build_jax_checkpoint_manager(
        tmp_path / "checkpoints",
        enable_async_checkpointing=False,
    )
    resumed_metrics = _run_jax_training_loop(
        config=cfg,
        datamodule=resumed_datamodule,
        model=resumed_model,
        logger=MetricLogger(),
        total_steps=2,
        checkpoint_manager=resumed_manager,
        resume_step=resumed_manager.latest_step(),
        checkpoint_metadata=metadata,
        metric_reduction="token_weighted",
        enable_msg_probe=False,
    )

    assert resumed_metrics["run/final_global_step"] == 2.0
    assert resumed_datamodule.loader_calls == [(1, 0)]
    assert resumed_manager.latest_step() == 2
    resumed_manager.close()


def test_train_routes_ar_spectra_jax_backend(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_train(config, workdir):
        calls.append((config, workdir))
        return {"run/device_backend": "jax", "run/training_task": "ar_spectra"}

    import spectra_learning.training.ar_spectra_jax as ar_spectra_jax

    monkeypatch.setattr(
        ar_spectra_jax,
        "train_and_evaluate_ar_spectra_jax",
        fake_train,
    )
    cfg = {
        "training_task": "ar_spectra",
        "device_backend": "jax",
    }

    assert train._train(cfg, tmp_path)["run/training_task"] == "ar_spectra"
    assert calls == [(cfg, tmp_path)]


def test_train_rejects_non_jax_ar_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires device_backend='jax'"):
        train._train(
            {
                "training_task": "ar_spectra",
                "device_backend": "torch",
            },
            tmp_path,
        )


def test_ar_jax_entrypoint_uses_canonical_jax_task_lifecycle(
    monkeypatch,
    tmp_path,
) -> None:
    from spectra_learning.training import ar_spectra_jax as ar_training

    captured = {}

    def fake_train(config, workdir, *, task):
        captured.update(config=config, workdir=workdir, task=task)
        return {"run/training_task": task.name}

    monkeypatch.setattr(ar_training, "train_and_evaluate_jax_task", fake_train)
    cfg = config_dict.ConfigDict()

    results = ar_training.train_and_evaluate_ar_spectra_jax(cfg, tmp_path)

    task = captured["task"]
    assert results == {"run/training_task": "ar_spectra"}
    assert captured["workdir"] == tmp_path
    assert task.metric_reduction == "token_weighted"
    assert task.enable_msg_probe is False
    assert task.checkpoint_contract is not None
    assert task.run_metadata is not None


def test_ar_jax_task_rejects_mae_msg_probe_configuration() -> None:
    task = JaxTrainingTask(
        name="ar_spectra",
        build_datamodule=lambda config, process_count, process_index: None,
        build_model=lambda config, datamodule: None,
        checkpoint_contract=lambda config, datamodule, total_steps: {},
    )
    cfg = config_dict.ConfigDict(
        {
            "msg_probe_every_n_steps": 0,
            "msg_probe_at_final_step": False,
        }
    )

    with pytest.raises(ValueError, match="does not support the MSG probe"):
        _validate_jax_task_probe_config(cfg, task)


def test_ar_jax_checkpoint_contract_tracks_tokenizer_semantics() -> None:
    from configs.ar_spectra_coarse_to_fine import get_config
    from spectra_learning.training.ar_spectra_jax import _ar_jax_checkpoint_contract

    cfg = get_config()
    cfg.config_path = "configs/ar_spectra_coarse_to_fine.py"

    def datamodule(tokenizer_config):
        return SimpleNamespace(
            ar_tokenizer_config=tokenizer_config,
            info={"source": "unit-test"},
            num_peaks_output=128,
            max_precursor_mz=1000.0,
            min_peak_intensity=1e-4,
            peak_drop_min_intensity=1e-4,
            peak_filtering="grouped",
            grouped_peak_shoulder_da=0.02,
            grouped_peak_isotope_charges=(1, 2, 3),
            peak_ordering="mz",
            precursor_peak_exclusion_window_da=0.0,
            global_batch_size=2048,
            gradient_accumulation_steps=1,
            drop_remainder=True,
            train_steps=10_000,
        )

    baseline = _ar_jax_checkpoint_contract(
        cfg,
        datamodule(SpectraARTokenizerConfig()),
        250_000,
    )
    changed = _ar_jax_checkpoint_contract(
        cfg,
        datamodule(SpectraARTokenizerConfig(mz_bin_widths=(10.0, 1.0))),
        250_000,
    )

    assert baseline["tokenizer"]["mz_bin_widths"] == [50.0, 25.0, 5.0, 1.0]
    assert changed["tokenizer"]["mz_bin_widths"] == [10.0, 1.0]
    assert baseline["config"]["dataloader_output_format"] == "numpy"
    assert "config_path" not in baseline["config"]
    assert baseline["training"] == {
        "seed": 0,
        "num_epochs": 6.0,
        "global_batch_size": 2048,
        "gradient_accumulation_steps": 1,
        "drop_remainder": True,
        "train_steps_per_epoch": 10_000,
        "total_steps": 250_000,
    }
    assert baseline != changed


def test_ar_jax_config_has_only_effective_checkpoint_and_model_fields() -> None:
    from configs.ar_spectra_coarse_to_fine import get_config

    cfg = get_config()

    assert "checkpoint_every_n_steps" not in cfg
    assert "ar_dropout" not in cfg
    assert cfg.checkpoint_every_steps == 1000
    assert cfg.dataloader_pin_memory is False
    assert cfg.dataloader_persistent_workers is False
    assert cfg.dataloader_multiprocessing_context == "forkserver"
    assert cfg.dataloader_output_format == "numpy"
