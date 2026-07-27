from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from spectra_learning.data.ar_spectra import SpectraARTokenizer, SpectraARTokenizerConfig
from spectra_learning.data.spectra import DEFAULT_MAX_PRECURSOR_MZ, PEAK_MZ_MAX
from spectra_learning.models.ar_spectra import (
    SpectraARTransformer,
    SpectraARTransformerConfig,
)
from spectra_learning.models.lora import LoRALinear, lora_state_dict
from spectra_learning.probes.massspec import ar_fluorine
from spectra_learning.probes.massspec.ar_fluorine import (
    FluorineLabelTokenHead,
    SpectraARFluorineModule,
    build_arg_parser,
    load_ar_checkpoint_model,
    _input_dim,
    _lora_config,
    apply_ar_fluorine_lora,
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
        "label": torch.tensor([1.0], dtype=torch.float32),
        "row_idx": torch.tensor([0], dtype=torch.long),
    }


def _model_and_tokenizer() -> tuple[SpectraARTransformer, SpectraARTokenizer]:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    model = SpectraARTransformer(
        SpectraARTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            num_token_kinds=tokenizer.num_token_kinds,
            max_sequence_length=tokenizer.sequence_length,
            pad_token_id=tokenizer.pad_token_id,
            model_dim=32,
            num_layers=2,
            num_heads=4,
            mlp_multiple=2.0,
            dropout=0.0,
        )
    )
    return model, tokenizer


def _training_batch(row_start: int) -> dict[str, torch.Tensor]:
    batch = {
        key: torch.cat([value, value.clone()], dim=0)
        for key, value in _batch().items()
    }
    batch["peak_mz"][1] = (
        torch.tensor([250.0, 625.0, 875.0, 0.0], dtype=torch.float32)
        / PEAK_MZ_MAX
    )
    batch["peak_intensity"][1] = torch.tensor(
        [0.80, 0.40, 0.20, 0.0],
        dtype=torch.float32,
    )
    batch["precursor_mz"][1] = 750.0 / DEFAULT_MAX_PRECURSOR_MZ
    batch["collision_energy"][1] = 0.65
    batch["charge"][1] = 1.0
    batch["label"] = torch.tensor([0.0, 1.0], dtype=torch.float32)
    batch["row_idx"] = torch.tensor(
        [row_start, row_start + 1],
        dtype=torch.long,
    )
    return batch


def _training_kwargs(
    *,
    mode: str,
    tmp_path: Path,
    model: SpectraARTransformer,
    tokenizer: SpectraARTokenizer,
    epochs: int = 1,
    patience: int = 1,
    progress_output_prefix: Path | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "state_path": tmp_path / f"ar_{mode}_state.pt",
        "model": model,
        "tokenizer": tokenizer,
        "config": {},
        "config_path": tmp_path / "config.py",
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "cache_dir": tmp_path / "cache",
        "device": torch.device("cpu"),
        "batch_size": 2,
        "num_workers": 0,
        "seed": 7,
        "epochs": epochs,
        "patience": patience,
        "model_learning_rate": 1e-2,
        "lora_rank": 2,
        "lora_alpha": 4.0,
        "lora_dropout": 0.0,
        "lora_learning_rate": 1e-2,
        "head_learning_rate": 1e-2,
        "weight_decay": 0.0,
        "autocast_dtype": None,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
        "select_metric": "average_precision",
        "progress_output_prefix": progress_output_prefix,
        "eval_test_every_epoch": True,
    }


def _assert_tensor_states_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value)


_AR_STATE_COMMON_KEYS = {
    "mode",
    "complete",
    "config_path",
    "checkpoint_path",
    "input_dim",
    "model_dim",
    "pooling",
    "pair_dim",
    "classifier_state",
    "best_epoch",
    "best_val",
    "test",
    "history",
    "hparams",
    "autocast_dtype",
    "focal_alpha",
    "focal_gamma",
    "finetune_cache_dir",
    "device_ids",
    "train_size",
    "train_positive",
    "val_size",
    "val_positive",
    "max_train_samples",
    "max_val_samples",
    "tokenizer_config",
}


def _training_data_and_loaders() -> tuple[
    Any,
    dict[str, list[dict[str, torch.Tensor]]],
]:
    data = SimpleNamespace(
        metadata={
            "train_size": 2,
            "train_positive": 1,
            "val_size": 2,
            "val_positive": 1,
        }
    )
    loaders = {
        "train": [_training_batch(100)],
        "val": [_training_batch(200)],
        "test": [_training_batch(300)],
    }
    return data, loaders


def _assert_ar_training_state(
    *,
    mode: str,
    model: SpectraARTransformer,
    state: dict[str, Any],
    saved_state: dict[str, Any],
    best_state: dict[str, Any],
) -> None:
    mode_keys = (
        {"lora_config", "lora_state"}
        if mode == "lora"
        else {"model_state"}
    )
    assert set(state) == _AR_STATE_COMMON_KEYS | mode_keys
    assert saved_state.keys() == state.keys()
    assert best_state.keys() == state.keys()
    assert state["mode"] == f"ar_{mode}"
    assert state["complete"] is True
    assert saved_state["complete"] is True
    assert best_state["complete"] is False
    assert best_state["test"] is None
    assert state["best_epoch"] == 1
    assert state["best_val"] == state["history"][0]["val"]
    assert len(state["history"]) == 1
    assert "test/average_precision" in state["history"][0]["test"]
    assert "test/average_precision" in state["test"]

    snapshot_key = "lora_state" if mode == "lora" else "model_state"
    _assert_tensor_states_equal(saved_state[snapshot_key], state[snapshot_key])
    _assert_tensor_states_equal(best_state[snapshot_key], state[snapshot_key])
    _assert_tensor_states_equal(
        saved_state["classifier_state"],
        state["classifier_state"],
    )
    _assert_tensor_states_equal(
        best_state["classifier_state"],
        state["classifier_state"],
    )
    actual_model_state = (
        lora_state_dict(model)
        if mode == "lora"
        else dict(model.state_dict())
    )
    _assert_tensor_states_equal(actual_model_state, state[snapshot_key])


_EXPECTED_TIE_EVENTS = [
    ("predict", "val"),
    ("metric", "val"),
    ("predict", "test"),
    ("metric", "test"),
    ("progress", 1),
    ("save", "ar_full_state.best.pt", False, 1),
    ("log", 1),
    ("predict", "val"),
    ("metric", "val"),
    ("predict", "test"),
    ("metric", "test"),
    ("progress", 2),
    ("log", 2),
    ("predict", "test"),
    ("metric", "test"),
    ("save", "ar_full_state.pt", True, 2),
]


class _TieTrainingRecorder:
    def __init__(
        self,
        loaders: dict[str, list[dict[str, torch.Tensor]]],
    ) -> None:
        self.loader_names = {id(loader): name for name, loader in loaders.items()}
        self.events: list[tuple[Any, ...]] = []
        self.prediction_states: list[dict[str, torch.Tensor]] = []
        self.prediction_modes: list[bool] = []
        self.original_predict = ar_fluorine.predict_ar_fluorine
        self.original_save = torch.save

    def predict(
        self,
        *,
        module: SpectraARFluorineModule,
        loader: Any,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
    ) -> tuple[Any, Any, Any]:
        self.events.append(("predict", self.loader_names[id(loader)]))
        self.prediction_states.append(
            {
                key: value.detach().cpu().clone()
                for key, value in module.model.state_dict().items()
            }
        )
        self.prediction_modes.append(module.training)
        return self.original_predict(
            module=module,
            loader=loader,
            device=device,
            autocast_dtype=autocast_dtype,
        )

    def metrics(
        self,
        _targets: Any,
        _logits: Any,
        prefix: str,
    ) -> dict[str, float]:
        self.events.append(("metric", prefix))
        return {
            f"{prefix}/roc_auc": 0.5,
            f"{prefix}/average_precision": 0.5,
            f"{prefix}/accuracy": 0.5,
            f"{prefix}/balanced_accuracy": 0.5,
            f"{prefix}/f1": 0.5,
            f"{prefix}/precision": 0.5,
            f"{prefix}/recall": 0.5,
            f"{prefix}/positive_rate": 0.5,
        }

    def progress(
        self,
        *,
        output_prefix: Any,
        history: list[dict[str, Any]],
    ) -> dict[str, None]:
        del output_prefix
        self.events.append(("progress", len(history)))
        return {"history": None, "training_curve_plot": None}

    def save(
        self,
        obj: dict[str, Any],
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.events.append(
            ("save", path.name, obj["complete"], len(obj["history"]))
        )
        self.original_save(obj, path, *args, **kwargs)

    def log(self, _message: str, *args: Any) -> None:
        self.events.append(("log", args[1]))


def _assert_tie_prediction_states(
    recorder: _TieTrainingRecorder,
    model: SpectraARTransformer,
) -> None:
    assert recorder.prediction_modes == [True, False, True, False, False]
    assert any(
        not torch.equal(
            recorder.prediction_states[0][key],
            recorder.prediction_states[2][key],
        )
        for key in recorder.prediction_states[0]
    )
    _assert_tensor_states_equal(
        recorder.prediction_states[0],
        recorder.prediction_states[1],
    )
    _assert_tensor_states_equal(
        recorder.prediction_states[2],
        recorder.prediction_states[3],
    )
    _assert_tensor_states_equal(
        recorder.prediction_states[0],
        recorder.prediction_states[4],
    )
    _assert_tensor_states_equal(
        dict(model.state_dict()),
        recorder.prediction_states[0],
    )


def test_ar_model_exposes_final_hidden_states() -> None:
    model, tokenizer = _model_and_tokenizer()
    tokenized = tokenizer.tokenize_batch(_batch())

    hidden = model.encode_batch(tokenized)

    assert hidden.shape == (1, tokenizer.sequence_length - 1, 32)


def test_ar_fluorine_lora_targets_attention_and_ffn_linears() -> None:
    model, _ = _model_and_tokenizer()
    model.requires_grad_(False)

    applied = apply_ar_fluorine_lora(
        model,
        _lora_config(rank=2, alpha=4.0, dropout=0.0),
    )

    assert "blocks.0.attention.qkv" in applied
    assert "blocks.0.attention.out_proj" in applied
    assert "blocks.0.ffn.0" in applied
    assert "blocks.0.ffn.3" in applied
    assert isinstance(model.blocks[0].attention.qkv, LoRALinear)
    assert isinstance(model.blocks[0].attention.out_proj, LoRALinear)
    assert isinstance(model.blocks[0].ffn[0], LoRALinear)
    assert isinstance(model.blocks[0].ffn[3], LoRALinear)
    assert not isinstance(model.lm_head, LoRALinear)


def test_ar_fluorine_head_forward_returns_one_logit_per_spectrum() -> None:
    model, tokenizer = _model_and_tokenizer()
    classifier = FluorineLabelTokenHead(input_dim=_input_dim(model))
    module = SpectraARFluorineModule(
        model=model,
        tokenizer=tokenizer,
        classifier=classifier,
    )

    logits = module(_batch())

    assert logits.shape == (1,)


def test_ar_fluorine_eos_pooling_uses_full_sequence_eos_state() -> None:
    model, tokenizer = _model_and_tokenizer()
    classifier = FluorineLabelTokenHead(input_dim=_input_dim(model))
    module = SpectraARFluorineModule(
        model=model,
        tokenizer=tokenizer,
        classifier=classifier,
    )
    tokenized = tokenizer.tokenize_batch(_batch())
    full_ids, full_kinds = module._full_sequence(tokenized)

    hidden = model.hidden_states(full_ids, full_kinds)
    features = module.features(_batch())
    eos_index = full_ids[0].eq(tokenizer.eos_token_id).to(dtype=torch.long).argmax()

    assert torch.allclose(features[0], hidden[0, eos_index].float())


def test_ar_fluorine_cli_accepts_full_finetune_mode() -> None:
    args = build_arg_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--mode",
            "full",
            "--model-learning-rate",
            "1e-5",
        ]
    )

    assert args.mode == "full"
    assert args.model_learning_rate == 1e-5


def test_ar_fluorine_loads_model_shape_from_checkpoint_config(tmp_path) -> None:
    model, tokenizer = _model_and_tokenizer()
    checkpoint_path = tmp_path / "step.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "global_step": 1,
            "config": {
                "num_peaks": 4,
                "ar_model_dim": 32,
                "ar_num_layers": 2,
                "ar_num_heads": 4,
                "ar_mlp_multiple": 2.0,
                "ar_dropout": 0.0,
            },
            "tokenizer_config": tokenizer.config.__dict__,
        },
        checkpoint_path,
    )
    config_path = tmp_path / "stale_config.py"
    config_path.write_text("ar_model_dim = 64\n")

    _config, loaded_tokenizer, loaded_model, _checkpoint = load_ar_checkpoint_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert loaded_model.config.model_dim == 32
    assert loaded_tokenizer.sequence_length == tokenizer.sequence_length


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_train_ar_fluorine_completes_one_epoch_and_restores_best_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    torch.manual_seed(0)
    model, tokenizer = _model_and_tokenizer()
    data, loaders = _training_data_and_loaders()
    loader_calls: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(
        ar_fluorine,
        "build_fluorine_data",
        lambda **_kwargs: data,
    )

    def make_loader(
        _data: Any,
        split: str,
        **kwargs: Any,
    ) -> list[dict[str, torch.Tensor]]:
        loader_calls.append((split, kwargs))
        return loaders[split]

    monkeypatch.setattr(ar_fluorine, "_make_loader", make_loader)
    kwargs = _training_kwargs(
        mode=mode,
        tmp_path=tmp_path,
        model=model,
        tokenizer=tokenizer,
    )

    state, returned_data, targets, logits, row_indices = (
        ar_fluorine.train_ar_fluorine(**kwargs)
    )

    assert returned_data is data
    assert [
        (split, call["shuffle"], call["seed"])
        for split, call in loader_calls
    ] == [
        ("train", True, 7),
        ("val", False, 10_007),
        ("test", False, 20_007),
    ]
    assert targets.tolist() == [0.0, 1.0]
    assert logits.shape == (2,)
    assert row_indices.tolist() == [300, 301]

    state_path = kwargs["state_path"]
    assert isinstance(state_path, Path)
    saved_state = torch.load(state_path, weights_only=False)
    best_state = torch.load(
        state_path.with_name(f"{state_path.stem}.best.pt"),
        weights_only=False,
    )
    _assert_ar_training_state(
        mode=mode,
        model=model,
        state=state,
        saved_state=saved_state,
        best_state=best_state,
    )
    assert model.training is False


def test_train_ar_fluorine_strict_tie_stops_and_restores_first_epoch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    model, tokenizer = _model_and_tokenizer()
    data, loaders = _training_data_and_loaders()
    recorder = _TieTrainingRecorder(loaders)

    monkeypatch.setattr(
        ar_fluorine,
        "build_fluorine_data",
        lambda **_kwargs: data,
    )
    monkeypatch.setattr(
        ar_fluorine,
        "_make_loader",
        lambda _data, split, **_kwargs: loaders[split],
    )
    monkeypatch.setattr(ar_fluorine, "predict_ar_fluorine", recorder.predict)
    monkeypatch.setattr(ar_fluorine, "_metric_dict", recorder.metrics)
    monkeypatch.setattr(
        ar_fluorine,
        "write_training_history_outputs",
        recorder.progress,
    )
    monkeypatch.setattr(torch, "save", recorder.save)
    monkeypatch.setattr(ar_fluorine.log, "info", recorder.log)
    kwargs = _training_kwargs(
        mode="full",
        tmp_path=tmp_path,
        model=model,
        tokenizer=tokenizer,
        epochs=5,
        patience=1,
        progress_output_prefix=tmp_path / "progress",
    )
    state_path = kwargs["state_path"]
    assert isinstance(state_path, Path)

    state, _data, _targets, _logits, row_indices = (
        ar_fluorine.train_ar_fluorine(**kwargs)
    )

    assert recorder.events == _EXPECTED_TIE_EVENTS
    assert len(state["history"]) == 2
    assert state["best_epoch"] == 1
    assert row_indices.tolist() == [300, 301]
    _assert_tie_prediction_states(recorder, model)

    best_state = torch.load(
        state_path.with_name(f"{state_path.stem}.best.pt"),
        weights_only=False,
    )
    _assert_tensor_states_equal(
        best_state["model_state"],
        recorder.prediction_states[0],
    )
    _assert_tensor_states_equal(
        state["model_state"],
        recorder.prediction_states[0],
    )
