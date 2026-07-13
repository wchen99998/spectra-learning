from __future__ import annotations

import torch

from spectra_learning.data.ar_spectra import SpectraARTokenizer, SpectraARTokenizerConfig
from spectra_learning.data.spectra import DEFAULT_MAX_PRECURSOR_MZ, PEAK_MZ_MAX
from spectra_learning.models.ar_spectra import (
    SpectraARTransformer,
    SpectraARTransformerConfig,
)
from spectra_learning.models.lora import LoRALinear
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
