import torch

from spectra_learning.config import load_config
from spectra_learning.models.factory import build_model_from_config


def test_mz_token_ablation_uses_fixed_two_gpu_adam_protocol() -> None:
    config = load_config("configs/mz_token_ablation.py")

    assert config.device_backend == "torch"
    assert config.optimizer == "adam"
    assert config.weight_decay == 0.0
    assert config.batch_size == 512
    assert config.gradient_accumulation_steps == 2
    assert config.training_max_steps == 2_500
    assert config.warmup_steps == 250
    assert config.model_dim == 448
    assert config.encoder_num_layers == 10
    assert config.encoder_num_heads == 7
    assert config.val_every_n_steps == 500
    assert config.val_num_steps == 64
    assert config.encoder_mz_embedding == "fourier"
    assert config.encoder_mz_token_bin_size == 0.02
    assert config.encoder_mz_token_embedding_dim == 38
    assert config.jepa_mae_mz_bin_size == 0.5
    assert config.compile_mode == "default"
    assert config.compile_scope == "blocks"
    assert config.dataloader_num_workers == 8
    assert config.device_prefetch_size == 4
    assert config.log_every_n_steps == 25
    assert config.disable_progress_bar
    assert not config.msg_probe_at_final_step


def test_fourier_and_token_ablation_models_are_parameter_matched() -> None:
    fourier_config = load_config("configs/mz_token_ablation.py")
    token_config = load_config(
        "configs/mz_token_ablation.py",
        {"encoder_mz_embedding": "token"},
    )

    torch.manual_seed(66)
    fourier = build_model_from_config(fourier_config)
    torch.manual_seed(66)
    token = build_model_from_config(token_config)

    fourier_count = sum(
        parameter.numel() for parameter in fourier.parameters() if parameter.requires_grad
    )
    token_count = sum(
        parameter.numel() for parameter in token.parameters() if parameter.requires_grad
    )
    assert fourier_count == 36_808_794
    assert token_count == 36_793_594
    assert fourier_count - token_count == 15_200

    token_parameters = dict(token.named_parameters())
    for name, parameter in fourier.named_parameters():
        if name.startswith(("encoder.embedder.mz_features.", "encoder.embedder.mz_ffn.")):
            continue
        torch.testing.assert_close(parameter, token_parameters[name])

    assert fourier.jepa_mae_num_mz_bins == token.jepa_mae_num_mz_bins == 2_000
    torch.testing.assert_close(
        fourier.jepa_mae_mz_head.weight,
        token.jepa_mae_mz_head.weight,
    )
