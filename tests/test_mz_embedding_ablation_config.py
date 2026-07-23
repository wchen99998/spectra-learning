import torch

from spectra_learning.config import load_config
from spectra_learning.models.factory import build_model_from_config


def test_mz_embedding_ablation_uses_fixed_two_gpu_adam_protocol() -> None:
    config = load_config("configs/mz_embedding_ablation.py")

    assert config.device_backend == "torch"
    assert config.optimizer == "adam"
    assert config.weight_decay == 0.0
    assert config.batch_size == 512
    assert config.gradient_accumulation_steps == 4
    assert config.training_max_steps == 1_000
    assert config.warmup_steps == 100
    assert config.val_every_n_steps == 250
    assert config.val_num_steps == 32
    assert config.encoder_mz_embedding == "fourier"
    assert config.encoder_discrete_mz_bin_size == 0.02
    assert config.encoder_discrete_mz_coarse_bin_size == 1.0
    assert config.encoder_discrete_mz_embedding_dim == 70
    assert config.compile_mode == "default"
    assert config.compile_scope == "blocks"
    assert config.dataloader_num_workers == 8
    assert config.device_prefetch_size == 4
    assert config.log_every_n_steps == 25
    assert config.disable_progress_bar
    assert not config.msg_probe_at_final_step


def test_fourier_and_discrete_ablation_models_are_parameter_matched() -> None:
    fourier_config = load_config("configs/mz_embedding_ablation.py")
    discrete_config = load_config(
        "configs/mz_embedding_ablation.py",
        {"encoder_mz_embedding": "discrete"},
    )

    torch.manual_seed(66)
    fourier = build_model_from_config(fourier_config)
    torch.manual_seed(66)
    discrete = build_model_from_config(discrete_config)

    fourier_count = sum(
        parameter.numel() for parameter in fourier.parameters() if parameter.requires_grad
    )
    discrete_count = sum(
        parameter.numel() for parameter in discrete.parameters() if parameter.requires_grad
    )
    assert fourier_count == 128_125_738
    assert discrete_count == 128_125_068
    assert fourier_count - discrete_count == 670

    discrete_parameters = dict(discrete.named_parameters())
    for name, parameter in fourier.named_parameters():
        if name.startswith(("encoder.embedder.mz_features.", "encoder.embedder.mz_ffn.")):
            continue
        torch.testing.assert_close(parameter, discrete_parameters[name])
