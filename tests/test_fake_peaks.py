import pytest
import torch

from configs.adversarial_fake_peaks_10m_50m import get_config
from spectra_learning.models.fake_peaks import FakePeakDiscriminator
from spectra_learning.training.routing import resolve_training_route


def test_adversarial_config_has_no_frozen_generator_fields() -> None:
    config = get_config()

    assert config.training_task == "adversarial_fake_peak"
    assert not {
        "generator_source_checkpoint",
        "generator_checkpoint_path",
        "generator_top_k",
        "generator_temperature",
        "generator_compile_mode",
    } & set(config)


def test_discriminator_detects_and_reconstructs_both_peak_values() -> None:
    config = get_config()
    config.model_dim = 16
    config.encoder_num_layers = 1
    config.encoder_num_heads = 2
    config.feature_mlp_hidden_dim = 32
    config.encoder_fourier_mlp_hidden_dim = 32
    config.encoder_fourier_mlp_num_layers = 2
    config.encoder_fourier_num_freqs = 4
    config.pairmixer_pair_dim = 8
    config.pairmixer_pair_feature_hidden_dim = 16
    config.pairmixer_fourier_num_freqs = 2
    config.predictor_dim = 16
    config.masked_latent_predictor_num_heads = 2
    model = FakePeakDiscriminator(config)
    fake = torch.tensor(
        [[False, True, False, True], [False, True, False, True]]
    )
    batch = {
        "peak_mz": torch.rand(2, 4),
        "peak_intensity": torch.rand(2, 4),
        "peak_valid_mask": torch.ones(2, 4, dtype=torch.bool),
        "student_visible_mask": torch.ones(2, 4, dtype=torch.bool),
        "detection_mask": torch.ones(2, 4, dtype=torch.bool),
        "fake_peak_mask": fake,
        "true_peak_mz": torch.rand(2, 4),
        "true_peak_intensity": torch.rand(2, 4),
        "target_masks": fake.unsqueeze(1),
        "generator_exact_bin_mask": torch.zeros(2, 4, dtype=torch.bool),
        "precursor_mz": torch.rand(2),
    }

    predictions = model.predict(batch)
    without_detection = dict(batch)
    del without_detection["detection_mask"]
    with pytest.raises(KeyError, match="detection_mask"):
        model(without_detection)
    metrics = model(batch)
    metrics["loss"].backward()

    assert predictions["fake_logits"].shape == (2, 4)
    assert predictions["mz_logits"].shape == (2, 4, 2000)
    assert predictions["intensity_logits"].shape == (2, 4, 10)
    assert torch.isfinite(metrics["loss"])
    assert model.fake_head.weight.grad is not None
    assert model.mz_head.weight.grad is not None
    assert model.intensity_head.weight.grad is not None


def test_legacy_fake_peak_training_route_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown training_task: fake_peak"):
        resolve_training_route(
            {"training_task": "fake_peak", "device_backend": "torch"}
        )
