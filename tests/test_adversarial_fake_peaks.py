from contextlib import nullcontext

import torch

import train
from configs.adversarial_fake_peaks_10m_50m import get_config
from spectra_learning.models.fake_peaks import (
    DynamicPeakGenerator,
    FakePeakDiscriminator,
)
from spectra_learning.training.adversarial_fake_peaks import (
    _evaluate,
    _generator_adversarial_loss,
    _prepare_adversarial_batch,
    _training_contract,
    adversarial_weight_at_step,
    generator_config,
    train_adversarial_microbatch,
)
from spectra_learning.training.routing import resolve_training_route


def _tiny_config():
    config = get_config()
    config.num_peaks = 8
    model = {
        "model_dim": 32,
        "encoder_num_layers": 1,
        "encoder_num_heads": 4,
        "feature_mlp_hidden_dim": 64,
        "encoder_fourier_mlp_hidden_dim": 64,
        "encoder_fourier_mlp_num_layers": 2,
        "encoder_fourier_num_freqs": 8,
        "pairmixer_pair_dim": 16,
        "pairmixer_pair_feature_hidden_dim": 32,
        "pairmixer_fourier_num_freqs": 4,
        "predictor_dim": 32,
        "masked_latent_predictor_num_layers": 1,
        "masked_latent_predictor_num_heads": 4,
    }
    config.update(model)
    config.generator_model.update(model)
    return config


def _batch() -> dict[str, torch.Tensor]:
    batch_size = 4
    num_peaks = 8
    return {
        "peak_mz": torch.rand(batch_size, num_peaks),
        "peak_intensity": torch.rand(batch_size, num_peaks),
        "peak_valid_mask": torch.ones(
            batch_size,
            num_peaks,
            dtype=torch.bool,
        ),
        "context_mask": torch.tensor(
            [[True, True, True, False, False, False, False, False]]
            * batch_size
        ),
        "target_masks": torch.tensor(
            [
                [
                    [
                        False,
                        False,
                        False,
                        True,
                        True,
                        True,
                        False,
                        False,
                    ]
                ]
            ]
            * batch_size
        ),
        "precursor_mz": torch.rand(batch_size),
    }


def test_joint_models_have_requested_parameter_sizes() -> None:
    config = get_config()
    generator = DynamicPeakGenerator(generator_config(config))
    assert sum(parameter.numel() for parameter in generator.parameters()) == 10_036_954
    assert generator.backbone.jepa_mae_mz_head is not None
    assert generator.backbone.jepa_mae_intensity_head is not None
    assert generator.backbone.teacher_encoder is None

    discriminator = FakePeakDiscriminator(config)
    assert (
        sum(parameter.numel() for parameter in discriminator.parameters())
        == 51_405_531
    )


def test_adversarial_batch_is_balanced_at_target_positions() -> None:
    config = _tiny_config()
    generator = DynamicPeakGenerator(generator_config(config))
    discriminator = FakePeakDiscriminator(config)
    fake, discriminator_batch, _, _ = _prepare_adversarial_batch(
        generator,
        _batch(),
        torch.device("cpu"),
        torch.device("cpu"),
        config,
    )

    target = fake["detection_mask"]
    assert torch.equal(
        fake["peak_mz"][~target],
        fake["true_peak_mz"][~target],
    )
    assert torch.equal(
        fake["peak_intensity"][~target],
        fake["true_peak_intensity"][~target],
    )
    assert discriminator_batch["detection_mask"].sum() == 12
    assert discriminator_batch["fake_peak_mask"].sum() == 6
    assert discriminator(discriminator_batch)["fake_fraction"] == 0.5


def test_detached_discriminator_then_frozen_generator_pass() -> None:
    config = _tiny_config()
    generator = DynamicPeakGenerator(generator_config(config))
    discriminator = FakePeakDiscriminator(config)
    fake, discriminator_batch, reconstruction_loss, _ = (
        _prepare_adversarial_batch(
            generator,
            _batch(),
            torch.device("cpu"),
            torch.device("cpu"),
            config,
        )
    )

    discriminator(discriminator_batch)["loss"].backward()
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert discriminator.fake_head.weight.grad is not None

    discriminator.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    adversarial_loss = _generator_adversarial_loss(discriminator, fake)
    (
        reconstruction_loss
        + config.generator_adversarial_loss_weight * adversarial_loss
    ).backward()
    assert generator.backbone.jepa_mae_mz_head.weight.grad is not None
    assert generator.backbone.jepa_mae_intensity_head.weight.grad is not None
    assert generator.backbone.latent_mask_token.grad is not None
    assert all(parameter.grad is None for parameter in discriminator.parameters())


def test_microbatch_trains_both_models_and_ramps_adversarial_weight() -> None:
    config = _tiny_config()
    config.generator_adversarial_warmup_steps = 10
    generator = DynamicPeakGenerator(generator_config(config))
    discriminator = FakePeakDiscriminator(config)

    metrics = train_adversarial_microbatch(
        generator,
        discriminator,
        _batch(),
        torch.device("cpu"),
        torch.device("cpu"),
        config,
        global_step=5,
        accumulation_steps=1,
    )

    assert adversarial_weight_at_step(config, 0) == 0.0
    assert adversarial_weight_at_step(config, 5) == 1e-4
    assert adversarial_weight_at_step(config, 10) == 2e-4
    assert torch.isfinite(metrics["generator/loss"])
    assert torch.isfinite(metrics["discriminator/loss"])
    assert generator.backbone.jepa_mae_mz_head.weight.grad is not None
    assert discriminator.fake_head.weight.grad is not None


def test_validation_preserves_training_rng() -> None:
    config = _tiny_config()
    generator = DynamicPeakGenerator(generator_config(config))
    discriminator = FakePeakDiscriminator(config)
    batch = _batch()
    torch.manual_seed(123)
    state = torch.get_rng_state()

    _evaluate(
        generator,
        discriminator,
        [batch],
        torch.device("cpu"),
        torch.device("cpu"),
        config,
        global_step=1,
        max_steps=1,
    )

    assert torch.equal(torch.get_rng_state(), state)


def test_validation_forks_both_model_cuda_rngs(monkeypatch) -> None:
    config = _tiny_config()
    generator = DynamicPeakGenerator(generator_config(config))
    discriminator = FakePeakDiscriminator(config)
    forked_devices = []

    def record_fork_rng(*, devices):
        forked_devices.extend(devices)
        return nullcontext()

    monkeypatch.setattr(torch.random, "fork_rng", record_fork_rng)
    _evaluate(
        generator,
        discriminator,
        [],
        torch.device("cuda:1"),
        torch.device("cuda:0"),
        config,
        global_step=1,
        max_steps=0,
    )

    assert forked_devices == [
        torch.device("cuda:1"),
        torch.device("cuda:0"),
    ]


def test_training_contract_tracks_intensity_aware_masking() -> None:
    config = _tiny_config()
    contract = _training_contract(config, generator_config(config))
    config.jepa_intensity_aware_alpha = 0.5
    changed = _training_contract(config, generator_config(config))

    assert contract != changed
    assert (
        contract["masking"]["jepa_intensity_aware_mask_config"]["alpha"]
        == 0.75
    )
    assert (
        changed["masking"]["jepa_intensity_aware_mask_config"]["alpha"]
        == 0.5
    )


def test_training_contract_tracks_optimizer_and_data_stream() -> None:
    config = _tiny_config()
    contract = _training_contract(config, generator_config(config))
    config.optimizer_fused = False
    config.drop_remainder = False
    config.gems_hdf5_rows_per_block = 4096
    config.dataloader_num_workers = 3
    changed = _training_contract(config, generator_config(config))

    assert contract != changed
    assert changed["optimization"]["optimizer_fused"] is False
    assert changed["data_stream"] == {
        "drop_remainder": False,
        "gems_hdf5_rows_per_block": 4096,
        "dataloader_num_workers": 3,
    }


def test_adversarial_fake_peak_training_routes_to_torch() -> None:
    assert resolve_training_route(
        {
            "training_task": "adversarial_fake_peak",
            "device_backend": "torch",
        }
    ) == ("adversarial_fake_peak", "torch")


def test_train_dispatches_adversarial_fake_peaks(monkeypatch, tmp_path) -> None:
    from spectra_learning.training import adversarial_fake_peaks

    calls = []

    def fake_train(config, workdir):
        calls.append((config, workdir))
        return {"run/training_task": "adversarial_fake_peak"}

    monkeypatch.setattr(
        adversarial_fake_peaks,
        "train_adversarial_fake_peaks",
        fake_train,
    )
    config = {
        "training_task": "adversarial_fake_peak",
        "device_backend": "torch",
    }

    assert train._train(config, tmp_path) == {
        "run/training_task": "adversarial_fake_peak"
    }
    assert calls == [(config, tmp_path)]
