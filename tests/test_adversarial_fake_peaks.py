from contextlib import nullcontext

import torch
from torch import nn

import train
from configs.adversarial_fake_peaks_equal_50m import get_config
from spectra_learning.models.fake_peaks import (
    DynamicPeakGenerator,
    FakePeakDiscriminator,
    logit_uniform_nll,
    sample_logit_uniform_residual,
)
from spectra_learning.training.adversarial_fake_peaks import (
    ADVERSARIAL_FAKE_PEAK_CHECKPOINT_FORMAT_VERSION,
    _clamped_class_targets,
    _evaluate,
    _generator_adversarial_loss,
    _merge_adversarial_gradients,
    _prepare_adversarial_batch,
    _training_contract,
    anchor_base_peak_in_context,
    adversarial_weight_at_step,
    generator_config,
    sample_adversarial_pair_masks,
    shuffle_peak_order,
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
        "peak_mz": torch.linspace(0.02, 0.8, num_peaks).repeat(
            batch_size,
            1,
        ),
        "peak_intensity": torch.linspace(1.0, 0.2, num_peaks).repeat(
            batch_size,
            1,
        ),
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
                        True,
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
    generator_params = sum(
        parameter.numel() for parameter in generator.parameters()
    )
    assert generator_params == 51_494_108
    assert generator.backbone.jepa_mae_mz_head is not None
    assert generator.backbone.jepa_mae_intensity_head is not None
    assert generator.backbone.teacher_encoder is None
    assert torch.count_nonzero(generator.mz_residual_head.weight) == 0
    assert torch.count_nonzero(generator.intensity_residual_head.weight) == 0

    discriminator = FakePeakDiscriminator(config)
    discriminator_params = sum(
        parameter.numel() for parameter in discriminator.parameters()
    )
    assert discriminator_params == 51_405_531
    assert abs(generator_params - discriminator_params) / discriminator_params < 0.01


def test_adversarial_batch_mixes_real_and_fake_targets_per_spectrum() -> None:
    config = _tiny_config()
    generator = DynamicPeakGenerator(generator_config(config))
    discriminator = FakePeakDiscriminator(config)
    mixed, discriminator_batch, _, _ = _prepare_adversarial_batch(
        generator,
        _batch(),
        torch.device("cpu"),
        torch.device("cpu"),
        config,
    )

    target = mixed["detection_mask"]
    fake = mixed["fake_peak_mask"]
    assert torch.equal(
        mixed["peak_mz"][~fake],
        mixed["true_peak_mz"][~fake],
    )
    assert torch.equal(
        mixed["peak_intensity"][~fake],
        mixed["true_peak_intensity"][~fake],
    )
    assert torch.equal(target.sum(dim=1), torch.full((4,), 4))
    assert torch.equal(fake.sum(dim=1), torch.full((4,), 2))
    assert torch.equal((target & ~fake).sum(dim=1), torch.full((4,), 2))
    assert discriminator_batch["detection_mask"].sum() == 16
    assert discriminator_batch["fake_peak_mask"].sum() == 8
    assert discriminator(discriminator_batch)["fake_fraction"] == 0.5


def test_base_peak_is_visible_and_never_generated() -> None:
    anchored = anchor_base_peak_in_context(_batch())

    assert anchored["context_mask"][:, 0].all()
    assert not anchored["target_masks"][:, :, 0].any()


def test_peak_shuffle_preserves_aligned_fields() -> None:
    batch = _batch()
    torch.manual_seed(5)

    shuffled = shuffle_peak_order(batch)
    restore = shuffled["peak_mz"].argsort(dim=1)

    assert not torch.equal(shuffled["peak_mz"], batch["peak_mz"])
    for key in (
        "peak_mz",
        "peak_intensity",
        "peak_valid_mask",
        "context_mask",
    ):
        assert torch.equal(
            torch.gather(shuffled[key], 1, restore),
            batch[key],
        )
    assert torch.equal(
        torch.gather(
            shuffled["target_masks"],
            2,
            restore.unsqueeze(1),
        ),
        batch["target_masks"],
    )


def test_adversarial_pair_masks_balance_each_spectrum() -> None:
    target = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    fake, detection = sample_adversarial_pair_masks(target)

    assert torch.equal(fake.sum(dim=1), torch.tensor([1, 1, 2]))
    assert torch.equal(detection.sum(dim=1), torch.tensor([2, 2, 4]))
    assert torch.equal(
        (detection & ~fake).sum(dim=1),
        torch.tensor([1, 1, 2]),
    )


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
    assert generator.mz_residual_head.weight.grad is not None
    assert generator.intensity_residual_head.weight.grad is not None
    assert generator.mz_residual_head.weight.grad.abs().sum() > 0
    assert generator.intensity_residual_head.weight.grad.abs().sum() > 0
    assert generator.backbone.latent_mask_token.grad is not None
    assert all(parameter.grad is None for parameter in discriminator.parameters())


def test_microbatch_trains_both_models_and_ramps_adversarial_weight() -> None:
    config = _tiny_config()
    config.generator_adversarial_start_step = 4
    config.generator_adversarial_warmup_steps = 6
    generator = DynamicPeakGenerator(generator_config(config))
    discriminator = FakePeakDiscriminator(config)

    generator_parameters = tuple(generator.parameters())
    metrics, adversarial_gradients = train_adversarial_microbatch(
        generator,
        discriminator,
        generator_parameters,
        _batch(),
        torch.device("cpu"),
        torch.device("cpu"),
        config,
        global_step=5,
        accumulation_steps=1,
    )

    assert adversarial_weight_at_step(config, 0) == 0.0
    assert adversarial_weight_at_step(config, 4) == 0.0
    assert adversarial_weight_at_step(config, 7) == 0.125
    assert adversarial_weight_at_step(config, 10) == 0.25
    assert torch.isfinite(metrics["generator/loss"])
    assert torch.isfinite(metrics["discriminator/loss"])
    assert torch.isfinite(metrics["generator/mz_residual_loss"])
    assert torch.isfinite(metrics["generator/intensity_residual_loss"])
    assert torch.isfinite(metrics["generator/mz_residual_mae"])
    assert torch.isfinite(metrics["generator/intensity_residual_mae"])
    assert torch.isfinite(metrics["generator/generated_mz_residual_mean"])
    assert torch.isfinite(metrics["generator/generated_mz_residual_std"])
    assert torch.isfinite(metrics["generator/target_mz_residual_mean"])
    assert torch.isfinite(metrics["generator/target_mz_residual_std"])
    assert torch.isfinite(
        metrics["generator/generated_intensity_residual_mean"]
    )
    assert torch.isfinite(
        metrics["generator/generated_intensity_residual_std"]
    )
    assert torch.isfinite(
        metrics["generator/target_intensity_residual_mean"]
    )
    assert torch.isfinite(
        metrics["generator/target_intensity_residual_std"]
    )
    assert generator.backbone.jepa_mae_mz_head.weight.grad is not None
    assert generator.mz_residual_head.weight.grad is not None
    assert generator.intensity_residual_head.weight.grad is not None
    assert discriminator.fake_head.weight.grad is not None
    assert any(
        gradient is not None and gradient.abs().sum() > 0
        for gradient in adversarial_gradients
    )


def test_adversarial_gradient_norm_is_capped_relative_to_reconstruction() -> None:
    first = nn.Parameter(torch.zeros(2))
    second = nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([3.0, 4.0])
    adversarial = [torch.tensor([30.0, 40.0]), torch.tensor([100.0])]

    raw_norm, scale, realized_ratio = _merge_adversarial_gradients(
        (first, second),
        adversarial,
        max_ratio=0.1,
    )

    assert torch.allclose(raw_norm, torch.tensor(12_500**0.5))
    assert torch.allclose(scale, torch.tensor(0.5 / 12_500**0.5))
    assert torch.allclose(realized_ratio, torch.tensor(0.1))
    assert torch.allclose(
        nn.utils.get_total_norm(
            [first.grad - torch.tensor([3.0, 4.0]), second.grad]
        ),
        torch.tensor(0.5),
    )


def test_generator_preserves_non_target_intensities() -> None:
    config = _tiny_config()
    generator = DynamicPeakGenerator(generator_config(config))

    generated = generator(_batch())
    target = generated["target_mask"]

    assert torch.equal(
        generated["peak_intensity"][~target],
        _batch()["peak_intensity"][~target],
    )
    assert (generated["predicted_mz"][target] >= 0.02).all()


def test_forbidden_mz_bins_do_not_poison_padded_slots() -> None:
    config = _tiny_config()
    generator = DynamicPeakGenerator(generator_config(config))
    batch = _batch()
    batch["peak_mz"][:, -1] = 0.0
    batch["peak_intensity"][:, -1] = 0.0
    batch["peak_valid_mask"][:, -1] = False

    _, _, reconstruction_loss, _ = _prepare_adversarial_batch(
        generator,
        batch,
        torch.device("cpu"),
        torch.device("cpu"),
        config,
    )

    assert torch.isfinite(reconstruction_loss)


def test_generator_preserves_finite_zero_placeholder() -> None:
    config = _tiny_config()
    generator = DynamicPeakGenerator(generator_config(config))
    batch = _batch()
    batch["peak_intensity"].zero_()
    batch["target_masks"].zero_()

    generated = generator(batch)

    assert torch.isfinite(generated["peak_intensity"]).all()
    assert torch.count_nonzero(generated["peak_intensity"]) == 0


def test_clamped_last_intensity_class_keeps_upper_residual() -> None:
    classes, residuals = _clamped_class_targets(
        torch.ones(1),
        value_scale=1.0,
        bin_size=0.1,
        num_classes=10,
    )

    assert classes.item() == 9
    assert residuals.item() == 1.0


def test_zero_shift_samples_uniform_residuals() -> None:
    shift = torch.zeros(65_536)
    torch.manual_seed(123)
    expected = torch.rand_like(shift)
    torch.manual_seed(123)

    residual = sample_logit_uniform_residual(shift)

    assert torch.allclose(residual, expected, atol=1e-6)
    assert abs(float(residual.mean()) - 0.5) < 0.005
    assert abs(float(residual.std()) - 12**-0.5) < 0.005


def test_logit_uniform_nll_is_exact_and_differentiable() -> None:
    shift = torch.zeros(4, requires_grad=True)
    target = torch.tensor([0.0, 0.2, 0.8, 1.0])

    nll = logit_uniform_nll(shift, target)

    assert torch.isfinite(nll).all()
    assert torch.allclose(nll, torch.zeros_like(nll), atol=1e-6)
    logit_uniform_nll(shift, torch.full_like(target, 0.2)).mean().backward()
    assert shift.grad is not None
    assert torch.all(shift.grad > 0)


def test_adversarial_checkpoint_format_is_hard_cut() -> None:
    assert ADVERSARIAL_FAKE_PEAK_CHECKPOINT_FORMAT_VERSION == 5


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
    config.generator_adversarial_grad_max_ratio = 0.2
    config.drop_remainder = False
    config.gems_hdf5_rows_per_block = 4096
    config.dataloader_num_workers = 3
    changed = _training_contract(config, generator_config(config))

    assert contract != changed
    assert changed["optimization"]["optimizer_fused"] is False
    assert changed["optimization"]["generator_adversarial_grad_max_ratio"] == 0.2
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
