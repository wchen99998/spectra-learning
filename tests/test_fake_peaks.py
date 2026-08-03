import torch

from configs.fake_peak_discriminator_50m import get_config
from scripts.export_jax_generator import torch_key
from spectra_learning.models.fake_peaks import FakePeakDiscriminator
from spectra_learning.training.fake_peaks import (
    build_fake_peak_batch,
    sample_fake_mz,
)
from spectra_learning.training.routing import resolve_training_route


def test_sample_fake_mz_replaces_every_target_without_true_residual() -> None:
    true_mz = torch.tensor([[0.1234, 0.4321]])
    target = torch.tensor([[True, True]])
    logits = torch.full((1, 2, 2000), -100.0)
    true_bins = torch.floor(true_mz * 2000).long()
    logits[0, 0, true_bins[0, 0]] = 100.0
    logits[0, 1, true_bins[0, 1] + 10] = 100.0

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        generated, exact = sample_fake_mz(
            logits,
            true_mz,
            target,
            bin_size=0.5,
            top_k=1,
            temperature=1.0,
        )

    assert exact.tolist() == [[True, False]]
    assert not torch.equal(generated, true_mz)
    assert not torch.equal(
        torch.frac(generated * 2000),
        torch.frac(true_mz * 2000),
    )


def test_fake_batch_uses_donor_intensity_and_preserves_true_labels() -> None:
    peak_mz = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    peak_intensity = torch.tensor([[1.0, 0.8, 0.6], [1.0, 0.7, 0.5]])
    target = torch.tensor([[[False, True, False]], [[False, True, False]]])
    batch = {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": torch.ones_like(peak_mz, dtype=torch.bool),
        "context_mask": torch.tensor(
            [[True, False, True], [True, False, True]]
        ),
        "target_masks": target,
    }
    generated_mz = peak_mz + target[:, 0] * 0.01
    generated = build_fake_peak_batch(
        batch,
        generated_mz,
        torch.zeros_like(peak_mz, dtype=torch.bool),
    )

    assert generated["fake_peak_mask"].sum() == 2
    assert torch.equal(
        generated["peak_intensity"].sort(dim=1, descending=True).values,
        generated["peak_intensity"],
    )
    assert torch.equal(
        generated["true_peak_mz"][generated["fake_peak_mask"]],
        torch.tensor([0.2, 0.5]),
    )
    assert torch.equal(
        generated["true_peak_intensity"][generated["fake_peak_mask"]],
        torch.tensor([0.8, 0.7]),
    )
    assert torch.equal(
        generated["peak_intensity"][generated["fake_peak_mask"]],
        torch.tensor([0.7, 0.8]),
    )
    single_batch = {key: value[:1] for key, value in batch.items()}
    single_generated = build_fake_peak_batch(
        single_batch,
        generated_mz[:1],
        torch.zeros_like(peak_mz[:1], dtype=torch.bool),
    )
    assert not torch.equal(
        single_generated["peak_intensity"][single_generated["fake_peak_mask"]],
        single_generated["true_peak_intensity"][
            single_generated["fake_peak_mask"]
        ],
    )


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
        "fake_peak_mask": fake,
        "true_peak_mz": torch.rand(2, 4),
        "true_peak_intensity": torch.rand(2, 4),
        "target_masks": fake.unsqueeze(1),
        "generator_exact_bin_mask": torch.zeros(2, 4, dtype=torch.bool),
        "precursor_mz": torch.rand(2),
    }

    predictions = model.predict(batch)
    metrics = model(batch)
    metrics["loss"].backward()

    assert predictions["fake_logits"].shape == (2, 4)
    assert predictions["mz_logits"].shape == (2, 4, 2000)
    assert predictions["intensity_logits"].shape == (2, 4, 10)
    assert torch.isfinite(metrics["loss"])
    assert model.fake_head.weight.grad is not None
    assert model.mz_head.weight.grad is not None
    assert model.intensity_head.weight.grad is not None


def test_jax_checkpoint_paths_map_to_torch_state_dict_paths() -> None:
    assert (
        torch_key(("encoder", "embedder", "mz_ffn", "layers", "1", "weight"))
        == "encoder.embedder.mz_ffn.2.weight"
    )
    assert (
        torch_key(("encoder", "pair_embedder", "raw_proj", "1", "bias"))
        == "encoder.pair_embedder.raw_proj.2.bias"
    )


def test_fake_peak_training_routes_to_torch() -> None:
    assert resolve_training_route(
        {"training_task": "fake_peak", "device_backend": "torch"}
    ) == ("fake_peak", "torch")
