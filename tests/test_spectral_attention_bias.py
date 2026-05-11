import tempfile

import pytest
import torch

from spectra_learning.models.model import PeakSetSIGReg
from spectra_learning.models.spectral_attention_bias import (
    HarmonicRelativeLossBias,
    SpectralGraphormerBias,
)
from spectra_learning.training.api import load_pretrained_weights


def _small_model(**overrides) -> PeakSetSIGReg:
    kwargs = dict(
        model_dim=32,
        encoder_num_layers=1,
        encoder_num_heads=4,
        encoder_num_kv_heads=4,
        attention_mlp_multiple=2.0,
        feature_mlp_hidden_dim=16,
        encoder_fourier_num_freqs=8,
        masked_latent_predictor_num_layers=1,
        masked_latent_predictor_num_heads=4,
        jepa_num_target_blocks=1,
        jepa_target_layers=[1],
        num_peaks=6,
    )
    kwargs.update(overrides)
    return PeakSetSIGReg(**kwargs)


def test_harmonic_relative_loss_bias_matches_direct_trig():
    bias = HarmonicRelativeLossBias(
        num_heads=2,
        num_freqs=3,
        init_std=0.0,
    )
    with torch.no_grad():
        bias.freqs.b.copy_(torch.tensor([0.05, 0.10, 0.25]))
        bias.cos_weight.copy_(
            torch.tensor(
                [
                    [0.20, -0.30, 0.50],
                    [-0.10, 0.40, 0.70],
                ]
            )
        )
        bias.sin_weight.copy_(
            torch.tensor(
                [
                    [0.60, 0.10, -0.20],
                    [0.30, -0.50, 0.20],
                ]
            )
        )

    mass_da = torch.tensor([[10.0, 17.5, 25.0]])
    actual = bias(mass_da)

    delta = mass_da.unsqueeze(2) - mass_da.unsqueeze(1)
    angles = 2.0 * torch.pi * delta.unsqueeze(-1) * bias.freqs.b
    scale = 1.0 / torch.sqrt(torch.tensor(3.0))
    expected = (
        torch.cos(angles).unsqueeze(1) * bias.cos_weight.view(1, 2, 1, 1, 3)
        + torch.sin(angles).unsqueeze(1) * bias.sin_weight.view(1, 2, 1, 1, 3)
    ).sum(dim=-1) * scale

    assert torch.allclose(actual, expected, atol=1e-6)


def test_spectral_graphormer_bias_zero_init_pads_special_tokens():
    module = SpectralGraphormerBias(
        num_heads=4,
        relative_kind="harmonic",
        use_precursor_bias=True,
        use_intensity_bias=True,
        num_freqs=8,
        init_std=0.0,
    )
    peak_mz = torch.rand(2, 5)
    peak_intensity = torch.rand(2, 5)
    precursor_mz = torch.rand(2)

    bias = module(
        peak_mz,
        peak_intensity=peak_intensity,
        precursor_mz=precursor_mz,
        num_special_tokens=2,
    )

    assert bias.shape == (2, 4, 7, 7)
    assert int(torch.count_nonzero(bias).item()) == 0


def test_precursor_only_bias_does_not_apply_to_special_query_tokens():
    precursor_only = SpectralGraphormerBias(
        num_heads=2,
        relative_kind="none",
        use_precursor_bias=True,
        num_freqs=4,
        init_std=0.0,
    )
    relative_and_precursor = SpectralGraphormerBias(
        num_heads=2,
        relative_kind="harmonic",
        use_precursor_bias=True,
        num_freqs=4,
        init_std=0.0,
    )
    with torch.no_grad():
        precursor_only.precursor_bias.cos_weight.copy_(
            torch.tensor(
                [
                    [0.20, -0.10, 0.30, 0.40],
                    [-0.50, 0.60, -0.20, 0.10],
                ]
            )
        )
        precursor_only.precursor_bias.sin_weight.copy_(
            torch.tensor(
                [
                    [0.70, 0.20, -0.40, 0.10],
                    [0.30, -0.80, 0.50, 0.20],
                ]
            )
        )
        relative_and_precursor.precursor_bias.cos_weight.copy_(
            precursor_only.precursor_bias.cos_weight
        )
        relative_and_precursor.precursor_bias.sin_weight.copy_(
            precursor_only.precursor_bias.sin_weight
        )

    peak_mz = torch.tensor([[0.10, 0.20, 0.35]])
    precursor_mz = torch.tensor([0.50])

    precursor_bias = precursor_only(
        peak_mz,
        precursor_mz=precursor_mz,
        num_special_tokens=2,
    )
    combined_bias = relative_and_precursor(
        peak_mz,
        precursor_mz=precursor_mz,
        num_special_tokens=2,
    )

    assert precursor_bias.shape == (1, 2, 5, 5)
    assert torch.count_nonzero(precursor_bias[:, :, :3, :3]).item() > 0
    assert int(torch.count_nonzero(precursor_bias[:, :, 3:, :]).item()) == 0
    assert int(torch.count_nonzero(precursor_bias[:, :, :, 3:]).item()) == 0
    assert torch.equal(precursor_bias, combined_bias)


def test_spectral_graphormer_bias_autocast_matches_fp32():
    torch.manual_seed(11)
    module = SpectralGraphormerBias(
        num_heads=4,
        relative_kind="harmonic",
        use_precursor_bias=True,
        use_intensity_bias=True,
        num_freqs=16,
        init_std=0.2,
    )
    peak_mz = torch.rand(2, 6)
    peak_intensity = torch.rand(2, 6)
    precursor_mz = torch.rand(2)

    expected = module(
        peak_mz,
        peak_intensity=peak_intensity,
        precursor_mz=precursor_mz,
        num_special_tokens=2,
    )
    with torch.autocast(device_type=peak_mz.device.type, dtype=torch.bfloat16):
        actual = module(
            peak_mz,
            peak_intensity=peak_intensity,
            precursor_mz=precursor_mz,
            num_special_tokens=2,
        )

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_zero_init_spectral_bias_preserves_encoder_output():
    torch.manual_seed(7)
    base = _small_model()
    biased = _small_model(
        spectral_bias_relative_kind="harmonic",
        spectral_bias_use_precursor=True,
        spectral_bias_use_intensity=True,
        spectral_bias_num_freqs=8,
        spectral_bias_init_std=0.0,
    )
    biased.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    biased.eval()

    peak_mz = torch.rand(2, 6)
    peak_intensity = torch.rand(2, 6)
    precursor_mz = torch.rand(2)

    with torch.no_grad():
        expected = base.encoder(
            peak_mz,
            peak_intensity,
            precursor_mz=precursor_mz,
        )
        actual = biased.encoder(
            peak_mz,
            peak_intensity,
            precursor_mz=precursor_mz,
        )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_load_pretrained_weights_rejects_missing_spectral_bias():
    source = _small_model()
    target = _small_model(
        spectral_bias_relative_kind="harmonic",
        spectral_bias_use_precursor=True,
        spectral_bias_num_freqs=8,
        spectral_bias_init_std=0.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/old.pt"
        torch.save(
            {"state_dict": source.state_dict()},
            path,
        )
        with pytest.raises(RuntimeError, match="Missing key"):
            load_pretrained_weights(target, path)
