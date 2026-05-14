import torch
import torch.nn.functional as F

from spectra_learning.models.losses import SIGReg, SlotwiseSIGReg
from spectra_learning.models.peak_features import FourierFeatures, PeakFeatureEmbedder


def _normalize_directions(directions: torch.Tensor) -> torch.Tensor:
    return directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)


def test_raw_fourier_features_are_autocast_sensitive():
    fourier = FourierFeatures(
        strategy="log_spaced",
        x_min=2e-4,
        x_max=1.0,
        num_freqs=32,
    )
    mass_da = torch.linspace(0.001, 1000.0, steps=120).reshape(2, 60, 1)

    expected = fourier(mass_da).float()
    with torch.autocast(device_type=mass_da.device.type, dtype=torch.bfloat16):
        actual = fourier(mass_da)

    error = (actual.float() - expected).abs()
    cosine = F.cosine_similarity(actual.float().flatten(), expected.flatten(), dim=0)
    assert error.mean() > 0.25
    assert cosine < 0.5


def test_peak_feature_embedder_runs_fourier_stem_in_fp32_under_autocast():
    torch.manual_seed(0)
    embedder = PeakFeatureEmbedder(
        model_dim=32,
        hidden_dim=16,
        fourier_strategy="log_spaced",
        fourier_x_min=2e-4,
        fourier_x_max=1.0,
        fourier_num_freqs=32,
    )
    peak_mz = torch.linspace(0.001, 1.0, steps=12).reshape(2, 6)
    peak_intensity = torch.rand(2, 6)

    expected = embedder(peak_mz, peak_intensity)
    with torch.autocast(device_type=peak_mz.device.type, dtype=torch.bfloat16):
        actual = embedder(peak_mz, peak_intensity)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_sigreg_runs_fp32_under_autocast():
    torch.manual_seed(0)
    sigreg = SIGReg(num_slices=16)
    proj = torch.randn(3, 5, 11)
    valid_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 1.0],
        ]
    )
    directions = _normalize_directions(torch.randn(11, sigreg.num_slices))

    expected = sigreg(proj, valid_mask=valid_mask, directions=directions)
    with torch.autocast(device_type=proj.device.type, dtype=torch.bfloat16):
        actual = sigreg(proj, valid_mask=valid_mask, directions=directions)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_slotwise_sigreg_runs_fp32_under_autocast():
    torch.manual_seed(0)
    sigreg = SlotwiseSIGReg(num_slices=16)
    proj = torch.randn(3, 2, 5, 11)
    valid_mask = torch.ones(3, 2, 5)
    valid_mask[:, :, 2] = 0.0
    valid_mask[1, 0, 4] = 0.0
    directions = _normalize_directions(torch.randn(11, sigreg.num_slices))

    expected = sigreg(proj, valid_mask=valid_mask, directions=directions)
    with torch.autocast(device_type=proj.device.type, dtype=torch.bfloat16):
        actual = sigreg(proj, valid_mask=valid_mask, directions=directions)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
