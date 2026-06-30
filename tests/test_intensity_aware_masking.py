import importlib

import torch

import spectra_learning.data.gems as gems
from spectra_learning.data.gems.intensity_aware import (
    AWARE_MIXED_MASK_CONFIG,
    _mixed_reliability_weights,
    sample_intensity_aware_masks_torch,
)


def _toy_spectra() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = torch.ones(2, 16, dtype=torch.bool)
    intensity = torch.tensor(
        [
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04],
            [1.0, 0.7, 0.65, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04],
        ],
        dtype=torch.float32,
    )
    mz_da = torch.arange(16, dtype=torch.float32).unsqueeze(0).repeat(2, 1) * 10.0
    return valid, intensity, mz_da


def _spectrum_metadata(
    collision_energy: float,
    charge: float,
) -> dict[str, torch.Tensor]:
    return {
        "collision_energy": torch.tensor(collision_energy, dtype=torch.float32),
        "charge": torch.tensor(charge, dtype=torch.float32),
    }


def test_mixed_reliability_weights_preserve_requested_epsilon_mix() -> None:
    reliability = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32)
    beta = 0.85
    eps = 0.2

    normalized_reliability = reliability.pow(beta)
    normalized_reliability = normalized_reliability / normalized_reliability.sum()
    expected = (1.0 - eps) * normalized_reliability + eps / reliability.numel()

    assert torch.allclose(
        _mixed_reliability_weights(reliability, beta=beta, eps=eps),
        expected,
    )


def test_intensity_aware_masks_are_disjoint_and_leave_unused_mass() -> None:
    valid, intensity, mz_da = _toy_spectra()

    torch.manual_seed(1)
    context, targets = sample_intensity_aware_masks_torch(
        valid,
        intensity,
        mz_da,
        num_target_blocks=2,
        **AWARE_MIXED_MASK_CONFIG,
    )

    p = intensity / intensity.sum(dim=1, keepdim=True)
    target_union = targets.any(dim=1)
    unused_mass = 1.0 - (p * context).sum(dim=1) - (p * target_union).sum(dim=1)

    assert context.shape == valid.shape
    assert targets.shape == (2, 2, 16)
    assert not (context & ~valid).any()
    assert not (targets & ~valid.unsqueeze(1)).any()
    assert not (targets & context.unsqueeze(1)).any()
    assert (targets.sum(dim=1) <= 1).all()
    assert torch.all(unused_mass >= AWARE_MIXED_MASK_CONFIG["min_unused_mass"])


def test_intensity_aware_masks_handle_sparse_rows() -> None:
    valid = torch.tensor([[True, False, False]], dtype=torch.bool)
    intensity = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    mz_da = torch.tensor([[100.0, 0.0, 0.0]], dtype=torch.float32)

    context, targets = sample_intensity_aware_masks_torch(
        valid,
        intensity,
        mz_da,
        num_target_blocks=2,
        **AWARE_MIXED_MASK_CONFIG,
    )

    assert context.shape == valid.shape
    assert targets.shape == (1, 2, 3)
    assert not (context & ~valid).any()
    assert not (targets & ~valid.unsqueeze(1)).any()
    assert not (targets & context.unsqueeze(1)).any()


def test_intensity_aware_masks_can_use_peak_count_fractions() -> None:
    valid, intensity, mz_da = _toy_spectra()

    torch.manual_seed(2)
    context, targets = sample_intensity_aware_masks_torch(
        valid,
        intensity,
        mz_da,
        num_target_blocks=1,
        context_fraction=0.7,
        target_fraction=0.3,
        tail_target_mix=1.0,
        local_gap_probability=0.0,
    )

    assert torch.equal(context.sum(dim=1), torch.full((2,), 11))
    assert torch.equal(targets.sum(dim=(1, 2)), torch.full((2,), 5))
    assert not (targets & context.unsqueeze(1)).any()


def test_gems_batch_collator_generates_intensity_aware_masks() -> None:
    collator = gems.GemsBatchCollator(
        augment=True,
        num_target_blocks=2,
        context_fraction=0.4,
        target_fraction=0.25,
        block_min_len=1,
        mask_strategy="intensity_aware",
        num_peaks=16,
        max_precursor_mz=1000.0,
        min_peak_intensity=1e-4,
        peak_drop_min_intensity=1e-4,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
        intensity_aware_mask_config=AWARE_MIXED_MASK_CONFIG,
    )
    samples = [
        {
            "spectra": torch.tensor(
                [
                    [100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0],
                    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35],
                ],
                dtype=torch.float32,
            ),
            "precursor_mz_raw": torch.tensor(500.0, dtype=torch.float32),
            **_spectrum_metadata(20.0, 1.0),
        },
        {
            "spectra": torch.tensor(
                [
                    [200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0, 340.0],
                    [1.0, 0.7, 0.65, 0.6, 0.5, 0.45, 0.4, 0.35],
                ],
                dtype=torch.float32,
            ),
            "precursor_mz_raw": torch.tensor(600.0, dtype=torch.float32),
            **_spectrum_metadata(40.0, 2.0),
        },
    ]

    torch.manual_seed(3)
    batch = collator(samples)

    assert batch["context_mask"].shape == (2, 16)
    assert batch["target_masks"].shape == (2, 2, 16)
    assert not (batch["context_mask"] & ~batch["peak_valid_mask"]).any()
    assert not (batch["target_masks"] & ~batch["peak_valid_mask"].unsqueeze(1)).any()
    assert not (batch["target_masks"] & batch["context_mask"].unsqueeze(1)).any()
    assert (batch["target_masks"].sum(dim=1) <= 1).all()


def test_wandb_pa645_config_uses_intensity_aware_mixed_params() -> None:
    module = importlib.import_module("configs.wandb_pa645zxs")
    cfg = module.get_config()

    assert cfg.jepa_mask_strategy == "intensity_aware"
    assert cfg.run_name_suffix == "ema-teacher-intensity-aware-mixed"
    for key in (
        "jepa_block_min_len",
        "jepa_context_fraction",
        "jepa_context_fraction_range",
        "jepa_target_fraction",
        "jepa_target_fraction_range",
        "jepa_mask_lengths",
        "jepa_mask_round_from",
    ):
        assert key not in cfg
    for key, value in module.aware_mixed.items():
        assert getattr(cfg, f"jepa_intensity_aware_{key}") == value
