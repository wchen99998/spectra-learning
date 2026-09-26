from pathlib import Path

import numpy as np
import pytest
import torch
from ml_collections import config_dict

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.datamodule import GemsDataModule
from spectra_learning.data.gems.settings import GemsDataConfig
from spectra_learning.data.gems.test_massive_v2 import _config, _write_artifact
from spectra_learning.models.fastmixer_capacity import (
    pairmixer_fast_stage_capacities,
    predictor_target_max_tokens,
)


def _collator(**overrides: object) -> GemsBatchCollator:
    kwargs = {
        "augment": True,
        "num_target_blocks": 1,
        "context_fraction": 0.9,
        "target_fraction": 0.1,
        "block_min_len": 1,
        "num_peaks": 4,
        "max_precursor_mz": 1000.0,
        "min_peak_intensity": 1e-4,
        "peak_drop_min_intensity": 1e-4,
        "peak_ordering": "mz",
        "precursor_peak_exclusion_window_da": 0.0,
        "mae_objective": "absorbing_diffusion",
    }
    kwargs.update(overrides)
    return GemsBatchCollator(**kwargs)


def _samples(counts: list[int]) -> dict[str, np.ndarray]:
    spectra = np.zeros((len(counts), 2, 4), dtype=np.float32)
    for row, count in enumerate(counts):
        spectra[row, 0, :count] = np.arange(count) * 10 + 100 + row
        spectra[row, 1, :count] = 1.0
    return {
        "spectra": spectra,
        "precursor_mz_raw": np.full(len(counts), 500, dtype=np.float32),
        "collision_energy": np.full(len(counts), 20, dtype=np.float32),
        "charge": np.ones(len(counts), dtype=np.float32),
        "sample_id": np.arange(len(counts), dtype=np.uint64) + 15,
    }


def test_uniform_counts_and_uniform_subsets_include_full_masking() -> None:
    # A gapped validity mask checks that padded peaks never become targets.
    valid = torch.tensor([[True, False, True, True, False, True]]).repeat(12_000, 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(451)
        context, targets = _collator()._sample_masks({"peak_valid_mask": valid})
    assert targets.shape == (12_000, 1, 6)
    selected = targets[:, 0]
    assert torch.equal(context | selected, valid)
    assert not (context & selected).any()
    counts = selected.sum(dim=1)
    frequencies = torch.bincount(counts, minlength=5)
    assert frequencies[0] == 0
    torch.testing.assert_close(
        frequencies[1:].float(), torch.full((4,), 3000.0), rtol=0.06, atol=0
    )
    assert not context[counts == 4].any()
    for count in (1, 2, 3):
        positions = selected[counts == count][:, valid[0]]
        _, subset_frequencies = torch.unique(positions, dim=0, return_counts=True)
        expected_subsets = 6 if count == 2 else 4
        assert len(subset_frequencies) == expected_subsets
        torch.testing.assert_close(
            subset_frequencies.float(),
            torch.full((expected_subsets,), len(positions) / expected_subsets),
            rtol=0.15,
            atol=0,
        )


@pytest.mark.parametrize("input_format", ["numpy", "torch"])
def test_singletons_are_preserved_and_empty_spectra_replaced(input_format: str) -> None:
    samples = _samples([1, 3, 0])
    if input_format == "torch":
        samples = [
            {
                key: torch.as_tensor(value[row])
                for key, value in samples.items()
                if key != "sample_id"
            }
            for row in range(3)
        ]
    batch = _collator()(samples)
    assert batch["peak_valid_mask"].sum(dim=1).tolist() == [1, 3, 1]
    assert not batch["context_mask"][0].any()
    assert batch["target_masks"][0].sum() == 1
    torch.testing.assert_close(batch["peak_mz"][0], batch["peak_mz"][2])
    assert batch["peak_mz"][0, 0] != batch["peak_mz"][1, 0]
    with pytest.raises(ValueError, match="at least 1 usable peaks"):
        _collator()(_samples([0, 0]))


def test_seeded_masks_are_stable_across_batch_partition_and_mask_schedules() -> None:
    samples = _samples([4] * 20)
    collator = _collator(augmentation_seed=17, augmentation_epoch=3)
    rng_state = torch.get_rng_state()
    expected = collator(samples)
    assert torch.equal(rng_state, torch.get_rng_state())
    for start, end in ((0, 3), (3, 12), (12, 20)):
        actual = collator({key: value[start:end] for key, value in samples.items()})
        for key in ("context_mask", "target_masks"):
            torch.testing.assert_close(actual[key], expected[key][start:end])
    collator.context_fraction = 0.1
    collator.target_fraction = 0.9
    changed_schedule = collator(samples)
    torch.testing.assert_close(changed_schedule["target_masks"], expected["target_masks"])
    collator.augmentation_epoch += 1
    assert not torch.equal(collator(samples)["target_masks"], expected["target_masks"])


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"mae_objective": "unknown"}, "Unknown mae_objective"),
        ({"training_mode": "jepa"}, "requires training_mode='mae'"),
        ({"jepa_num_target_blocks": 2}, "requires jepa_num_target_blocks=1"),
    ],
)
def test_data_config_rejects_invalid_diffusion_settings(overrides: dict, error: str) -> None:
    config = config_dict.ConfigDict(
        {
            "training_mode": "mae",
            "mae_objective": "absorbing_diffusion",
            "jepa_num_target_blocks": 1,
            **overrides,
        }
    )
    with pytest.raises(ValueError, match=error):
        GemsDataConfig.from_config(config)


def test_datamodule_wires_diffusion_and_accepts_single_peak_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.training_mode = "mae"
    config.mae_objective = "absorbing_diffusion"
    config.num_peaks = 1
    config.batch_size = 2
    _write_artifact(
        Path(config.artifact_dir) / "gems" / "unit--massive-v2--unit-revision"
    )
    module = GemsDataModule(config, seed=17)
    try:
        collator = module._collator(augment=True)
        assert collator.mae_objective == "absorbing_diffusion"
        batch = next(iter(module.train_loader_for_epoch(0)))
        assert not batch["context_mask"].any()
        assert batch["target_masks"].all()
    finally:
        for dataset in module._datasets.values():
            dataset.close()


def test_diffusion_capacities_cover_all_counts_at_every_schedule_stage() -> None:
    config = {
        "training_mode": "mae",
        "mae_objective": "absorbing_diffusion",
        "num_peaks": 128,
        "encoder_use_cls_token": True,
        "jepa_mask_strategy": "all",
        "jepa_num_target_blocks": 1,
        "jepa_context_fraction_schedule": (0.1, 0.9),
        "jepa_target_fraction_schedule": (0.9, 0.1),
    }
    assert pairmixer_fast_stage_capacities(config) == (
        (128, 129, 128),
        (128, 129, 128),
    )
    assert predictor_target_max_tokens(config) == 128
