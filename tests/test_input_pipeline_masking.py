import torch

import input_pipeline


def test_sample_block_masks_ragged_blocks_alias_matches_ragged() -> None:
    peak_valid_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    kwargs = dict(
        num_target_blocks=2,
        context_fraction=0.4,
        target_fraction=0.25,
        block_min_len=1,
        mask_lengths=(1, 2, 4),
        mask_round_from=2,
    )

    torch.manual_seed(7)
    context_mask, target_masks = input_pipeline._sample_block_masks_torch(
        peak_valid_mask,
        mask_strategy="ragged",
        **kwargs,
    )
    torch.manual_seed(7)
    alias_context_mask, alias_target_masks = input_pipeline._sample_block_masks_torch(
        peak_valid_mask,
        mask_strategy="ragged_blocks",
        **kwargs,
    )

    assert torch.equal(context_mask, alias_context_mask)
    assert torch.equal(target_masks, alias_target_masks)


def test_sample_block_masks_ragged_stays_within_valid_support() -> None:
    peak_valid_mask = torch.tensor(
        [
            [True, False, True, True, False, True, False, False],
            [True, True, False, True, True, False, True, False],
        ],
        dtype=torch.bool,
    )

    torch.manual_seed(11)
    context_mask, target_masks = input_pipeline._sample_block_masks_torch(
        peak_valid_mask,
        num_target_blocks=2,
        context_fraction=0.4,
        target_fraction=0.25,
        block_min_len=1,
        mask_strategy="ragged",
        mask_lengths=(1, 2, 3),
        mask_round_from=2,
    )

    assert context_mask.shape == peak_valid_mask.shape
    assert target_masks.shape == (2, 2, 8)
    assert not (context_mask & ~peak_valid_mask).any()
    assert not (target_masks & ~peak_valid_mask.unsqueeze(1)).any()
    assert not (target_masks & context_mask.unsqueeze(1)).any()
    assert int(context_mask.sum().item()) > 0
    assert int(target_masks.sum().item()) > 0


def test_gems_batch_collator_generates_ragged_context_and_target_masks() -> None:
    collator = input_pipeline._GemsBatchCollator(
        augment=True,
        num_target_blocks=2,
        context_fraction=0.4,
        target_fraction=0.25,
        block_min_len=1,
        mask_strategy="ragged",
        mask_lengths=(1, 2, 4),
        mask_round_from=2,
        use_precursor_token=False,
        num_peaks=8,
        max_precursor_mz=1000.0,
        min_peak_intensity=1e-4,
        peak_drop_min_intensity=1e-4,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
    )
    samples = [
        {
            "spectra": torch.tensor(
                [
                    [100.0, 120.0, 140.0, 160.0, 180.0, 0.0, 0.0, 0.0],
                    [1.0, 0.9, 0.8, 0.7, 0.6, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "precursor_mz_raw": torch.tensor(500.0, dtype=torch.float32),
        },
        {
            "spectra": torch.tensor(
                [
                    [200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 0.0, 0.0],
                    [1.0, 0.95, 0.85, 0.75, 0.65, 0.55, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "precursor_mz_raw": torch.tensor(600.0, dtype=torch.float32),
        },
    ]

    torch.manual_seed(19)
    batch = collator(samples)

    assert batch["context_mask"].shape == (2, 8)
    assert batch["target_masks"].shape == (2, 2, 8)
    assert not (batch["context_mask"] & ~batch["peak_valid_mask"]).any()
    assert not (
        batch["target_masks"] & ~batch["peak_valid_mask"].unsqueeze(1)
    ).any()
    assert not (
        batch["target_masks"] & batch["context_mask"].unsqueeze(1)
    ).any()
