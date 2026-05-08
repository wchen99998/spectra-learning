from unittest import mock

import torch

import spectra_learning.data.gems.masking as gems_masking
import spectra_learning.data.gems.visualization as gems_visualization
from spectra_learning.data.gems.collate import GemsBatchCollator
import spectra_learning.data.gems.collate as gems_collate


def test_sample_ragged_block_mask_uses_full_block_length() -> None:
    active_positions = torch.tensor(
        [True, True, True, True, True, False],
        dtype=torch.bool,
    )

    torch.manual_seed(4)
    mask = gems_masking._sample_ragged_block_mask_1d_torch(
        active_positions,
        masked_fraction=3.0 / 5.0,
        lengths=(3,),
        round_from=1,
    )

    assert int(mask.sum().item()) == 3
    assert not (mask & ~active_positions).any()


def test_sample_ragged_block_mask_caps_length_to_active_support() -> None:
    active_positions = torch.tensor(
        [True, False, True, True, False],
        dtype=torch.bool,
    )

    torch.manual_seed(4)
    mask = gems_masking._sample_ragged_block_mask_1d_torch(
        active_positions,
        masked_fraction=1.0,
        lengths=(5,),
        round_from=1,
    )

    assert torch.equal(mask, active_positions)


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
    context_mask, target_masks = gems_masking._sample_block_masks_torch(
        peak_valid_mask,
        mask_strategy="ragged",
        **kwargs,
    )
    torch.manual_seed(7)
    alias_context_mask, alias_target_masks = gems_masking._sample_block_masks_torch(
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
    context_mask, target_masks = gems_masking._sample_block_masks_torch(
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


def test_sample_block_masks_context_and_targets_are_disjoint_for_all_modes() -> None:
    peak_valid_mask = torch.tensor(
        [
            [True, False, True, True, False, True, True, False],
            [False, True, True, False, True, True, False, True],
            [True, True, False, True, False, True, True, False],
        ],
        dtype=torch.bool,
    )
    kwargs = dict(
        num_target_blocks=2,
        context_fraction=0.4,
        target_fraction=0.25,
        block_min_len=1,
        mask_lengths=(1, 2, 3),
        mask_round_from=2,
    )

    for seed, strategy in enumerate(("contiguous", "ragged", "random"), start=31):
        torch.manual_seed(seed)
        context_mask, target_masks = gems_masking._sample_block_masks_torch(
            peak_valid_mask,
            mask_strategy=strategy,
            **kwargs,
        )

        assert not (context_mask & ~peak_valid_mask).any()
        assert not (target_masks & ~peak_valid_mask.unsqueeze(1)).any()
        assert not (target_masks & context_mask.unsqueeze(1)).any()

    torch.manual_seed(37)
    context_mask, target_masks = gems_masking._sample_block_masks_torch(
        peak_valid_mask,
        mask_strategy="all",
        **kwargs,
    )

    assert not (context_mask & ~peak_valid_mask).any()
    assert not (target_masks & ~peak_valid_mask.unsqueeze(1)).any()
    assert not (target_masks & context_mask.unsqueeze(1)).any()


def test_sample_block_masks_random_samples_exact_count_masks() -> None:
    peak_valid_mask = torch.ones((2, 8), dtype=torch.bool)

    torch.manual_seed(23)
    context_mask, target_masks = gems_masking._sample_block_masks_torch(
        peak_valid_mask,
        num_target_blocks=2,
        context_fraction=0.375,
        target_fraction=0.25,
        block_min_len=1,
        mask_strategy="random",
        mask_lengths=(1, 2, 4),
        mask_round_from=2,
    )

    assert torch.equal(context_mask.sum(dim=1), torch.tensor([3, 3]))
    assert torch.equal(target_masks.sum(dim=2), torch.full((2, 2), 2))
    assert not (target_masks & context_mask.unsqueeze(1)).any()


def test_sample_block_masks_all_uses_balanced_random_row_modes() -> None:
    peak_valid_mask = torch.ones((3, 8), dtype=torch.bool)

    with mock.patch.object(
        gems_masking,
        "_sample_all_mask_strategies_torch",
        return_value=["contiguous", "ragged", "random"],
    ) as sample_strategies:
        torch.manual_seed(29)
        context_mask, target_masks = gems_masking._sample_block_masks_torch(
            peak_valid_mask,
            num_target_blocks=2,
            context_fraction=0.375,
            target_fraction=0.25,
            block_min_len=1,
            mask_strategy="all",
            mask_lengths=(1, 2, 4),
            mask_round_from=2,
        )

    assert sample_strategies.call_count == 1
    assert context_mask.shape == peak_valid_mask.shape
    assert target_masks.shape == (3, 2, 8)
    assert not (context_mask & ~peak_valid_mask).any()
    assert not (target_masks & ~peak_valid_mask.unsqueeze(1)).any()
    assert not (target_masks & context_mask.unsqueeze(1)).any()


def test_sample_all_mask_strategies_balances_modes() -> None:
    torch.manual_seed(29)
    strategies = gems_masking._sample_all_mask_strategies_torch(
        6,
        device=torch.device("cpu"),
    )

    counts = {
        strategy: strategies.count(strategy)
        for strategy in gems_masking.JEPA_MASK_STRATEGIES
    }
    assert counts == {
        "contiguous": 2,
        "ragged": 2,
        "random": 2,
    }


def test_resolve_visualization_strategies_includes_supported_modes() -> None:
    assert gems_visualization._resolve_visualization_strategies("ragged") == (
        "contiguous",
        "ragged",
        "random",
    )


def test_resolve_visualization_strategies_keeps_all_meta_mode() -> None:
    assert gems_visualization._resolve_visualization_strategies("all") == (
        "contiguous",
        "ragged",
        "random",
        "all",
    )


def test_gems_batch_collator_generates_ragged_context_and_target_masks() -> None:
    collator = GemsBatchCollator(
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


def test_gems_batch_collator_samples_after_prepending_precursor() -> None:
    collator = GemsBatchCollator(
        augment=True,
        num_target_blocks=2,
        context_fraction=0.4,
        target_fraction=0.25,
        block_min_len=1,
        mask_strategy="ragged",
        use_precursor_token=True,
        num_peaks=4,
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
                    [100.0, 120.0, 140.0, 160.0],
                    [1.0, 0.9, 0.8, 0.7],
                ],
                dtype=torch.float32,
            ),
            "precursor_mz_raw": torch.tensor(500.0, dtype=torch.float32),
        },
        {
            "spectra": torch.tensor(
                [
                    [200.0, 220.0, 240.0, 260.0],
                    [1.0, 0.95, 0.85, 0.75],
                ],
                dtype=torch.float32,
            ),
            "precursor_mz_raw": torch.tensor(600.0, dtype=torch.float32),
        },
    ]
    sampled_context = torch.tensor(
        [
            [False, True, False, False, False],
            [True, False, True, False, False],
        ],
        dtype=torch.bool,
    )
    sampled_targets = torch.tensor(
        [
            [
                [True, False, False, False, False],
                [False, False, False, True, False],
            ],
            [
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
        ],
        dtype=torch.bool,
    )
    captured: dict[str, torch.Tensor] = {}

    def fake_sample_masks(
        peak_valid_mask: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured["peak_valid_mask"] = peak_valid_mask.detach().clone()
        return sampled_context.clone(), sampled_targets.clone()

    with mock.patch.object(
        gems_collate,
        "_sample_block_masks_torch",
        side_effect=fake_sample_masks,
    ):
        batch = collator(samples)

    assert torch.equal(captured["peak_valid_mask"], batch["peak_valid_mask"])
    assert batch["peak_mz"].shape == (2, 5)
    assert batch["peak_valid_mask"][:, 0].all()
    assert not batch["context_mask"][0, 0]
    assert batch["context_mask"][1, 0]
    assert batch["target_masks"][0, 0, 0]
    assert torch.equal(batch["context_mask"], sampled_context)
    assert torch.equal(batch["target_masks"], sampled_targets)


def test_gems_batch_collator_allows_precursor_to_follow_sampled_masks() -> None:
    collator = GemsBatchCollator(
        augment=True,
        num_target_blocks=2,
        context_fraction=0.4,
        target_fraction=0.25,
        block_min_len=1,
        mask_strategy="all",
        mask_lengths=(1, 2, 4),
        mask_round_from=2,
        use_precursor_token=True,
        num_peaks=6,
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
                    [100.0, 120.0, 140.0, 160.0, 180.0, 200.0],
                    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                ],
                dtype=torch.float32,
            ),
            "precursor_mz_raw": torch.tensor(500.0, dtype=torch.float32),
        },
        {
            "spectra": torch.tensor(
                [
                    [200.0, 220.0, 240.0, 260.0, 280.0, 300.0],
                    [1.0, 0.95, 0.85, 0.75, 0.65, 0.55],
                ],
                dtype=torch.float32,
            ),
            "precursor_mz_raw": torch.tensor(600.0, dtype=torch.float32),
        },
    ]

    sampled_context = torch.tensor(
        [
            [False, True, True, False, False, False, False],
            [True, False, True, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    sampled_targets = torch.zeros((2, 2, 7), dtype=torch.bool)
    sampled_targets[0, 0, 0] = True
    sampled_targets[0, 1, 3] = True
    sampled_targets[1, 0, 3] = True
    sampled_targets[1, 1, 4] = True

    def fake_sample_masks(
        peak_valid_mask: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert peak_valid_mask.shape == (2, 7)
        return sampled_context.clone(), sampled_targets.clone()

    with mock.patch.object(
        gems_collate,
        "_sample_block_masks_torch",
        side_effect=fake_sample_masks,
    ):
        batch = collator(samples)

    assert batch["peak_valid_mask"][:, 0].all()
    assert torch.equal(batch["context_mask"], sampled_context)
    assert torch.equal(batch["target_masks"], sampled_targets)


def test_mask_block_ranges_reports_absolute_slot_runs() -> None:
    mask = torch.tensor(
        [False, True, True, False, True, True, True, False],
        dtype=torch.bool,
    )

    assert gems_visualization._mask_block_ranges(mask) == [(1, 2), (4, 6)]


def test_mask_block_ranges_in_active_order_compresses_context_gap() -> None:
    active_positions = torch.tensor(
        [True, True, True, False, False, False, True, True, True],
        dtype=torch.bool,
    )
    mask = torch.tensor(
        [True, True, True, False, False, False, True, False, False],
        dtype=torch.bool,
    )

    assert gems_visualization._mask_block_ranges(mask) == [(0, 2), (6, 6)]
    assert gems_visualization._mask_block_ranges_in_active_order(
        mask,
        active_positions,
    ) == [(0, 3)]
