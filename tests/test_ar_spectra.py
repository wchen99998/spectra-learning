import math

import numpy as np
import pytest
import torch

from spectra_learning.data.ar_spectra import (
    SpectraARGemsBatchCollator,
    SpectraARTokenKind,
    SpectraARTokenizer,
    SpectraARTokenizerConfig,
)
from spectra_learning.data.spectra import (
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    PEAK_MZ_MAX,
)
from spectra_learning.models.ar_spectra import (
    SpectraARTransformer,
    SpectraARTransformerConfig,
)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "peak_mz": torch.tensor(
            [[138.75, 938.50, 466.25, 0.0]],
            dtype=torch.float32,
        )
        / PEAK_MZ_MAX,
        "peak_intensity": torch.tensor([[0.10, 1.00, 0.50, 0.0]], dtype=torch.float32),
        "peak_valid_mask": torch.tensor([[True, True, True, False]]),
        "precursor_mz": torch.tensor([512.50], dtype=torch.float32)
        / DEFAULT_MAX_PRECURSOR_MZ,
        "collision_energy": torch.tensor([0.35], dtype=torch.float32),
        "charge": torch.tensor([2.0], dtype=torch.float32),
    }


def _full_sequence(
    tokenized: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    full_ids = torch.cat(
        [tokenized["input_token_ids"][:, :1], tokenized["target_token_ids"]],
        dim=1,
    )
    full_kinds = torch.cat(
        [tokenized["input_token_kinds"][:, :1], tokenized["target_token_kinds"]],
        dim=1,
    )
    return full_ids, full_kinds


def test_tokenizer_sorts_fragments_by_descending_mz() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    tokenized = tokenizer.tokenize_batch(_batch())
    full_ids, full_kinds = _full_sequence(tokenized)

    decoded_mz = []
    for offset in range(0, 3 * tokenizer.tokens_per_peak, tokenizer.tokens_per_peak):
        start = tokenizer.prefix_length + offset
        assert full_kinds[0, start].item() == int(
            SpectraARTokenKind.FRAGMENT_MZ_LEVEL_0
        )
        level_0 = tokenizer.token_value(
            int(full_ids[0, start]),
            SpectraARTokenKind.FRAGMENT_MZ_LEVEL_0,
        )
        level_1 = tokenizer.token_value(
            int(full_ids[0, start + 1]),
            SpectraARTokenKind.FRAGMENT_MZ_LEVEL_1,
        )
        level_2 = tokenizer.token_value(
            int(full_ids[0, start + 2]),
            SpectraARTokenKind.FRAGMENT_MZ_LEVEL_2,
        )
        level_3 = tokenizer.token_value(
            int(full_ids[0, start + 3]),
            SpectraARTokenKind.FRAGMENT_MZ_LEVEL_3,
        )
        residual = tokenizer.token_value(
            int(full_ids[0, start + 4]),
            SpectraARTokenKind.FRAGMENT_MZ_RESIDUAL,
        )
        decoded_mz.append(
            tokenizer.decode_fragment_mz(
                level_0,
                level_1,
                level_2,
                level_3,
                residual,
            )
        )

    assert decoded_mz == [938.5, 466.25, 138.75]


def test_tokenizer_uses_intermediate_mz_levels() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    tokenized = tokenizer.tokenize_batch(_batch())
    full_ids, _ = _full_sequence(tokenized)
    first_peak = tokenizer.prefix_length

    level_values = [
        tokenizer.token_value(
            int(full_ids[0, first_peak + offset]),
            kind,
        )
        for offset, kind in enumerate(tokenizer.fragment_mz_level_kinds)
    ]
    residual = tokenizer.token_value(
        int(full_ids[0, first_peak + len(tokenizer.fragment_mz_level_kinds)]),
        SpectraARTokenKind.FRAGMENT_MZ_RESIDUAL,
    )

    assert level_values == [18, 1, 2, 3]
    assert residual == 50


def test_tokenizer_uses_metadata_prefix_without_metadata_loss() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    tokenized = tokenizer.tokenize_batch(_batch())
    full_ids, full_kinds = _full_sequence(tokenized)

    assert full_ids[0, 0].item() == tokenizer.bos_token_id
    assert full_kinds[0, 1].item() == int(SpectraARTokenKind.PRECURSOR_MZ_LEVEL_0)
    assert full_kinds[0, 5].item() == int(SpectraARTokenKind.PRECURSOR_MZ_RESIDUAL)
    assert full_kinds[0, 6].item() == int(SpectraARTokenKind.COLLISION_ENERGY)
    assert full_kinds[0, 7].item() == int(SpectraARTokenKind.CHARGE)
    assert not tokenized["target_loss_mask"][0, : tokenizer.prefix_length - 1].any()
    assert tokenized["target_loss_mask"][0, tokenizer.prefix_length - 1]

    eos_full_index = tokenizer.prefix_length + 3 * tokenizer.tokens_per_peak
    assert full_ids[0, eos_full_index].item() == tokenizer.eos_token_id
    assert tokenized["target_loss_mask"][0, eos_full_index - 1]
    assert not tokenized["target_loss_mask"][0, eos_full_index:].any()


def test_teacher_forcing_inputs_are_shifted_ground_truth_tokens() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    tokenized = tokenizer.tokenize_batch(_batch())
    full_ids, full_kinds = _full_sequence(tokenized)

    assert torch.equal(tokenized["input_token_ids"], full_ids[:, :-1])
    assert torch.equal(tokenized["target_token_ids"], full_ids[:, 1:])
    assert torch.equal(tokenized["input_token_kinds"], full_kinds[:, :-1])
    assert torch.equal(tokenized["target_token_kinds"], full_kinds[:, 1:])

    first_fragment_target_position = tokenizer.prefix_length - 1
    assert tokenized["input_token_kinds"][0, first_fragment_target_position].item() == int(
        SpectraARTokenKind.CHARGE
    )
    assert tokenized["target_token_kinds"][0, first_fragment_target_position].item() == int(
        SpectraARTokenKind.FRAGMENT_MZ_LEVEL_0
    )
    assert tokenized["target_loss_mask"][0, first_fragment_target_position]


def test_tokenizer_places_eos_after_each_rows_valid_fragments() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=3))
    batch = {
        "peak_mz": torch.tensor(
            [
                [100.0, 200.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        / PEAK_MZ_MAX,
        "peak_intensity": torch.tensor(
            [
                [1.0, 0.5, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "peak_valid_mask": torch.tensor(
            [
                [True, True, False],
                [False, False, False],
            ]
        ),
        "precursor_mz": torch.tensor([512.50, 256.0], dtype=torch.float32)
        / DEFAULT_MAX_PRECURSOR_MZ,
        "collision_energy": torch.tensor([0.35, 0.10], dtype=torch.float32),
        "charge": torch.tensor([2.0, 1.0], dtype=torch.float32),
    }

    tokenized = tokenizer.tokenize_batch(batch)
    full_ids, full_kinds = _full_sequence(tokenized)
    row0_eos = tokenizer.prefix_length + 2 * tokenizer.tokens_per_peak
    row1_eos = tokenizer.prefix_length

    assert full_ids[0, row0_eos].item() == tokenizer.eos_token_id
    assert full_ids[1, row1_eos].item() == tokenizer.eos_token_id
    assert full_kinds[0, row0_eos + 1 :].eq(int(SpectraARTokenKind.PAD)).all()
    assert full_kinds[1, row1_eos + 1 :].eq(int(SpectraARTokenKind.PAD)).all()
    assert tokenized["target_loss_mask"][0].sum().item() == (
        2 * tokenizer.tokens_per_peak + 1
    )
    assert tokenized["target_loss_mask"][1].sum().item() == 1


def test_ar_transformer_forward_returns_finite_loss() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    tokenized = tokenizer.tokenize_batch(_batch())
    model = SpectraARTransformer(
        SpectraARTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            num_token_kinds=tokenizer.num_token_kinds,
            max_sequence_length=tokenizer.sequence_length - 1,
            pad_token_id=tokenizer.pad_token_id,
            model_dim=32,
            num_layers=2,
            num_heads=4,
            mlp_multiple=2.0,
            dropout=0.0,
        )
    )

    output = model(tokenized)

    assert output["logits"].shape == (
        1,
        tokenizer.sequence_length - 1,
        tokenizer.vocab_size,
    )
    assert torch.isfinite(output["loss"])
    assert output["target_tokens"].item() == 19


def test_ar_transformer_uses_rope_without_learned_absolute_positions() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    model = SpectraARTransformer(
        SpectraARTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            num_token_kinds=tokenizer.num_token_kinds,
            max_sequence_length=tokenizer.sequence_length - 1,
            pad_token_id=tokenizer.pad_token_id,
            model_dim=32,
            num_layers=2,
            num_heads=4,
            mlp_multiple=2.0,
            dropout=0.0,
        )
    )

    assert not hasattr(model, "position_embedding")
    assert hasattr(model.blocks[0].attention, "rope")


def test_causal_attention_does_not_read_future_tokens() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=4))
    tokenized = tokenizer.tokenize_batch(_batch())
    changed = {key: value.clone() for key, value in tokenized.items()}
    changed["input_token_ids"][0, -1] = tokenizer.eos_token_id
    model = SpectraARTransformer(
        SpectraARTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            num_token_kinds=tokenizer.num_token_kinds,
            max_sequence_length=tokenizer.sequence_length - 1,
            pad_token_id=tokenizer.pad_token_id,
            model_dim=32,
            num_layers=1,
            num_heads=4,
            mlp_multiple=2.0,
            dropout=0.0,
        )
    )
    model.eval()

    logits = model(tokenized)["logits"]
    changed_logits = model(changed)["logits"]

    assert torch.allclose(logits[:, :-1], changed_logits[:, :-1], atol=1e-6)


def test_ar_gems_collator_tokenizes_raw_spectra_samples() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=3))
    collator = SpectraARGemsBatchCollator(
        tokenizer=tokenizer,
        num_peaks=3,
        max_precursor_mz=DEFAULT_MAX_PRECURSOR_MZ,
        min_peak_intensity=DEFAULT_MIN_PEAK_INTENSITY,
        peak_drop_min_intensity=DEFAULT_MIN_PEAK_INTENSITY,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
    )
    samples = [
        {
            "spectra": np.asarray(
                [[100.0, 750.0, 300.0], [0.2, 1.0, 0.4]],
                dtype=np.float32,
            ),
            "precursor_mz_raw": np.float32(800.0),
            "collision_energy": np.float32(35.0),
            "charge": np.float32(1.0),
        }
    ]

    tokenized = collator(samples)
    full_ids, full_kinds = _full_sequence(tokenized)

    assert tokenized["input_token_ids"].shape == (1, tokenizer.sequence_length - 1)
    first_peak = tokenizer.prefix_length
    assert full_kinds[0, first_peak].item() == int(
        SpectraARTokenKind.FRAGMENT_MZ_LEVEL_0
    )
    first_coarse = tokenizer.token_value(
        int(full_ids[0, first_peak]),
        SpectraARTokenKind.FRAGMENT_MZ_LEVEL_0,
    )
    assert math.isclose(first_coarse * tokenizer.config.mz_bin_widths[0], 750.0)


def test_ar_gems_collator_emits_numpy_batches_for_jax() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=3))
    collator = SpectraARGemsBatchCollator(
        tokenizer=tokenizer,
        num_peaks=3,
        max_precursor_mz=DEFAULT_MAX_PRECURSOR_MZ,
        min_peak_intensity=DEFAULT_MIN_PEAK_INTENSITY,
        peak_drop_min_intensity=DEFAULT_MIN_PEAK_INTENSITY,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
        output_format="numpy",
    )
    sample = {
        "spectra": np.asarray(
            [[100.0, 750.0, 300.0], [0.2, 1.0, 0.4]],
            dtype=np.float32,
        ),
        "precursor_mz_raw": np.float32(800.0),
        "collision_energy": np.float32(35.0),
        "charge": np.float32(1.0),
    }

    tokenized = collator([sample])

    assert tokenized
    assert all(isinstance(value, np.ndarray) for value in tokenized.values())


def test_ar_gems_collator_rejects_unknown_output_format() -> None:
    tokenizer = SpectraARTokenizer(SpectraARTokenizerConfig(max_num_peaks=3))
    collator = SpectraARGemsBatchCollator(
        tokenizer=tokenizer,
        num_peaks=3,
        max_precursor_mz=DEFAULT_MAX_PRECURSOR_MZ,
        min_peak_intensity=DEFAULT_MIN_PEAK_INTENSITY,
        peak_drop_min_intensity=DEFAULT_MIN_PEAK_INTENSITY,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
        output_format="unknown",
    )
    sample = {
        "spectra": np.asarray(
            [[100.0, 750.0, 300.0], [0.2, 1.0, 0.4]],
            dtype=np.float32,
        ),
        "precursor_mz_raw": np.float32(800.0),
        "collision_energy": np.float32(35.0),
        "charge": np.float32(1.0),
    }

    with pytest.raises(ValueError, match="Unknown dataloader output format"):
        collator([sample])
