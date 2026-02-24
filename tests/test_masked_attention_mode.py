import torch

from networks.transformer_torch import build_masked_attention_allow_matrix


def test_build_masked_attention_allow_matrix_rules():
    valid_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)
    masked_positions = torch.tensor([[False, True, False, True]], dtype=torch.bool)
    allow = build_masked_attention_allow_matrix(valid_mask, masked_positions)[0]

    expected = torch.tensor(
        [
            [True, False, True, False],
            [True, True, True, True],
            [True, False, True, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(allow, expected)


def test_build_masked_attention_allow_matrix_respects_padding():
    valid_mask = torch.tensor([[True, True, False, True]], dtype=torch.bool)
    masked_positions = torch.tensor([[False, True, False, True]], dtype=torch.bool)
    allow = build_masked_attention_allow_matrix(valid_mask, masked_positions)[0]

    assert not allow[2].any()
    assert not allow[:, 2].any()
