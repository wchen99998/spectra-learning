"""Separate, identity-initialized query/key maps for FLARE local compatibility."""

import torch
from torch import nn


class QueryKeyCompatibility(nn.Module):
    """Row vectors map as q = peaks @ W_Q and k = atoms @ W_K.

    Inputs have already undergone FLARE's original L2 normalization. These
    projections add no bias, output normalization, or attention scaling.
    """

    def __init__(self, dimension: int):
        super().__init__()
        self.query_weight = nn.Parameter(torch.eye(dimension))
        self.key_weight = nn.Parameter(torch.eye(dimension))

    def forward(
        self, peaks: torch.Tensor, atoms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return peaks @ self.query_weight, atoms @ self.key_weight
