"""Frozen pretrained atom storage and its linear FLARE compatibility map."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FrozenAtomCache:
    """Memory-mapped atom rows in the original molecule/atom order.

    Graph tuples retain FLARE's batching interface. All molecular context is
    already in the pretrained rows, so the graph carries no edges.
    """

    def __init__(self, root: Path | str):
        root = Path(root)
        self.smiles = json.loads((root / "smiles.json").read_text())
        if not isinstance(self.smiles, list) or any(
            not isinstance(smiles, str) or not smiles for smiles in self.smiles
        ):
            raise ValueError("smiles.json must contain a list of nonempty SMILES strings")
        if any(first >= second for first, second in zip(self.smiles, self.smiles[1:])):
            raise ValueError("smiles.json must be sorted and contain no duplicates")
        self.offsets = np.load(root / "offsets.npy", mmap_mode="r", allow_pickle=False)
        self.values = np.load(root / "atom_embeddings.float16.npy", mmap_mode="r", allow_pickle=False)
        if self.offsets.dtype != np.int64 or self.offsets.shape != (len(self.smiles) + 1,):
            raise ValueError("offsets.npy must be an int64 vector with one more entry than SMILES")
        if self.values.dtype != np.float16 or self.values.ndim != 2 or self.values.shape[1] == 0:
            raise ValueError("atom_embeddings.float16.npy must be a float16 matrix with positive width")
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.values):
            raise ValueError("offsets must start at zero and end at the number of atom rows")
        counts = np.diff(self.offsets)
        if np.any(counts <= 0) or np.any(counts > 128):
            raise ValueError("each molecule must have between 1 and 128 atom rows; truncation is forbidden")
        self.in_dim = self.values.shape[1]
        self.row_of = {smiles: row for row, smiles in enumerate(self.smiles)}

    def __contains__(self, smiles: str) -> bool:
        return smiles in self.row_of

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, smiles: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.row_of[smiles]
        start, stop = self.offsets[row:row + 2]
        atoms = torch.from_numpy(np.array(self.values[start:stop], dtype=np.float32))
        return atoms, torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float32)


class AtomCompatibility(nn.Module):
    """A normalized linear map of contextual pretrained atom embeddings."""

    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        # FLARE's candidate encoder reads out.out_features to size its store.
        self.out = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.normalize(self.out(self.norm(x)), dim=-1)
