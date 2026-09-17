"""Online Mole-BERT fine-tuning with cached discrete molecular graphs."""
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from molebert_atom_encoder import MoleBERTAtomEncoder, MoleBERTGraphBatch


class MoleBERTGraphCache:
    """Compact graph inputs only; learned atom representations are never cached here."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.smiles = json.loads((root / 'smiles.json').read_text())
        if not self.smiles or any(a >= b for a, b in zip(self.smiles, self.smiles[1:])):
            raise ValueError('Graph SMILES must be nonempty, sorted, and unique')
        self.row_of = {s: i for i, s in enumerate(self.smiles)}
        self.node_offsets = np.load(root / 'node_offsets.npy', mmap_mode='r')
        self.edge_offsets = np.load(root / 'edge_offsets.npy', mmap_mode='r')
        self.nodes = np.load(root / 'nodes.npy', mmap_mode='r')
        self.edges = np.load(root / 'edges.npy', mmap_mode='r')
        self.edge_attrs = np.load(root / 'edge_attrs.npy', mmap_mode='r')
        for values, offsets in [(self.nodes, self.node_offsets), (self.edges, self.edge_offsets)]:
            if values.dtype != np.uint8 or values.ndim != 2 or values.shape[1] != 2:
                raise ValueError('Graph values must be uint8 matrices of width two')
            if offsets.dtype != np.int64 or offsets.shape != (len(self.smiles) + 1,):
                raise ValueError('Invalid graph offset shape or dtype')
            if offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) < 0):
                raise ValueError('Graph offsets must span the complete values array')
        if self.edge_attrs.dtype != np.uint8 or self.edge_attrs.shape != self.edges.shape:
            raise ValueError('Each directed edge must have two uint8 attributes')
        counts = np.diff(self.node_offsets)
        if np.any(counts < 1) or np.any(counts > 128):
            raise ValueError('Graph atom counts must be 1..128; truncation is forbidden')

    def __len__(self):
        return len(self.smiles)

    def __contains__(self, smiles):
        return smiles in self.row_of

    def __getitem__(self, smiles):
        i = self.row_of[smiles]
        a, b = self.node_offsets[i:i + 2]
        e, f = self.edge_offsets[i:i + 2]
        return (
            torch.from_numpy(np.array(self.nodes[a:b], dtype=np.int64)),
            torch.from_numpy(np.array(self.edges[e:f], dtype=np.int64).T.copy()),
            torch.from_numpy(np.array(self.edge_attrs[e:f], dtype=np.int64)),
        )


class TrainableMoleBERT(nn.Module):
    """Original atom projection with an unfrozen, pretrained Mole-BERT backbone."""

    def __init__(self, checkpoint: str | Path, out_dim: int = 512):
        super().__init__()
        # Keep the projection's seed-0 initialization identical to the frozen run.
        self.norm = nn.LayerNorm(300, elementwise_affine=False)
        self.out = nn.Linear(300, out_dim, bias=False)
        # Initializing weights that are immediately overwritten must not perturb
        # the rest of the retrieval loop's random-number stream.
        with torch.random.fork_rng(devices=[]):
            self.backbone = MoleBERTAtomEncoder.from_checkpoint(checkpoint, frozen=False)

    def forward(self, x, edge_index, edge_attr, batch=None):
        atoms = self.backbone(MoleBERTGraphBatch(x, edge_index, edge_attr, ()))
        return F.normalize(self.out(self.norm(atoms)), dim=-1)
