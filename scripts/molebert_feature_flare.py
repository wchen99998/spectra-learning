"""Original FLARE atom features augmented with frozen pretrained Mole-BERT vectors."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from molebert_atom_encoder import MOLEBERT_CHECKPOINT_SHA256
from trainable_molebert_flare import MoleBERTGraphCache


class MoleBERTFeatureGraphs:
    """Aligned chemical features, frozen atom embeddings, and original bonds."""

    def __init__(self, chemical: Path, topology: Path, embeddings: Path):
        self.topology = MoleBERTGraphCache(topology)
        metadata = json.loads((embeddings / 'metadata.json').read_text())
        chemical_metadata = json.loads((chemical / 'metadata.json').read_text())
        if not metadata['complete'] or metadata['smoke_limit'] is not None:
            raise ValueError('Frozen molecular embeddings must cover the complete benchmark')
        if metadata['checkpoint_sha256'] != MOLEBERT_CHECKPOINT_SHA256:
            raise ValueError('Expected the released pretrained Mole-BERT atom embeddings')
        if not chemical_metadata['all_graph_ordering_verified']:
            raise ValueError('Chemical features must have verified atom and bond ordering')
        if json.loads((embeddings / 'smiles.json').read_text()) != self.topology.smiles:
            raise ValueError('Embedding and graph molecule ordering differ')
        offsets = np.load(embeddings / 'offsets.npy', mmap_mode='r')
        if not np.array_equal(offsets, self.topology.node_offsets):
            raise ValueError('Embedding and graph atom offsets differ')
        self.chemical = np.load(chemical / 'nodes.npy', mmap_mode='r')
        self.bonds = np.load(chemical / 'edge_attrs.npy', mmap_mode='r')
        self.embeddings = np.load(embeddings / 'atom_embeddings.float16.npy', mmap_mode='r')
        if self.chemical.shape != (len(self.topology.nodes), 49) or self.chemical.dtype != np.float32:
            raise ValueError('Expected all 49 original float32 atom features')
        if self.embeddings.shape != (len(self.chemical), 300) or self.embeddings.dtype != np.float16:
            raise ValueError('Expected one frozen 300-dimensional vector per atom')
        if self.bonds.shape != (len(self.topology.edges), 8) or self.bonds.dtype != np.float32:
            raise ValueError('Expected all eight original float32 bond features')

    def __len__(self):
        return len(self.topology)

    def __contains__(self, smiles):
        return smiles in self.topology

    def __getitem__(self, smiles):
        i = self.topology.row_of[smiles]
        a, b = self.topology.node_offsets[i:i + 2]
        e, f = self.topology.edge_offsets[i:i + 2]
        x = np.concatenate((self.chemical[a:b], self.embeddings[a:b].astype(np.float32)), axis=1)
        return (torch.from_numpy(x),
                torch.from_numpy(np.array(self.topology.edges[e:f].T, dtype=np.int64)),
                torch.from_numpy(np.array(self.bonds[e:f])))


def widen_atom_input(original: nn.Linear, extra_dim: int) -> nn.Linear:
    """Keep chemical initialization and the rest of the recipe's RNG unchanged."""
    with torch.random.fork_rng(devices=[]):
        expanded = nn.Linear(original.in_features + extra_dim, original.out_features)
    with torch.no_grad():
        expanded.weight[:, :original.in_features].copy_(original.weight)
        expanded.bias.copy_(original.bias)
    return expanded


def feature_model_class(original_class, atom_dim=49, embedding_dim=300):
    class MoleBERTFeatureFLARE(original_class):
        def __init__(self, hidden=512, out_dim=512, n_layers=3, dropout=.2, feature_set='basic'):
            if feature_set != 'basic':
                raise ValueError('This experiment retains the original basic chemical features')
            super().__init__(hidden, out_dim, n_layers, dropout, feature_set=feature_set)
            self.node_enc = widen_atom_input(self.node_enc, embedding_dim)

        def forward(self, x, edge_index, edge_attr, batch=None):
            # Normalize only the added feature block; original chemical columns stay untouched.
            augmented = torch.cat((x[:, :atom_dim], F.normalize(x[:, atom_dim:], dim=-1)), dim=-1)
            return super().forward(augmented, edge_index, edge_attr, batch)

    return MoleBERTFeatureFLARE
