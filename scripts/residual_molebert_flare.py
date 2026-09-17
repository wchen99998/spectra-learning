"""Add a gated pretrained atom branch to the original FLARE molecular encoder."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from molebert_atom_encoder import MoleBERTAtomEncoder, MoleBERTGraphBatch
from trainable_molebert_flare import MoleBERTGraphCache


class ResidualGraphCache:
    """Original chemical features and discrete Mole-BERT inputs in identical atom order."""

    def __init__(self, original: Path, pretrained: Path):
        self.pretrained = MoleBERTGraphCache(pretrained)
        self.nodes = np.load(original / 'nodes.npy', mmap_mode='r')
        self.edges = np.load(original / 'edge_attrs.npy', mmap_mode='r')
        if self.nodes.dtype != np.float32 or self.edges.dtype != np.float32:
            raise ValueError('Original chemical features must retain float32 precision')
        if len(self.nodes) != len(self.pretrained.nodes) or len(self.edges) != len(self.pretrained.edges):
            raise ValueError('Original and pretrained graph caches must have matching atoms and bonds')

    def __len__(self):
        return len(self.pretrained)

    def __contains__(self, smiles):
        return smiles in self.pretrained

    def __getitem__(self, smiles):
        i = self.pretrained.row_of[smiles]
        a, b = self.pretrained.node_offsets[i:i + 2]
        e, f = self.pretrained.edge_offsets[i:i + 2]
        x, edge_index, edge_attr = self.pretrained[smiles]
        original_x = torch.from_numpy(np.array(self.nodes[a:b]))
        original_e = torch.from_numpy(np.array(self.edges[e:f]))
        return torch.cat((original_x, x.float()), -1), edge_index, torch.cat((original_e, edge_attr.float()), -1)


def residual_model_class(original_class, molecular_checkpoint: Path, atom_dim: int, bond_dim: int):
    class ResidualMoleBERT(original_class):
        def __init__(self, hidden=512, out_dim=512, n_layers=3, dropout=.2, feature_set='basic'):
            if feature_set != 'basic':
                raise ValueError('This experiment preserves the original basic chemical features')
            super().__init__(hidden, out_dim, n_layers, dropout, feature_set=feature_set)
            with torch.random.fork_rng(devices=[]):
                self.pretrained = MoleBERTAtomEncoder.from_checkpoint(molecular_checkpoint, frozen=False)
                self.residual_norm = nn.LayerNorm(300, elementwise_affine=False)
                self.residual_projection = nn.Linear(300, out_dim, bias=False)
            self.residual_gate = nn.Parameter(torch.zeros(()))

        def forward(self, x, edge_index, edge_attr, batch=None):
            # Identical operations and parameter names to the original MolGNN.
            h = self.node_enc(x[:, :atom_dim].contiguous())
            e = self.edge_enc(edge_attr[:, :bond_dim].contiguous())
            for conv, norm in zip(self.convs, self.norms):
                h = h + self.drop(F.relu(norm(conv(h, edge_index, e))))
            original = self.out(h)
            graph = MoleBERTGraphBatch(x[:, atom_dim:].long(), edge_index,
                                      edge_attr[:, bond_dim:].long(), ())
            pretrained = self.pretrained(graph)
            residual = self.residual_projection(self.residual_norm(pretrained))
            return F.normalize(original + self.residual_gate * residual, dim=-1)

        def optimizer_groups(self, encoder, agreement, lr: float):
            backbone_ids = {id(p) for p in self.pretrained.parameters()}
            parameters = [*encoder.parameters(), *self.parameters(), *agreement.parameters()]
            return [
                {'params': [p for p in parameters if p.requires_grad and id(p) not in backbone_ids], 'lr': lr},
                {'params': list(self.pretrained.parameters()), 'lr': lr * .1},
            ]

    return ResidualMoleBERT


def extend_checkpoint(original: dict, model: nn.Module) -> dict:
    """Strictly retain every original GNN parameter and add only the new branch."""
    state = model.state_dict()
    added = {k for k in state if k.startswith(('pretrained.', 'residual_'))}
    if set(original['gnn']) != set(state) - added:
        raise ValueError('Checkpoint is not the expected original FLARE molecular architecture')
    state.update(original['gnn'])
    model.load_state_dict(state, strict=True)
    return original | {'gnn': model.state_dict(), 'epoch': -1}
