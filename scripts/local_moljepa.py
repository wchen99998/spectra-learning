"""Local FLARE encoders with zero-initialized global and Mol-JEPA residuals.

The local spectrum and three-hop GINE paths reach the output directly. Every
attention contribution has its own scalar gate, initialized to zero, so the
initial model preserves the original local computation exactly.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

from dgt_moljepa import AtomAgreement, GraphBatch


class SelfAttentionResidual(nn.Module):
    """A narrow attention/FFN branch, with no normalization on the local skip."""

    def __init__(self, width: int, context_width: int, heads: int, dropout: float, scale: float = .25):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, context_width))
        self.attention = nn.MultiheadAttention(context_width, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(context_width), nn.Linear(context_width, 2 * context_width),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * context_width, context_width),
        )
        self.output = nn.Linear(context_width, width)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(()))
        self.scale = scale

    def forward(self, local: torch.Tensor, valid: torch.Tensor, local_norm: torch.Tensor) -> torch.Tensor:
        reduced = self.input(local)
        attended = self.attention(reduced, reduced, reduced, key_padding_mask=~valid, need_weights=False)[0]
        delta = self.drop(attended)
        delta = delta + self.drop(self.ffn(reduced + delta))
        delta = F.normalize(self.output(delta).float(), dim=-1).masked_fill(~valid[..., None], 0)
        return local + self.scale * self.gate.tanh() * local_norm * delta


class MolJepaResidual(nn.Module):
    """Atom queries read frozen Mol-JEPA tokens through a separate gated branch."""

    def __init__(self, width: int, context_width: int, heads: int, dropout: float):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, context_width))
        self.memory = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, context_width))
        self.attention = nn.MultiheadAttention(context_width, heads, dropout=dropout, batch_first=True)
        self.output = nn.Linear(context_width, width)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(()))
        self.scale = .125

    def forward(self, local: torch.Tensor, valid: torch.Tensor, moljepa: torch.Tensor, local_norm: torch.Tensor) -> torch.Tensor:
        memory = self.memory(moljepa)
        delta = self.attention(self.query(local), memory, memory, need_weights=False)[0]
        delta = F.normalize(self.drop(self.output(delta)).float(), dim=-1).masked_fill(~valid[..., None], 0)
        return local + self.scale * self.gate.tanh() * local_norm * delta


class SpectrumEncoder(nn.Module):
    def __init__(self, width: int = 512, context_width: int = 256, heads: int = 8, dropout: float = .2):
        super().__init__()
        self.token_projection = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, width))
        self.peak_projection = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        self.local_block = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * width, width),
        )
        self.context = nn.ModuleList([
            SelfAttentionResidual(width, context_width, heads, dropout) for _ in range(2)
        ])

    def local_features(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # The final slot is CLS, with no physical m/z or intensity measurement.
        raw = x[..., 2048:].clone()
        raw[:, -1] = 0
        local = self.token_projection(x[..., :2048]) + self.peak_projection(raw)
        return (local + self.local_block(local)).masked_fill(~valid[..., None], 0)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid.clone()
        valid[:, -1] = True
        local = self.local_features(x, valid)
        local_norm = local.float().norm(dim=-1, keepdim=True)
        for block in self.context:
            local = block(local, valid, local_norm)
        return F.normalize(local.float(), dim=-1).masked_fill(~valid[..., None], 0)


class GlobalHead(nn.Sequential):
    def __init__(self, width: int, dropout: float):
        super().__init__(
            nn.LayerNorm(2048), nn.Linear(2048, 2 * width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x[..., :2048])


class MolecularEncoder(nn.Module):
    def __init__(self, width: int = 512, context_width: int = 256, heads: int = 8, dropout: float = .2):
        super().__init__()
        self.environments = nn.ModuleList([nn.Embedding(32768, 16, padding_idx=0) for _ in range(4)])
        self.node_enc = nn.Linear(49, width)
        self.edge_enc = nn.Linear(8, width)
        self.extra_node_enc = nn.Linear(78 + 4 * 16, width)
        self.extra_edge_enc = nn.Linear(49, width)
        self.atom_feature_gate = nn.Parameter(torch.zeros(()))
        self.bond_feature_gate = nn.Parameter(torch.zeros(()))
        self.feature_residual_scale = .25
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)), edge_dim=width)
            for _ in range(3)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(3)])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(width, width)
        self.context = nn.ModuleList([
            SelfAttentionResidual(width, context_width, heads, dropout, scale=.125) for _ in range(2)
        ])
        self.moljepa_context = nn.ModuleList([
            MolJepaResidual(width, context_width, heads, dropout) for _ in range(2)
        ])
        self.width = width

    def local_features(self, g: GraphBatch) -> torch.Tensor:
        # Pack only real atoms so neither GINE nor BatchNorm sees padding.
        atoms = g.atoms[g.atom_valid]
        extra_features = torch.cat((atoms[:, 49:127], *[
            embedding(atoms[:, 127 + i].long()) for i, embedding in enumerate(self.environments)
        ]), dim=-1)
        h = self.node_enc(atoms[:, :49])
        extra_atoms = F.normalize(self.extra_node_enc(extra_features).float(), dim=-1)
        atom_delta = self.feature_residual_scale * self.atom_feature_gate.tanh() * h.float().norm(dim=-1, keepdim=True) * extra_atoms
        h = h + atom_delta.to(h.dtype)
        packed_index = torch.full(g.atom_valid.shape, -1, device=h.device, dtype=torch.long)
        packed_index[g.atom_valid] = torch.arange(len(h), device=h.device)
        batch_index, bond_index = g.bond_valid.nonzero(as_tuple=True)
        endpoints = g.endpoints[batch_index, bond_index]
        u = packed_index[batch_index, endpoints[:, 0]]
        v = packed_index[batch_index, endpoints[:, 1]]
        # Cache orientation zero describes u->v, orientation one describes v->u.
        edge_index = torch.stack((torch.stack((u, v), -1).flatten(), torch.stack((v, u), -1).flatten()))
        directed_bonds = g.bonds[g.bond_valid].reshape(-1, 57)
        e = self.edge_enc(directed_bonds[:, :8])
        extra_bonds = F.normalize(self.extra_edge_enc(directed_bonds[:, 8:]).float(), dim=-1)
        bond_delta = self.feature_residual_scale * self.bond_feature_gate.tanh() * e.float().norm(dim=-1, keepdim=True) * extra_bonds
        e = e + bond_delta.to(e.dtype)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(F.relu(norm(conv(h, edge_index, e))))
        local = h.new_zeros((*g.atom_valid.shape, self.width))
        local[g.atom_valid] = self.out(h)
        return local

    def forward(self, g: GraphBatch) -> torch.Tensor:
        local = self.local_features(g)
        local_norm = local.float().norm(dim=-1, keepdim=True)
        for attention, cross in zip(self.context, self.moljepa_context):
            local = attention(local, g.atom_valid, local_norm)
            local = cross(local, g.atom_valid, g.moljepa, local_norm)
        return F.normalize(local.float(), dim=-1).masked_fill(~g.atom_valid[..., None], 0)


class Model(nn.Module):
    def __init__(self, width: int = 512, context_width: int = 256, heads: int = 8, dropout: float = .2):
        super().__init__()
        self.spectrum = SpectrumEncoder(width, context_width, heads, dropout)
        self.molecule = MolecularEncoder(width, context_width, heads, dropout)
        self.global_head = GlobalHead(width, dropout)
        self.mass = AtomAgreement()
        self.experiment_config = {
            'architecture': 'local_flare_gated_moljepa', 'width': width, 'gnn_type': 'GINEConv',
            'gnn_layers': 3, 'gnn_heads': None, 'gnn_normalization': 'BatchNorm1d_valid_atoms_only',
            'atom_input_features': 131, 'directed_bond_features': 57, 'morgan_embedding_width': 16,
            'base_atom_features': 49, 'base_bond_features': 8, 'extra_chemistry_gated': True,
            'spectrum_input_features': 2050, 'spectrum_raw_mz_intensity': True,
            'spectrum_readout_layers': 2, 'atom_global_blocks': 2, 'moljepa_cross_attention_blocks': 2,
            'context_width': context_width, 'attention_heads': heads, 'residual_gate_initial_value': 0.,
            'context_residual_norm_bound': .5, 'spectrum_gate_scale': .25,
            'atom_global_gate_scale': .125, 'moljepa_gate_scale': .125, 'extra_feature_gate_scale': .25,
            'moljepa_frozen': True, 'cls_ordinary_peak': True, 'dropout': dropout,
        }
