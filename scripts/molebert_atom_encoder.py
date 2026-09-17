"""Frozen Mole-BERT GIN atom representations, with no PyG dependency.

Mole-BERT checkpoint and atom-token pretraining: https://github.com/junxia97/Mole-BERT
Pinned upstream revision: 2feff8a33e3634b66b7408e2e2780fc9d960909f
Return the final 300-dimensional node states (upstream JK='last').

The operators reuse the mathematics of the MIT-licensed MolCLR port, with
Mole-BERT's expanded mask-token vocabularies. Upstream Mole-BERT does not
provide a separate checkpoint license; this notice covers the port's code.

Copyright (c) 2021 Yuyang Wang
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from rdkit import Chem
from torch import nn


MOLEBERT_CHECKPOINT_SHA256 = "375cd40af9f21d2a92ed1acbdea9efad14254c36703bb0e3a7e433e09e624ce1"
MOLEBERT_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/junxia97/Mole-BERT/"
    "2feff8a33e3634b66b7408e2e2780fc9d960909f/"
    "model_gin/Mole-BERT.pth"
)
_CHIRALITY = {
    Chem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}
_BOND_TYPE = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
_BOND_DIRECTION = {
    Chem.BondDir.NONE: 0,
    Chem.BondDir.ENDUPRIGHT: 1,
    Chem.BondDir.ENDDOWNRIGHT: 2,
}



@dataclass(frozen=True)
class MoleBERTGraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    counts: tuple[int, ...]

    def to(self, device: torch.device | str) -> "MoleBERTGraphBatch":
        return MoleBERTGraphBatch(
            self.x.to(device), self.edge_index.to(device),
            self.edge_attr.to(device), self.counts,
        )


def featurize_molecules(molecules: list[Chem.Mol]) -> MoleBERTGraphBatch:
    """Preserve RDKit atom indices and every atom, including disconnected ones."""
    atoms: list[tuple[int, int]] = []
    edges: list[tuple[int, int]] = []
    attributes: list[tuple[int, int]] = []
    counts: list[int] = []
    for molecule_index, mol in enumerate(molecules):
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Molecule {molecule_index} is invalid or empty")
        offset = len(atoms)
        counts.append(mol.GetNumAtoms())
        for atom in mol.GetAtoms():
            number, chiral = atom.GetAtomicNum(), atom.GetChiralTag()
            if number < 1 or number > 118 or chiral not in _CHIRALITY:
                raise ValueError(
                    f"Molecule {molecule_index}, atom {atom.GetIdx()}: "
                    f"unsupported Mole-BERT atomic number/chirality {number}/{chiral}"
                )
            atoms.append((number - 1, _CHIRALITY[chiral]))
        for bond in mol.GetBonds():
            bond_type, direction = bond.GetBondType(), bond.GetBondDir()
            if bond_type not in _BOND_TYPE or direction not in _BOND_DIRECTION:
                raise ValueError(
                    f"Molecule {molecule_index}, bond {bond.GetIdx()}: "
                    f"unsupported Mole-BERT bond type/direction {bond_type}/{direction}"
                )
            start = offset + bond.GetBeginAtomIdx()
            end = offset + bond.GetEndAtomIdx()
            edges.extend(((start, end), (end, start)))
            attr = (_BOND_TYPE[bond_type], _BOND_DIRECTION[direction])
            attributes.extend((attr, attr))
    return MoleBERTGraphBatch(
        torch.tensor(atoms, dtype=torch.long).reshape(-1, 2),
        torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
        torch.tensor(attributes, dtype=torch.long).reshape(-1, 2),
        tuple(counts),
    )


def featurize_smiles(smiles: list[str]) -> MoleBERTGraphBatch:
    molecules = []
    for index, value in enumerate(smiles):
        if not isinstance(value, str):
            raise ValueError(f"SMILES {index} must be a string")
        mol = Chem.MolFromSmiles(value)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"SMILES {index} is invalid or empty: {value!r}")
        molecules.append(mol)
    return featurize_molecules(molecules)


class _GINEConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(300, 600), nn.ReLU(), nn.Linear(600, 300))
        self.edge_embedding1 = nn.Embedding(6, 300)
        self.edge_embedding2 = nn.Embedding(3, 300)

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Match upstream's ordering: directed bonds, followed by all self loops.
        loop_nodes = torch.arange(h.shape[0], device=h.device)
        edge_index = torch.cat((edge_index, torch.stack((loop_nodes, loop_nodes))), dim=1)
        loops = edge_attr.new_zeros((h.shape[0], 2))
        loops[:, 0] = 4
        edge_attr = torch.cat((edge_attr, loops))
        edge_embeddings = (
            self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        )
        messages = h[edge_index[0]] + edge_embeddings
        aggregated = torch.zeros_like(h).index_add_(0, edge_index[1], messages)
        return self.mlp(aggregated)


class MoleBERTAtomEncoder(nn.Module):
    """Pretrained backbone; frozen by default, optionally fine-tuned end to end."""

    output_dim = 300

    def __init__(self, frozen: bool = True) -> None:
        super().__init__()
        self.frozen = frozen
        self.x_embedding1 = nn.Embedding(120, 300)
        self.x_embedding2 = nn.Embedding(3, 300)
        self.gnns = nn.ModuleList([_GINEConv() for _ in range(5)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(300) for _ in range(5)])
        self.requires_grad_(not frozen)
        self.train(not frozen)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, frozen: bool = True) -> "MoleBERTAtomEncoder":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        encoder = cls(frozen=frozen)
        encoder.load_state_dict(checkpoint, strict=True)
        return encoder

    def train(self, mode: bool = True) -> "MoleBERTAtomEncoder":
        # Frozen extraction keeps its original behavior; fine-tuning updates BN.
        super().train(mode and not self.frozen)
        return self

    def forward(self, graph: MoleBERTGraphBatch) -> torch.Tensor:
        h = self.x_embedding1(graph.x[:, 0]) + self.x_embedding2(graph.x[:, 1])
        for layer, (conv, norm) in enumerate(zip(self.gnns, self.batch_norms)):
            h = norm(conv(h, graph.edge_index, graph.edge_attr))
            if layer < 4:
                h = h.relu()
        return h

    def encode_smiles(self, smiles: list[str]) -> list[torch.Tensor]:
        graph = featurize_smiles(smiles).to(self.x_embedding1.weight.device)
        return list(self(graph).split(graph.counts))
