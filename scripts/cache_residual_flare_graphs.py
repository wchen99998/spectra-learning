"""Cache the unchanged original FLARE chemical features alongside Mole-BERT graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from torch.utils.data import DataLoader

from trainable_molebert_flare import MoleBERTGraphCache


def featurize(values):
    import flare_mol
    return [(s, flare_mol.mol_to_graph(s)) for s in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--pretrained-graphs', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import flare_mol
    original = MoleBERTGraphCache(args.pretrained_graphs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / 'metadata.json').exists():
        raise FileExistsError('Completed residual chemical cache already exists')
    nodes = np.lib.format.open_memmap(args.out_dir / 'nodes.npy', mode='w+', dtype=np.float32,
                                     shape=(len(original.nodes), flare_mol.ATOM_FDIM))
    edges = np.lib.format.open_memmap(args.out_dir / 'edge_attrs.npy', mode='w+', dtype=np.float32,
                                     shape=(len(original.edges), flare_mol.BOND_FDIM))
    loader = DataLoader(original.smiles, batch_size=256, num_workers=8, collate_fn=featurize,
                        multiprocessing_context='spawn')
    started = time.monotonic()
    i = 0
    for batch in loader:
        for smiles, graph in batch:
            if graph is None or smiles != original.smiles[i]:
                raise ValueError('Invalid or reordered original chemical graph')
            x, edge_index, edge_attr = graph
            a, b = original.node_offsets[i:i + 2]
            e, f = original.edge_offsets[i:i + 2]
            if len(x) != b-a or not np.array_equal(edge_index.T, original.edges[e:f]):
                raise ValueError('Original and pretrained graphs disagree on atom/bond ordering')
            nodes[a:b] = x
            edges[e:f] = edge_attr
            i += 1
        if i % 51200 == 0 or i == len(original):
            print(json.dumps({'molecules': i, 'total': len(original), 'seconds': time.monotonic()-started}), flush=True)
    nodes.flush(); edges.flush()
    metadata = json.loads((args.pretrained_graphs / 'metadata.json').read_text())
    metadata.update(format='original_flare_plus_molebert_graphs_v1', atom_dim=flare_mol.ATOM_FDIM,
                    bond_dim=flare_mol.BOND_FDIM, all_graph_ordering_verified=True,
                    original_featurizer_sha256=hashlib.sha256(Path(flare_mol.__file__).read_bytes()).hexdigest(),
                    elapsed_seconds=time.monotonic()-started)
    (args.out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
