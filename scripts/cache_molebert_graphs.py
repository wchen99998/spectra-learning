"""Cache discrete Mole-BERT inputs for the already verified candidate universe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from torch.utils.data import DataLoader

from molebert_atom_encoder import featurize_smiles


def write_npy(raw: Path, target: Path, shape: tuple[int, int]):
    with raw.open('rb') as source, target.open('wb') as output:
        np.lib.format.write_array_header_2_0(output, {
            'descr': np.lib.format.dtype_to_descr(np.dtype(np.uint8)),
            'fortran_order': False, 'shape': shape,
        })
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def build_cache(smiles: list[str], output: Path, *, batch_size: int, workers: int):
    output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(smiles, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=featurize_smiles,
                        multiprocessing_context='spawn' if workers else None)
    node_counts, edge_counts = [], []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='molebert-graphs-') as directory:
        paths = {name: Path(directory) / name for name in ['nodes', 'edges', 'edge_attrs']}
        with paths['nodes'].open('wb') as nodes, paths['edges'].open('wb') as edges, paths['edge_attrs'].open('wb') as attrs:
            for step, graph in enumerate(loader):
                counts = np.asarray(graph.counts, dtype=np.int64)
                if np.any(counts < 1) or np.any(counts > 128):
                    raise ValueError('Graph cache requires 1..128 atoms; no truncation')
                offsets = np.concatenate(([0], np.cumsum(counts)))
                edge_index = graph.edge_index.numpy().T
                owner = np.searchsorted(offsets[1:], edge_index[:, 0], side='right')
                local_edges = edge_index - offsets[owner, None]
                if np.any(local_edges < 0) or np.any(local_edges >= counts[owner, None]):
                    raise ValueError('Edges must connect atoms within their own molecule')
                for handle, array in [(nodes, graph.x.numpy()), (edges, local_edges),
                                      (attrs, graph.edge_attr.numpy())]:
                    handle.write(np.ascontiguousarray(array, dtype=np.uint8).tobytes())
                node_counts.extend(counts.tolist())
                edge_counts.extend(np.bincount(owner, minlength=len(counts)).tolist())
                if step % 200 == 0 or len(node_counts) == len(smiles):
                    print(json.dumps({'molecules':len(node_counts), 'total':len(smiles),
                                      'elapsed_seconds':time.monotonic()-started}), flush=True)
        for name, count in [('nodes', sum(node_counts)), ('edges', sum(edge_counts)), ('edge_attrs', sum(edge_counts))]:
            write_npy(paths[name], output / (name + '.npy'), (count, 2))
    np.save(output / 'node_offsets.npy', np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(node_counts, dtype=np.int64))))
    np.save(output / 'edge_offsets.npy', np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(edge_counts, dtype=np.int64))))
    (output / 'smiles.json').write_text(json.dumps(smiles))
    return {'molecules':len(smiles), 'atoms':sum(node_counts), 'directed_edges':sum(edge_counts),
            'max_atoms':max(node_counts), 'truncated_molecules':0,
            'elapsed_seconds':time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--universe', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or (args.limit is not None and args.limit < 1):
        parser.error('Invalid batch-size, workers, or limit')
    if (args.out_dir / 'metadata.json').exists():
        raise FileExistsError('Completed graph cache already exists')
    source = json.loads((args.universe / 'metadata.json').read_text())
    if not source['complete'] or source['smoke_limit'] is not None:
        raise ValueError('Source candidate universe must be complete')
    smiles = json.loads((args.universe / 'smiles.json').read_text())
    if len(smiles) != source['required_molecules'] or any(a >= b for a,b in zip(smiles,smiles[1:])):
        raise ValueError('Source candidate universe must be complete, unique, and sorted')
    result = build_cache(smiles[:args.limit] if args.limit else smiles, args.out_dir,
                         batch_size=args.batch_size, workers=args.workers)
    result.update({key:source[key] for key in ['dataset_tsv_sha256','candidate_pools_sha256','negative_banks','validation_rows','split_sizes']})
    result.update(complete=True, format='molebert_discrete_graphs_v1', smoke_limit=args.limit,
                  required_molecules=len(smiles), atom_order='Original RDKit indices; no canonicalization or truncation',
                  cache_contents='Discrete atom and bond features only; NO learned embeddings',
                  featurizer_source_sha256=hashlib.sha256(Path(__file__).with_name('molebert_atom_encoder.py').read_bytes()).hexdigest())
    (args.out_dir / 'metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GRAPH CACHE COMPLETE', json.dumps(result | {'validation_rows':'recorded in metadata'}), flush=True)


if __name__ == '__main__':
    main()
