"""Run the original FLARE recipe with frozen JEPA and trainable Mole-BERT."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from molebert_atom_encoder import MOLEBERT_CHECKPOINT_SHA256
from trainable_molebert_flare import MoleBERTGraphCache, TrainableMoleBERT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--graph-cache', required=True, type=Path)
    parser.add_argument('--molecular-checkpoint', required=True, type=Path)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    if hashlib.sha256(args.molecular_checkpoint.read_bytes()).hexdigest() != MOLEBERT_CHECKPOINT_SHA256:
        raise ValueError('Unexpected pretrained Mole-BERT checkpoint')
    metadata = json.loads((args.graph_cache / 'metadata.json').read_text())
    if not metadata['complete'] or metadata['format'] != 'molebert_discrete_graphs_v1':
        raise ValueError('Expected a complete discrete graph cache, not cached embeddings')
    cache = MoleBERTGraphCache(args.graph_cache)
    sys.path.insert(0, str(args.source.parent))
    spec = importlib.util.spec_from_file_location('trainable_flare_stage', args.source)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    hybrid = stage if args.source.name == '09_jepa_flare.py' else stage.hybrid

    def graph_cache(smiles_list, *unused, **options):
        for smiles in smiles_list:
            if smiles not in cache:
                raise ValueError(f'Missing molecular graph: {smiles}')
        print(f'Discrete Mole-BERT graph cache: {len(cache)} molecules', flush=True)
        return cache

    def molecule_model(hidden, out_dim, layers, dropout, feature_set='basic'):
        if feature_set != 'basic':
            raise ValueError('Mole-BERT uses its original pretrained graph features')
        return TrainableMoleBERT(args.molecular_checkpoint, out_dim)

    hybrid.fm.MolGNN = molecule_model
    hybrid.fm.build_graph_cache = graph_cache
    sys.argv = [str(args.source), *forwarded]
    print('Frozen JEPA; original FLARE readout; full Mole-BERT backbone fine-tuning', flush=True)
    stage.main()


if __name__ == '__main__':
    main()
