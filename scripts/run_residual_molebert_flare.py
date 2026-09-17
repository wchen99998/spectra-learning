"""Continue original FLARE with a zero-initialized pretrained atom residual."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from molebert_atom_encoder import MOLEBERT_CHECKPOINT_SHA256
from residual_molebert_flare import ResidualGraphCache, residual_model_class


def load_stage(source: Path, molecular_checkpoint: Path):
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location('residual_flare_stage', source)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    original_class = stage.fm.MolGNN
    model_class = residual_model_class(original_class, molecular_checkpoint, stage.fm.ATOM_FDIM, stage.fm.BOND_FDIM)
    stage.fm.MolGNN = model_class
    return stage, original_class, model_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--original-graphs', required=True, type=Path)
    parser.add_argument('--pretrained-graphs', required=True, type=Path)
    parser.add_argument('--molecular-checkpoint', required=True, type=Path)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    if hashlib.sha256(args.molecular_checkpoint.read_bytes()).hexdigest() != MOLEBERT_CHECKPOINT_SHA256:
        raise ValueError('Unexpected Mole-BERT checkpoint')
    metadata = json.loads((args.original_graphs / 'metadata.json').read_text())
    if not metadata['complete'] or not metadata['all_graph_ordering_verified']:
        raise ValueError('Both original and pretrained graph features must be complete and aligned')
    cache = ResidualGraphCache(args.original_graphs, args.pretrained_graphs)
    stage, _, _ = load_stage(args.source, args.molecular_checkpoint)

    def graph_cache(smiles_list, *unused, **options):
        if any(s not in cache for s in smiles_list):
            raise ValueError('Missing required molecular graph')
        return cache

    stage.fm.build_graph_cache = graph_cache
    sys.argv = [str(args.source), *forwarded]
    stage.main()


if __name__ == '__main__':
    main()
