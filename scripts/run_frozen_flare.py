"""Keep the recorded FLARE spectrum readout and match frozen pretrained atoms."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from frozen_flare_adapter import AtomCompatibility, FrozenAtomCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--atom-cache', type=Path, required=True)
    parser.add_argument('--expected-checkpoint-sha256', required=True)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    cache = FrozenAtomCache(args.atom_cache)
    metadata = json.loads((args.atom_cache / 'metadata.json').read_text())
    if metadata['checkpoint_sha256'] != args.expected_checkpoint_sha256:
        raise ValueError('Atom cache does not match the expected pretrained checkpoint SHA-256')
    sys.path.insert(0, str(args.source.parent))
    spec = importlib.util.spec_from_file_location('frozen_flare_stage', args.source)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    hybrid = stage if args.source.name == '09_jepa_flare.py' else stage.hybrid

    def graph_cache(smiles_list, *unused, **options):
        for smiles in smiles_list:
            if smiles not in cache:
                raise ValueError(f'Missing frozen candidate atoms: {smiles}')
        print(f'Frozen atom cache: {len(cache)} molecules, {cache.in_dim} dimensions', flush=True)
        return cache

    def molecule_compatibility(hidden, out_dim, layers, dropout, feature_set='basic'):
        if feature_set != 'basic':
            raise ValueError('Frozen compatibility experiment uses pretrained atoms without added features')
        return AtomCompatibility(cache.in_dim, out_dim)

    hybrid.fm.MolGNN = molecule_compatibility
    hybrid.fm.build_graph_cache = graph_cache
    sys.argv = [str(args.source), *forwarded]
    print('Original FLARE spectrum readout; deterministic linear map of frozen pretrained atoms', flush=True)
    stage.main()


if __name__ == '__main__':
    main()
