"""Run the original four-stage FLARE recipe with pretrained atom vectors as extra inputs."""
import argparse
import importlib.util
from pathlib import Path
import sys

from molebert_feature_flare import MoleBERTFeatureGraphs, feature_model_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    cache = MoleBERTFeatureGraphs(
        Path('/results/ftjepa530500_residual_molebert_flare/graphs'),
        Path('/results/ftjepa530500_trainable_molebert_flare/graphs'),
        Path('/results/ftjepa530500_frozen_molebert_flare/atoms'))
    sys.path.insert(0, str(args.source.parent))
    spec = importlib.util.spec_from_file_location('feature_flare_stage', args.source)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    hybrid = stage if args.source.name == '09_jepa_flare.py' else stage.hybrid
    hybrid.fm.MolGNN = feature_model_class(hybrid.fm.MolGNN)

    def graph_cache(smiles_list, *unused, **options):
        if any(s not in cache for s in smiles_list):
            raise ValueError('Missing required augmented molecular graph')
        print(f'Original chemical features + frozen Mole-BERT: {len(cache)} molecules, 349 atom inputs', flush=True)
        return cache

    hybrid.fm.build_graph_cache = graph_cache
    sys.argv = [str(args.source), *forwarded]
    stage.main()


if __name__ == '__main__':
    main()
