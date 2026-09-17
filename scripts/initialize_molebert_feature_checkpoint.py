"""Add inactive Mole-BERT input columns to a trained chemical-feature FLARE model."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def extend_checkpoint(checkpoint: dict) -> dict:
    weight = checkpoint['gnn']['node_enc.weight']
    if weight.shape != (512, 49):
        raise ValueError('Expected the original width-512 FLARE with 49 chemical inputs')
    result = dict(checkpoint)
    result['gnn'] = dict(checkpoint['gnn'])
    result['gnn']['node_enc.weight'] = torch.cat((weight, weight.new_zeros(512, 300)), dim=1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = torch.load(args.source, map_location='cpu', weights_only=True)
    converted = extend_checkpoint(original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        torch.save(converted, handle)
    provenance = {
        'source': str(args.source.resolve()),
        'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'source_epoch_zero_based': original['epoch'],
        'source_validation_metrics': original['validation_metrics'],
        'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'change': 'Append 300 zero weight columns; preserve every trained parameter and buffer.',
    }
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
