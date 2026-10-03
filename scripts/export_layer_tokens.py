"""Export intermediate-layer peak tokens of a frozen JEPA encoder as retrieval token caches.

Each requested block output goes through the encoder's own final norm/adaLN (as the last
layer does) and is written as a new cache root with the same layout as the source cache
(`covariance_embeddings.float16.npy`, `embedding_metadata.json`, `data`/`msg15` links),
so `run_retrieval_jax.py` trains on it unchanged. The last block must reproduce the
source cache within the historical export tolerance (relative L2 < 0.005).
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from scripts.export_retrieval_tokens import INPUT_NAMES, padded_inputs, restore_encoder
from scripts.export_salt_glacier_tokens import BATCH_SIZE, atomic_json, link_original_inputs
from spectra_learning.models.encoder_jax import _active_indices, _gather_pair
from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch


def build_layer_encoder(schema: str, layers: tuple[int, ...]):
    @nnx.jit
    def encode(encoder, mz, intensity, valid, precursor, energy, energy_present,
               charge, charge_present, polarity, instrument):
        condition = jax_spectrum_metadata_from_batch({
            'precursor_mz': precursor, 'collision_energy': energy, 'collision_energy_present': energy_present,
            'charge': charge, 'charge_present': charge_present, 'polarity_id': polarity,
            'instrument_family_id': instrument}, schema=schema)
        # Mirrors PeakSetEncoder.__call__ (single path) / forward_with_pair (fastmixer path).
        x, visible, embedding = encoder._embed_peaks(mz, intensity, valid, valid, condition)
        if encoder.use_pair_path:
            if not encoder.use_fastmixer:
                raise ValueError('Only fastmixer pair-path encoders are supported')
            z = encoder.pair_embedder(encoder._quantize_mz(mz), intensity, x, visible, precursor_mz=precursor)
            z = encoder._append_cls_pair_tokens(z)
        x = encoder._append_cls_token(x, embedding if encoder.metadata_conditioning != "adaln_zero" else None)
        token_mask = encoder._append_cls_mask(visible)
        if encoder.use_pair_path:
            idx, compact_mask = _active_indices(token_mask, encoder.pairmixer_fast_max_visible_tokens)
            z = _gather_pair(z, idx)
        outputs = []
        for number, block in enumerate(encoder.blocks, start=1):
            if encoder.use_pair_path:
                x, z = block.fastmixer_compact_call(x, z, idx, compact_mask, token_mask)
            else:
                x = block(x, token_mask, None, embedding)
            if number in layers:
                outputs.append(encoder._finalize(x, embedding).astype(jnp.float32))
        return outputs
    return encode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--layers', required=True, help='comma-separated 1-based block numbers')
    parser.add_argument('--out-prefix', required=True, help='output root is f"{prefix}{layer}_flare_raw"')
    parser.add_argument('--limit', type=int, default=0, help='export only the first rows (smoke tests)')
    args = parser.parse_args(argv)
    source = json.loads((args.source_root / 'embedding_metadata.json').read_text())
    config = json.loads(Path(source['config_metadata_path']).read_text())['task_contract']['config']
    depth = int(config['encoder_num_layers'])
    layers = tuple(sorted({int(v) for v in args.layers.split(',')} | {depth}))
    if not all(1 <= layer <= depth for layer in layers) or not config['encoder_use_cls_token']:
        raise ValueError(f'Layers must lie in 1..{depth} of a CLS-token encoder')
    with np.load(Path(source['data_root']) / 'prepared_massspecgym.npz', allow_pickle=False) as saved:
        data = {name: saved[name] for name in INPUT_NAMES}
    reference = np.load(args.source_root / 'covariance_embeddings.float16.npy', mmap_mode='r')
    count = args.limit or len(reference)
    shape = (count, *reference.shape[1:])
    jax.config.update('jax_default_matmul_precision', 'highest')
    encoder = restore_encoder(config, Path(source['checkpoint']))
    encode = build_layer_encoder(config['encoder_metadata_schema'], layers)

    rows = np.arange(min(BATCH_SIZE, count))
    final = np.asarray(encode(encoder, *padded_inputs(data, rows))[-1], np.float16)[:len(rows), :-1]
    valid = data['peak_valid_mask'][rows]
    expected = np.asarray(reference[rows], np.float32)[valid]
    relative = float(np.linalg.norm(final[valid].astype(np.float32) - expected) / np.linalg.norm(expected))
    print(json.dumps({'stage': 'parity', 'layer': depth, 'valid_relative_l2': relative}), flush=True)
    if not np.isfinite(relative) or relative > .005:
        raise ValueError(f'Final layer does not reproduce the source cache: relative L2 {relative}')

    exported = [layer for layer in layers if layer != depth]
    roots = {layer: Path(f'{args.out_prefix}{layer}_flare_raw') for layer in exported}
    arrays = {}
    for layer, root in roots.items():
        root.mkdir(parents=True, exist_ok=True)
        link_original_inputs(root, args.source_root)
        arrays[layer] = np.lib.format.open_memmap(root / 'covariance_embeddings.float16.partial.npy',
                                                  mode='w+', dtype=np.float16, shape=shape)
    started = time.monotonic()
    for start in range(0, count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, count)
        outputs = encode(encoder, *padded_inputs(data, np.arange(start, end)))
        for layer, value in zip(layers, outputs):
            if layer in arrays:
                block = np.asarray(value, np.float16)[:end - start, :-1]
                if not np.isfinite(block).all():
                    raise ValueError(f'Non-finite float16 tokens at layer {layer}, rows {start}:{end}')
                arrays[layer][start:end] = block
        if (start // BATCH_SIZE + 1) % 128 == 0 or end == count:
            print(json.dumps({'stage': 'export', 'rows': end, 'total': count,
                              'seconds': round(time.monotonic() - started, 1)}), flush=True)
    for layer, root in roots.items():
        arrays[layer].flush()
        del arrays[layer]
        (root / 'covariance_embeddings.float16.partial.npy').replace(root / 'covariance_embeddings.float16.npy')
        atomic_json(root / 'embedding_metadata.json', {
            **source, 'samples': count, 'embedding_shape': list(shape), 'encoder_layer': layer,
            'encoder_depth': depth, 'source_embedding_root': str(args.source_root),
            'final_layer_parity_relative_l2': relative, 'extraction_seconds': time.monotonic() - started})
    print(json.dumps({'stage': 'complete', 'layers': exported, 'shape': shape}), flush=True)


if __name__ == '__main__':
    main()
