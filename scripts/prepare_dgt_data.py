"""Extend frozen Mol-JEPA coverage and cache all existing chemical feature families."""
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from dgt_chemistry import features

ROOT = Path('/results/dgt_moljepa_atom_latest')
OLD = Path('/results/moljepa_attention_latest')
BANK = Path('/results/jepa_fps_ftjepa_1p8b_step125000_mining_v2/cache/bank_s0_k128_smoke0.json')

def prepare():
    ROOT.mkdir(exist_ok=True)
    previous = json.loads((OLD/'mol_tokens/smiles.json').read_text())
    bank = json.loads(BANK.read_text())
    added = sorted({s for values in bank.values() for s in values} - set(previous))
    (ROOT/'smiles.json').write_text(json.dumps(previous + added))
    (ROOT/'extra_tokens').mkdir(exist_ok=True)
    (ROOT/'extra_tokens/smiles.json').write_text(json.dumps(added))
    result = {'original_molecules': len(previous), 'additional_molecules': len(added),
              'molecules': len(previous)+len(added), 'hard_bank': str(BANK),
              'validation_sha256': json.loads((OLD/'data.json').read_text())['validation_sha256']}
    (ROOT/'data.json').write_text(json.dumps(result, indent=2))
    return result

def cache(shard: int, shards: int = 8):
    out = ROOT/f'graphs/{shard}'; out.mkdir(parents=True, exist_ok=True)
    meta = out/'metadata.json'
    if meta.exists(): return json.loads(meta.read_text())
    smiles = json.loads((ROOT/'smiles.json').read_text())
    lo, hi = len(smiles)*shard//shards, len(smiles)*(shard+1)//shards
    node_offsets, edge_offsets = [0], [0]
    start = time.monotonic()
    with (out/'nodes.bin').open('wb') as nodes, (out/'bonds.bin').open('wb') as bonds, (out/'edges.bin').open('wb') as edges, (out/'fractions.bin').open('wb') as fractions, mp.get_context('spawn').Pool(16) as pool:
        for k, (x, ei, ea, af) in enumerate(pool.imap(features, smiles[lo:hi], chunksize=128)):
            nodes.write(x.tobytes()); bonds.write(ea.tobytes()); edges.write(ei.tobytes()); fractions.write(af.tobytes())
            node_offsets.append(node_offsets[-1]+len(x)); edge_offsets.append(edge_offsets[-1]+len(ei))
            if (k+1)%16384 == 0:
                progress = {'rows': k+1, 'total': hi-lo, 'seconds': time.monotonic()-start}
                (out/'progress.json').write_text(json.dumps(progress)); print(json.dumps(progress), flush=True)
    np.save(out/'node_offsets.npy', np.array(node_offsets, dtype=np.int64))
    np.save(out/'edge_offsets.npy', np.array(edge_offsets, dtype=np.int64))
    result = {'lo': lo, 'hi': hi, 'atoms': node_offsets[-1], 'bonds': edge_offsets[-1],
              'atom_dim': x.shape[1], 'directed_bond_dim': ea.shape[1], 'complete': True,
              'seconds': time.monotonic()-start}
    meta.write_text(json.dumps(result, indent=2)); return result
