"""Compact original retained fractions for the enlarged molecular universe."""
from pathlib import Path
import json
import numpy as np
import torch
from joblib import Parallel,delayed
from flare_mol_agreement import atom_retained_fractions
ROOT=Path('/results/moljepa_attention_latest')
OLD=Path('/results/jepa_fps_ftjepa_1p8b_step530500_upstream_agreement/cache')

def run():
    out=ROOT/'mass';out.mkdir(exist_ok=True)
    meta=out/'metadata.json'
    if meta.exists():return json.loads(meta.read_text())
    smiles=json.loads((ROOT/'mol_tokens/smiles.json').read_text())
    cached=torch.load(OLD/'atom_fractions_F6.pt',map_location='cpu',weights_only=False)
    missing=[s for s in smiles if s not in cached]
    print(json.dumps({'cached':len(cached),'missing':len(missing)}),flush=True)
    values=Parallel(n_jobs=16,batch_size=256)(delayed(atom_retained_fractions)(s,6) for s in missing)
    for s,v in zip(missing,values,strict=True):
        if v is None:raise ValueError(f'Missing molecular fractions for {s}')
        cached[s]=torch.from_numpy(v)
    sizes=np.array([len(cached[s]) for s in smiles],dtype=np.int64)
    offsets=np.concatenate(([0],sizes.cumsum()));np.save(out/'offsets.npy',offsets)
    fractions=np.lib.format.open_memmap(out/'atom_fractions.npy',mode='w+',dtype=np.float32,shape=(int(offsets[-1]),6))
    for i,s in enumerate(smiles):fractions[offsets[i]:offsets[i+1]]=cached[s]
    fractions.flush()
    fp=np.load(OLD/'peak_retained_fraction_f32.npy')
    if fp.shape!=(231104,127) or not np.isfinite(fp).all():raise ValueError('Wrong physical peak fraction cache')
    np.save(out/'peak_fractions.npy',np.concatenate((fp,np.zeros((len(fp),1),dtype=np.float32)),axis=1))
    result={'molecules':len(smiles),'atoms':int(offsets[-1]),'max_atoms':int(sizes.max()),'fraction_slots':6,'new_molecules':len(missing),'source':'original upstream incident-bridge retained fractions','cls_bonus':'zero; CLS remains valid in the readout and reductions','complete':True}
    meta.write_text(json.dumps(result,indent=2));return result
