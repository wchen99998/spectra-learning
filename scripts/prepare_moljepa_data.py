"""Fix a 5,000-spectrum validation split and enumerate complete evaluation pools."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/results/moljepa_attention_latest')
DATA=Path('/results/jepa_fps_ftjepa_1p8b_step530500_flare_raw')
BANK=Path('/results/jepa_fps_ftjepa_1p8b_step125000_candidate_v1/cache/bank_s0_k32_smoke0.json')

def run():
    metadata=ROOT/'data.json'
    if metadata.exists():return json.loads(metadata.read_text())
    frame=pd.read_csv(DATA/'msg15/MassSpecGym1.5.tsv',sep='\t',usecols=['smiles','fold'])
    smiles=frame.smiles.to_numpy();fold=frame.fold.to_numpy()
    train=np.flatnonzero(fold=='train');val=np.flatnonzero(fold=='val');test=np.flatnonzero(fold=='test')
    val=np.sort(np.random.default_rng(0).choice(val,5000,replace=False))
    bank=json.loads(BANK.read_text());pool=json.loads((DATA/'msg15/MassSpecGym1.5_retrieval_candidates_formula.json').read_text())
    needed=set(smiles[train]);needed.update(c for cs in bank.values() for c in cs)
    for row in np.concatenate([val,test]):
        if smiles[row] not in pool[smiles[row]]:raise ValueError('Truth is missing from candidate pool')
        needed.update(pool[smiles[row]])
    informative=np.array([i for i in train if bank[smiles[i]]],dtype=np.int64)
    (ROOT/'mol_tokens').mkdir(parents=True,exist_ok=True)
    universe=sorted(needed)
    (ROOT/'mol_tokens/smiles.json').write_text(json.dumps(universe))
    np.savez(ROOT/'splits.npz',train=informative,validation=val,test=test)
    result={'train':len(train),'informative_train':len(informative),'validation_total':int((fold=='val').sum()),'validation_selection':5000,'test':len(test),'molecules':len(universe),'validation_seed':0,'validation_rows':val.tolist(),'validation_sha256':hashlib.sha256(val.tobytes()).hexdigest(),'data_root':str(DATA),'negative_bank':str(BANK),'reduction':'uniform bidirectional MaxSim','candidate_training':'truth + eight random negatives from original planar-filtered bank32'}
    metadata.write_text(json.dumps(result,indent=2));return {k:v for k,v in result.items() if k!='validation_rows'}
