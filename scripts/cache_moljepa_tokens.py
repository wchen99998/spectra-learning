"""Cache the released model's 13 latent modality tokens, with exact public-API parity."""
import importlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

MODEL='Flogrammer/Mol-JEPA'
REVISION='4c912b450175f31b5ba913a5dc921c03b27b985a'
ROOT=Path('/results/moljepa_attention_latest/mol_tokens')
TOPOLOGY=Path('/results/ftjepa530500_trainable_molebert_flare/graphs')

class GraphCollator:
    def __init__(self,module_name):
        self.module_name=module_name
        self.featurizer=None
    def __call__(self,smiles):
        mod=importlib.import_module(self.module_name)
        if self.featurizer is None:self.featurizer=mod.GraphFeaturizer()
        data=[]
        for s in smiles:
            features=self.featurizer.encode(s)
            if features['x'].shape[1]!=82 or features['edge_attr'].shape[1]!=17:
                raise ValueError('Mol-JEPA graph feature dimensions differ from the checkpoint')
            item=mod.MultimodalData()
            item.add_graph('graph',x=features['x'],edge_index=features['edge_index'].long(),edge_attr=features['edge_attr'])
            data.append(item)
        return mod.from_data_list(mod.Batch,data,follow_batch=['graph_x'])

@torch.no_grad()
def encode(model,batch):
    batch=batch.to(model.device)
    modalities=[f"{m['name']}_x_ptr" if f"{m['name']}_x_ptr" in batch else None for m in model.model.modalities_spec]
    count=len(batch['graph_x_ptr'])-1
    active=torch.stack([batch[k][1:]-batch[k][:-1] if k is not None else torch.zeros(count,device=model.device) for k in modalities]).T>0
    full=model.model.encode(batch,modalities,active)
    return model.model.predict(full,active,active)[2]

def run(shard,shards,smoke):
    from transformers import AutoModel
    torch.set_num_threads(1)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION,trust_remote_code=True).eval().cuda()
    model.requires_grad_(False)
    module=model.__class__.__module__
    probe=['CCO','CC(=O)Oc1ccccc1C(=O)O','[Na+].[Cl-]','C']
    collator=GraphCollator(module)
    expected=model(probe).embeddings
    actual=encode(model,collator(probe))
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    if actual.shape!=(4,13,512) or not torch.isfinite(actual).all():raise ValueError('Invalid latent tokens')
    if torch.allclose(actual[0],actual[1]):raise ValueError('Molecular tokens lost input dependence')
    parity={'public_api_max_abs':float((actual-expected).abs().max()),'shape':list(actual.shape),'model':MODEL,'revision':REVISION}
    ROOT.mkdir(parents=True,exist_ok=True)
    if smoke:
        (ROOT/'preflight.json').write_text(json.dumps(parity,indent=2))
        return parity
    smiles=json.loads((ROOT/'smiles.json').read_text())
    lo=len(smiles)*shard//shards;hi=len(smiles)*(shard+1)//shards
    meta=ROOT/f'shard_{shard}.json'
    if meta.exists():return json.loads(meta.read_text())
    path=ROOT/f'tokens_{shard}.npy'
    progress=ROOT/f'progress_{shard}.json'
    start=json.loads(progress.read_text())['rows'] if progress.exists() else 0
    cache=np.lib.format.open_memmap(path,mode='r+' if start else 'w+',dtype=np.float16,shape=(hi-lo,13,512))
    loader=DataLoader(smiles[lo+start:hi],batch_size=128,num_workers=12,collate_fn=collator,multiprocessing_context='spawn',prefetch_factor=2)
    written=start;t0=time.monotonic();last_commit=t0
    for batch in loader:
        values=encode(model,batch).float().cpu().numpy()
        if not np.isfinite(values).all():raise ValueError('Non-finite molecule embeddings')
        cache[written:written+len(values)]=values;written+=len(values)
        if written%4096==0 or written==hi-lo:
            cache.flush()
            data={'shard':shard,'rows':written,'total':hi-lo,'seconds':time.monotonic()-t0,'rows_per_second':(written-start)/max(time.monotonic()-t0,1)}
            progress.write_text(json.dumps(data));print(json.dumps(data),flush=True)
            if time.monotonic()-last_commit>600:
                import modal
                modal.Volume.from_name('spectra-massspecgym-maccs-probe').commit();last_commit=time.monotonic()
    result={**parity,'shard':shard,'shards':shards,'lo':lo,'hi':hi,'total_molecules':len(smiles),'complete':True}
    cache.flush();meta.write_text(json.dumps(result,indent=2))
    return result
