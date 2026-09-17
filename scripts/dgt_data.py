"""Aligned candidate chemistry and frozen spectra/Mol-JEPA cache loading."""
from pathlib import Path
import json
import numpy as np
import torch
from dgt_moljepa import GraphBatch

ROOT = Path('/results/dgt_moljepa_atom_latest')

def pack_graphs(graphs, moljepa, device='cuda'):
    n = max(len(g[0]) for g in graphs); e = max(1,max(len(g[1]) for g in graphs)); b = len(graphs)
    atoms = np.zeros((b,n,131),np.float32); bonds = np.zeros((b,e,2,57),np.float32)
    av = np.zeros((b,n),bool); ev = np.zeros((b,e),bool); edges = np.zeros((b,e,2),np.int64)
    fractions = np.full((b,n,6),-1,np.float32)
    for i,(x,ei,ea,af) in enumerate(graphs):
        atoms[i,:len(x)] = x; av[i,:len(x)] = True
        bonds[i,:len(ei)] = ea.reshape(-1,2,57); ev[i,:len(ei)] = True; edges[i,:len(ei)] = ei
        fractions[i,:len(x)] = af
    return GraphBatch(*[torch.from_numpy(v).to(device) for v in (atoms,av,bonds,ev,edges)],moljepa,torch.from_numpy(fractions).to(device))

class GraphCache:
    def __init__(self):
        self.shards=[]
        for i in range(8):
            path=ROOT/f'graphs/{i}'; meta=json.loads((path/'metadata.json').read_text())
            if not meta['complete'] or (meta['atom_dim'],meta['directed_bond_dim']) != (131,57):
                raise ValueError('Incomplete or wrong chemical feature cache')
            self.shards.append((meta['lo'],meta['hi'],
                np.load(path/'node_offsets.npy',mmap_mode='r'),np.load(path/'edge_offsets.npy',mmap_mode='r'),
                np.memmap(path/'nodes.bin',mode='r',dtype=np.float32,shape=(meta['atoms'],131)),
                np.memmap(path/'edges.bin',mode='r',dtype=np.int16,shape=(meta['bonds'],2)),
                np.memmap(path/'bonds.bin',mode='r',dtype=np.float32,shape=(meta['bonds']*2,57)),
                np.memmap(path/'fractions.bin',mode='r',dtype=np.float32,shape=(meta['atoms'],6))))
        self.ends=np.array([s[1] for s in self.shards])
        self.counts=np.concatenate([np.diff(s[2]) for s in self.shards])
    def __getitem__(self,i):
        lo,hi,no,eo,x,ei,ea,af=self.shards[np.searchsorted(self.ends,i,side='right')]
        j=i-lo; a,b=no[j:j+2]; e,f=eo[j:j+2]
        return x[a:b],ei[e:f],ea[2*e:2*f],af[a:b]

def load_data():
    from run_moljepa_attention import Data
    class AtomData(Data):
        def __init__(self):
            super().__init__()
            old_count=len(self.smiles)
            self.smiles=json.loads((ROOT/'smiles.json').read_text())
            self.row_of={s:i for i,s in enumerate(self.smiles)}
            for k in range(4):
                path=ROOT/'extra_tokens'; meta=json.loads((path/f'shard_{k}.json').read_text())
                if not meta['complete'] or meta['revision']!='4c912b450175f31b5ba913a5dc921c03b27b985a':
                    raise ValueError('Incomplete extended Mol-JEPA cache')
                self.shards.append((old_count+meta['lo'],old_count+meta['hi'],np.load(path/f'tokens_{k}.npy',mmap_mode='r')))
            bank=json.loads(Path(json.loads((ROOT/'data.json').read_text())['hard_bank']).read_text())
            self.hard_bank={s:np.array([self.row_of[c] for c in cs],np.int32) for s,cs in bank.items()}
            self.graphs=GraphCache()
            self.peak_fractions=np.load('/results/moljepa_attention_latest/mass/peak_fractions.npy',mmap_mode='r')
            from rdkit import Chem
            self.warmup_train=np.flatnonzero(self.frame.fold.to_numpy()=='train')
            keys={s:Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for s in set(self.query[self.warmup_train])}
            key_ids={s:i for i,s in enumerate(sorted(set(keys.values())))}
            self.structure_ids=np.full(len(self.query),-1,np.int64)
            self.structure_ids[self.warmup_train]=[key_ids[keys[s]] for s in self.query[self.warmup_train]]
        def graph(self,ids):
            z,_,_=self.molecule(ids)
            return pack_graphs([self.graphs[int(i)] for i in ids],z)
        def selected(self,rows,rng,hard=False):
            count=32 if hard else 8
            selected=np.zeros((len(rows),count+1),np.int32); valid=np.zeros(selected.shape,bool)
            bank=self.hard_bank if hard else self.bank
            for k,row in enumerate(rows):
                negatives=bank[self.query[row]]
                decoys=rng.choice(negatives,min(count,len(negatives)),replace=False)
                selected[k,0]=self.truth_id[row]; selected[k,1:len(decoys)+1]=decoys
                valid[k,:len(decoys)+1]=True
            return selected,torch.from_numpy(valid).cuda()
    return AtomData()
