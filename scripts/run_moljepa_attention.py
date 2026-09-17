"""Direct PyTorch training and full-pool evaluation for the simplified two-readout trial."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from moljepa_attention import ChemicalBatch,RetrievalModel,maxsim,spectrum_tokens

ROOT=Path('/results/moljepa_attention_latest')
RAW=Path('/results/jepa_fps_ftjepa_1p8b_step1190000_flare_raw')
OLD=Path('/results/jepa_fps_ftjepa_1p8b_step530500_flare_raw')
TOPO=Path('/results/ftjepa530500_trainable_molebert_flare/graphs')
CHEM=Path('/results/ftjepa530500_residual_molebert_flare/graphs')

def write_json(path,value):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def save_checkpoint(path,value):
    tmp=path.with_suffix('.partial');torch.save(value,tmp);tmp.replace(path)

def metrics(ranks):
    r=np.asarray(ranks)
    return {'n':len(r),**{f'hit{k}':float((r<=k).mean()) for k in [1,5,10,20]},'mrr':float((1/r).mean()),'median_rank':float(np.median(r))}

class Data:
    def __init__(self,chemical=False,agreement=False):
        self.agreement=agreement
        self.smiles=json.loads((ROOT/'mol_tokens/smiles.json').read_text())
        self.row_of={s:i for i,s in enumerate(self.smiles)}
        self.shards=[]
        for k in range(4):
            meta=json.loads((ROOT/f'mol_tokens/shard_{k}.json').read_text())
            if not meta['complete'] or meta['revision']!='4c912b450175f31b5ba913a5dc921c03b27b985a':raise ValueError('Incomplete or wrong molecular cache')
            self.shards.append((meta['lo'],meta['hi'],np.load(ROOT/f'mol_tokens/tokens_{k}.npy',mmap_mode='r')))
        self.peaks=np.load(RAW/'covariance_embeddings.float16.npy',mmap_mode='r')
        self.cls=np.load(RAW/'cls_embeddings.float16.npy',mmap_mode='r')
        self.mask=np.load(OLD/'data/token_valid_mask.npy',mmap_mode='r')[:,:127]
        if self.peaks.shape!=(231104,127,2048) or self.cls.shape!=(231104,2048):raise ValueError('Wrong spectrum cache shapes')
        self.frame=pd.read_csv(OLD/'msg15/MassSpecGym1.5.tsv',sep='\t',usecols=['smiles','fold'])
        pool=json.loads((OLD/'msg15/MassSpecGym1.5_retrieval_candidates_formula.json').read_text())
        self.query=self.frame.smiles.to_numpy()
        split=np.load(ROOT/'splits.npz');self.train=split['train'];self.val=split['validation'];self.test=split['test']
        self.pool={s:np.array([self.row_of[c] for c in pool[s]],dtype=np.int32) for s in set(self.query[np.concatenate((self.val,self.test))])}
        self.truth_id=np.array([self.row_of.get(s,-1) for s in self.query],dtype=np.int32)
        if (self.truth_id[np.concatenate((self.train,self.val,self.test))]<0).any():raise ValueError('Required truth missing from cache')
        bank=json.loads(Path(json.loads((ROOT/'data.json').read_text())['negative_bank']).read_text())
        self.bank={s:np.array([self.row_of[c] for c in cs],dtype=np.int32) for s,cs in bank.items()}
        self.chemical=chemical
        if agreement:
            self.fraction_offsets=np.load(ROOT/'mass/offsets.npy',mmap_mode='r')
            self.atom_fractions=np.load(ROOT/'mass/atom_fractions.npy',mmap_mode='r')
            self.peak_fractions=np.load(ROOT/'mass/peak_fractions.npy',mmap_mode='r')
        if chemical:
            oldsmiles=json.loads((TOPO/'smiles.json').read_text());oldrow={s:i for i,s in enumerate(oldsmiles)}
            self.old_index=np.array([oldrow.get(s,-1) for s in self.smiles],dtype=np.int32)
            self.nodes=np.load(CHEM/'nodes.npy',mmap_mode='r');self.bonds=np.load(CHEM/'edge_attrs.npy',mmap_mode='r')
            self.noff=np.load(TOPO/'node_offsets.npy',mmap_mode='r');self.eoff=np.load(TOPO/'edge_offsets.npy',mmap_mode='r');self.edges=np.load(TOPO/'edges.npy',mmap_mode='r')
            self.extra_graphs={}
    def fractions(self,ids,cap):
        fractions=np.full((len(ids),cap,6),-1,dtype=np.float32)
        for j,i in enumerate(ids):
            a,b=self.fraction_offsets[i:i+2]
            if b-a>cap:raise ValueError('Agreement atom count exceeds chemical graph size')
            fractions[j,:b-a]=self.atom_fractions[a:b]
        return torch.from_numpy(fractions).cuda()
    def physical_peaks(self,rows):
        return torch.from_numpy(np.array(self.peak_fractions[rows],dtype=np.float32)).cuda()
    def spectrum(self,rows):
        peaks=torch.from_numpy(np.array(self.peaks[rows],dtype=np.float32)).cuda()
        cls=torch.from_numpy(np.array(self.cls[rows],dtype=np.float32)).cuda()
        mask=torch.from_numpy(np.array(self.mask[rows],dtype=bool)).cuda()
        return spectrum_tokens(peaks,cls,mask)
    def molecule(self,ids):
        out=np.empty((len(ids),13,512),dtype=np.float32)
        for lo,hi,values in self.shards:
            use=(ids>=lo)&(ids<hi)
            if use.any():out[use]=values[ids[use]-lo]
        z=torch.from_numpy(out).cuda();valid=torch.ones(len(ids),13,dtype=torch.bool,device='cuda')
        return z,valid,self.chemistry(ids) if self.chemical else None
    def chemistry(self,ids):
        graphs=[]
        for i in ids:
            j=self.old_index[i]
            if j<0:
                if i not in self.extra_graphs:
                    from flare_mol import mol_to_graph
                    graph=mol_to_graph(self.smiles[i])
                    if graph is None:raise ValueError(f'Invalid chemistry: {self.smiles[i]}')
                    self.extra_graphs[i]=graph
                graphs.append(self.extra_graphs[i])
            else:
                a,b=self.noff[j:j+2];e,f=self.eoff[j:j+2]
                graphs.append((self.nodes[a:b],self.edges[e:f].T,self.bonds[e:f]))
        cap=max(len(g[0]) for g in graphs)
        if self.agreement:
            counts=self.fraction_offsets[ids+1]-self.fraction_offsets[ids]
            if not np.array_equal(counts,[len(g[0]) for g in graphs]):raise ValueError('Chemical and mass-cache atom counts differ')
        # Do not truncate molecules added by the larger validation subset.
        x=np.zeros((len(ids),cap,49),dtype=np.float32);valid=np.zeros((len(ids),cap),dtype=bool)
        edge=[];bond=[]
        for k,(atoms,edges,bonds) in enumerate(graphs):
            x[k,:len(atoms)]=atoms;valid[k,:len(atoms)]=True
            edge.append(edges.astype(np.int64)+k*cap);bond.append(bonds)
        return ChemicalBatch(torch.from_numpy(x).cuda(),torch.from_numpy(valid).cuda(),torch.from_numpy(np.concatenate(edge,axis=1)).cuda(),torch.from_numpy(np.concatenate(bond,axis=0)).cuda())
    def candidates(self,rows,rng):
        selected=np.zeros((len(rows),9),dtype=np.int32);valid=np.zeros((len(rows),9),dtype=bool)
        for k,row in enumerate(rows):
            negatives=self.bank[self.query[row]]
            decoys=rng.choice(negatives,min(8,len(negatives)),replace=False)
            selected[k,0]=self.truth_id[row];selected[k,1:len(decoys)+1]=decoys
            valid[k,:len(decoys)+1]=True
        unique,mapping=np.unique(selected,return_inverse=True)
        return unique,torch.from_numpy(mapping.reshape(selected.shape)).cuda(),torch.from_numpy(valid).cuda()

@torch.no_grad()
def evaluate(model,data,rows,out_dir,name,agreement_active=True):
    model.eval();start=time.monotonic()
    needed=np.unique(np.concatenate([data.pool[s] for s in set(data.query[rows])]))
    lookup=np.full(len(data.smiles),-1,dtype=np.int32);lookup[needed]=np.arange(len(needed))
    encoded=torch.empty((len(needed),13,512),device='cuda',dtype=torch.float16)
    mass=model.mass if agreement_active else None
    if mass is not None:
        cap=int(np.diff(data.fraction_offsets)[needed].max())
        atom_weights=torch.zeros(len(needed),13,cap,device='cuda',dtype=torch.float16)
        fractions=torch.full((len(needed),cap,6),-1,device='cuda',dtype=torch.float32)
        atom_valid=torch.zeros(len(needed),cap,device='cuda',dtype=torch.bool)
    for i in range(0,len(needed),256):
        ids=needed[i:i+256];z,valid,chem=data.molecule(ids)
        with torch.autocast('cuda',dtype=torch.bfloat16):value=model.molecule(z,valid,chem)
        encoded[i:i+len(ids)]=value.half()
        if mass is not None:
            a=chem.valid.shape[1]
            with torch.autocast('cuda',dtype=torch.bfloat16):weights=mass.atom_weights(value,chem)
            atom_weights[i:i+len(ids),:,:a]=weights.half()
            fractions[i:i+len(ids)]=data.fractions(ids,cap)
            atom_valid[i:i+len(ids),:a]=chem.valid
    stable=[];optimistic=[];pairs=[];records=[]
    for begin in range(0,len(rows),32):
        chunk=rows[begin:begin+32];x,mask=data.spectrum(chunk)
        with torch.autocast('cuda',dtype=torch.bfloat16):peaks=model.spectrum(x,mask)
        peaks=peaks.half().float()
        for k,row in enumerate(chunk):
            candidates=data.pool[data.query[row]]
            # The complete formula pool is scored, preserving its supplied order.
            scores=[]
            for c in range(0,len(candidates),512):
                idx=torch.from_numpy(lookup[candidates[c:c+512]].astype(np.int64)).cuda()
                bonus=None if mass is None else mass.bonus(data.physical_peaks([row]),fractions[idx][None],atom_valid[idx][None],atom_weights[idx][None])
                scores.append(maxsim(peaks[k:k+1],mask[k:k+1],encoded[idx][None].float(),bonus=bonus)[0])
            score=torch.cat(scores)
            if not torch.isfinite(score).all():raise ValueError('Nonfinite evaluation score')
            truth=int(np.flatnonzero(candidates==data.truth_id[row])[0])
            rank=1+int((score>score[truth]).sum());rank_stable=rank+int((score[:truth]==score[truth]).sum())
            pred=int(candidates[int(score.argmax())]);stable.append(rank_stable);optimistic.append(rank)
            pairs.append((data.query[row],data.smiles[pred]))
            records.append({'row':int(row),'stable_rank':rank_stable,'optimistic_rank':rank,'prediction':data.smiles[pred],'candidates':len(candidates)})
    result={'stable':metrics(stable),'optimistic':metrics(optimistic),'seconds':time.monotonic()-start,'unique_candidates':len(needed)}
    write_json(out_dir/f'{name}_ranks.json',records)
    del encoded
    return result,pairs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chemical',action='store_true');parser.add_argument('--agreement',action='store_true');parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--patience',type=int,default=10);parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--eval-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(8);torch.manual_seed(0);np.random.seed(0)
    torch.set_float32_matmul_precision('high')
    tag='chemical_cross_attention' if args.chemical else 'moljepa_only'
    parent=ROOT/tag/'best_model.pt'
    if args.agreement:tag+='_agreement'
    out=ROOT/tag;out.mkdir(exist_ok=True,parents=True)
    data=Data(args.chemical or args.agreement,args.agreement);model=RetrievalModel(args.chemical,agreement=args.agreement).cuda()
    checkpoint=out/'best_model.pt';last=out/'last_model.pt';history_path=out/'history.json'
    config={**vars(args),'width':512,'layers':2,'heads':8,'dropout':.1,'batch':16,'accumulation':4,'temperature':.03,'weight_decay':1e-5,'seed':0,'validation_n':5000,'spectrum_step':1190000,'spectrum_cls':'always-valid ordinary token','molecular_tokens':'13 latent modality/CLS tokens, all valid','frozen_encoders':True,'evidence_weighting':False,'mass_agreement':args.agreement,'hard_mining':False}
    write_json(out/'config.json',config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    best=-1.;best_epoch=-1;stale=0;first=0;history=[]
    if args.agreement and not last.exists():
        initial=torch.load(parent,map_location='cuda',weights_only=False)
        missing,unexpected=model.load_state_dict(initial['model'],strict=False)
        if unexpected or any(not k.startswith('mass.') for k in missing):raise ValueError('Agreement parent architecture mismatch')
        # The parent is the reference; select only checkpoints with agreement enabled.
        if not args.smoke:
            write_json(out/'initial_validation.json',initial['validation'])
    if last.exists() and not args.smoke:
        saved=torch.load(last,map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        best=saved['best'];best_epoch=saved['best_epoch'];stale=saved['stale'];first=saved['epoch']+1
        history=json.loads(history_path.read_text())
    if args.smoke:
        rows=data.train[:16];unique,mapping,cv=data.candidates(rows,np.random.default_rng(0))
        x,mask=data.spectrum(rows);z,mv,chem=data.molecule(unique)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            q=model.spectrum(x,mask);m=model.molecule(z,mv,chem)
            bonus=None if model.mass is None else model.mass.bonus(data.physical_peaks(rows),data.fractions(unique,chem.valid.shape[1])[mapping],chem.valid[mapping],model.mass.atom_weights(m,chem)[mapping])
            loss=F.cross_entropy(maxsim(q,mask,m[mapping],cv,bonus)/.03,torch.zeros(len(rows),dtype=torch.long,device='cuda'))
        loss.backward()
        if not torch.isfinite(loss):raise ValueError('Nonfinite smoke loss')
        optimizer.step()
        val,_=evaluate(model,data,data.val[:16],out,'smoke')
        write_json(out/'preflight.json',{'loss':loss.item(),'validation':val,'parameters':sum(p.numel() for p in model.parameters()),'smoke_only':True})
        print(json.dumps({'smoke_loss':loss.item(),'validation':val}),flush=True);return
    for epoch in range(first,args.epochs if not args.eval_only and stale<args.patience else first):
        t0=time.monotonic();last_commit=t0;model.train();rng=np.random.default_rng(np.random.SeedSequence([0,epoch]))
        order=rng.permutation(data.train);total_loss=0.;seen=0
        write_json(out/'status.json',{'phase':'training','epoch':epoch+1,'time':datetime.now(timezone.utc).isoformat()})
        for group_start in range(0,len(order),64):
            group=order[group_start:group_start+64];optimizer.zero_grad(set_to_none=True)
            for i in range(0,len(group),16):
                rows=group[i:i+16];unique,mapping,cv=data.candidates(rows,rng)
                x,mask=data.spectrum(rows);z,mv,chem=data.molecule(unique)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    q=model.spectrum(x,mask);m=model.molecule(z,mv,chem)
                    bonus=None if model.mass is None else model.mass.bonus(data.physical_peaks(rows),data.fractions(unique,chem.valid.shape[1])[mapping],chem.valid[mapping],model.mass.atom_weights(m,chem)[mapping])
                    scores=maxsim(q,mask,m[mapping],cv,bonus)
                    loss=F.cross_entropy(scores/.03,torch.zeros(len(rows),dtype=torch.long,device='cuda'))
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                (loss*len(rows)/len(group)).backward();total_loss+=float(loss.detach())*len(rows);seen+=len(rows)
            optimizer.step()
            if group_start%16384==0:
                report={'phase':'training','epoch':epoch+1,'examples':seen,'total':len(order),'loss':total_loss/seen,'seconds':time.monotonic()-t0}
                write_json(out/'status.json',report);print(json.dumps(report),flush=True)
                if time.monotonic()-last_commit>=600:
                    import modal
                    modal.Volume.from_name('spectra-massspecgym-maccs-probe').commit();last_commit=time.monotonic()
        write_json(out/'status.json',{'phase':'validation','epoch':epoch+1,'time':datetime.now(timezone.utc).isoformat()})
        validation,_=evaluate(model,data,data.val,out,'validation_latest')
        row={'epoch':epoch+1,'train_loss':total_loss/seen,'validation':validation,'seconds':time.monotonic()-t0}
        score=validation['stable']['hit1'];improved=score>best
        if improved:
            best=score;best_epoch=epoch+1;stale=0
            save_checkpoint(checkpoint,{'model':model.state_dict(),'epoch':epoch+1,'validation':validation,'config':config})
            write_json(out/'selected_validation.json',validation)
            (out/'validation_best_ranks.json').write_bytes((out/'validation_latest_ranks.json').read_bytes())
        else:stale+=1
        history.append(row);write_json(history_path,history)
        save_checkpoint(last,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
        import modal
        modal.Volume.from_name('spectra-massspecgym-maccs-probe').commit()
        print(json.dumps({**row,'selected_epoch':best_epoch,'patience':stale}),flush=True)
        if stale>=args.patience:break
    selected=torch.load(checkpoint,map_location='cuda',weights_only=False);model.load_state_dict(selected['model'])
    write_json(out/'status.json',{'phase':'full_test','selected_epoch':selected['epoch']})
    test,pairs=evaluate(model,data,data.test,out,'test',selected.get('agreement_active',True))
    write_json(out/'metrics_retrieval.json',test)
    spec=importlib.util.spec_from_file_location('mces_eval',Path(__file__).parent/'03_retrieval.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    write_json(out/'status.json',{'phase':'mces','selected_epoch':selected['epoch']})
    distance=module.mces_at_1(pairs,15,str(out/'mces_cache.json'),16)
    test['stable'].update(distance);test['optimistic'].update(distance)
    result={'test':test,'validation':selected['validation'],'selected_epoch':selected['epoch'],'epochs_run':len(history),'config':config,'agreement_active':args.agreement and selected.get('agreement_active',True),'correct_top1':round(test['stable']['hit1']*len(data.test)),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    write_json(out/'summary.json',result)
    write_json(out/'status.json',{'phase':'complete','time':datetime.now(timezone.utc).isoformat()})
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
