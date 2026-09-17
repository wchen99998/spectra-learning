"""Direct four-stage FLARE training of one atom-level DGT + Mol-JEPA model."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from dgt_moljepa import Model,compile_graph_blocks
from dgt_data import ROOT,load_data,pack_graphs
from flare_candidate import candidate_scores,hard_negative_indices

def write_json(path,value):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def save(path,value):
    tmp=path.with_suffix('.partial');torch.save(value,tmp);tmp.replace(path)

def metrics(ranks):
    r=np.asarray(ranks)
    return {'n':len(r),**{f'hit{k}':float((r<=k).mean()) for k in [1,5,10,20]},
            'mrr':float((1/r).mean()),'median_rank':float(np.median(r))}

def components(q,pm,a,am,bonus=None):
    with torch.autocast(q.device.type,enabled=False):
        similarity=torch.einsum('bpd,bcad->bcpa',q.float(),a.float())
        if bonus is not None: similarity=similarity+bonus
        peak=similarity.masked_fill(~am[:,:,None],-1e4).amax(-1)
        atom=similarity.masked_fill(~pm[:,None,:,None],-1e4).amax(-2)
        atom=(atom*am).sum(-1)/am.sum(-1).clamp_min(1)
        return peak,atom

def atom_mean(a,valid):
    return F.normalize((a*valid[...,None]).sum(-2)/valid.sum(-1,keepdim=True).clamp_min(1),dim=-1)

def info_nce(scores,structure):
    same=structure[:,None]==structure[None,:]
    off=same & ~torch.eye(len(structure),dtype=torch.bool,device=scores.device)
    logits=(scores/.03).masked_fill(off,-1e4); target=torch.arange(len(structure),device=scores.device)
    return .5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))

def encode_molecules(model,data,ids,batch_size=32):
    order=np.argsort(data.graphs.counts[ids],kind='stable')
    restore=torch.from_numpy(np.argsort(order)).cuda(); ids=ids[order]
    cap=int(data.graphs.counts[ids].max()); values=[]; masks=[]; fractions=[]
    for start in range(0,len(ids),batch_size):
        g=data.graph(ids[start:start+batch_size])
        with torch.autocast('cuda',dtype=torch.bfloat16): value=model.molecule(g)
        pad=cap-value.shape[1]
        values.append(F.pad(value,(0,0,0,pad))); masks.append(F.pad(g.atom_valid,(0,pad)))
        fractions.append(F.pad(g.fractions,(0,0,0,pad),value=-1))
    return torch.cat(values)[restore],torch.cat(masks)[restore],torch.cat(fractions)[restore]

def candidate_forward(model,data,rows,selected,valid,mode,mass):
    ids,mapping=np.unique(selected,return_inverse=True)
    mapping=torch.from_numpy(mapping.reshape(selected.shape)).cuda()
    a,am,af=encode_molecules(model,data,ids)
    x,pm=data.spectrum(rows)
    with torch.autocast('cuda',dtype=torch.bfloat16):q=model.spectrum(x,pm)
    bonus=model.mass(data.physical_peaks(rows),af[mapping]) if mass else None
    peak,atom=components(q,pm,a[mapping],am[mapping],bonus)
    return candidate_scores(peak,atom,pm,valid,mode)

def warmup_loss(model,data,rows):
    a,am,_=encode_molecules(model,data,data.truth_id[rows]); x,pm=data.spectrum(rows)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        q=model.spectrum(x,pm)
        global_embedding=F.normalize(model.global_head(x[:,-1]).float(),dim=-1)
    peak,atom=components(q,pm,a[None].expand(len(rows),-1,-1,-1),am[None].expand(len(rows),-1,-1))
    cv=torch.ones((len(rows),len(rows)),dtype=torch.bool,device='cuda')
    score=candidate_scores(peak,atom,pm,cv,'uniform')
    codes=torch.from_numpy(data.structure_ids[rows]).cuda()
    return info_nce(score,codes)+info_nce(global_embedding@atom_mean(a,am).T,codes)

@torch.no_grad()
def evaluate(model,data,rows,out,name,mode,mass,global_grid=(0.,)):
    model.eval(); start=time.monotonic()
    ids=np.unique(np.concatenate([data.pool[s] for s in set(data.query[rows])]))
    ids=ids[np.argsort(data.graphs.counts[ids],kind='stable')]
    lookup=np.full(len(data.smiles),-1,np.int64); lookup[ids]=np.arange(len(ids))
    counts=data.graphs.counts[ids]; offsets=np.concatenate(([0],counts.cumsum()))
    required=int(offsets[-1])*512*2
    # CUDA's free-memory counter excludes unused PyTorch allocator reservations.
    # Release those reservations before checking space for the packed cache.
    torch.cuda.empty_cache()
    free,_=torch.cuda.mem_get_info()
    if required>free-12*1024**3: raise ValueError(f'Packed evaluation cache needs {required} bytes; GPU has {free} free')
    encoded=torch.empty(int(offsets[-1]),512,device='cuda',dtype=torch.float16)
    fractions=torch.empty(int(offsets[-1]),6,device='cuda',dtype=torch.float32) if mass else None
    for begin in range(0,len(ids),64):
        batch_ids=ids[begin:begin+64]; graph=data.graph(batch_ids)
        with torch.autocast('cuda',dtype=torch.bfloat16): a=model.molecule(graph)
        lo,hi=int(offsets[begin]),int(offsets[begin+len(batch_ids)])
        encoded[lo:hi]=a[graph.atom_valid].half()
        if mass: fractions[lo:hi]=graph.fractions[graph.atom_valid]
        if begin%32768==0:
            print(json.dumps({'phase':name+'_encoding','molecules':begin,'total':len(ids),'seconds':time.monotonic()-start}),flush=True)
    offsets=torch.from_numpy(offsets).cuda(); sizes=torch.from_numpy(counts).cuda()
    stable={w:[] for w in global_grid}; optimistic={w:[] for w in global_grid}
    records={w:[] for w in global_grid}; pairs={w:[] for w in global_grid}
    for begin in range(0,len(rows),32):
        chunk=rows[begin:begin+32]; x,pm=data.spectrum(chunk)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            q=model.spectrum(x,pm).half().float()
            sg=F.normalize(model.global_head(x[:,-1]).float(),dim=-1).half().float()
        for j,row in enumerate(chunk):
            candidates=data.pool[data.query[row]]; peaks=[]; atoms=[]; global_scores=[]
            for k in range(0,len(candidates),256):
                positions=lookup[candidates[k:k+256]]
                count=int(counts[positions].max()); pos=torch.from_numpy(positions).cuda()
                local=torch.arange(count,device='cuda')[None]
                valid=local<sizes[pos,None]
                idx=(offsets[pos,None]+local).clamp_max(len(encoded)-1)
                a=encoded[idx].float()
                bonus=model.mass(data.physical_peaks([row]),fractions[idx][None].masked_fill(~valid[None,:,:,None],-1)) if mass else None
                p,t=components(q[j:j+1],pm[j:j+1],a[None],valid[None],bonus)
                peaks.append(p); atoms.append(t)
                global_scores.append(atom_mean(a,valid)@sg[j])
            # Evidence normalization sees the complete candidate pool, regardless
            # of the chunks used to bound similarity memory.
            p,t=torch.cat(peaks,1),torch.cat(atoms,1)
            valid=torch.ones((1,len(candidates)),dtype=torch.bool,device='cuda')
            score=candidate_scores(p,t,pm[j:j+1],valid,mode)[0]
            global_score=torch.cat(global_scores)
            truth=int(np.flatnonzero(candidates==data.truth_id[row])[0])
            for weight in global_grid:
                s=score+weight*global_score
                if not torch.isfinite(s).all():raise ValueError('Nonfinite evaluation scores')
                rank=1+int((s>s[truth]).sum()); rank_stable=rank+int((s[:truth]==s[truth]).sum())
                pred=data.smiles[int(candidates[int(s.argmax())])]
                stable[weight].append(rank_stable); optimistic[weight].append(rank)
                pairs[weight].append((data.query[row],pred))
                records[weight].append({'row':int(row),'stable_rank':rank_stable,'optimistic_rank':rank,'prediction':pred,'candidates':len(candidates)})
    weight=max(global_grid,key=lambda w:metrics(stable[w])['hit1'])
    result={'stable':metrics(stable[weight]),'optimistic':metrics(optimistic[weight]),
            'global_weight':weight,'global_grid':{str(w):metrics(stable[w]) for w in global_grid},
            'seconds':time.monotonic()-start,'unique_candidates':len(ids)}
    write_json(out/f'{name}_ranks.json',records[weight]); return result,pairs[weight]

def synthetic_smoke():
    from dgt_chemistry import features
    torch.manual_seed(0)
    probe=['CC(=O)Oc1ccccc1C(=O)O','CCN(CC)C(=O)c1ccc(O)cc1','[Na+].[Cl-]','C']*8
    g=pack_graphs([features(s) for s in probe],torch.randn(32,13,512,device='cuda'))
    model=Model().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    times=[]
    for step in range(4):
        start=time.monotonic();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): a=model.molecule(g)
        loss=(a*torch.randn_like(a)).sum()/g.atom_valid.sum();loss.backward();opt.step();torch.cuda.synchronize()
        if not torch.isfinite(loss):raise ValueError('Nonfinite GPU smoke loss')
        times.append(time.monotonic()-start)
    result={'parameters':sum(p.numel() for p in model.parameters()),'step_seconds':times,
            'peak_gpu_gb':torch.cuda.max_memory_allocated()/1024**3,
            'context_gates':[float(b.context_gate.detach().tanh()) for b in model.molecule.blocks if b.cross],
            'finite_backward':True,'smoke_only':True}
    write_json(ROOT/'synthetic_preflight.json',result);print(json.dumps(result),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['warmup','evidence','hard','agreement'],default='warmup')
    parser.add_argument('--synthetic-smoke',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--compile-blocks',action='store_true')
    args=parser.parse_args(); torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    torch.manual_seed(0);np.random.seed(0)
    if args.synthetic_smoke: synthetic_smoke();return
    data=load_data(); model=Model().cuda()
    if args.compile_blocks:compile_graph_blocks(model)
    stage=args.stage; warmup=stage=='warmup'; mass=stage=='agreement'; hard=stage in ['hard','agreement']
    mode='uniform' if warmup else 'evidence'; epochs=80 if warmup else 40; patience=10 if warmup else 8
    lr=1e-4 if warmup else 3e-5
    out=ROOT/(stage+'_compiled_smoke' if args.smoke and args.compile_blocks else stage)
    out.mkdir(parents=True,exist_ok=True)
    config={'stage':stage,'spectrum_step':1190000,'validation_n':5000,'width':512,'gnn_layers':8,
            'gnn_heads':64,'spectrum_readout_layers':2,'dropout':.2,'learning_rate':lr,'epochs':epochs,
            'patience':patience,'effective_batch':64,'query_microbatch':64 if warmup else 16,
            'molecule_microbatch':32,'weight_decay':1e-5,'temperature':.03,'seed':0,
            'spectrum_cls':'always valid ordinary peak; no physical bonus',
            'chemical_features':'basic + chem_v1 + fragment_v1 + Morgan radius 0..3',
            'frozen':'JEPA and public Mol-JEPA','trainable_parameters':sum(p.numel() for p in model.parameters()),
            'precision':'bfloat16 encoder matmuls; float32 similarity and loss; float16 evaluation cache',
            'compiled_graph_blocks':args.compile_blocks,
            'mass_agreement':mass,'hard_mining':hard,'evidence_weighting':not warmup}
    config.update(getattr(model, 'experiment_config', {}))
    if not args.eval_only:write_json(out/'config.json',config)
    checkpoint=out/'best_model.pt'; last=out/'last_model.pt'; history_path=out/'history.json'
    pending_path=out/'pending_validation.pt';pending=None
    parent={'evidence':'warmup','hard':'evidence','agreement':'hard'}.get(stage)
    if parent:
        previous=torch.load(ROOT/parent/'best_model.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(previous['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    best=-1.; best_epoch=0; stale=0; first=0; history=[]
    if last.exists() and not args.smoke:
        state=torch.load(last,map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        best=state['best'];best_epoch=state['best_epoch'];stale=state['stale'];first=state['epoch']
        history=json.loads(history_path.read_text())
    if pending_path.exists() and not args.smoke and not args.eval_only:
        saved=torch.load(pending_path,map_location='cuda',weights_only=False)
        if saved['epoch']>first:
            if saved['epoch']!=first+1:raise ValueError('Pending checkpoint is not the next unvalidated epoch')
            pending=saved
            model.load_state_dict(pending['model']);optimizer.load_state_dict(pending['optimizer'])
            torch.set_rng_state(pending['cpu_rng'].cpu());torch.cuda.set_rng_state(pending['cuda_rng'].cpu())
            # The first compiled warmup worker wrote the earlier schema; its
            # selection metadata is already available in last_model.pt.
            if 'selection' in pending:
                best,best_epoch,stale=pending['selection']
                history=pending['history']
    if args.smoke:
        from flare_mol import mol_to_graph
        from flare_mol_agreement import atom_retained_fractions
        for i in np.random.default_rng(0).choice(len(data.smiles),16,replace=False):
            x,ei,ea,af=data.graphs[int(i)]; bx,bei,bea=mol_to_graph(data.smiles[int(i)])
            np.testing.assert_array_equal(x[:,:49],bx)
            np.testing.assert_array_equal(ea[:,:8],bea)
            directed=np.stack((ei,ei[:,::-1]),axis=1).reshape(-1,2).T
            np.testing.assert_array_equal(directed,bei)
            np.testing.assert_array_equal(af,atom_retained_fractions(data.smiles[int(i)]))
        smoke_start=time.monotonic(); rows=data.warmup_train[:64] if warmup else data.train[:16]
        model.train()
        if warmup: loss=warmup_loss(model,data,rows)
        else:
            selected,valid=data.selected(rows,np.random.default_rng(0),hard)
            if hard:
                model.eval()
                with torch.no_grad():scores=candidate_forward(model,data,rows,selected,valid,mode,mass)
                keep=hard_negative_indices(scores,valid,8)
                selected=np.take_along_axis(selected,keep.cpu().numpy(),axis=1);valid=valid.gather(1,keep)
                model.train()
            scores=candidate_forward(model,data,rows,selected,valid,mode,mass)
            loss=F.cross_entropy(scores/.03,torch.zeros(len(rows),dtype=torch.long,device='cuda'))
        loss.backward();optimizer.step()
        shape_checks=[]
        if args.compile_blocks:
            # Exercise fresh batch/graph sizes and checkpointed stochastic
            # backward before trusting the dynamically compiled training path.
            for count in [17,32,64]:
                optimizer.zero_grad(set_to_none=True)
                probe_rows=data.warmup_train[4096:4096+count]
                probe_loss=warmup_loss(model,data,probe_rows)
                probe_loss.backward()
                if not torch.isfinite(probe_loss):raise ValueError('Nonfinite compiled shape probe')
                shape_checks.append({'batch':count,'loss':float(probe_loss.detach())})
        train_seconds=time.monotonic()-smoke_start
        validation,_=evaluate(model,data,data.val[:16],out,'smoke',mode,mass)
        result={'loss':float(loss.detach()),'validation':validation,'train_step_seconds':train_seconds,
                'peak_gpu_gb':torch.cuda.max_memory_allocated()/1024**3,
                'original_feature_and_fraction_parity':True,
                'compiled_shape_checks':shape_checks,
                'finite_gradients':all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),'smoke_only':True}
        write_json(out/'preflight.json',result);print(json.dumps(result),flush=True);return
    def select(epoch,validation):
        nonlocal best,best_epoch,stale
        if validation['stable']['hit1']>best:
            best=validation['stable']['hit1']; best_epoch=epoch; stale=0
            save(checkpoint,{'model':model.state_dict(),'epoch':epoch,'validation':validation,'config':config})
            write_json(out/'selected_validation.json',validation)
            (out/'validation_best_ranks.json').write_bytes((out/'validation_latest_ranks.json').read_bytes())
        else:stale+=1
    if hard and not last.exists() and pending is None and not args.eval_only:
        validation,_=evaluate(model,data,data.val,out,'validation_latest',mode,mass)
        write_json(out/'initial_validation.json',validation);select(0,validation)
    worker_start=time.monotonic(); last_commit=worker_start
    for epoch in range(first,epochs if not args.eval_only and stale<patience else first):
        model.train();rng=np.random.default_rng(np.random.SeedSequence([0,epoch]))
        resume_validation=pending is not None and pending['epoch']==epoch+1
        order=np.empty(0,np.int64) if resume_validation else rng.permutation(data.warmup_train if warmup else data.train)
        start=time.monotonic()-(pending['train_seconds'] if resume_validation else 0)
        total=pending['total_loss'] if resume_validation else 0.
        seen=pending['seen'] if resume_validation else 0
        for group_start in range(0,len(order),64):
            group=order[group_start:group_start+64];optimizer.zero_grad(set_to_none=True)
            for begin in range(0,len(group),64 if warmup else 16):
                rows=group[begin:begin+(64 if warmup else 16)]
                if warmup:
                    loss=warmup_loss(model,data,rows)
                else:
                    selected,valid=data.selected(rows,rng,hard)
                    if hard:
                        model.eval()
                        with torch.no_grad():scores=candidate_forward(model,data,rows,selected,valid,mode,mass)
                        keep=hard_negative_indices(scores,valid,8)
                        selected=np.take_along_axis(selected,keep.cpu().numpy(),axis=1);valid=valid.gather(1,keep)
                        model.train()
                    scores=candidate_forward(model,data,rows,selected,valid,mode,mass)
                    loss=F.cross_entropy(scores/.03,torch.zeros(len(rows),dtype=torch.long,device='cuda'))
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                (loss*len(rows)/len(group)).backward();total+=float(loss.detach())*len(rows);seen+=len(rows)
            optimizer.step()
            if group_start%4096==0:
                report={'phase':'training','stage':stage,'epoch':epoch+1,'examples':seen,'total':len(order),
                        'loss':total/seen,'seconds':time.monotonic()-start,'time':datetime.now(timezone.utc).isoformat()}
                write_json(out/'status.json',report); print(json.dumps(report),flush=True)
                if time.monotonic()-last_commit>=600:
                    import modal
                    modal.Volume.from_name('spectra-massspecgym-maccs-probe').commit();last_commit=time.monotonic()
        if not resume_validation:
            save(pending_path,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
                'total_loss':total,'seen':seen,'train_seconds':time.monotonic()-start,
                'selection':(best,best_epoch,stale),'history':history,
                'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'config':config})
            import modal
            modal.Volume.from_name('spectra-massspecgym-maccs-probe').commit()
        write_json(out/'status.json',{'phase':'validation','epoch':epoch+1,'stage':stage})
        grid=(0.,.25,.5,1.,2.) if warmup else (0.,)
        validation,_=evaluate(model,data,data.val,out,'validation_latest',mode,mass,grid)
        select(epoch+1,validation)
        row={'epoch':epoch+1,'train_loss':total/seen,'validation':validation,'seconds':time.monotonic()-start}
        history.append(row);write_json(history_path,history)
        save(last,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'best':best,
                   'best_epoch':best_epoch,'stale':stale,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
        pending=None
        pending_path.unlink(missing_ok=True)
        import modal
        modal.Volume.from_name('spectra-massspecgym-maccs-probe').commit()
        print(json.dumps({**row,'selected_epoch':best_epoch,'stale':stale}),flush=True)
        if stale>=patience:break
        if time.monotonic()-worker_start>72000 and epoch+1<epochs:
            write_json(out/'status.json',{'phase':'resume_needed','epoch':epoch+1});return
    if args.eval_only:
        state=torch.load(checkpoint,map_location='cuda',weights_only=False);model.load_state_dict(state['model'])
        config=state['config']
        test,pairs=evaluate(model,data,data.test,out,'test',mode,mass)
        write_json(out/'metrics_retrieval.json',test)
        spec=importlib.util.spec_from_file_location('mces_eval',Path(__file__).parent/'03_retrieval.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        write_json(out/'status.json',{'phase':'mces','stage':stage})
        distance=module.mces_at_1(pairs,15,str(out/'mces_cache.json'),16)
        test['stable'].update(distance);test['optimistic'].update(distance)
        result={'test':test,'validation':state['validation'],'selected_epoch':state['epoch'],
                'epochs_run':len(history),'config':config,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        write_json(out/'summary.json',result);print(json.dumps(result),flush=True)
    write_json(out/'status.json',{'phase':'complete','stage':stage,'selected_epoch':best_epoch,'epochs_run':len(history)})

if __name__=='__main__':main()
