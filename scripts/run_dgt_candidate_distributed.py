"""Direct DDP candidate training; global batch and retrieval protocol stay fixed."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from dgt_data import ROOT, load_data
from dgt_moljepa import Model
from run_dgt_flare import candidate_forward, evaluate, hard_negative_indices, save, write_json


class CandidateLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data, rows, selected, valid, mass):
        scores = candidate_forward(self.model, data, rows, selected, valid, 'evidence', mass)
        return F.cross_entropy(scores/.03, torch.zeros(len(rows), dtype=torch.long, device='cuda'))


def init_distributed():
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('high')
    dist.init_process_group('nccl', device_id=torch.device('cuda', torch.cuda.current_device()))
    return rank, world


def rng_states(world):
    states = [None]*world
    dist.all_gather_object(states, {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()})
    return states


def commit():
    import modal
    modal.Volume.from_name('spectra-massspecgym-maccs-probe').commit()


def train_step(wrapped, data, group, rng, hard, mass, rank, world):
    # Every rank consumes the same RNG stream, preserving the global row order
    # and negative draws of the serial protocol before partitioning queries.
    selected, valid = data.selected(group, rng, hard)
    indices = np.array_split(np.arange(len(group)), world)[rank]
    if not len(indices):
        raise ValueError('A global batch must contain at least one query per rank')
    rows = group[indices]; selected = selected[indices]; valid = valid[indices]
    model = wrapped.module.model
    if hard:
        model.eval()
        with torch.no_grad():
            scores = candidate_forward(model, data, rows, selected, valid, 'evidence', mass)
        keep = hard_negative_indices(scores, valid, 8)
        selected = np.take_along_axis(selected, keep.cpu().numpy(), axis=1)
        valid = valid.gather(1, keep)
    model.train()
    loss = wrapped(data, rows, selected, valid, mass)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite distributed loss')
    # DDP averages rank gradients; this also handles the final uneven batch.
    (loss*(world*len(rows)/len(group))).backward()
    return float(loss.detach())*len(rows), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['evidence', 'hard', 'agreement'], required=True)
    args = parser.parse_args()
    rank, world = init_distributed()
    try:
        run(args.stage, rank, world)
    finally:
        dist.destroy_process_group()


def run(stage, rank, world):
    torch.manual_seed(0)
    data = load_data(); model = Model().cuda()
    hard = stage != 'evidence'; mass = stage == 'agreement'
    out = ROOT/stage; out.mkdir(parents=True, exist_ok=True)
    parent = {'evidence':'warmup', 'hard':'evidence', 'agreement':'hard'}[stage]
    previous = torch.load(ROOT/parent/'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(previous['model']); del previous
    wrapped = DDP(CandidateLoss(model), device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)
    torch.manual_seed(rank)
    config = {'stage':stage, 'spectrum_step':1190000, 'validation_n':5000, 'width':512,
              'gnn_layers':8, 'gnn_heads':64, 'spectrum_readout_layers':2, 'dropout':.2,
              'learning_rate':3e-5, 'epochs':40, 'patience':8, 'effective_batch':64,
              'query_microbatch':64//world, 'molecule_microbatch':32,
              'weight_decay':1e-5, 'temperature':.03, 'seed':0, 'world_size':world,
              'rank_dropout_seed':'rank; restored per rank on resume',
              'spectrum_cls':'always valid ordinary peak; no physical bonus',
              'chemical_features':'basic + chem_v1 + fragment_v1 + Morgan radius 0..3',
              'frozen':'JEPA and public Mol-JEPA',
              'trainable_parameters':sum(p.numel() for p in model.parameters()),
              'precision':'bfloat16 encoder matmuls; float32 similarity and loss; float16 evaluation cache',
              'compiled_graph_blocks':False, 'mass_agreement':mass,
              'hard_mining':hard, 'evidence_weighting':True}
    checkpoint = out/'best_model.pt'; last = out/'last_model.pt'; pending_path = out/'pending_validation.pt'
    best = -1.; best_epoch = 0; stale = 0; first = 0; history = []; pending = None
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['world_size'] != world:
            raise ValueError('Resumption requires the saved distributed world size')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['distributed_rng'][rank]['cpu'])
        torch.cuda.set_rng_state(state['distributed_rng'][rank]['cuda'])
        best = state['best']; best_epoch = state['best_epoch']; stale = state['stale']; first = state['epoch']
        history = json.loads((out/'history.json').read_text()); del state
    if pending_path.exists():
        state = torch.load(pending_path, map_location='cpu', weights_only=False)
        if state['epoch'] > first:
            if state['epoch'] != first+1 or state['world_size'] != world:
                raise ValueError('Pending checkpoint does not match the next epoch/world size')
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['distributed_rng'][rank]['cpu'])
            torch.cuda.set_rng_state(state['distributed_rng'][rank]['cuda'])
            best, best_epoch, stale = state['selection']; history = state['history']
            pending = {k:state[k] for k in ['epoch','train_seconds','total_loss','seen']}
        del state
    if rank == 0:
        write_json(out/'config.json', config)

    def select(epoch, validation):
        nonlocal best, best_epoch, stale
        if validation['stable']['hit1'] > best:
            best = validation['stable']['hit1']; best_epoch = epoch; stale = 0
            save(checkpoint, {'model':model.state_dict(), 'epoch':epoch, 'validation':validation, 'config':config})
            write_json(out/'selected_validation.json', validation)
            (out/'validation_best_ranks.json').write_bytes((out/'validation_latest_ranks.json').read_bytes())
        else:
            stale += 1

    if hard and not last.exists() and pending is None:
        if rank == 0:
            validation, _ = evaluate(model, data, data.val, out, 'validation_latest', 'evidence', mass)
            write_json(out/'initial_validation.json', validation); select(0, validation)
        dist.barrier()
    selection = [(best,best_epoch,stale) if rank == 0 else None]
    dist.broadcast_object_list(selection, src=0); best,best_epoch,stale = selection[0]
    worker_start = time.monotonic(); last_commit = worker_start
    for epoch in range(first, 40 if stale < 8 else first):
        resume_validation = pending is not None and pending['epoch'] == epoch+1
        rng = np.random.default_rng(np.random.SeedSequence([0,epoch]))
        order = np.empty(0,np.int64) if resume_validation else rng.permutation(data.train)
        start = time.monotonic()-(pending['train_seconds'] if resume_validation else 0)
        total = pending['total_loss'] if resume_validation else 0.
        seen = pending['seen'] if resume_validation else 0
        for begin in range(0,len(order),64):
            group = order[begin:begin+64]; optimizer.zero_grad(set_to_none=True)
            value, count = train_step(wrapped, data, group, rng, hard, mass, rank, world)
            optimizer.step(); total += value; seen += count
            if begin % 4096 == 0 and rank == 0:
                report = {'phase':'training', 'stage':stage, 'epoch':epoch+1,
                          'examples':begin+len(group), 'total':len(order), 'world_size':world,
                          'rank0_loss':total/seen, 'seconds':time.monotonic()-start,
                          'time':datetime.now(timezone.utc).isoformat()}
                write_json(out/'status.json',report); print(json.dumps(report),flush=True)
                if time.monotonic()-last_commit >= 600:
                    commit(); last_commit = time.monotonic()
        if not resume_validation:
            totals = torch.tensor([total,seen],device='cuda',dtype=torch.float64)
            dist.all_reduce(totals); total,seen = totals.tolist(); seen = int(seen)
            states = rng_states(world)
            if rank == 0:
                save(pending_path, {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                     'epoch':epoch+1, 'total_loss':total, 'seen':seen, 'train_seconds':time.monotonic()-start,
                     'selection':(best,best_epoch,stale), 'history':history,
                     'distributed_rng':states, 'world_size':world, 'config':config})
                commit()
        dist.barrier()
        if rank == 0:
            write_json(out/'status.json', {'phase':'validation','stage':stage,'epoch':epoch+1})
            validation, _ = evaluate(model,data,data.val,out,'validation_latest','evidence',mass)
            select(epoch+1,validation)
            row = {'epoch':epoch+1, 'train_loss':total/seen, 'validation':validation,
                   'seconds':time.monotonic()-start}
            history.append(row); write_json(out/'history.json',history)
        dist.barrier()
        selection = [(best,best_epoch,stale) if rank == 0 else None]
        dist.broadcast_object_list(selection,src=0); best,best_epoch,stale = selection[0]
        states = rng_states(world)
        if rank == 0:
            save(last, {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch+1,
                 'best':best,'best_epoch':best_epoch,'stale':stale,'world_size':world,
                 'distributed_rng':states,'cpu_rng':states[0]['cpu'],'cuda_rng':states[0]['cuda']})
            pending_path.unlink(missing_ok=True); commit()
            print(json.dumps({**row,'selected_epoch':best_epoch,'stale':stale}),flush=True)
        pending = None
        dist.barrier()
        pause = [time.monotonic()-worker_start > 72000 if rank == 0 else None]
        dist.broadcast_object_list(pause,src=0)
        if stale >= 8: break
        if pause[0] and epoch+1 < 40:
            if rank == 0:
                write_json(out/'status.json',{'phase':'resume_needed','epoch':epoch+1});commit()
            return
    if rank == 0:
        write_json(out/'status.json', {'phase':'complete','stage':stage,
                   'selected_epoch':best_epoch,'epochs_run':len(history)});commit()


if __name__ == '__main__':
    main()
