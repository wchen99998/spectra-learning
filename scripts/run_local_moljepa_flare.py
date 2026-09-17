"""Four-stage FLARE with a preserved local path and gated Mol-JEPA context."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from local_moljepa import Model as LocalModel
from local_moljepa_data import load_data, RAW_FEATURE_ROOT
import run_dgt_flare as trainer


ROOT = Path('/results/local_moljepa_atom_latest')
_evaluate = trainer.evaluate
_encode_molecules = trainer.encode_molecules


class Model(LocalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_config.update({
            'world_size': 1,
            'molecule_microbatch': 'complete training query microbatch; 32 during mining',
            'warmup_molecule_batch': 64,
            'candidate_molecule_batch_max': 144,
            'raw_peak_feature_root': str(RAW_FEATURE_ROOT),
            'raw_peak_preprocessing': 'unchanged canonical float16 cache promoted to float32',
            'cls_raw_mz_intensity': [0., 0.],
            'representation_probe_n': 64,
            'representation_probe_seed': 73,
            'representation_probe_sampling': 'first 64 sorted rows of fixed random 512 validation rows',
        })


def encode_molecules(model, data, ids, batch_size=32):
    if not model.training:
        return _encode_molecules(model, data, ids, batch_size)
    # GINE BatchNorm must see the complete original training microbatch:
    # 64 positive molecules in warmup, all unique candidates for 16 queries later.
    graph = data.graph(ids)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        atoms = model.molecule(graph)
    return atoms, graph.atom_valid, graph.fractions


def candidate_forward(model, data, rows, selected, valid, mode, mass):
    # Padding must not introduce unrelated molecule ID zero into BatchNorm.
    real = valid.cpu().numpy()
    ids = np.unique(selected[real])
    indices = np.zeros(selected.shape, dtype=np.int64)
    indices[real] = np.searchsorted(ids, selected[real])
    mapping = torch.from_numpy(indices).cuda()
    a, am, af = encode_molecules(model, data, ids)
    x, pm = data.spectrum(rows)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        q = model.spectrum(x, pm)
    bonus = model.mass(data.physical_peaks(rows), af[mapping]) if mass else None
    peak, atom = trainer.components(q, pm, a[mapping], am[mapping], bonus)
    return trainer.candidate_scores(peak, atom, pm, valid, mode)


def within_set_cosines(values: torch.Tensor, valid: torch.Tensor) -> list[float]:
    """Mean off-diagonal cosine per set; omit sets with fewer than two tokens."""
    with torch.autocast(values.device.type, enabled=False):
        normalized = F.normalize(values.float(), dim=-1).masked_fill(~valid[..., None], 0)
        count = valid.sum(-1)
        pair_sum = normalized.sum(-2).square().sum(-1) - normalized.square().sum((-2, -1))
        keep = count > 1
        values = pair_sum[keep] / (count[keep] * (count[keep] - 1))
    return values.cpu().tolist()


def residual_gates(model) -> dict[str, dict[str, float]]:
    raw, effective = {}, {}
    for name, value in model.named_parameters():
        if not name.endswith('gate'):
            continue
        module = model.get_submodule(name.rsplit('.', 1)[0])
        scale = module.feature_residual_scale if name.endswith('_feature_gate') else module.scale
        raw[name] = float(value.detach().tanh().cpu())
        effective[name] = scale * raw[name]
    return {'tanh_residual_gates': raw, 'effective_residual_gates': effective}


@torch.no_grad()
def representation_diagnostics(model, data):
    # This is the exact 64-query cohort from the paused DGT investigation.
    rows = np.sort(np.random.default_rng(73).choice(data.val, min(512, len(data.val)), replace=False))[:64]
    trainer.write_json(ROOT / 'representation_probe_rows.json', rows.tolist())
    atoms, peaks, atom_pairs, peak_pairs = [], [], [], []
    for start in range(0, len(rows), 32):
        current = rows[start:start + 32]
        graph = data.graph(data.truth_id[current])
        x, valid = data.spectrum(current)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            a = model.molecule(graph)
            q = model.spectrum(x, valid)
        atoms.extend(within_set_cosines(a, graph.atom_valid))
        atom_count = graph.atom_valid.sum(-1)
        atom_pairs.extend((atom_count[atom_count > 1] * (atom_count[atom_count > 1] - 1)).cpu().tolist())
        # CLS participates in retrieval, but has no physical peak counterpart.
        peaks.extend(within_set_cosines(q[:, :-1], valid[:, :-1]))
        peak_count = valid[:, :-1].sum(-1)
        peak_pairs.extend((peak_count[peak_count > 1] * (peak_count[peak_count > 1] - 1)).cpu().tolist())
    return {
        'n': len(rows), 'seed': 73,
        'within_molecule_atom_cosine_macro': float(np.mean(atoms)),
        'within_molecule_atom_cosine_pair_weighted': float(np.average(atoms, weights=atom_pairs)),
        'within_spectrum_physical_peak_cosine_macro': float(np.mean(peaks)),
        'within_spectrum_physical_peak_cosine_pair_weighted': float(np.average(peaks, weights=peak_pairs)),
        'molecules_with_multiple_atoms': len(atoms),
        'spectra_with_multiple_physical_peaks': len(peaks),
        **residual_gates(model),
    }


def evaluate(model, data, rows, out, name, mode, mass, global_grid=(0.,)):
    result, pairs = _evaluate(model, data, rows, out, name, mode, mass, global_grid)
    if name == 'validation_latest':
        geometry = representation_diagnostics(model, data)
        result['representation_diagnostics'] = geometry
        trainer.write_json(out / f'{name}_representation.json', geometry)
        print(json.dumps({'phase': 'representation_diagnostics', **geometry}), flush=True)
    return result, pairs


def synthetic_smoke():
    from dgt_chemistry import features
    from dgt_data import pack_graphs
    smiles = ['CC(=O)Oc1ccccc1C(=O)O', 'CCN(CC)C(=O)c1ccc(O)cc1', '[Na+].[Cl-]', 'C'] * 8
    graph = pack_graphs([features(s) for s in smiles], torch.randn(32, 13, 512, device='cuda'))
    model = Model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    x = torch.randn(32, 128, 2050, device='cuda')
    x[:, -1, -2:] = 0
    valid = torch.ones(32, 128, dtype=torch.bool, device='cuda')
    times = []
    for _ in range(4):
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            a = model.molecule(graph)
            q = model.spectrum(x, valid)
            global_embedding = F.normalize(model.global_head(x[:, -1]).float(), dim=-1)
        loss = (a * torch.randn_like(a)).sum() / graph.atom_valid.sum()
        loss = loss + (q * torch.randn_like(q)).mean() + global_embedding.square().mean()
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite synthetic training loss or gradient')
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.monotonic() - start)
    result = {
        'parameters': sum(p.numel() for p in model.parameters()),
        'step_seconds': times, 'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1024 ** 3,
        **residual_gates(model),
        'finite_backward': True, 'smoke_only': True,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    trainer.write_json(ROOT / 'synthetic_preflight.json', result)
    print(json.dumps(result), flush=True)


def main():
    if '--compile-blocks' in sys.argv:
        raise SystemExit('This experiment uses the eager local GINE path; DGT block compilation does not apply.')
    trainer.ROOT = ROOT
    trainer.Model = Model
    trainer.load_data = load_data
    trainer.encode_molecules = encode_molecules
    trainer.candidate_forward = candidate_forward
    trainer.evaluate = evaluate
    trainer.synthetic_smoke = synthetic_smoke
    trainer.main()


if __name__ == '__main__':
    main()
