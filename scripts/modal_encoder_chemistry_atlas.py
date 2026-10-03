"""Supplemental layer atlas; token readouts remain the primary chemistry test.

Ridge probes use one fixed 256-dimensional random projection, selected penalties
on validation only, and molecule-held-out folds. Token geometry and input probes
are descriptive and are not fragment annotations or causal explanations.
"""
from __future__ import annotations

import os
from pathlib import Path
import json
import time

import modal
import numpy as np

from scripts.modal_encoder_chemistry import LOCAL, PROJECT, ROOT, TOKEN_DEPTHS, error_metrics, image, volume, write_json

app = modal.App("spectra-native-chemistry-atlas")
PENALTIES = np.array([.1, 1., 10., 100., 1000.])


def ridge_predictions(features, targets, train, val):
    center = features[train].mean(0)
    scale = np.maximum(features[train].std(0), 1e-6)
    x = ((features - center) / scale).astype(np.float64)
    ymean = targets[train].mean(0)
    xty = x[train].T @ (targets[train] - ymean)
    eigen, vectors = np.linalg.eigh(x[train].T @ x[train])
    base = vectors.T @ xty
    candidates = np.stack([x @ (vectors @ (base / (eigen[:, None] + p))) + ymean for p in PENALTIES])
    loss = ((candidates[:, val] - targets[None, val]) ** 2).mean(1)
    best = loss.argmin(0)
    prediction = candidates[best, :, np.arange(targets.shape[1])].T
    return prediction.astype(np.float32), PENALTIES[best].tolist()


def classification_metrics(prediction, labels, support):
    from sklearn.metrics import roc_auc_score
    auc = np.full(labels.shape[1], np.nan)
    for k in np.flatnonzero(support):
        if len(np.unique(labels[:, k])) == 2:
            auc[k] = roc_auc_score(labels[:, k], prediction[:, k])
    brier = ((np.clip(prediction, 0, 1) - labels) ** 2).mean(0)
    mean = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else None
    return {"auroc": [None if not np.isfinite(x) else float(x) for x in auc],
            "brier": brier.tolist(), "maccs_auroc": mean(auc[:166]),
            "element_auroc": mean(auc[166:173]), "functional_group_auroc": mean(auc[173:]),
            "maccs_brier": float(brier[:166].mean()), "concept_brier": float(brier[166:].mean())}


@app.function(image=image, cpu=16, memory=65536, timeout=7200, volumes={"/results": volume},
              env={"OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "8", "JAX_PLATFORMS": "cpu"})
def analyze(name: str) -> dict:
    os.chdir(PROJECT)
    import ml_dtypes
    from threadpoolctl import threadpool_limits
    threadpool_limits(8)
    directory = ROOT / name
    extraction = json.loads((directory / "extraction.json").read_text())
    if not extraction["complete"]:
        raise ValueError("Incomplete extraction")
    cached = np.load(directory / "tokens_bf16.npy", mmap_mode="r")
    reference = np.load(directory / "pilot_tokens.npz")
    parity = error_metrics(cached[-1, reference["rows"]].view(ml_dtypes.bfloat16),
                           reference["native"], reference["valid"])
    write_json(directory / "cache_parity.json", parity)
    if not parity["finite"] or parity["relative_l2"] >= .01:
        volume.commit()
        raise ValueError("Final extracted tokens differ from the native pilot beyond the numerical gate")
    data = np.load(ROOT / "panel/inputs.npz")
    labels = np.load(ROOT / "panel/labels.npz")
    y = np.concatenate([labels["maccs"], labels["binary"]], 1).astype(np.float32)
    fold = data["fold"]
    train, val, test = (fold == f for f in ("train", "val", "test"))
    # Identical support rules and random projection for the two checkpoints.
    support = (y[train].sum(0) >= 20) & ((1-y[train]).sum(0) >= 20)
    support &= (y[val].sum(0) >= 5) & ((1-y[val]).sum(0) >= 5)
    support &= (y[test].sum(0) >= 5) & ((1-y[test]).sum(0) >= 5)
    rng = np.random.default_rng(20260930)
    projection = (rng.standard_normal((2048, 256)) / np.sqrt(256)).astype(np.float32)
    output = directory / "atlas"
    output.mkdir(exist_ok=True)
    started = time.monotonic()
    records, predictions = [], []
    for pool in ("mean", "max", "cls"):
        features = np.load(directory / f"{pool}.npy", mmap_mode="r")
        for depth in range(28):
            z = features[depth] @ projection
            prediction, penalties = ridge_predictions(z, y, train, val)
            metrics = classification_metrics(prediction[test], y[test], support)
            records.append({"pool": pool, "depth": depth, "penalties": penalties,
                            "test": metrics, "validation": classification_metrics(prediction[val], y[val], support)})
            predictions.append(prediction[test])
        print(name, pool, "layer ridge complete", round(time.monotonic()-started, 1), flush=True)
    np.savez_compressed(output / "test_predictions.npz", scores=np.stack(predictions), labels=y[test],
                        rows=np.flatnonzero(test), pool=np.array([r["pool"] for r in records]),
                        depth=np.array([r["depth"] for r in records]))
    # Compare against acquisition features alone, with exactly the same ridge selection.
    mass = data["precursor_mz"]
    energy = data["collision_energy"]
    nuisance = np.concatenate([np.stack([mass, mass**2, mass**3, energy, energy**2,
        data["peak_valid_mask"].sum(1)/127, data["collision_energy_present"], data["charge"],
        data["charge_present"]], 1), np.exp(-((mass[:, None] - np.linspace(0, 1, 32))/.03)**2),
        np.eye(int(data["instrument_family_id"].max())+1)[data["instrument_family_id"]],
        np.eye(int(data["polarity_id"].max())+1)[data["polarity_id"]]], 1).astype(np.float32)
    baseline, penalties = ridge_predictions(nuisance, y, train, val)
    baseline_record = {"features": "mass polynomial/RBF, collision energy, peak count, instrument, polarity, charge/presence",
                       "penalties": penalties, "test": classification_metrics(baseline[test], y[test], support)}
    np.savez_compressed(output / "nuisance_predictions.npz", scores=baseline[test], labels=y[test])
    tokens = np.load(directory / "tokens_bf16.npy", mmap_mode="r")
    mask = data["peak_valid_mask"]
    count = mask.sum(1)
    geometry = []
    # Fixed quantile positions give each spectrum at most eight equally weighted peak samples.
    positions = [np.unique(np.linspace(0, int(n)-1, min(int(n), 8)).astype(int)) for n in count]
    sampled_rows = np.concatenate([np.repeat(i, len(p)) for i, p in enumerate(positions)])
    sampled_slots = np.concatenate([np.flatnonzero(mask[i])[p] for i, p in enumerate(positions)])
    for token_index, depth in enumerate(TOKEN_DEPTHS):
        pair_cosine, variance_fraction, norms = [], [], []
        for first in range(0, len(mask), 128):
            last = min(first+128, len(mask))
            x = tokens[token_index, first:last, :-1].view(ml_dtypes.bfloat16).astype(np.float32)
            valid = mask[first:last, :, None]
            n = count[first:last]
            squared = (x*x).sum(-1)
            norm = np.sqrt(squared)
            unit = x / np.maximum(norm[..., None], 1e-12) * valid
            pair_cosine.extend((((unit.sum(1)**2).sum(-1)-n)/(n*(n-1))).tolist())
            average = (x*valid).sum(1)/n[:, None]
            energy_total = (squared*valid[...,0]).sum(1)/n
            variance_fraction.extend((1-(average**2).sum(-1)/np.maximum(energy_total,1e-12)).tolist())
            norms.extend(((norm*valid[...,0]).sum(1)/n).tolist())
        samples = tokens[token_index, sampled_rows, sampled_slots].view(ml_dtypes.bfloat16).astype(np.float32)
        z = samples @ projection
        observable = np.stack([data["peak_mz"][sampled_rows,sampled_slots],
                               data["peak_intensity"][sampled_rows,sampled_slots]], 1)
        observed_prediction, selected = ridge_predictions(z, observable, train[sampled_rows], val[sampled_rows])
        use = test[sampled_rows]
        r2 = 1-((observed_prediction[use]-observable[use])**2).sum(0)/((observable[use]-observable[use].mean(0))**2).sum(0)
        centered = z-z.mean(0)
        covariance = centered.T@centered/len(z)
        eig = np.maximum(np.linalg.eigvalsh(covariance),0)
        q = eig/np.maximum(eig.sum(),1e-12)
        geometry.append({"depth":depth,"mean_valid_peak_cosine":float(np.mean(pair_cosine)),
            "median_within_spectrum_variance_fraction":float(np.median(variance_fraction)),
            "median_token_l2":float(np.median(norms)),"projected_token_effective_rank":float(np.exp(-np.sum(q*np.log(np.maximum(q,1e-30))))),
            "input_linear_r2_mz":float(r2[0]),"input_linear_r2_intensity":float(r2[1]),
            "input_probe_penalties":selected,"input_test_tokens":int(use.sum())})
        np.savez_compressed(output / f"token_geometry_depth{depth}.npz",mean_pair_cosine=np.asarray(pair_cosine),
                            variance_fraction=np.asarray(variance_fraction),mean_l2=np.asarray(norms))
        print(name,"token geometry",depth,round(time.monotonic()-started,1),flush=True)
    result = {"name":name,"records":records,"token_geometry":geometry,"nuisance_baseline":baseline_record,
        "target_names":[*[f"MACCS_{i}" for i in range(1,167)],*labels["binary_names"].tolist()],
        "target_support":support.tolist(),"positive_counts":{f:y[fold==f].sum(0).astype(int).tolist() for f in ("train","val","test")},
        "projection_dimension":256,"projection_seed":20260930,"penalty_selection":"per-target validation MSE",
        "scope":"Supplemental projected pooled probes; not token-capacity bounds. Input probes decode observed m/z/intensity, not chemical fragments.",
        "seconds":time.monotonic()-started,"extraction_manifest":extraction,"cache_parity":parity}
    write_json(output / "atlas.json",result)
    volume.commit()
    return result


@app.local_entrypoint()
def main(name: str = "both"):
    names = ("parent_1190k","periodic_1209k") if name == "both" else (name,)
    calls = {n:analyze.spawn(n) for n in names}
    for n,call in calls.items():
        write_json(LOCAL / n / "atlas.json",call.get())
