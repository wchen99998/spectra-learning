"""Fresh, molecule-balanced audit of the initial and periodically promoted teachers.

Frozen inference uses the repository's existing JAX-to-PyTorch encoder exports;
no pretraining path or checkpoint is changed. Run with the existing environment:
  .venv/bin/python -m modal run scripts/modal_periodic_semantics.py --output-dir DIR
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import modal
import numpy as np

from scripts.modal_layerwise_semantic_probe import (
    ENCODERS, LOCAL_TSV, PROJECT_ROOT, VOLUME_ROOT, REGRESSION_KEYS,
    _binned_spectra, _load_encoder, _molecule_targets, image, volume,
)
from scripts.periodic_semantics_stats import macro_auc, nearest_mass, rank_correlation

app = modal.App("periodic-teacher-semantic-audit")
ARMS = ("mae_666k", "ftjepa_1190k", "periodic_student_1200k", "periodic_student_1209k")
SEED = 20260929
LAYERS = (13, 19, "final")
METADATA = ("precursor_mz", "collision_energy", "collision_energy_present", "charge",
            "charge_present", "polarity_id", "instrument_family_id")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(encoder, data, config, indices=None, perturbation=None):
    import torch
    from spectra_learning.models.spectrum_metadata import torch_spectrum_metadata_from_batch

    indices = np.arange(len(data["fold"])) if indices is None else indices
    features = {f"mean_{l}": torch.empty((len(indices), config.model_dim), dtype=torch.bfloat16) for l in LAYERS}
    features["cls_final"] = torch.empty_like(features["mean_final"])
    captured = {}
    hooks = [encoder.blocks[l - 1].register_forward_hook(
        lambda _m, _a, output, l=l: captured.__setitem__(l, output)
    ) for l in LAYERS if isinstance(l, int)]
    rng = np.random.default_rng(SEED)
    with torch.inference_mode():
        for start in range(0, len(indices), 256):
            ids = indices[start:start + 256]
            batch = {k: torch.from_numpy(data[k][ids]).to("cuda") for k in (*METADATA, "peak_mz", "peak_intensity", "peak_valid_mask")}
            if perturbation == "drop20":
                keep = rng.random(batch["peak_valid_mask"].shape) >= .2
                # Always preserve the strongest valid peak, and never add peaks.
                strongest = data["peak_intensity"][ids].argmax(1)
                keep[np.arange(len(ids)), strongest] = True
                batch["peak_valid_mask"] &= torch.from_numpy(keep).to("cuda")
            elif perturbation == "intensity_jitter":
                noise = torch.from_numpy(rng.lognormal(0, .2, batch["peak_intensity"].shape).astype(np.float32)).to("cuda")
                batch["peak_intensity"] *= noise
                batch["peak_intensity"] /= batch["peak_intensity"].amax(1, keepdim=True).clamp(min=1e-8)
            elif perturbation == "metadata_zero":
                for k in METADATA:
                    if k != "precursor_mz":
                        batch[k] = torch.zeros_like(batch[k])
            metadata = torch_spectrum_metadata_from_batch(batch, schema=str(config.encoder_metadata_schema))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                captured["final"] = encoder(batch["peak_mz"], batch["peak_intensity"],
                    valid_mask=batch["peak_valid_mask"], visible_mask=batch["peak_valid_mask"],
                    precursor_mz=batch["precursor_mz"], spectrum_metadata=metadata)
            mask = batch["peak_valid_mask"].unsqueeze(-1).float()
            width = mask.shape[1]
            for layer in LAYERS:
                pooled = (captured[layer][:, :width].float() * mask).sum(1) / mask.sum(1).clamp(min=1)
                features[f"mean_{layer}"][start:start + len(ids)] = pooled.bfloat16().cpu()
            features["cls_final"][start:start + len(ids)] = captured["final"][:, width].bfloat16().cpu()
            if start % 25600 == 0:
                print(f"{perturbation or 'full'} embedded {start + len(ids)}/{len(indices)}", flush=True)
    for hook in hooks:
        hook.remove()
    return features


def aggregate(features, molecule, count):
    import torch
    x = features.to("cuda", torch.float32) if isinstance(features, torch.Tensor) else torch.as_tensor(features, device="cuda", dtype=torch.float32)
    if x.ndim == 1:
        x = x[:, None]
    index = torch.as_tensor(molecule, device="cuda", dtype=torch.long)
    summed = torch.zeros((count, x.shape[1]), device="cuda")
    summed.index_add_(0, index, x)
    return summed / torch.bincount(index, minlength=count)[:, None]


def ridge(features, split, binary, regression, fraction=1.):
    import torch
    ids = split["train"]
    if fraction < 1:
        ids = np.sort(np.random.default_rng(SEED).choice(ids, max(2, int(len(ids) * fraction)), replace=False))
    mean, std = features[ids].mean(0), features[ids].std(0).clamp(min=1e-6)
    x = {s: ((features[i] - mean) / std).double() for s, i in split.items()}
    tr = ((features[ids] - mean) / std).double()
    eig, vec = torch.linalg.eigh(tr.T @ tr / len(ids))
    scale = float(eig.mean())
    result, saved = {}, {}
    for group, target in (("binary", binary), ("regression", regression)):
        y = torch.as_tensor(target, device="cuda", dtype=torch.float64)
        ym, ys = y[ids].mean(0), y[ids].std(0).clamp(min=1e-6)
        if group == "binary":
            ys = torch.ones_like(ys)
        projected = vec.T @ (tr.T @ ((y[ids] - ym) / ys) / len(ids))
        best = None
        for factor in (1e-4, 1e-3, 1e-2, .1, 1., 10., 100., 1000.):
            weights = (vec / (eig.clamp(min=0) + factor * scale)) @ projected
            prediction = (x["val"] @ weights * ys + ym).cpu().numpy()
            truth = target[split["val"]]
            score = macro_auc(prediction[:, :166], truth[:, :166]) if group == "binary" else float(np.mean(1 - ((prediction - truth)**2).sum(0) / ((truth - truth.mean(0))**2).sum(0)))
            if best is None or score > best[0]:
                best = (score, factor, weights)
        score, factor, weights = best
        prediction = (x["test"] @ weights * ys + ym).cpu().numpy().astype(np.float32)
        truth = target[split["test"]]
        result[group + "_lambda"] = factor
        result[group + "_val_score"] = score
        if group == "binary":
            result.update(maccs_auc=macro_auc(prediction[:, :166], truth[:, :166]),
                          elements_auc=macro_auc(prediction[:, 166:173], truth[:, 166:173]),
                          fg_auc=macro_auc(prediction[:, 173:], truth[:, 173:]))
        else:
            r2 = 1 - ((prediction - truth)**2).sum(0) / ((truth - truth.mean(0))**2).sum(0)
            result.update(r2=dict(zip(REGRESSION_KEYS, map(float, r2))), r2_mean=float(r2.mean()))
        saved[group] = prediction
    result["train_molecules"] = len(ids)
    return result, saved


def fingerprint_similarity(a, b):
    intersection = (a * b).sum(-1)
    return intersection / (a.sum(-1) + b.sum(-1) - intersection).clamp(min=1)


def geometry(features, split, fingerprints, binned, mass, pairs, variant):
    import torch
    import torch.nn.functional as F
    tr, te = split["train"], split["test"]
    x = features.clone()
    train = x[tr]
    if variant != "raw":
        x -= train.mean(0)
    if variant == "standardized":
        x /= train.std(0).clamp(min=1e-6)
    elif variant == "whiten_0.01":
        centered = train.double() - train.double().mean(0)
        cov = centered.T @ centered / (len(train) - 1)
        eig, vec = torch.linalg.eigh(cov)
        transform = vec * (eig.clamp(min=0) + .01 * eig.mean()).rsqrt()
        x = (x.double() @ transform).float()
    x = F.normalize(x, dim=1)
    nearest, top10, mass_nearest, eligible = [], [], [], []
    for start in range(0, len(te), 512):
        q = te[start:start + 512]
        sims = x[q] @ x[tr].T
        nb = tr[sims.topk(10, dim=1).indices.cpu().numpy()]
        tan = fingerprint_similarity(fingerprints[q, None], fingerprints[nb])
        nearest.append(tan[:, 0].cpu().numpy())
        top10.append(tan.mean(1).cpu().numpy())
        allowed = abs(mass[q, None] - mass[tr][None, :]) <= 5.
        eligible.append(allowed.any(1))
        sims.masked_fill_(~torch.as_tensor(allowed, device="cuda"), -torch.inf)
        nbm = tr[sims.argmax(1).cpu().numpy()]
        mass_nearest.append(fingerprint_similarity(fingerprints[q], fingerprints[nbm]).cpu().numpy())
    a, b = pairs
    cosine = (x[te[a]] * x[te[b]]).sum(1).cpu().numpy()
    tan = fingerprint_similarity(fingerprints[te[a]], fingerprints[te[b]]).cpu().numpy()
    spectral = (F.normalize(binned[te[a]], dim=1) * F.normalize(binned[te[b]], dim=1)).sum(1).cpu().numpy()
    controls = np.column_stack([spectral, abs(mass[te[a]] - mass[te[b]])])
    values, matched, eligible = np.concatenate(nearest), np.concatenate(mass_nearest), np.concatenate(eligible)
    matched[~eligible] = np.nan
    metrics = {"knn1_tanimoto": float(values.mean()), "knn10_tanimoto": float(np.concatenate(top10).mean()),
               "knn1_mass_window_5Da": float(np.nanmean(matched)), "mass_window_eligible": int(eligible.sum()),
               "pair_spearman": rank_correlation(cosine, tan),
               "pair_partial_spearman": rank_correlation(cosine, tan, controls),
               "pair_count": len(a)}
    return metrics, {"knn1": values, "knn_mass5": matched, "pair_cosine": cosine, "pair_tanimoto": tan}


def collapse(features, train):
    import torch
    x = features[train].double()
    centered = x - x.mean(0)
    eig = torch.linalg.eigvalsh(centered.T @ centered / (len(x) - 1)).clamp(min=0)
    p = eig / eig.sum()
    positive = p > 0
    return {"effective_rank": float(torch.exp(-(p[positive] * p[positive].log()).sum())),
            "participation_ratio": float(1 / p.square().sum()),
            "pc1_fraction": float(p[-1]), "total_variance": float(eig.sum()),
            "mean_pair_cosine": float(torch.nn.functional.normalize(x, dim=1).mean(0).square().sum()),
            "feature_norm_mean": float(x.norm(dim=1).mean())}


@app.function(image=image, gpu="H100", cpu=16, memory=120 * 1024, timeout=5400,
              volumes={str(VOLUME_ROOT): volume})
def audit(name: str, targets: dict) -> dict:
    os.chdir(PROJECT_ROOT)
    import torch
    from ml_collections import config_dict
    from scripts.modal_export_jepa_fps_embeddings import _prepare_massspecgym_inputs

    started = time.time()
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(8)
    directory = VOLUME_ROOT / name
    config = config_dict.ConfigDict(json.loads((directory / "config.json").read_text()))
    data = _prepare_massspecgym_inputs(VOLUME_ROOT / LOCAL_TSV.name, config)
    molecule = targets["molecule_index"]
    count = len(targets["maccs"])
    first = np.full(count, len(molecule))
    np.minimum.at(first, molecule, np.arange(len(molecule)))
    fold = data["fold"][first]
    if not np.array_equal(fold[molecule], data["fold"]):
        raise ValueError("Molecule crosses evaluation folds")
    split = {s: np.flatnonzero(fold == s) for s in ("train", "val", "test")}
    test_spectra = np.flatnonzero(data["fold"] == "test")
    bins = _binned_spectra(data)
    binned = aggregate(bins, molecule, count)
    # Preprocessing normalizes precursor m/z by PEAK_MZ_MAX, just like peaks.
    # Use deterministic float64 aggregation before applying a physical window.
    mass = np.bincount(molecule, weights=data["precursor_mz"].astype(np.float64) * 1000.) / np.bincount(molecule)
    fp = torch.as_tensor(np.unpackbits(targets["morgan_packed"], axis=1), device="cuda", dtype=torch.float32)
    binary = np.concatenate([targets["maccs"], targets["binary"]], axis=1)
    rng = np.random.default_rng(SEED)
    a = rng.integers(len(split["test"]), size=100000)
    b = rng.integers(len(split["test"]), size=100000)
    pairs = (a[a != b], b[a != b])
    encoder = _load_encoder(directory, config)
    print(name, "loaded", flush=True)
    pooled = extract(encoder, data, config)
    # Same molecule, distinct measured spectrum; the gallery has one per molecule.
    gallery, queries = [], []
    for m in split["test"]:
        rows = np.flatnonzero(molecule == m)
        if len(rows) >= 2:
            gallery.append(rows[0]); queries.append(rows[-1])
    gallery, queries = np.asarray(gallery), np.asarray(queries)
    subset = first[split["test"]][:2048]
    initial_mean = pooled["mean_final"][subset].float().to("cuda")
    train_mu = pooled["mean_final"][np.flatnonzero(data["fold"] == "train")].float().mean(0).to("cuda")
    reference = torch.nn.functional.normalize(initial_mean - train_mu, dim=1)
    robustness = {}
    for perturb in ("drop20", "intensity_jitter", "metadata_zero"):
        altered = extract(encoder, data, config, subset, perturb)["mean_final"].float().to("cuda")
        transformed = torch.nn.functional.normalize(altered - train_mu, dim=1)
        sim = transformed @ reference.T
        robustness[perturb] = {"mean_centered_cosine": float((transformed * reference).sum(1).mean()),
                              "self_retrieval_top1": float((sim.argmax(1) == torch.arange(len(subset), device="cuda")).float().mean()),
                              "spectra": len(subset)}
    del encoder
    torch.cuda.empty_cache()
    out = {"name": name, "checkpoint": ENCODERS[name][0], "source_module": ENCODERS[name][1],
           "encoder_sha256": sha256(directory / "encoder.pt"), "config_sha256": sha256(directory / "config.json"),
           "dataset_sha256": sha256(VOLUME_ROOT / LOCAL_TSV.name),
           "molecules": {s: len(i) for s, i in split.items()},
           "spectra": {s: int((data["fold"] == s).sum()) for s in split},
           "robustness": robustness, "features": {},
           "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "numpy": np.__version__}}
    arrays = {"test_molecule": split["test"], "test_binary": binary[split["test"]],
              "test_regression": targets["regression"][split["test"]]}
    random_nb = rng.choice(split["train"], len(split["test"]))
    mass_nb = split["train"][nearest_mass(mass[split["train"]], mass[split["test"]])]
    out["baselines"] = {
        "random_molecule_tanimoto": float(fingerprint_similarity(fp[split["test"]], fp[random_nb]).mean()),
        "nearest_mass_tanimoto": float(fingerprint_similarity(fp[split["test"]], fp[mass_nb]).mean()),
    }
    if name == "ftjepa_1190k":
        pooled["binned"] = torch.from_numpy(bins)
    for label, features in pooled.items():
        print(name, "evaluating", label, flush=True)
        means = aggregate(features, molecule, count)
        entry = {"geometry": {}, "collapse": collapse(means, split["train"]), "ridge": {}}
        variants = ("raw", "centered", "standardized", "whiten_0.01") if label in ("mean_final", "mean_19") else ("raw", "standardized")
        for variant in variants:
            entry["geometry"][variant], samples = geometry(means, split, fp, binned, mass, pairs, variant)
            for key in ("knn1", "knn_mass5"):
                arrays[f"{label}_{variant}_{key}"] = samples[key]
        if label == "mean_final":
            arrays["test_mean_features"] = means[split["test"]].half().cpu().numpy()
        for fraction in ((.01, .1, 1.) if label == "mean_final" else (1.,)):
            entry["ridge"][str(fraction)], predictions = ridge(means, split, binary, targets["regression"], fraction)
            if fraction == 1.:
                for group, prediction in predictions.items():
                    arrays[f"{label}_{group}_prediction"] = prediction
        if label == "mean_final":
            # Label-permutation control, with the same validation selection.
            shuffled = binary.copy()
            shuffled[split["train"]] = shuffled[rng.permutation(split["train"])]
            entry["permuted_label_ridge"], _ = ridge(means, split, shuffled, targets["regression"])
        if label in ("mean_final", "cls_final"):
            x = features.float().to("cuda")
            mean = x[np.flatnonzero(data["fold"] == "train")].mean(0)
            q = torch.nn.functional.normalize(x[queries] - mean, dim=1)
            g = torch.nn.functional.normalize(x[gallery] - mean, dim=1)
            sim = q @ g.T
            entry["replicate_retrieval"] = {"molecules": len(queries),
                "top1": float((sim.argmax(1) == torch.arange(len(queries), device="cuda")).float().mean()),
                "same_molecule_cosine": float((q * g).sum(1).mean()),
                "different_molecule_cosine": float((q * g.roll(1, 0)).sum(1).mean()),
                "different_instrument_pairs": int((data["instrument_family_id"][queries] != data["instrument_family_id"][gallery]).sum()),
                "different_collision_energy_pairs": int((data["collision_energy"][queries] != data["collision_energy"][gallery]).sum())}
        out["features"][label] = entry
    out["seconds"] = time.time() - started
    dest = VOLUME_ROOT / "periodic_audit_20260929"
    dest.mkdir(exist_ok=True)
    (dest / f"{name}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(dest / f"{name}_samples.npz", **arrays)
    volume.commit()
    print(name, "complete", out["seconds"], flush=True)
    return out


@app.function(image=image, gpu="H100", cpu=16, memory=120 * 1024, timeout=5400,
              volumes={str(VOLUME_ROOT): volume})
def mass_and_single_spectrum_audit(name: str, targets: dict, single_spectrum_only: bool = False) -> dict:
    """Correct physical mass windows and add an identity-free single-row test.

    Reuses measured spectra but reruns frozen inference because the first audit
    retained predictions, not the full train embeddings. Cache centroids here.
    For newly measured encoders with correct physical units, single_spectrum_only
    reuses their geometry and extracts just the first row per molecule.
    """
    os.chdir(PROJECT_ROOT)
    import torch
    from ml_collections import config_dict
    from scripts.modal_export_jepa_fps_embeddings import _prepare_massspecgym_inputs

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(8)
    config = config_dict.ConfigDict(json.loads((VOLUME_ROOT / name / "config.json").read_text()))
    data = _prepare_massspecgym_inputs(VOLUME_ROOT / LOCAL_TSV.name, config)
    molecule = targets["molecule_index"]
    count = len(targets["maccs"])
    first = np.full(count, len(molecule))
    np.minimum.at(first, molecule, np.arange(len(molecule)))
    split = {s: np.flatnonzero(data["fold"][first] == s) for s in ("train", "val", "test")}
    bins = _binned_spectra(data)
    binned = aggregate(bins, molecule, count)
    mass = np.bincount(molecule, weights=data["precursor_mz"].astype(np.float64) * 1000.) / np.bincount(molecule)
    fp = torch.as_tensor(np.unpackbits(targets["morgan_packed"], axis=1), device="cuda", dtype=torch.float32)
    rng = np.random.default_rng(SEED)
    a, b = rng.integers(len(split["test"]), size=(2, 100000))
    pairs = (a[a != b], b[a != b])
    encoder = _load_encoder(VOLUME_ROOT / name, config)
    pooled = extract(encoder, data, config, indices=first if single_spectrum_only else None)
    del encoder
    dest = VOLUME_ROOT / "periodic_audit_20260929"
    out = json.loads((dest / f"{name}.json").read_text())
    with np.load(dest / f"{name}_samples.npz") as old:
        arrays = dict(old)
    cache = {}
    if name == "ftjepa_1190k" and not single_spectrum_only:
        pooled["binned"] = torch.from_numpy(bins)
    for label, features in (() if single_spectrum_only else pooled.items()):
        print(name, "physical mass window", label, flush=True)
        means = aggregate(features, molecule, count)
        cache[label] = means.half().cpu().numpy()
        for variant in out["features"][label]["geometry"]:
            metrics, sample = geometry(means, split, fp, binned, mass, pairs, variant)
            # All geometry is recomputed so the saved record corresponds to the
            # corrected source; existing chemical probes remain unchanged.
            out["features"][label]["geometry"][variant] = metrics
            arrays[f"{label}_{variant}_knn1"] = sample["knn1"]
            arrays[f"{label}_{variant}_knn_mass5"] = sample["knn_mass5"]
    single = (pooled["mean_final"] if single_spectrum_only else pooled["mean_final"][first]).float().to("cuda")
    single_bins = torch.as_tensor(bins[first], device="cuda")
    single_mass = data["precursor_mz"][first].astype(np.float64) * 1000.
    out["single_spectrum"] = {}
    for variant in ("raw", "standardized", "whiten_0.01"):
        metrics, sample = geometry(single, split, fp, single_bins, single_mass, pairs, variant)
        out["single_spectrum"][variant] = metrics
        arrays[f"single_{variant}_knn1"] = sample["knn1"]
        arrays[f"single_{variant}_knn_mass5"] = sample["knn_mass5"]
    mass_nb = split["train"][nearest_mass(mass[split["train"]], mass[split["test"]])]
    out["baselines"]["nearest_mass_tanimoto"] = float(fingerprint_similarity(fp[split["test"]], fp[mass_nb]).mean())
    out["mass_window_correction"] = {"precursor_input_unit": "m/z divided by 1000", "window_Th": 5.,
        "centroid_mass_unit": "Th", "mean_mass_range_Th": [float(mass.min()), float(mass.max())],
        "single_spectrum_protocol": "First file row per molecule; 22,746 train candidates, 2,998 test queries; no averaging"}
    (dest / f"{name}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(dest / f"{name}_samples.npz", **arrays)
    if not single_spectrum_only:
        np.savez_compressed(dest / f"{name}_centroids.npz", **cache)
    volume.commit()
    return out


@app.local_entrypoint()
def main(output_dir: str, encoders: str = ",".join(ARMS)):
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
    targets = _molecule_targets(LOCAL_TSV)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for result in audit.map(encoders.split(","), kwargs={"targets": targets}):
        (output / f"{result['name']}.json").write_text(json.dumps(result, indent=2))
        path = f"/periodic_audit_20260929/{result['name']}_samples.npz"
        with (output / Path(path).name).open("wb") as handle:
            for chunk in volume.read_file(path):
                handle.write(chunk)


@app.local_entrypoint()
def physical_mass_windows(output_dir: str, encoders: str = ",".join(ARMS), single_spectrum_only: bool = False):
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
    targets = _molecule_targets(LOCAL_TSV)
    output = Path(output_dir)
    for result in mass_and_single_spectrum_audit.map(encoders.split(","), kwargs={"targets": targets, "single_spectrum_only": single_spectrum_only}):
        (output / f"{result['name']}.json").write_text(json.dumps(result, indent=2))
        path = f"/periodic_audit_20260929/{result['name']}_samples.npz"
        with (output / Path(path).name).open("wb") as handle:
            for chunk in volume.read_file(path):
                handle.write(chunk)
