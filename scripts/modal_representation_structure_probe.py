"""Label-free and acquisition-aware structure probes of frozen encoders on MassSpecGym 1.5.

The layerwise semantic probe asks how much chemistry is linearly decodable.
This probe asks how the representation is organised, using no chemical labels
beyond molecule identity (InChIKey). Per depth and pooling it measures:

- geometry: effective rank, participation ratio, PCs for 90% variance, and the
  TwoNN intrinsic dimension of the test fold;
- identity vs acquisition: the share of variance explained by molecule identity,
  same-molecule retrieval across acquisition conditions (collision energy,
  adduct, instrument), and the same retrieval restricted to isobaric candidates
  (same adduct, precursor within 10 mDa);
- shortcut content: how much of the representation is linearly explained by
  low-level spectrum statistics (peak count, spectral entropy, base-peak m/z,
  intensity-weighted m/z, precursor m/z);
- perturbation sensitivity: squared distance to the clean embedding after
  dropping weak peaks, jittering intensities, hiding acquisition metadata, or
  dropping the three strongest peaks, relative to the distance between random
  spectra and to the distance between spectra of the same molecule.

Mean-pooled test-fold features of every layer (and the final CLS) are saved on
the volume, and `cka` compares encoders layer by layer with linear CKA.

    .venv/bin/modal run scripts/modal_representation_structure_probe.py::main \
        --encoders ctrl_50k,mrlfixed_50k --output-dir DIR
    .venv/bin/modal run scripts/modal_representation_structure_probe.py::cka \
        --encoders mae_666k,ctrl_50k,mrlfixed_50k --output-dir DIR

Encoders must already be staged with modal_layerwise_semantic_probe.py::stage.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.modal_layerwise_semantic_probe import (  # noqa: E402
    ENCODERS,
    LOCAL_TSV,
    PROJECT_ROOT,
    VOLUME_ROOT,
    _binned_spectra,
    _load_encoder,
    _pooled_layer_features,
    image,
    volume,
)

app = modal.App("spectra-representation-structure-probe")

FEATURE_DIR = "structure_features"
TWONN_SAMPLES = 8192
ISOBARIC_WINDOW_DA = 0.01
NUM_PAIRS = 200_000
RIDGE_TRAIN_SAMPLES = 60_000
PERTURBATIONS = ("drop_weak_half", "jitter_intensity", "hide_metadata", "drop_top3")


def _spectrum_labels(tsv_path: Path) -> dict[str, np.ndarray]:
    """Molecule identity and acquisition-condition ids per spectrum."""
    import pyarrow.csv as csv

    rows = csv.read_csv(
        tsv_path,
        parse_options=csv.ParseOptions(delimiter="\t"),
        convert_options=csv.ConvertOptions(
            include_columns=[
                "inchikey", "collision_energy", "adduct", "instrument_type", "precursor_mz"
            ]
        ),
    ).to_pydict()
    _, molecule = np.unique(np.asarray(rows["inchikey"]), return_inverse=True)
    condition_keys = [
        f"{ce}|{adduct}|{instrument}"
        for ce, adduct, instrument in zip(
            rows["collision_energy"], rows["adduct"], rows["instrument_type"]
        )
    ]
    _, condition = np.unique(np.asarray(condition_keys), return_inverse=True)
    _, adduct = np.unique(np.asarray(rows["adduct"]), return_inverse=True)
    return {
        "molecule": molecule.astype(np.int64),
        "condition": condition.astype(np.int64),
        "adduct": adduct.astype(np.int64),
        "precursor_mz_da": np.asarray(rows["precursor_mz"], dtype=np.float64),
    }


def _lowlevel_statistics(data: dict[str, np.ndarray]) -> np.ndarray:
    """Per-spectrum shortcut statistics: count, entropy, base-peak m/z, weighted m/z, precursor."""
    valid = data["peak_valid_mask"]
    intensity = np.where(valid, data["peak_intensity"], 0.0).astype(np.float64)
    mz = data["peak_mz"].astype(np.float64)
    total = np.maximum(intensity.sum(1, keepdims=True), 1e-12)
    p = intensity / total
    entropy = -(p * np.log(np.where(p > 0, p, 1.0))).sum(1)
    base_peak = mz[np.arange(len(mz)), intensity.argmax(1)]
    weighted_mz = (p * mz).sum(1)
    return np.stack(
        [valid.sum(1), entropy, base_peak, weighted_mz, data["precursor_mz"]], axis=1
    ).astype(np.float32)


def _perturbed(data: dict[str, np.ndarray], index: np.ndarray, kind: str, seed: int = 0):
    """Copy of the spectra at `index` with one nuisance or content perturbation."""
    out = {key: value[index].copy() for key, value in data.items() if key != "fold"}
    valid = out["peak_valid_mask"]
    intensity = out["peak_intensity"]
    rng = np.random.default_rng(seed)
    if kind == "hide_metadata":
        for key in ("collision_energy", "collision_energy_present", "charge_present"):
            out[key][:] = 0
        out["polarity_id"][:] = 0
        out["instrument_family_id"][:] = 0
        return out
    if kind == "jitter_intensity":
        jittered = intensity * rng.lognormal(0.0, 0.3, intensity.shape).astype(np.float32)
        jittered = np.where(valid, jittered, 0.0)
        out["peak_intensity"] = (
            jittered / np.maximum(jittered.max(1, keepdims=True), 1e-12)
        ).astype(np.float32)
        return out
    # Peak removal: rank valid peaks by intensity, strongest first.
    order = np.argsort(np.where(valid, -intensity, np.inf), axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(order.shape[1])[None].repeat(len(order), 0), 1)
    count = valid.sum(1, keepdims=True)
    if kind == "drop_weak_half":
        keep = rank < np.maximum((count + 1) // 2, 1)
    elif kind == "drop_top3":
        keep = (rank >= 3) | (count <= 3)
    else:
        raise ValueError(kind)
    out["peak_valid_mask"] = valid & keep
    return out


def _standardize(x, train_mean, train_std):
    return (x - train_mean) / train_std


class _StructureEvaluator:
    def __init__(self, labels: dict, data: dict, fold: np.ndarray):
        import torch

        self.train = np.flatnonzero(fold == "train")
        self.val = np.flatnonzero(fold == "val")
        self.test = np.flatnonzero(fold == "test")
        rng = np.random.default_rng(0)
        self.ridge_train = np.sort(
            rng.choice(self.train, min(RIDGE_TRAIN_SAMPLES, len(self.train)), replace=False)
        )
        cuda = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda")  # noqa: E731
        molecule = labels["molecule"][self.test]
        condition = labels["condition"][self.test]
        adduct = labels["adduct"][self.test]
        precursor = labels["precursor_mz_da"][self.test]
        self.molecule = cuda(molecule)
        self.condition = cuda(condition)
        n = len(self.test)
        # Candidates exclude the query and same-molecule copies under the same condition.
        same_mol = molecule[:, None] == molecule[None]
        same_cond = condition[:, None] == condition[None]
        exclude = (same_mol & same_cond) | np.eye(n, dtype=bool)
        positive = same_mol & ~same_cond
        self.exclude = cuda(exclude)
        self.positive = cuda(positive)
        self.cross_queries = cuda(np.flatnonzero(positive.any(1)))
        isobaric = (adduct[:, None] == adduct[None]) & (
            np.abs(precursor[:, None] - precursor[None]) < ISOBARIC_WINDOW_DA
        )
        iso_candidates = isobaric & ~exclude
        iso_positive = iso_candidates & positive
        iso_negative = iso_candidates & ~same_mol
        self.iso_exclude = cuda(~iso_candidates)
        self.iso_positive = cuda(iso_positive)
        self.iso_queries = cuda(np.flatnonzero(iso_positive.any(1) & iso_negative.any(1)))
        del same_mol, same_cond, exclude, positive, isobaric
        # Molecules with at least two test spectra, for the variance decomposition.
        counts = np.bincount(molecule)
        self.multi = cuda(np.flatnonzero(counts[molecule] >= 2))
        self.random_pairs = cuda(rng.integers(0, n, (2, NUM_PAIRS)))
        rows = np.flatnonzero(np.bincount(molecule)[molecule] >= 2)
        partner = np.full(n, -1)
        by_molecule: dict[int, list[int]] = {}
        for i in rows:
            by_molecule.setdefault(int(molecule[i]), []).append(i)
        for members in by_molecule.values():
            for j, i in enumerate(members):
                partner[i] = members[(j + 1) % len(members)]
        self.molecule_pairs = cuda(np.stack([rows, partner[rows]]))
        # One spectrum per molecule: replicate spectra would collapse TwoNN's nearest distances.
        shuffled = rng.permutation(n)
        _, first = np.unique(molecule[shuffled], return_index=True)
        one_per_molecule = shuffled[first]
        self.twonn = cuda(np.sort(one_per_molecule[: min(TWONN_SAMPLES, len(first))]))
        stats = _lowlevel_statistics(data)
        stats = np.concatenate([stats, stats**2], axis=1)
        stats_mean, stats_std = stats[self.train].mean(0), stats[self.train].std(0) + 1e-6
        self.stats = {
            name: cuda(((stats[idx] - stats_mean) / stats_std).astype(np.float32))
            for name, idx in (("train", self.ridge_train), ("val", self.val), ("test", self.test))
        }

    def evaluate(
        self, features, perturbed: dict | None = None, standardize: bool = True
    ) -> dict[str, float]:
        import torch

        x = {
            name: features[torch.from_numpy(idx)].to("cuda", torch.float32)
            for name, idx in (("train", self.ridge_train), ("val", self.val), ("test", self.test))
        }
        mean = x["train"].mean(0)
        # Floor per-channel std so near-constant channels cannot dominate distances.
        std = x["train"].std(0)
        std = std.clamp(min=0.1 * float(std.mean())) if standardize else torch.ones_like(std)
        metrics = self._geometry(x["test"])
        z = {name: _standardize(v, mean, std) for name, v in x.items()}
        metrics.update(self._identity(z["test"]))
        metrics.update(self._shortcut(z))
        if perturbed is not None:
            metrics.update(self._perturbation(z["test"], perturbed, mean, std))
        return metrics

    def _geometry(self, x) -> dict[str, float]:
        import torch

        centered = x - x.mean(0)
        eig = torch.linalg.eigvalsh(centered.double().T @ centered.double() / len(x)).clamp(
            min=0
        )
        eig = eig.flip(0)
        p = eig / eig.sum()
        entropy = -(p * torch.log(torch.where(p > 0, p, torch.ones_like(p)))).sum()
        cumulative = torch.cumsum(p, 0)
        sample = centered[self.twonn]
        distance = torch.cdist(sample, sample)
        distance.fill_diagonal_(torch.inf)
        nearest = distance.topk(2, dim=1, largest=False).values
        ratio = nearest[:, 1] / nearest[:, 0]
        ratio = ratio[torch.isfinite(ratio) & (nearest[:, 0] > 0)]
        return {
            "effective_rank": float(torch.exp(entropy)),
            "participation_ratio": float(eig.sum() ** 2 / (eig**2).sum()),
            "pcs_for_90pct_variance": int((cumulative < 0.9).sum()) + 1,
            "top1_pc_variance_fraction": float(p[0]),
            "twonn_intrinsic_dimension": float(len(ratio) / torch.log(ratio).sum()),
        }

    def _identity(self, z) -> dict[str, float]:
        import torch

        multi = z[self.multi]
        molecule = self.molecule[self.multi]
        _, inverse = torch.unique(molecule, return_inverse=True)
        sums = torch.zeros((int(inverse.max()) + 1, z.shape[1]), device="cuda").index_add_(
            0, inverse, multi
        )
        counts = torch.bincount(inverse).float()[:, None]
        within = ((multi - (sums / counts)[inverse]) ** 2).sum()
        total = ((multi - multi.mean(0)) ** 2).sum()
        unit = torch.nn.functional.normalize(z, dim=1).half()
        out = {"identity_variance_fraction": float(1 - within / total)}
        for prefix, queries, excluded, positive in (
            ("cross_condition", self.cross_queries, self.exclude, self.positive),
            ("isobaric", self.iso_queries, self.iso_exclude, self.iso_positive),
        ):
            hits, reciprocal = [], []
            for start in range(0, len(queries), 2048):
                q = queries[start : start + 2048]
                similarity = (unit[q] @ unit.T).float()
                similarity[excluded[q]] = -torch.inf
                order = similarity.argsort(dim=1, descending=True)
                is_positive = torch.gather(positive[q], 1, order)
                first = is_positive.float().argmax(1)
                hits.append(is_positive[:, 0].float())
                reciprocal.append(1.0 / (first.float() + 1))
            out[f"{prefix}_recall_at_1"] = float(torch.cat(hits).mean())
            out[f"{prefix}_mrr"] = float(torch.cat(reciprocal).mean())
            out[f"{prefix}_queries"] = len(queries)
        return out

    def _shortcut(self, z) -> dict[str, float]:
        """Share of standardized representation variance linearly explained by shortcut stats."""
        import torch

        s_train, s_test = self.stats["train"].double(), self.stats["test"].double()
        design = torch.cat([s_train, torch.ones(len(s_train), 1, device="cuda", dtype=s_train.dtype)], 1)
        beta = torch.linalg.lstsq(design, z["train"].double()).solution
        test_design = torch.cat([s_test, torch.ones(len(s_test), 1, device="cuda", dtype=s_test.dtype)], 1)
        residual = z["test"].double() - test_design @ beta
        centered = z["test"].double() - z["test"].double().mean(0)
        return {
            "shortcut_explained_fraction": float(1 - (residual**2).sum() / (centered**2).sum())
        }

    def _perturbation(self, z, perturbed, mean, std) -> dict[str, float]:
        random_distance = ((z[self.random_pairs[0]] - z[self.random_pairs[1]]) ** 2).sum(1).mean()
        molecule_distance = (
            ((z[self.molecule_pairs[0]] - z[self.molecule_pairs[1]]) ** 2).sum(1).mean()
        )
        out = {"same_molecule_over_random_distance": float(molecule_distance / random_distance)}
        for kind, features in perturbed.items():
            zp = _standardize(features.to("cuda", z.dtype), mean, std)
            shift = ((z - zp) ** 2).sum(1).mean()
            out[f"perturb_{kind}_over_random"] = float(shift / random_distance)
            out[f"perturb_{kind}_over_same_molecule"] = float(shift / molecule_distance)
        return out


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=200 * 1024,
    timeout=4 * 60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def probe_encoder(name: str, labels: dict[str, np.ndarray], batch_size: int = 256) -> dict:
    os.chdir(PROJECT_ROOT)

    import torch
    from ml_collections import config_dict

    from scripts.modal_export_jepa_fps_embeddings import _prepare_massspecgym_inputs

    started = time.time()
    checkpoint, source_module = ENCODERS[name]
    directory = VOLUME_ROOT / name
    config = config_dict.ConfigDict(json.loads((directory / "config.json").read_text()))
    data = _prepare_massspecgym_inputs(VOLUME_ROOT / LOCAL_TSV.name, config)
    evaluator = _StructureEvaluator(labels, data, data["fold"])
    results: dict[str, Any] = {"encoder": name, "checkpoint": checkpoint, "layers": {}}
    if source_module == "binned":
        features = torch.from_numpy(_binned_spectra(data))
        perturbed = {
            kind: torch.from_numpy(_binned_spectra(_perturbed(data, evaluator.test, kind)))
            for kind in PERTURBATIONS
            if kind != "hide_metadata"
        }
        results["layers"]["binned"] = {"mean": evaluator.evaluate(features, perturbed, standardize=False)}
        results["seconds"] = time.time() - started
        return results
    encoder = _load_encoder(directory, config)
    pooled = _pooled_layer_features(encoder, data, config, batch_size)
    perturbed_pooled = {
        kind: _pooled_layer_features(
            encoder, _perturbed(data, evaluator.test, kind), config, batch_size
        )
        for kind in PERTURBATIONS
    }
    del encoder
    torch.cuda.empty_cache()
    num_layers = pooled["mean"].shape[0]
    feature_dir = VOLUME_ROOT / FEATURE_DIR / name
    feature_dir.mkdir(parents=True, exist_ok=True)
    test = torch.from_numpy(evaluator.test)
    for layer in range(num_layers):
        label = "embed" if layer == 0 else "final" if layer == num_layers - 1 else str(layer)
        results["layers"][label] = {}
        for pool in pooled:
            perturbed = {kind: values[pool][layer] for kind, values in perturbed_pooled.items()}
            results["layers"][label][pool] = evaluator.evaluate(pooled[pool][layer], perturbed)
            if pool == "mean" or label == "final":
                np.save(
                    feature_dir / f"{pool}_{label}.npy",
                    pooled[pool][layer][test].to(torch.float16).numpy(),
                )
        print(
            name,
            label,
            {
                pool: (
                    round(values["cross_condition_recall_at_1"], 4),
                    round(values["isobaric_recall_at_1"], 4),
                    round(values["effective_rank"], 1),
                )
                for pool, values in results["layers"][label].items()
            },
            flush=True,
        )
    volume.commit()
    results["seconds"] = time.time() - started
    return results


def _linear_cka(x, y) -> float:
    x = x - x.mean(0)
    y = y - y.mean(0)
    cross = (x.T @ y).norm() ** 2
    return float(cross / ((x.T @ x).norm() * (y.T @ y).norm()))


@app.function(
    image=image,
    gpu="H100",
    memory=128 * 1024,
    timeout=2 * 60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def cka_matrix(names: list[str]) -> dict:
    """Linear CKA between every saved layer of every pair of encoders (test fold, mean pool)."""
    import torch

    def layers(name):
        directory = VOLUME_ROOT / FEATURE_DIR / name
        files = sorted(
            directory.glob("mean_*.npy"),
            key=lambda p: (
                -1 if p.stem == "mean_embed" else 10**6 if p.stem == "mean_final" else int(p.stem[5:])
            ),
        )
        return {
            p.stem[5:]: torch.from_numpy(np.load(p)).to("cuda", torch.float32) for p in files
        }

    loaded = {name: layers(name) for name in names}
    out: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[f"{a}__{b}"] = {
                la: {lb: _linear_cka(xa, xb) for lb, xb in loaded[b].items()}
                for la, xa in loaded[a].items()
            }
            print("cka", a, b, flush=True)
    return out


@app.local_entrypoint()
def main(encoders: str, output_dir: str = "") -> None:
    names = encoders.split(",")
    labels = _spectrum_labels(LOCAL_TSV)
    output = Path(output_dir) if output_dir else None
    for result in probe_encoder.map(names, kwargs={"labels": labels}):
        print(json.dumps({"encoder": result["encoder"], "seconds": result["seconds"]}))
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{result['encoder']}.json").write_text(json.dumps(result, indent=2))


@app.local_entrypoint()
def cka(encoders: str, output_dir: str) -> None:
    result = cka_matrix.remote(encoders.split(","))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"cka_{'_'.join(encoders.split(','))}.json").write_text(
        json.dumps(result, indent=2)
    )
