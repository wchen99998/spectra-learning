"""Layerwise semantic probes of frozen JEPA encoders on MassSpecGym 1.5.

For every encoder depth (embedder output, each block, final normed output) and
pooling (mean over valid peaks, CLS), this measures:

- ridge linear probes (lambda picked on val, scored on the MCES-disjoint test
  fold) for MACCS bits, heteroatom / functional-group presence, and RDKit
  properties;
- training-free structure: Morgan Tanimoto of the nearest train-fold neighbour
  of each test spectrum, and the Spearman correlation between embedding cosine
  and Tanimoto on random test pairs, with a partial correlation that removes
  binned-spectrum cosine and precursor m/z difference.

Encoders are restored from GCS on the launching machine and staged as torch
state dicts on a Modal volume, so the GPU containers need no GCP credentials:

    .venv/bin/modal run scripts/modal_layerwise_semantic_probe.py::stage
    .venv/bin/modal run scripts/modal_layerwise_semantic_probe.py::main --output-dir DIR
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import modal
import numpy as np


BUCKET = "gs://metal-repeater-411410-spectra-checkpoints"
PERIODIC_RUN = "1p8b-ptjepa-t1190k-p50k-b1024-v6e32-20260922-062250"
FTJEPA_RUN = "1p8b-ftjepa-t1p8b-s666k-perleaf-b1024-v6e32-20260905-115335"
MAE_RUN = "1p8b-sm-hd256-adam-v6e64-b6144-ga1-666k-3d-mr6-20260828-024511"
PERIODIC_CHECKPOINT = f"{BUCKET}/skypilot/{PERIODIC_RUN}/checkpoints/orbax/1209000"
# name -> (checkpoint, state subtree); None checkpoint means random init.
ENCODERS: dict[str, tuple[str | None, str]] = {
    "binned_spectrum": (None, "binned"),
    "random_init": (None, "encoder"),
    "mae_666k": (f"{BUCKET}/skypilot/{MAE_RUN}/checkpoints/orbax/666000", "encoder"),
    # Generation-0 teacher of the periodic run; trained against the MAE teacher.
    "ftjepa_1190k": (
        f"{BUCKET}/skypilot/{FTJEPA_RUN}/checkpoints/orbax/1190000",
        "encoder",
    ),
    # Frozen teacher inside the 1209000 checkpoint = student at step 1200000.
    "periodic_student_1200k": (PERIODIC_CHECKPOINT, "teacher_encoder"),
    "periodic_student_1209k": (PERIODIC_CHECKPOINT, "encoder"),
    # 1B regularizer comparisons: MAE vs SJEPA (CLS invariance + SIGReg), both
    # 2M steps on the same 127-peak m/z-ordered backbone.
    "mae1b_2m": (
        f"{BUCKET}/skypilot/1b-sm-hd256-adam-v6e64-b2048-ga2-2m-3d-mr6-20260826-144653"
        "/checkpoints/orbax/2000000",
        "encoder",
    ),
    "sjepa_sigreg_2m": (
        f"{BUCKET}/skypilot/1b-sjepa-adam-v6e32-b2048-v2g75-v3l20-sigfix-s9-20260901-033604"
        "/checkpoints/orbax/2000000",
        "encoder",
    ),
    # Grouped JEPA (63 intensity-ordered peaks) fine-tuned from a 350k JEPA init,
    # with and without VISReg on the student embeddings.
    "grouped_init_350k": (
        f"{BUCKET}/skypilot/1b-singlemixer-n64-muon-v6e64-b4096-ga2-ctx60-t25-use5a-datapart-20260811-045140"
        "/checkpoints/orbax/350000",
        "encoder",
    ),
    "grouped_jepa_95k": (
        f"{BUCKET}/skypilot/1b-grouped-jepa-init350k-v6e32-ew4-py314-b256-ga2-la-20260815-125636"
        "/checkpoints/orbax/95000",
        "encoder",
    ),
    "grouped_visreg_300k": (
        f"{BUCKET}/skypilot/1b-grouped-jepa-visreg-300k-v6e32-20260818-015358"
        "/checkpoints/orbax/300000",
        "encoder",
    ),
}
# Paired 100k-step runs: frozen-MAE-teacher student without and with SIGReg on
# its context readouts (checkpoints kept every 25k steps).
REGULARIZER_RUNS = {
    "ctrl": "1p8b-ftjepa-ctrl-100k-b1024-v6e32-20260928-141110",
    "ctxsigreg": "1p8b-ftjepa-ctxsigreg-100k-b1024-v6e32-20260928-141110",
    # Target-side variants: teacher layer-19 targets, contiguous m/z masking.
    "teacherl19": "1p8b-ftjepa-teacherl19fz-100k-b1024-v6e32-20260929-031039",
    "contig": "1p8b-ftjepa-contig-100k-b1024-v6e32-20260929-022426",
    # Cross-acquisition pairs: data control and cross-view invariance.
    "paired": "1p8b-ftjepa-paired-100k-b1024-v6e32-20260929-082912",
    "crossview": "1p8b-ftjepa-crossview-100k-b1024-v6e32-20260929-082912",
}
for _arm, _run in REGULARIZER_RUNS.items():
    for _step in (25_000, 50_000, 75_000, 100_000):
        ENCODERS[f"{_arm}_{_step // 1000}k"] = (
            f"{BUCKET}/skypilot/{_run}/checkpoints/orbax/{_step}",
            "encoder",
        )
# Progressive Matryoshka: the predictor sees a shrinking prefix (2048 -> 256)
# of the encoder output on a fixed or adaptive schedule. Trained on Isambard
# GH200 and copied back locally, so these restore from local disk.
ISAMBARD_RUNS = "/mnt/data/isambard_stage/runs"
for _arm in ("mrlfixed", "mrladaptive"):
    for _step in (25_000, 50_000, 75_000, 100_000):
        ENCODERS[f"{_arm}_{_step // 1000}k"] = (
            f"{ISAMBARD_RUNS}/1p8b-ftjepa-{_arm}-100k-b1024-gh200x8-20260930-151628"
            f"/checkpoints/orbax/{_step}",
            "encoder",
        )
# Group-SupCon fine-tunes of ftjepa step 1,190,000.
FINETUNE_RUNS = {
    "gcontrast": "1p8b-gcontrast-ft1190k-25k-b1024-v6e32-20260929-141041",
    "gcsameprec": "1p8b-gcsameprec-anchor1-ft1190k-25k-v6e32-20260929-171735",
    "gcnoprec": "1p8b-gcnoprec-anchor1-ft1190k-25k-v6e32-20260930-000450",
    "gchardpos": "1p8b-gchardpos-noprec-anchor1-lr1e5-ft1190k-25k-v6e32-20260930-011457",
    "gccrossce": "1p8b-gccrossce-noprec-anchor1-lr1e5-ga2-ft1190k-25k-v6e16-20260930-052316",
}
for _arm, _run in FINETUNE_RUNS.items():
    for _step in (5_000, 10_000, 15_000, 20_000, 25_000):
        ENCODERS[f"{_arm}_{_step // 1000}k"] = (
            f"{BUCKET}/skypilot/{_run}/checkpoints/orbax/{_step}",
            "encoder",
        )
# ftjepa 1.19M continued with its JEPA loss plus a CLS cross-view term, trained
# on Isambard GH200 (checkpoints pulled to ISAMBARD_RUNS).
WARM_START_RUNS = {
    "warmxv": "1p8b-ftjepa-warmxv-cls-w1e-1-10k-b1024-gh200x16-20261001-010230",
    "warmxvw1": "1p8b-ftjepa-warmxv-cls-w1-10k-b1024-gh200x16-20261001-032809",
    "warmxvmix": "1p8b-ftjepa-warmxv-cls-w1-mixed8-10k-b1024-gh200x16-20261001-033032",
    "warmctrl": "1p8b-ftjepa-warm-nopair-10k-b1024-gh200x16-20261001-055533",
}
for _arm, _run in WARM_START_RUNS.items():
    for _step in (2_500, 5_000, 7_500, 10_000):
        ENCODERS[f"{_arm}_{_step / 1000:g}k"] = (
            f"{ISAMBARD_RUNS}/{_run}/checkpoints/orbax/{_step}",
            "encoder",
        )
# Token-level cross-CE SupCon fine-tunes of ftjepa 1.19M (matched fragment
# peaks are positives), trained on Isambard GH200.
TOKEN_RUNS = {
    "gctoken": "1p8b-gctoken-crossce-ft1190k-10k-isb16-20261001-090654",
    "gctokena01": "1p8b-gctokenanchor0p1-crossce-ft1190k-10k-isb16-20261001-090654",
    "gctokencls": "1p8b-gctokencls-crossce-ft1190k-10k-isb16-20261001-090654",
}
for _arm, _run in TOKEN_RUNS.items():
    for _step in (2_500, 5_000, 7_500, 10_000):
        ENCODERS[f"{_arm}_{_step / 1000:g}k"] = (
            f"{ISAMBARD_RUNS}/{_run}/checkpoints/orbax/{_step}",
            "encoder",
        )
# ftjepa 1.19M continued with JEPA plus cross-CE token prediction (each paired
# spectrum predicts its partner's teacher tokens), Isambard GH200.
CROSS_CE_TOKEN_RUNS = {
    "xctw03": "1p8b-ftjepa-warm-crosscetoken-w0p3-mixed4-10k-gh200x16-20261001-111226",
    "xctw1": "1p8b-ftjepa-warm-crosscetoken-w1p0-mixed4-10k-gh200x16-20261001-111226",
    "xctw3": "1p8b-ftjepa-warm-crosscetoken-w3p0-mixed4-10k-gh200x16-20261001-111226",
}
for _arm, _run in CROSS_CE_TOKEN_RUNS.items():
    for _step in (2_500, 5_000, 7_500, 10_000):
        ENCODERS[f"{_arm}_{_step / 1000:g}k"] = (
            f"{ISAMBARD_RUNS}/{_run}/checkpoints/orbax/{_step}",
            "encoder",
        )
# Only the encoder is probed; these objectives share the JEPA encoder builder.
ENCODER_ONLY_TRAINING_MODES = {
    "sjepa": "jepa",
    "grouped_jepa": "jepa",
    "group_contrastive": "jepa",
}
LOCAL_TSV = Path("data/massspecgym/data/MassSpecGym1.5.tsv")
PROJECT_ROOT = Path("/root/spectra-learning")
VOLUME_ROOT = Path("/vol")
BIN_MAX_MZ = 1000
NUM_PAIRS = 200_000
KNN_K = 10
REGRESSION_KEYS = (
    "logp",
    "num_rings",
    "tpsa",
    "fraction_csp3",
    "num_aromatic_rings",
    "num_hbd",
    "num_hba",
)
ELEMENTS = ("N", "O", "F", "P", "S", "Cl", "Br")
# Final-output interfaces of width k: the first k channels (what a Matryoshka
# predictor sees) and the rank-k train-fold PCA projection (post-hoc control).
INTERFACE_WIDTHS = (256, 512, 1024)


def _ignore_project_file(path: Path) -> bool:
    parts = path.absolute().relative_to(Path.cwd().absolute()).parts
    return parts[0] not in {"configs", "scripts", "spectra_learning", "pyproject.toml"} or (
        "__pycache__" in parts
    )


image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("libgomp1")
    .pip_install_from_pyproject("pyproject.toml")
    .uv_pip_install(
        "torch==2.13.0",
        index_url="https://download.pytorch.org/whl/cu130",
        extra_index_url="https://pypi.org/simple",
    )
    .add_local_dir(
        ".",
        str(PROJECT_ROOT),
        copy=True,
        ignore=_ignore_project_file,
    )
    .env(
        {
            "PYTHONPATH": str(PROJECT_ROOT),
            "JAX_PLATFORMS": "cpu",
        }
    )
)

app = modal.App("spectra-layerwise-semantic-probe")
volume = modal.Volume.from_name("spectra-layerwise-semantic-probe", create_if_missing=True)


def _molecule_targets(tsv_path: Path) -> dict[str, np.ndarray]:
    """Per-spectrum molecule index plus per-molecule targets."""
    import pyarrow.csv as csv
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    from spectra_learning.data.massspec_targets import (
        FG_SMARTS,
        compute_maccs_fingerprint_bits_for_smiles,
        compute_morgan_fingerprint_bits_for_smiles,
    )

    rows = csv.read_csv(
        tsv_path,
        parse_options=csv.ParseOptions(delimiter="\t"),
        convert_options=csv.ConvertOptions(include_columns=["smiles", "inchikey"]),
    ).to_pydict()
    keys, molecule_index = np.unique(
        np.asarray(rows["inchikey"]), return_inverse=True
    )
    first = np.zeros(len(keys), dtype=np.int64)
    first[molecule_index[::-1]] = np.arange(len(molecule_index))[::-1]
    smiles = [rows["smiles"][i] for i in first]

    maccs, maccs_valid = compute_maccs_fingerprint_bits_for_smiles(smiles)
    morgan, morgan_valid = compute_morgan_fingerprint_bits_for_smiles(smiles)
    assert maccs_valid.all() and morgan_valid.all()
    patterns = {name: Chem.MolFromSmarts(s) for name, s in FG_SMARTS.items()}
    binary_names = [f"elem_{e}" for e in ELEMENTS] + [f"fg_{n}" for n in FG_SMARTS]
    binary = np.zeros((len(smiles), len(binary_names)), dtype=np.uint8)
    regression = np.zeros((len(smiles), len(REGRESSION_KEYS)), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        binary[i, : len(ELEMENTS)] = [e in symbols for e in ELEMENTS]
        binary[i, len(ELEMENTS) :] = [
            mol.HasSubstructMatch(p) for p in patterns.values()
        ]
        regression[i] = (
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
        )
    return {
        "molecule_index": molecule_index.astype(np.int32),
        "maccs": maccs.astype(np.uint8),
        "morgan_packed": np.packbits(morgan.astype(np.uint8), axis=1),
        "binary": binary,
        "binary_names": np.asarray(binary_names),
        "regression": regression,
    }


def _binned_spectra(data: dict[str, np.ndarray]) -> np.ndarray:
    """sqrt-intensity 1 Da bins, L2-normalized."""
    from spectra_learning.data.spectra import PEAK_MZ_MAX

    count = len(data["peak_mz"])
    bins = np.zeros((count, BIN_MAX_MZ), dtype=np.float32)
    rows = np.repeat(np.arange(count), data["peak_mz"].shape[1])
    mz = data["peak_mz"].reshape(-1) * PEAK_MZ_MAX
    cols = np.clip(mz.astype(np.int64), 0, BIN_MAX_MZ - 1)
    values = np.sqrt(
        np.where(data["peak_valid_mask"], data["peak_intensity"], 0.0)
    ).reshape(-1)
    np.add.at(bins, (rows, cols), values)
    bins /= np.maximum(np.linalg.norm(bins, axis=1, keepdims=True), 1e-8)
    return bins


def _checkpoint_config(checkpoint: str) -> dict[str, Any]:
    import etils.epath as epath

    metadata = epath.Path(f"{checkpoint}/metadata/metadata").read_text()
    return json.loads(metadata)["task_contract"]["config"]


def _stage_encoder(name: str, directory: Path) -> None:
    """Restores one encoder from GCS on CPU and writes a torch state dict."""
    import gc

    import torch
    from flax import nnx
    from ml_collections import config_dict

    from spectra_learning.models.factory_jax import build_model_from_config
    from spectra_learning.training.checkpointing_jax import (
        restore_jax_encoder_state,
        torch_encoder_state_dict_from_jax,
    )

    checkpoint, source_module = ENCODERS[name]
    config = _checkpoint_config(checkpoint or PERIODIC_CHECKPOINT)
    config["training_mode"] = ENCODER_ONLY_TRAINING_MODES.get(
        config["training_mode"], config["training_mode"]
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    if checkpoint is None:
        return
    model = build_model_from_config(config_dict.ConfigDict(config))
    template = nnx.as_pure(nnx.state(model.encoder))
    del model
    gc.collect()
    restored = restore_jax_encoder_state(
        checkpoint, template, source_module=source_module
    )
    del template
    torch.save(torch_encoder_state_dict_from_jax(restored), directory / "encoder.pt")


def _load_encoder(directory: Path, config: Any) -> Any:
    import torch

    from spectra_learning.models.factory import build_model_from_config

    if not (directory / "encoder.pt").exists():
        torch.manual_seed(0)
        return build_model_from_config(config).encoder.eval().to("cuda")
    # Encoder-only probing: objectives without JEPA target blocks (SJEPA) still
    # need the builder's target-block validation to pass.
    config.jepa_num_target_blocks = max(int(config.jepa_num_target_blocks), 1)
    with torch.device("meta"):
        model = build_model_from_config(config)
    encoder = model.encoder
    encoder.load_state_dict(
        torch.load(directory / "encoder.pt", mmap=True), assign=True
    )
    return encoder.eval().to("cuda")


def _pooled_layer_features(encoder: Any, data: dict, config: Any, batch_size: int):
    """Returns bf16 CPU tensors [layers, N, D] for mean and CLS pooling.

    Layer 0 is the embedder output entering block 0, layers 1..L are raw block
    outputs (residual stream), and layer L+1 is the final normed encoder output.
    """
    import torch

    from spectra_learning.models.spectrum_metadata import (
        torch_spectrum_metadata_from_batch,
    )

    count = len(data["peak_mz"])
    num_layers = len(encoder.blocks) + 2
    dim = int(config.model_dim)
    pooled = {
        pool: torch.empty((num_layers, count, dim), dtype=torch.bfloat16)
        for pool in ("mean", "cls")
    }
    captured: list[torch.Tensor] = []
    hooks = [
        encoder.blocks[0].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0])
        )
    ]
    hooks += [
        block.register_forward_hook(
            lambda _module, _args, output: captured.append(output)
        )
        for block in encoder.blocks
    ]
    metadata_names = (
        "precursor_mz",
        "collision_energy",
        "collision_energy_present",
        "charge",
        "charge_present",
        "polarity_id",
        "instrument_family_id",
    )
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            batch = {
                name: torch.from_numpy(data[name][start:end]).to("cuda")
                for name in (*metadata_names, "peak_mz", "peak_intensity", "peak_valid_mask")
            }
            spectrum_metadata = torch_spectrum_metadata_from_batch(
                batch, schema=str(config.encoder_metadata_schema)
            )
            captured.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = encoder(
                    batch["peak_mz"],
                    batch["peak_intensity"],
                    valid_mask=batch["peak_valid_mask"],
                    visible_mask=batch["peak_valid_mask"],
                    precursor_mz=batch["precursor_mz"],
                    spectrum_metadata=spectrum_metadata,
                )
            mask = batch["peak_valid_mask"].unsqueeze(-1).float()
            denominator = mask.sum(dim=1).clamp(min=1.0)
            for layer, tokens in enumerate([*captured, output]):
                tokens = tokens.float()
                num_peaks = mask.shape[1]
                mean = (tokens[:, :num_peaks] * mask).sum(dim=1) / denominator
                pooled["mean"][layer, start:end] = mean.bfloat16().cpu()
                pooled["cls"][layer, start:end] = tokens[:, num_peaks].bfloat16().cpu()
            if (start // batch_size) % 100 == 0:
                print(f"embedded {end}/{count}", flush=True)
    for hook in hooks:
        hook.remove()
    return pooled


def _auroc(scores, labels):
    """Column-wise AUROC; columns lacking either class give NaN."""
    import torch

    ranks = torch.empty_like(scores)
    order = scores.argsort(dim=0)
    ranks.scatter_(
        0,
        order,
        torch.arange(1, len(scores) + 1, device=scores.device, dtype=scores.dtype)
        .unsqueeze(1)
        .expand_as(scores)
        .contiguous(),
    )
    positives = labels.sum(dim=0)
    negatives = len(labels) - positives
    rank_sum = (ranks * labels).sum(dim=0)
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return torch.where((positives > 0) & (negatives > 0), auc, torch.nan)


def _rank(values):
    import torch

    ranks = torch.empty_like(values)
    ranks[values.argsort()] = torch.arange(
        len(values), device=values.device, dtype=values.dtype
    )
    return ranks


def _residualize(y, controls):
    y = y.double() - y.double().mean()
    design = controls.double() - controls.double().mean(0)
    beta = (design.T @ design).inverse() @ (design.T @ y)
    return (y - design @ beta).float()


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def _tanimoto(fp_a, fp_b):
    intersection = (fp_a * fp_b).sum(dim=-1)
    union = fp_a.sum(dim=-1) + fp_b.sum(dim=-1) - intersection
    return intersection / union.clamp(min=1.0)


class _Evaluator:
    def __init__(self, targets: dict, fold: np.ndarray, binned: np.ndarray, precursor):
        import torch

        self.split = {name: np.flatnonzero(fold == name) for name in ("train", "val", "test")}
        molecule = targets["molecule_index"]
        morgan = np.unpackbits(targets["morgan_packed"], axis=1)
        self.morgan = torch.from_numpy(morgan).to("cuda", torch.float16)
        self.molecule = torch.from_numpy(molecule.astype(np.int64)).to("cuda")
        binary = np.concatenate([targets["maccs"], targets["binary"]], axis=1)
        self.num_maccs = targets["maccs"].shape[1]
        self.binary_names = list(targets["binary_names"])
        self.binary = {
            name: torch.from_numpy(binary[molecule[idx]]).to("cuda", torch.float32)
            for name, idx in self.split.items()
        }
        regression = targets["regression"][molecule]
        mean = regression[self.split["train"]].mean(0)
        std = regression[self.split["train"]].std(0)
        self.regression = {
            name: torch.from_numpy((regression[idx] - mean) / std).to("cuda")
            for name, idx in self.split.items()
        }
        rng = np.random.default_rng(0)
        test = self.split["test"]
        a, b = rng.choice(test, NUM_PAIRS), rng.choice(test, NUM_PAIRS)
        keep = molecule[a] != molecule[b]
        self.pairs = (
            torch.from_numpy(a[keep]).to("cuda"),
            torch.from_numpy(b[keep]).to("cuda"),
        )
        binned_t = torch.from_numpy(binned).to("cuda")
        precursor_t = torch.from_numpy(precursor).to("cuda")
        spectral_cosine = (binned_t[self.pairs[0]] * binned_t[self.pairs[1]]).sum(-1)
        mass_gap = (precursor_t[self.pairs[0]] - precursor_t[self.pairs[1]]).abs()
        self.pair_tanimoto = _tanimoto(
            self.morgan[self.molecule[self.pairs[0]]].float(),
            self.morgan[self.molecule[self.pairs[1]]].float(),
        )
        self.pair_controls = torch.stack([_rank(spectral_cosine), _rank(mass_gap)], 1)
        self.pair_tanimoto_residual = _residualize(
            _rank(self.pair_tanimoto), self.pair_controls
        )
        self.precursor = precursor_t
        del binned_t

    def baselines(self) -> dict[str, float]:
        import torch

        train, test = (torch.from_numpy(self.split[s]).to("cuda") for s in ("train", "test"))
        random_nb = train[torch.randint(len(train), (len(test),), device="cuda")]
        train_mass = self.precursor[train]
        order = train_mass.argsort()
        pos = torch.searchsorted(train_mass[order], self.precursor[test]).clamp(
            max=len(train) - 1
        )
        mass_nb = train[order[pos]]
        return {
            "knn_top1_tanimoto_random": float(self._pair_tanimoto(test, random_nb).mean()),
            "knn_top1_tanimoto_precursor_mz": float(
                self._pair_tanimoto(test, mass_nb).mean()
            ),
        }

    def _pair_tanimoto(self, a, b):
        return _tanimoto(
            self.morgan[self.molecule[a]].float(), self.morgan[self.molecule[b]].float()
        )

    def evaluate(self, features) -> dict[str, float]:
        import torch

        x = {
            name: features[torch.from_numpy(idx)].to("cuda", torch.float32)
            for name, idx in self.split.items()
        }
        mean = x["train"].mean(0)
        std = x["train"].std(0).clamp(min=1e-6)
        x = {name: (v - mean) / std for name, v in x.items()}
        metrics = self._ridge(x)
        metrics.update(self._knn(x))
        metrics.update(self._pairwise(features))
        return metrics

    def _ridge(self, x) -> dict[str, float]:
        import torch

        xtr = x["train"].double()
        n, d = xtr.shape
        eigenvalues, eigenvectors = torch.linalg.eigh(xtr.T @ xtr / n)
        scale = float(eigenvalues.mean())
        lambdas = [scale * f for f in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)]
        metrics: dict[str, float] = {}
        groups = {"binary": self.binary, "regression": self.regression}
        for group, targets in groups.items():
            ytr = targets["train"].double()
            y_mean = ytr.mean(0)
            projected = eigenvectors.T @ (xtr.T @ (ytr - y_mean) / n)
            best = None
            for lam in lambdas:
                weights = (eigenvectors / (eigenvalues + lam)) @ projected
                weights = weights.float()
                val = self._score(group, x["val"] @ weights + y_mean.float(), targets["val"])
                if best is None or val["select"] > best[0]:
                    best = (val["select"], lam, weights)
            _, lam, weights = best
            test = self._score(group, x["test"] @ weights + y_mean.float(), targets["test"])
            test.pop("select")
            metrics.update(test)
            metrics[f"{group}_lambda_over_mean_eig"] = lam / scale
        return metrics

    def _score(self, group, prediction, target) -> dict[str, float]:
        import torch

        if group == "regression":
            r2 = 1 - ((prediction - target) ** 2).sum(0) / (
                (target - target.mean(0)) ** 2
            ).sum(0)
            out = {f"r2_{k}": float(v) for k, v in zip(REGRESSION_KEYS, r2)}
            out["r2_mean"] = float(r2.mean())
            out["select"] = out["r2_mean"]
            return out
        auc = _auroc(prediction, target)
        maccs = auc[: self.num_maccs]
        out = {"auc_maccs_macro": float(torch.nanmean(maccs))}
        for name, value in zip(self.binary_names, auc[self.num_maccs :]):
            out[f"auc_{name}"] = float(value)
        out["auc_fg_macro"] = float(
            torch.nanmean(auc[self.num_maccs + len(ELEMENTS) :])
        )
        out["select"] = out["auc_maccs_macro"]
        return out

    def _knn(self, x) -> dict[str, float]:
        import torch

        database = torch.nn.functional.normalize(x["train"], dim=1).half()
        queries = torch.nn.functional.normalize(x["test"], dim=1).half()
        train = torch.from_numpy(self.split["train"]).to("cuda")
        test = torch.from_numpy(self.split["test"]).to("cuda")
        top1, topk = [], []
        for start in range(0, len(queries), 2048):
            similarity = queries[start : start + 2048] @ database.T
            neighbours = train[similarity.topk(KNN_K, dim=1).indices]
            query = test[start : start + 2048]
            tanimoto = self._pair_tanimoto(
                query[:, None].expand_as(neighbours), neighbours
            )
            top1.append(tanimoto[:, 0])
            topk.append(tanimoto.mean(1))
        return {
            "knn_top1_tanimoto": float(torch.cat(top1).mean()),
            f"knn_top{KNN_K}_tanimoto": float(torch.cat(topk).mean()),
        }

    def _pairwise(self, features) -> dict[str, float]:
        import torch

        test = torch.from_numpy(self.split["test"])
        x = features[test].to("cuda", torch.float32)
        x = (x - x.mean(0)) / x.std(0).clamp(min=1e-6)
        lookup = torch.full((int(test.max()) + 1,), -1, dtype=torch.long, device="cuda")
        lookup[test.to("cuda")] = torch.arange(len(test), device="cuda")
        a = torch.nn.functional.normalize(x[lookup[self.pairs[0]]], dim=1)
        b = torch.nn.functional.normalize(x[lookup[self.pairs[1]]], dim=1)
        cosine_rank = _rank((a * b).sum(-1))
        return {
            "pair_spearman_tanimoto": _pearson(cosine_rank, _rank(self.pair_tanimoto)),
            "pair_partial_spearman_tanimoto": _pearson(
                _residualize(cosine_rank, self.pair_controls),
                self.pair_tanimoto_residual,
            ),
        }


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=200 * 1024,
    timeout=4 * 60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def probe_encoder(name: str, targets: dict[str, np.ndarray], batch_size: int = 256) -> dict:
    # The export module builds its Modal image from ./pyproject.toml at import.
    os.chdir(PROJECT_ROOT)

    import torch
    from ml_collections import config_dict

    from scripts.modal_export_jepa_fps_embeddings import _prepare_massspecgym_inputs

    started = time.time()
    checkpoint, source_module = ENCODERS[name]
    directory = VOLUME_ROOT / name
    config = config_dict.ConfigDict(json.loads((directory / "config.json").read_text()))
    data = _prepare_massspecgym_inputs(VOLUME_ROOT / LOCAL_TSV.name, config)
    binned = _binned_spectra(data)
    evaluator = _Evaluator(targets, data["fold"], binned, data["precursor_mz"])
    results: dict[str, Any] = {
        "encoder": name,
        "checkpoint": checkpoint,
        "source_module": source_module,
        "baselines": evaluator.baselines(),
        "layers": {},
    }
    if source_module == "binned":
        features = torch.from_numpy(
            np.concatenate([binned, data["precursor_mz"][:, None] / 1000.0], axis=1)
        )
        results["layers"]["binned"] = {"mean": evaluator.evaluate(features)}
    else:
        encoder = _load_encoder(directory, config)
        pooled = _pooled_layer_features(encoder, data, config, batch_size)
        del encoder
        torch.cuda.empty_cache()
        num_layers = pooled["mean"].shape[0]
        for layer in range(num_layers):
            label = (
                "embed" if layer == 0 else "final" if layer == num_layers - 1 else str(layer)
            )
            results["layers"][label] = {
                pool: evaluator.evaluate(pooled[pool][layer]) for pool in pooled
            }
            print(
                name,
                label,
                {
                    pool: round(values["auc_maccs_macro"], 4)
                    for pool, values in results["layers"][label].items()
                },
                flush=True,
            )
        train = torch.from_numpy(evaluator.split["train"])
        for pool in pooled:
            final = pooled[pool][-1].float()
            center = final[train].mean(0)
            _, _, basis = torch.linalg.svd(
                (final[train] - center).to("cuda"), full_matrices=False
            )
            for width in INTERFACE_WIDTHS:
                components = basis[:width].T.cpu()
                interfaces = {
                    f"final_prefix{width}": final[:, :width],
                    f"final_pca{width}": (final - center) @ components @ components.T
                    + center,
                }
                for label, features in interfaces.items():
                    results["layers"].setdefault(label, {})[pool] = evaluator.evaluate(
                        features
                    )
    results["seconds"] = time.time() - started
    return results


@app.local_entrypoint()
def stage(encoders: str = ",".join(ENCODERS), staging_dir: str = "") -> None:
    import tempfile

    root = Path(staging_dir or tempfile.mkdtemp())
    for name in encoders.split(","):
        _stage_encoder(name, root / name)
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(root / name), f"/{name}")
        for path in (root / name).iterdir():
            path.unlink()
        print(f"staged {name}", flush=True)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(LOCAL_TSV), f"/{LOCAL_TSV.name}")


@app.local_entrypoint()
def main(encoders: str = ",".join(ENCODERS), output_dir: str = "") -> None:
    names = encoders.split(",")
    targets = _molecule_targets(LOCAL_TSV)
    output = Path(output_dir) if output_dir else None
    for result in probe_encoder.map(names, kwargs={"targets": targets}):
        print(json.dumps({"encoder": result["encoder"], "seconds": result["seconds"]}))
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{result['encoder']}.json").write_text(json.dumps(result, indent=2))
