from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import modal


RUN_ID = "1b-grouped-jepa-xattn-init350k-v6e32-ew4-b256-ga2-la-20260816-022147"
CHECKPOINT = (
    "gs://metal-repeater-411410-spectra-checkpoints/skypilot/"
    f"{RUN_ID}/checkpoints/orbax/100000"
)
PARENT_CHECKPOINT = (
    "gs://metal-repeater-411410-spectra-checkpoints/skypilot/"
    "1b-singlemixer-n64-muon-v6e64-b4096-ga2-ctx60-t25-use5a-"
    "datapart-20260811-045140/checkpoints/orbax/350000"
)
DATASET_REPO = "roman-bushuiev/MassSpecGym"
DATASET_REVISION = "d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a"
DATASET_FILE = "data/MassSpecGym1.5.tsv"
DATA_ROOT = Path("/results/massspecgym_maccs_cls_step100000")
DEFAULT_RESULT_NAME = "massspecgym_maccs_cls_step100000"
DEFAULT_CONFIG_METADATA = "artifacts/checkpoint_metadata/training_metadata"
PROJECT_ROOT = Path("/root/spectra-learning")


def _ignore_project_file(path: Path) -> bool:
    parts = path.absolute().relative_to(Path.cwd().absolute()).parts
    return parts[0] in {".git", ".venv", ".pytest_cache", "data"} or "__pycache__" in parts


image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("libgomp1")
    .pip_install_from_pyproject("pyproject.toml")
    .uv_pip_install(
        "torch==2.13.0",
        index_url="https://download.pytorch.org/whl/cu130",
        extra_index_url="https://pypi.org/simple",
    )
    .uv_pip_install(
        "jax[cuda13]>=0.11.0",
        "flax>=0.12.8",
        "optax>=0.2.8",
        "orbax-checkpoint>=0.12.4",
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
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        }
    )
)

app = modal.App("spectra-massspecgym-maccs-cls-probe")
volume = modal.Volume.from_name("spectra-massspecgym-maccs-probe", create_if_missing=True)
gcp_secret = modal.Secret.from_name("spectra-probe-gcp-adc-20260818")


def _configure_google_credentials() -> None:
    keys = (
        "client_id",
        "client_secret",
        "refresh_token",
        "type",
        "quota_project_id",
        "universe_domain",
        "account",
    )
    credentials = {key: os.environ[key] for key in keys if key in os.environ}
    path = Path("/tmp/gcp-adc.json")
    path.write_text(json.dumps(credentials))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)


def _load_run_config(
    config_metadata_path: str = DEFAULT_CONFIG_METADATA,
) -> dict[str, Any]:
    payload = json.loads(
        (PROJECT_ROOT / config_metadata_path).read_text()
    )
    return payload["task_contract"]["config"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_dataset(
    tsv_path: Path,
    output_path: Path,
    config_metadata_path: str = DEFAULT_CONFIG_METADATA,
) -> dict[str, Any]:
    import numpy as np
    import pyarrow.csv as csv
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys

    from spectra_learning.data.spectra import (
        COLLISION_ENERGY_MAX,
        preprocess_peak_batch_numpy,
    )

    table = csv.read_csv(tsv_path, parse_options=csv.ParseOptions(delimiter="\t"))
    rows = table.to_pydict()
    count = table.num_rows
    config = _load_run_config(config_metadata_path)

    peak_mz = np.empty((count, int(config["num_peaks"])), dtype=np.float32)
    peak_intensity = np.empty_like(peak_mz)
    peak_valid_mask = np.empty_like(peak_mz, dtype=bool)
    precursor_mz = np.asarray(rows["precursor_mz"], dtype=np.float32)

    chunk_size = 4096
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        mz_lists = [np.fromstring(value, sep=",", dtype=np.float32) for value in rows["mzs"][start:end]]
        intensity_lists = [
            np.fromstring(value, sep=",", dtype=np.float32)
            for value in rows["intensities"][start:end]
        ]
        width = max(len(value) for value in mz_lists)
        spectra = np.zeros((end - start, 2, width), dtype=np.float32)
        for row_idx, (mz, intensity) in enumerate(
            zip(mz_lists, intensity_lists, strict=True)
        ):
            spectra[row_idx, 0, : len(mz)] = mz
            spectra[row_idx, 1, : len(intensity)] = intensity
        batch = preprocess_peak_batch_numpy(
            spectra,
            precursor_mz[start:end],
            num_peaks=int(config["num_peaks"]),
            peak_drop_min_intensity=float(config["peak_drop_min_intensity"]),
            peak_ordering=str(config["peak_ordering"]),
            max_precursor_mz=float(config["max_precursor_mz"]),
            precursor_peak_exclusion_window_da=float(
                config["precursor_peak_exclusion_window_da"]
            ),
            min_peak_intensity=float(config["min_peak_intensity"]),
        )
        peak_mz[start:end] = batch["peak_mz"]
        peak_intensity[start:end] = batch["peak_intensity"]
        peak_valid_mask[start:end] = batch["peak_valid_mask"]
        precursor_mz[start:end] = batch["precursor_mz"]

    unique_smiles, inverse = np.unique(np.asarray(rows["smiles"], dtype=str), return_inverse=True)
    unique_targets = np.empty((len(unique_smiles), 166), dtype=np.uint8)
    full = np.empty(167, dtype=np.int8)
    for idx, smiles in enumerate(unique_smiles):
        mol = Chem.MolFromSmiles(smiles)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), full)
        unique_targets[idx] = full[1:]
    targets = unique_targets[inverse]

    collision_energy = np.asarray(
        [0.0 if value is None else value for value in rows["collision_energy"]],
        dtype=np.float32,
    )
    collision_energy = np.clip(collision_energy, 0.0, COLLISION_ENERGY_MAX)
    collision_energy /= COLLISION_ENERGY_MAX
    charge = np.ones(count, dtype=np.float32)
    split = np.asarray(
        [{"train": 0, "val": 1, "test": 2}[value] for value in rows["fold"]],
        dtype=np.uint8,
    )

    np.savez(
        output_path,
        peak_mz=peak_mz,
        peak_intensity=peak_intensity,
        peak_valid_mask=peak_valid_mask,
        precursor_mz=precursor_mz,
        collision_energy=collision_energy,
        charge=charge,
        targets=targets,
        split=split,
    )
    return {
        "samples": count,
        "unique_smiles": len(unique_smiles),
        "train_samples": int(np.count_nonzero(split == 0)),
        "val_samples": int(np.count_nonzero(split == 1)),
        "test_samples": int(np.count_nonzero(split == 2)),
        "target_bits": 166,
        "tsv_sha256": _sha256(tsv_path),
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=6 * 60 * 60,
    secrets=[gcp_secret],
    volumes={"/results": volume},
)
def extract_embeddings(
    batch_size: int = 128,
    checkpoint: str = CHECKPOINT,
    result_name: str = DEFAULT_RESULT_NAME,
    representation: str = "cls",
    run_id: str = RUN_ID,
    checkpoint_step: int = 100000,
    checkpoint_role: str = "grouped_jepa_final",
    config_metadata_path: str = DEFAULT_CONFIG_METADATA,
) -> dict[str, Any]:
    _configure_google_credentials()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from huggingface_hub import hf_hub_download
    from ml_collections import config_dict

    from spectra_learning.models.factory_jax import build_model_from_config
    from spectra_learning.models.spectrum_metadata import jax_spectrum_metadata_from_batch
    from spectra_learning.training.checkpointing_jax import restore_jax_encoder_state

    started = time.time()
    result_root = Path("/results") / result_name
    result_root.mkdir(parents=True, exist_ok=True)
    dataset_path = DATA_ROOT / "prepared_massspecgym.npz"
    preparation_metadata_path = DATA_ROOT / "preparation_metadata.json"
    metadata_path = result_root / "embedding_metadata.json"
    embeddings_path = result_root / f"{representation}_embeddings.float16.npy"

    tsv_path = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=DATASET_FILE,
            local_dir="/tmp/massspecgym",
        )
    )
    preparation_started = time.time()
    if preparation_metadata_path.exists():
        data_metadata = json.loads(preparation_metadata_path.read_text())
    else:
        data_metadata = _prepare_dataset(
            tsv_path,
            dataset_path,
            config_metadata_path=config_metadata_path,
        )
        preparation_metadata_path.write_text(
            json.dumps(data_metadata, indent=2, sort_keys=True)
        )
        volume.commit()
    preparation_seconds = time.time() - preparation_started
    data = np.load(dataset_path, mmap_mode="r")

    config = config_dict.ConfigDict(_load_run_config(config_metadata_path))
    model = build_model_from_config(config)
    encoder_state = nnx.as_pure(nnx.state(model.encoder))
    restored = restore_jax_encoder_state(checkpoint, encoder_state)
    nnx.update(model.encoder, restored)
    encoder = model.encoder
    del model, restored, encoder_state

    @nnx.jit
    def encode_tokens(
        encoder: nnx.Module,
        mz: jax.Array,
        intensity: jax.Array,
        valid_mask: jax.Array,
        precursor: jax.Array,
        collision_energy: jax.Array,
        charge: jax.Array,
    ) -> jax.Array:
        batch = {
            "collision_energy": collision_energy,
            "charge": charge,
        }
        metadata = jax_spectrum_metadata_from_batch(batch)
        tokens = encoder(
            mz,
            intensity,
            valid_mask=valid_mask,
            visible_mask=valid_mask,
            precursor_mz=precursor,
            spectrum_metadata=metadata,
        )
        return tokens.astype(jnp.float32)

    count = int(data["peak_mz"].shape[0])
    num_peak_tokens = int(data["peak_mz"].shape[1])
    embedding_shape = (
        (count, int(config.model_dim))
        if representation == "cls"
        else (count, num_peak_tokens, int(config.model_dim))
    )
    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float16,
        shape=embedding_shape,
    )
    cls_embeddings = (
        np.lib.format.open_memmap(
            result_root / "cls_embeddings.float16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(count, int(config.model_dim)),
        )
        if representation != "cls"
        else None
    )
    extraction_started = time.time()
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        take = end - start

        def padded(name: str) -> np.ndarray:
            value = np.asarray(data[name][start:end])
            if take == batch_size:
                return value
            return np.pad(value, ((0, batch_size - take),) + ((0, 0),) * (value.ndim - 1))

        tokens = encode_tokens(
            encoder,
            jnp.asarray(padded("peak_mz")),
            jnp.asarray(padded("peak_intensity")),
            jnp.asarray(padded("peak_valid_mask")),
            jnp.asarray(padded("precursor_mz")),
            jnp.asarray(padded("collision_energy")),
            jnp.asarray(padded("charge")),
        )
        selected = (
            tokens[:take, -1]
            if representation == "cls"
            else tokens[:take, :num_peak_tokens]
        )
        embeddings[start:end] = np.asarray(selected, dtype=np.float16)
        if cls_embeddings is not None:
            cls_embeddings[start:end] = np.asarray(tokens[:take, -1], dtype=np.float16)
        if start == 0 or (start // batch_size) % 100 == 0:
            print(f"embedded {end}/{count}", flush=True)
    embeddings.flush()
    if cls_embeddings is not None:
        cls_embeddings.flush()
    extraction_seconds = time.time() - extraction_started

    metadata = {
        **data_metadata,
        "run_id": run_id,
        "run_url": f"https://wandb.ai/iclac/jepa-finalrun/runs/{run_id}",
        "checkpoint": checkpoint,
        "checkpoint_step": checkpoint_step,
        "checkpoint_role": checkpoint_role,
        "config_metadata_path": config_metadata_path,
        "encoder": "student",
        "embedding": "final_cls" if representation == "cls" else "valid_peak_tokens",
        "embedding_shape": list(embedding_shape),
        "embedding_dim": int(config.model_dim),
        "embedding_dtype": "float16",
        "also_saved_final_cls": cls_embeddings is not None,
        "batch_size": batch_size,
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_file": DATASET_FILE,
        "preprocessing_seconds": preparation_seconds,
        "extraction_seconds": extraction_seconds,
        "total_seconds": time.time() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    volume.commit()
    return metadata


class ProbeMetrics:
    @staticmethod
    def compute(
        targets: Any,
        probabilities: Any,
        thresholds: Any,
    ) -> tuple[dict[str, float], list[dict[str, float | int]]]:
        import numpy as np
        from sklearn.metrics import average_precision_score, roc_auc_score

        targets = np.asarray(targets, dtype=np.uint8)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        predictions = probabilities >= np.asarray(thresholds)[None]
        positives = targets.sum(axis=0)
        valid = (positives > 0) & (positives < len(targets))
        target_bool = targets.astype(bool)
        tp = np.count_nonzero(predictions & target_bool, axis=0)
        fp = np.count_nonzero(predictions & ~target_bool, axis=0)
        fn = np.count_nonzero(~predictions & target_bool, axis=0)
        precision = np.divide(tp, tp + fp, out=np.zeros(166), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros(166), where=(tp + fn) > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(166),
            where=(precision + recall) > 0,
        )
        auc = np.full(166, np.nan)
        ap = np.full(166, np.nan)
        for bit in np.flatnonzero(valid):
            auc[bit] = roc_auc_score(targets[:, bit], probabilities[:, bit])
            ap[bit] = average_precision_score(targets[:, bit], probabilities[:, bit])

        tp_micro = tp.sum()
        fp_micro = fp.sum()
        fn_micro = fn.sum()
        precision_micro = tp_micro / (tp_micro + fp_micro)
        recall_micro = tp_micro / (tp_micro + fn_micro)
        intersection = np.count_nonzero(predictions & target_bool, axis=1)
        union = np.count_nonzero(predictions | target_bool, axis=1)
        target_norm = np.linalg.norm(targets, axis=1)
        probability_norm = np.linalg.norm(probabilities, axis=1)
        eps = 1e-7
        clipped = probabilities.clip(eps, 1.0 - eps)
        metrics = {
            "roc_auc_macro": float(np.nanmean(auc)),
            "roc_auc_weighted": float(np.average(auc[valid], weights=positives[valid])),
            "roc_auc_micro": float(roc_auc_score(targets.ravel(), probabilities.ravel())),
            "average_precision_macro": float(np.nanmean(ap)),
            "average_precision_weighted": float(np.average(ap[valid], weights=positives[valid])),
            "average_precision_micro": float(
                average_precision_score(targets.ravel(), probabilities.ravel())
            ),
            "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()),
            "f1_macro": float(f1.mean()),
            "precision_micro": float(precision_micro),
            "recall_micro": float(recall_micro),
            "f1_micro": float(
                2 * precision_micro * recall_micro / (precision_micro + recall_micro)
            ),
            "sample_tanimoto": float(np.mean(intersection / np.maximum(union, 1))),
            "sample_cosine": float(
                np.mean(
                    np.sum(probabilities * targets, axis=1)
                    / np.maximum(probability_norm * target_norm, 1e-12)
                )
            ),
            "exact_match_accuracy": float(np.mean(np.all(predictions == target_bool, axis=1))),
            "hamming_loss": float(np.mean(predictions != target_bool)),
            "binary_cross_entropy": float(
                np.mean(-(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))
            ),
            "brier_score": float(np.mean((probabilities - targets) ** 2)),
            "label_prevalence": float(targets.mean()),
            "predicted_positive_rate": float(predictions.mean()),
            "valid_roc_auc_bits": int(valid.sum()),
            "samples": int(len(targets)),
        }
        per_bit = [
            {
                "bit": bit + 1,
                "positives": int(positives[bit]),
                "prevalence": float(positives[bit] / len(targets)),
                "roc_auc": float(auc[bit]),
                "average_precision": float(ap[bit]),
                "precision": float(precision[bit]),
                "recall": float(recall[bit]),
                "f1": float(f1[bit]),
                "threshold": float(np.asarray(thresholds)[bit]),
            }
            for bit in range(166)
        ]
        return metrics, per_bit


def _validation_f1_thresholds(targets: Any, probabilities: Any) -> Any:
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    thresholds = np.full(166, 0.5, dtype=np.float32)
    for bit in range(166):
        precision, recall, candidates = precision_recall_curve(
            targets[:, bit], probabilities[:, bit]
        )
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
            precision[:-1] + recall[:-1], 1e-12
        )
        thresholds[bit] = candidates[int(np.argmax(f1))]
    return thresholds


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    timeout=4 * 60 * 60,
    volumes={"/results": volume},
)
def train_probes(
    seeds: tuple[int, ...] = (66, 67, 68),
    batch_size: int = 4096,
    max_epochs: int = 100,
    result_name: str = DEFAULT_RESULT_NAME,
    artifact_prefix: str = "",
) -> dict[str, Any]:
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    started = time.time()
    result_root = Path("/results") / result_name
    data = np.load(DATA_ROOT / "prepared_massspecgym.npz")
    embeddings = np.load(result_root / "cls_embeddings.float16.npy", mmap_mode="r")
    targets = data["targets"].astype(np.float32)
    split = data["split"]
    indices = {name: np.flatnonzero(split == value) for name, value in {"train": 0, "val": 1, "test": 2}.items()}

    train_embeddings = np.asarray(embeddings[indices["train"]], dtype=np.float32)
    mean = train_embeddings.mean(axis=0)
    std = train_embeddings.std(axis=0).clip(min=1e-6)
    x = {
        name: torch.from_numpy(
            ((np.asarray(embeddings[idx], dtype=np.float32) - mean) / std)
        ).to("cuda")
        for name, idx in indices.items()
    }
    y = {
        name: torch.from_numpy(targets[idx]).to("cuda")
        for name, idx in indices.items()
    }
    del train_embeddings

    results: dict[str, Any] = {
        "methodology": {
            "frozen_encoder": True,
            "embedding_standardization": "per-dimension train mean/std",
            "loss": "unweighted binary cross entropy with logits",
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "scheduler": "0.5-epoch linear warmup then cosine to 1% of initial LR",
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "early_stopping": "validation macro AUROC; minimum 20 epochs; patience 5",
            "model_selection": "best validation macro AUROC",
            "linear_architecture": "Linear(1536, 166)",
            "mlp_architecture": "Linear(1536, 256), SiLU, Linear(256, 166)",
            "seeds": list(seeds),
            "default_threshold": 0.5,
            "tuned_threshold": "per-bit validation F1 optimum",
        },
        "probes": {},
    }

    output_dir = result_root / f"{artifact_prefix}probe_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_size = len(indices["train"])
    steps_per_epoch = math.ceil(train_size / batch_size)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = max(1, round(0.5 * steps_per_epoch))

    for probe_name in ("linear", "mlp"):
        probe_runs = []
        for seed in seeds:
            torch.manual_seed(seed)
            if probe_name == "linear":
                model = torch.nn.Linear(1536, 166, device="cuda")
            else:
                model = torch.nn.Sequential(
                    torch.nn.Linear(1536, 256),
                    torch.nn.SiLU(),
                    torch.nn.Linear(256, 166),
                ).to("cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)

            def lr_factor(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
            best_auc = -math.inf
            best_epoch = -1
            best_state = None
            stale_epochs = 0
            epoch_history = []
            generator = torch.Generator(device="cuda").manual_seed(seed)

            for epoch in range(max_epochs):
                model.train()
                permutation = torch.randperm(train_size, generator=generator, device="cuda")
                loss_sum = 0.0
                for start in range(0, train_size, batch_size):
                    batch_idx = permutation[start : start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(x["train"][batch_idx])
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, y["train"][batch_idx]
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    loss_sum += float(loss) * len(batch_idx)

                model.eval()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    val_probabilities = torch.sigmoid(model(x["val"])).float().cpu().numpy()
                val_targets = targets[indices["val"]]
                positive = val_targets.sum(axis=0)
                valid = (positive > 0) & (positive < len(val_targets))
                val_auc = float(
                    roc_auc_score(
                        val_targets[:, valid],
                        val_probabilities[:, valid],
                        average="macro",
                    )
                )
                epoch_history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": loss_sum / train_size,
                        "val_roc_auc_macro": val_auc,
                    }
                )
                print(
                    f"{probe_name} seed={seed} epoch={epoch + 1} "
                    f"loss={loss_sum / train_size:.6f} val_auc={val_auc:.6f}",
                    flush=True,
                )
                if val_auc > best_auc + 1e-4:
                    best_auc = val_auc
                    best_epoch = epoch + 1
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if epoch + 1 >= 20 and stale_epochs >= 5:
                    break

            model.load_state_dict(best_state)
            model.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                val_probabilities = torch.sigmoid(model(x["val"])).float().cpu().numpy()
                test_probabilities = torch.sigmoid(model(x["test"])).float().cpu().numpy()
            val_targets = targets[indices["val"]]
            test_targets = targets[indices["test"]]
            tuned_thresholds = _validation_f1_thresholds(val_targets, val_probabilities)
            default_metrics, default_per_bit = ProbeMetrics.compute(
                test_targets, test_probabilities, np.full(166, 0.5)
            )
            tuned_metrics, tuned_per_bit = ProbeMetrics.compute(
                test_targets, test_probabilities, tuned_thresholds
            )
            run_result = {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_roc_auc_macro": best_auc,
                "test_default_threshold": default_metrics,
                "test_val_f1_threshold": tuned_metrics,
                "history": epoch_history,
            }
            probe_runs.append(run_result)
            torch.save(best_state, output_dir / f"{probe_name}_seed{seed}.pt")
            np.savez_compressed(
                output_dir / f"{probe_name}_seed{seed}_predictions.npz",
                test_indices=indices["test"],
                targets=test_targets.astype(np.uint8),
                probabilities=test_probabilities.astype(np.float32),
                val_f1_thresholds=tuned_thresholds,
            )
            (output_dir / f"{probe_name}_seed{seed}_per_bit.json").write_text(
                json.dumps(
                    {
                        "default_threshold": default_per_bit,
                        "val_f1_threshold": tuned_per_bit,
                    },
                    indent=2,
                    allow_nan=True,
                )
            )

        aggregate: dict[str, dict[str, float]] = {}
        for threshold_name in ("test_default_threshold", "test_val_f1_threshold"):
            keys = probe_runs[0][threshold_name].keys()
            aggregate[threshold_name] = {}
            for key in keys:
                values = np.asarray([run[threshold_name][key] for run in probe_runs], dtype=float)
                aggregate[threshold_name][f"{key}_mean"] = float(values.mean())
                aggregate[threshold_name][f"{key}_std"] = float(values.std(ddof=1))
        results["probes"][probe_name] = {"runs": probe_runs, "aggregate": aggregate}

    results["runtime_seconds"] = time.time() - started
    results["torch_version"] = torch.__version__
    results["device"] = torch.cuda.get_device_name()
    (result_root / f"{artifact_prefix}results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=True)
    )
    volume.commit()
    return results


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=6 * 60 * 60,
    volumes={"/results": volume},
)
def train_covariance_probes(
    seeds: tuple[int, ...] = (66, 67, 68),
    batch_size: int = 1024,
    max_epochs: int = 100,
    result_name: str = DEFAULT_RESULT_NAME,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    class CovarianceProbe(torch.nn.Module):
        def __init__(self, probe_name: str) -> None:
            super().__init__()
            self.left = torch.nn.Linear(1536, 64, bias=False)
            self.right = torch.nn.Linear(1536, 64, bias=False)
            torch.nn.init.xavier_normal_(self.left.weight)
            torch.nn.init.xavier_normal_(self.right.weight)
            self.head = (
                torch.nn.Linear(4096, 166)
                if probe_name == "linear"
                else torch.nn.Sequential(
                    torch.nn.Linear(4096, 256),
                    torch.nn.SiLU(),
                    torch.nn.Linear(256, 166),
                )
            )

        def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
            tokens = tokens.float()
            mask = valid_mask.unsqueeze(-1).float()
            left = self.left(tokens) * mask
            right = self.right(tokens) * mask
            covariance = left.transpose(1, 2) @ right
            covariance = covariance / mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            return self.head(covariance.flatten(start_dim=1))

    started = time.time()
    torch.set_float32_matmul_precision("high")
    result_root = Path("/results") / result_name
    data = np.load(DATA_ROOT / "prepared_massspecgym.npz")
    embeddings = np.load(
        result_root / "covariance_embeddings.float16.npy",
        mmap_mode="r",
    )
    tokens = torch.from_numpy(embeddings).to("cuda")
    valid_mask = torch.from_numpy(data["peak_valid_mask"]).to("cuda")
    targets = data["targets"].astype(np.float32)
    target_tensor = torch.from_numpy(targets).to("cuda")
    split = data["split"]
    indices = {
        name: np.flatnonzero(split == value)
        for name, value in {"train": 0, "val": 1, "test": 2}.items()
    }
    device_indices = {
        name: torch.from_numpy(value).to("cuda") for name, value in indices.items()
    }

    results: dict[str, Any] = {
        "methodology": {
            "frozen_encoder": True,
            "representation": "all valid final-layer peak-token embeddings; CLS excluded",
            "covariance_pooling": (
                "uncentered 64x64 cross-covariance of independently learned left/right "
                "projections, averaged over valid tokens"
            ),
            "covariance_input_dim": 1536,
            "covariance_compressed_dim": 64,
            "covariance_output_dim": 4096,
            "covariance_pooler_trainable": True,
            "embedding_standardization": "none",
            "loss": "unweighted binary cross entropy with logits",
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "scheduler": "0.5-epoch linear warmup then cosine to 1% of initial LR",
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "early_stopping": "validation macro AUROC; minimum 20 epochs; patience 5",
            "model_selection": "best validation macro AUROC",
            "linear_architecture": "CovariancePool(1536, 64) then Linear(4096, 166)",
            "mlp_architecture": (
                "CovariancePool(1536, 64) then Linear(4096, 256), SiLU, "
                "Linear(256, 166)"
            ),
            "seeds": list(seeds),
            "default_threshold": 0.5,
            "tuned_threshold": "per-bit validation F1 optimum",
        },
        "probes": {},
    }

    output_dir = result_root / "probe_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_size = len(indices["train"])
    steps_per_epoch = math.ceil(train_size / batch_size)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = max(1, round(0.5 * steps_per_epoch))

    def predict(model: torch.nn.Module, split_name: str) -> np.ndarray:
        probabilities = []
        model.eval()
        with torch.inference_mode():
            split_indices = device_indices[split_name]
            for start in range(0, len(split_indices), batch_size):
                batch_idx = split_indices[start : start + batch_size]
                logits = model(tokens[batch_idx], valid_mask[batch_idx])
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probabilities)

    for probe_name in ("linear", "mlp"):
        probe_runs = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = CovarianceProbe(probe_name).to("cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)

            def lr_factor(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
            best_auc = -math.inf
            best_epoch = -1
            best_state = None
            stale_epochs = 0
            epoch_history = []
            generator = torch.Generator(device="cuda").manual_seed(seed)

            for epoch in range(max_epochs):
                model.train()
                permutation = torch.randperm(train_size, generator=generator, device="cuda")
                loss_sum = 0.0
                for start in range(0, train_size, batch_size):
                    order = permutation[start : start + batch_size]
                    batch_idx = device_indices["train"][order]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(tokens[batch_idx], valid_mask[batch_idx])
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits,
                        target_tensor[batch_idx],
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    loss_sum += float(loss) * len(batch_idx)

                val_probabilities = predict(model, "val")
                val_targets = targets[indices["val"]]
                positive = val_targets.sum(axis=0)
                valid = (positive > 0) & (positive < len(val_targets))
                val_auc = float(
                    roc_auc_score(
                        val_targets[:, valid],
                        val_probabilities[:, valid],
                        average="macro",
                    )
                )
                epoch_history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": loss_sum / train_size,
                        "val_roc_auc_macro": val_auc,
                    }
                )
                print(
                    f"covariance {probe_name} seed={seed} epoch={epoch + 1} "
                    f"loss={loss_sum / train_size:.6f} val_auc={val_auc:.6f}",
                    flush=True,
                )
                if val_auc > best_auc + 1e-4:
                    best_auc = val_auc
                    best_epoch = epoch + 1
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if epoch + 1 >= 20 and stale_epochs >= 5:
                    break

            model.load_state_dict(best_state)
            val_probabilities = predict(model, "val")
            test_probabilities = predict(model, "test")
            val_targets = targets[indices["val"]]
            test_targets = targets[indices["test"]]
            tuned_thresholds = _validation_f1_thresholds(val_targets, val_probabilities)
            default_metrics, default_per_bit = ProbeMetrics.compute(
                test_targets,
                test_probabilities,
                np.full(166, 0.5),
            )
            tuned_metrics, tuned_per_bit = ProbeMetrics.compute(
                test_targets,
                test_probabilities,
                tuned_thresholds,
            )
            run_result = {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_roc_auc_macro": best_auc,
                "test_default_threshold": default_metrics,
                "test_val_f1_threshold": tuned_metrics,
                "history": epoch_history,
            }
            probe_runs.append(run_result)
            torch.save(best_state, output_dir / f"{probe_name}_seed{seed}.pt")
            np.savez_compressed(
                output_dir / f"{probe_name}_seed{seed}_predictions.npz",
                test_indices=indices["test"],
                targets=test_targets.astype(np.uint8),
                probabilities=test_probabilities.astype(np.float32),
                val_f1_thresholds=tuned_thresholds,
            )
            (output_dir / f"{probe_name}_seed{seed}_per_bit.json").write_text(
                json.dumps(
                    {
                        "default_threshold": default_per_bit,
                        "val_f1_threshold": tuned_per_bit,
                    },
                    indent=2,
                    allow_nan=True,
                )
            )
            del model, optimizer
            torch.cuda.empty_cache()

        aggregate: dict[str, dict[str, float]] = {}
        for threshold_name in ("test_default_threshold", "test_val_f1_threshold"):
            keys = probe_runs[0][threshold_name].keys()
            aggregate[threshold_name] = {}
            for key in keys:
                values = np.asarray(
                    [run[threshold_name][key] for run in probe_runs],
                    dtype=float,
                )
                aggregate[threshold_name][f"{key}_mean"] = float(values.mean())
                aggregate[threshold_name][f"{key}_std"] = float(values.std(ddof=1))
        results["probes"][probe_name] = {
            "runs": probe_runs,
            "aggregate": aggregate,
        }

    results["runtime_seconds"] = time.time() - started
    results["torch_version"] = torch.__version__
    results["device"] = torch.cuda.get_device_name()
    (result_root / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=True)
    )
    volume.commit()
    return results


@app.local_entrypoint()
def main(
    batch_size: int = 128,
    probe_batch_size: int = 4096,
    max_epochs: int = 100,
    checkpoint: str = CHECKPOINT,
    result_name: str = DEFAULT_RESULT_NAME,
    representation: str = "cls",
    compare_cls: bool = False,
    run_id: str = RUN_ID,
    checkpoint_step: int = 100000,
    checkpoint_role: str = "grouped_jepa_final",
    config_metadata_path: str = DEFAULT_CONFIG_METADATA,
) -> None:
    embedding_metadata = extract_embeddings.remote(
        batch_size=batch_size,
        checkpoint=checkpoint,
        result_name=result_name,
        representation=representation,
        run_id=run_id,
        checkpoint_step=checkpoint_step,
        checkpoint_role=checkpoint_role,
        config_metadata_path=config_metadata_path,
    )
    print(json.dumps(embedding_metadata, indent=2, sort_keys=True))
    train = train_probes if representation == "cls" else train_covariance_probes
    results = train.remote(
        batch_size=probe_batch_size,
        max_epochs=max_epochs,
        result_name=result_name,
    )
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=True))
    if compare_cls:
        cls_results = train_probes.remote(
            batch_size=4096,
            max_epochs=max_epochs,
            result_name=result_name,
            artifact_prefix="cls_",
        )
        print(json.dumps(cls_results, indent=2, sort_keys=True, allow_nan=True))
