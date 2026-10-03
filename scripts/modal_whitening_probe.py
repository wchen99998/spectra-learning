"""Does post-hoc whitening recover the geometry a SIGReg-style regularizer buys?

Ridge probes are nearly invariant to invertible linear maps, but cosine kNN and
cosine-vs-Tanimoto rank correlation are not. For selected layers of staged
encoders, this evaluates raw mean-pooled features against train-fold whitened
ones (shrunk covariance ``C + eps * tr(C) / D * I``) with the same evaluator as
`modal_layerwise_semantic_probe.py`.

    .venv/bin/modal run scripts/modal_whitening_probe.py --output-dir DIR
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal
import numpy as np

from scripts.modal_layerwise_semantic_probe import (
    LOCAL_TSV,
    PROJECT_ROOT,
    VOLUME_ROOT,
    _binned_spectra,
    _Evaluator,
    _load_encoder,
    _molecule_targets,
    _pooled_layer_features,
    image,
    volume,
)

ENCODERS = ("ftjepa_1190k", "mae1b_2m", "sjepa_sigreg_2m")
# Block outputs are 1-indexed; -1 is the final normed output.
LAYERS = (13, 19, -1)
SHRINKAGE = (1e-3, 1e-1)

app = modal.App("spectra-whitening-probe")


def _whitened(features, train, eps: float):
    import torch

    x = features.to("cuda", torch.float64)
    mean = x[train].mean(0)
    centered = x[train] - mean
    covariance = centered.T @ centered / (len(centered) - 1)
    dim = covariance.shape[0]
    covariance += eps * covariance.trace() / dim * torch.eye(
        dim, dtype=covariance.dtype, device=covariance.device
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    transform = eigenvectors * eigenvalues.rsqrt()
    return ((x - mean) @ transform).float().cpu()


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=128 * 1024,
    timeout=3 * 60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def whitening_probe(name: str, targets: dict[str, np.ndarray], batch_size: int = 256) -> dict:
    os.chdir(PROJECT_ROOT)

    import torch
    from ml_collections import config_dict

    from scripts.modal_export_jepa_fps_embeddings import _prepare_massspecgym_inputs

    started = time.time()
    directory = VOLUME_ROOT / name
    config = config_dict.ConfigDict(json.loads((directory / "config.json").read_text()))
    data = _prepare_massspecgym_inputs(VOLUME_ROOT / LOCAL_TSV.name, config)
    evaluator = _Evaluator(
        targets, data["fold"], _binned_spectra(data), data["precursor_mz"]
    )
    encoder = _load_encoder(directory, config)
    pooled = _pooled_layer_features(encoder, data, config, batch_size)["mean"]
    del encoder
    torch.cuda.empty_cache()

    train = torch.from_numpy(data["fold"] == "train")
    results: dict = {"encoder": name, "layers": {}}
    for layer in LAYERS:
        label = "final" if layer == -1 else str(layer)
        features = pooled[layer].float()
        variants = {"raw": features}
        for eps in SHRINKAGE:
            variants[f"whiten_{eps:g}"] = _whitened(features, train, eps)
        results["layers"][label] = {
            variant: evaluator.evaluate(values) for variant, values in variants.items()
        }
        print(
            name,
            label,
            {
                variant: (
                    round(metrics["auc_maccs_macro"], 4),
                    round(metrics["knn_top1_tanimoto"], 4),
                    round(metrics["pair_spearman_tanimoto"], 4),
                    round(metrics["pair_partial_spearman_tanimoto"], 4),
                )
                for variant, metrics in results["layers"][label].items()
            },
            flush=True,
        )
    results["seconds"] = time.time() - started
    return results


@app.local_entrypoint()
def main(encoders: str = ",".join(ENCODERS), output_dir: str = "") -> None:
    targets = _molecule_targets(LOCAL_TSV)
    output = Path(output_dir) if output_dir else None
    for result in whitening_probe.map(encoders.split(","), kwargs={"targets": targets}):
        print(json.dumps({"encoder": result["encoder"], "seconds": result["seconds"]}))
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{result['encoder']}.json").write_text(json.dumps(result, indent=2))
