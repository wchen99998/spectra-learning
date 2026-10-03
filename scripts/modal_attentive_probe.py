"""Attentive (PMA cross-attention) probes of frozen encoders on MassSpecGym 1.5.

Reuses the encoders staged by `modal_layerwise_semantic_probe.py::stage`. For
each selected depth, valid peak tokens plus CLS are cached, then two probes are
trained with the same optimizer and early stopping on val MACCS AUROC:

- attentive: MsgPmaPool (learned seeds cross-attending over tokens) + linear;
- mean_linear: linear head on mean-pooled tokens (the SGD counterpart of the
  ridge probe, isolating the gain from attention pooling).

    .venv/bin/modal run scripts/modal_attentive_probe.py --output-dir DIR
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import modal
import numpy as np

from scripts.modal_layerwise_semantic_probe import (
    LOCAL_TSV,
    PROJECT_ROOT,
    REGRESSION_KEYS,
    VOLUME_ROOT,
    _auroc,
    _load_encoder,
    _molecule_targets,
    image,
    volume,
)

ENCODERS = ("mae_666k", "ftjepa_1190k", "periodic_student_1209k")
# Block outputs are 1-indexed; 0 would be the embedder output.
LAYERS = (13, 19, "final")
SEEDS = (0, 1)
EPOCHS = 12
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
NUM_SEEDS = 4
NUM_HEADS = 16

app = modal.App("spectra-attentive-probe")


def _packed_layer_tokens(encoder: Any, data: dict, config: Any, batch_size: int):
    """bf16 CPU [total_tokens, D] per layer: each spectrum's valid peaks, then CLS."""
    import torch

    from spectra_learning.models.spectrum_metadata import (
        torch_spectrum_metadata_from_batch,
    )

    valid = data["peak_valid_mask"]
    lengths = valid.sum(1) + 1
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    dim = int(config.model_dim)
    tokens = {
        layer: torch.empty((int(offsets[-1]), dim), dtype=torch.bfloat16)
        for layer in LAYERS
    }
    captured: dict[Any, torch.Tensor] = {}
    hooks = [
        encoder.blocks[layer - 1].register_forward_hook(
            lambda _module, _args, output, layer=layer: captured.__setitem__(layer, output)
        )
        for layer in LAYERS
        if layer != "final"
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
    count = len(valid)
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                captured["final"] = encoder(
                    batch["peak_mz"],
                    batch["peak_intensity"],
                    valid_mask=batch["peak_valid_mask"],
                    visible_mask=batch["peak_valid_mask"],
                    precursor_mz=batch["precursor_mz"],
                    spectrum_metadata=spectrum_metadata,
                )
            # Appending the CLS slot as valid keeps spectra contiguous in order.
            keep = torch.cat(
                [batch["peak_valid_mask"], torch.ones_like(batch["peak_valid_mask"][:, :1])],
                dim=1,
            )
            rows = slice(int(offsets[start]), int(offsets[end]))
            for layer in LAYERS:
                tokens[layer][rows] = captured[layer][keep].bfloat16().cpu()
            if (start // batch_size) % 100 == 0:
                print(f"embedded {end}/{count}", flush=True)
    for hook in hooks:
        hook.remove()
    return tokens, offsets


class _Probe:
    def __init__(self, kind: str, dim: int, num_outputs: int, seed: int) -> None:
        import torch

        from spectra_learning.probes.massspec.msg_modules import MsgPmaPool

        torch.manual_seed(seed)
        self.kind = kind
        self.pool = (
            MsgPmaPool(input_dim=dim, num_seeds=NUM_SEEDS, num_heads=NUM_HEADS)
            if kind == "attentive"
            else None
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim), torch.nn.Linear(dim, num_outputs)
        )
        modules = [m for m in (self.pool, self.head) if m is not None]
        self.module = torch.nn.ModuleList(modules).to("cuda")

    def __call__(self, tokens, mask):
        if self.pool is None:
            weights = mask.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * weights).sum(1) / weights.sum(1)
        else:
            pooled = self.pool(tokens, mask)
        return self.head(pooled.float())


def _train_probe(kind, tokens, offsets, split, binary, regression, seed) -> dict:
    import torch
    import torch.nn.functional as F

    dim = tokens.shape[1]
    num_binary = binary.shape[1]
    probe = _Probe(kind, dim, num_binary + regression.shape[1], seed)
    optimizer = torch.optim.AdamW(
        probe.module.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = len(split["train"]) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    warmup = steps_per_epoch // 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup)
        * 0.5
        * (1 + np.cos(np.pi * min(step, total_steps) / total_steps)),
    )
    offsets_t = torch.from_numpy(offsets).to("cuda")

    def gather(indices):
        start = offsets_t[indices]
        lengths = offsets_t[indices + 1] - start
        width = int(lengths.max())
        positions = start[:, None] + torch.arange(width, device="cuda")
        mask = torch.arange(width, device="cuda")[None] < lengths[:, None]
        batch = tokens[positions.clamp(max=tokens.shape[0] - 1)]
        return F.layer_norm(batch.float(), (dim,)).bfloat16(), mask

    def predict(indices):
        outputs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(indices), 2048):
                chunk = indices[start : start + 2048]
                outputs.append(probe(*gather(chunk)).float())
        return torch.cat(outputs)

    def score(indices) -> dict[str, float]:
        prediction = predict(indices)
        auc = _auroc(prediction[:, :num_binary], binary[indices])
        target = regression[indices]
        residual = ((prediction[:, num_binary:] - target) ** 2).sum(0)
        r2 = 1 - residual / ((target - target.mean(0)) ** 2).sum(0)
        return {
            "auc_maccs_macro": float(torch.nanmean(auc[:166])),
            "auc_elem_F": float(auc[166 + 2]),
            "auc_elem_S": float(auc[166 + 4]),
            "auc_fg_macro": float(torch.nanmean(auc[166 + 7 :])),
            "r2_mean": float(r2.mean()),
            **{f"r2_{k}": float(v) for k, v in zip(REGRESSION_KEYS, r2)},
        }

    generator = torch.Generator(device="cuda").manual_seed(seed)
    best: dict[str, Any] = {"val": {"auc_maccs_macro": -1.0}}
    for epoch in range(EPOCHS):
        order = split["train"][
            torch.randperm(len(split["train"]), device="cuda", generator=generator)
        ]
        probe.module.train()
        for step in range(steps_per_epoch):
            indices = order[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = probe(*gather(indices)).float()
            loss = F.binary_cross_entropy_with_logits(
                output[:, :num_binary], binary[indices]
            ) + F.mse_loss(output[:, num_binary:], regression[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.module.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        probe.module.eval()
        val = score(split["val"])
        if val["auc_maccs_macro"] > best["val"]["auc_maccs_macro"]:
            best = {"epoch": epoch, "val": val, "test": score(split["test"])}
        print(kind, seed, epoch, round(float(loss), 4), round(val["auc_maccs_macro"], 4), flush=True)
    return best


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=200 * 1024,
    timeout=6 * 60 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def attentive_probe(name: str, targets: dict[str, np.ndarray], batch_size: int = 256) -> dict:
    os.chdir(PROJECT_ROOT)

    import torch
    from ml_collections import config_dict

    from scripts.modal_export_jepa_fps_embeddings import _prepare_massspecgym_inputs

    started = time.time()
    directory = VOLUME_ROOT / name
    config = config_dict.ConfigDict(json.loads((directory / "config.json").read_text()))
    data = _prepare_massspecgym_inputs(VOLUME_ROOT / LOCAL_TSV.name, config)
    encoder = _load_encoder(directory, config)
    tokens, offsets = _packed_layer_tokens(encoder, data, config, batch_size)
    del encoder
    torch.cuda.empty_cache()

    molecule = targets["molecule_index"]
    binary = np.concatenate([targets["maccs"], targets["binary"]], axis=1)[molecule]
    regression = targets["regression"][molecule]
    train = data["fold"] == "train"
    regression = (regression - regression[train].mean(0)) / regression[train].std(0)
    binary_t = torch.from_numpy(binary).to("cuda", torch.float32)
    regression_t = torch.from_numpy(regression).to("cuda", torch.float32)
    split = {
        fold: torch.from_numpy(np.flatnonzero(data["fold"] == fold)).to("cuda")
        for fold in ("train", "val", "test")
    }
    results: dict[str, Any] = {"encoder": name, "layers": {}}
    for layer in LAYERS:
        layer_tokens = tokens.pop(layer).to("cuda")
        results["layers"][str(layer)] = {
            kind: [
                _train_probe(
                    kind, layer_tokens, offsets, split, binary_t, regression_t, seed
                )
                for seed in SEEDS
            ]
            for kind in ("attentive", "mean_linear")
        }
        del layer_tokens
        torch.cuda.empty_cache()
    results["seconds"] = time.time() - started
    return results


@app.local_entrypoint()
def main(encoders: str = ",".join(ENCODERS), output_dir: str = "") -> None:
    targets = _molecule_targets(LOCAL_TSV)
    output = Path(output_dir) if output_dir else None
    for result in attentive_probe.map(encoders.split(","), kwargs={"targets": targets}):
        print(json.dumps({"encoder": result["encoder"], "seconds": result["seconds"]}))
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{result['encoder']}.json").write_text(json.dumps(result, indent=2))
