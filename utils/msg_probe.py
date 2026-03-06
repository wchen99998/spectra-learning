from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ml_collections import config_dict
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import r2_score, roc_auc_score

from input_pipeline import TfLightningDataModule
from models.model import PeakSetSIGReg
from utils.massspec_probe_data import (
    MassSpecProbeData,
    download_massspec_tsv,
    numpy_batch_to_torch,
)
from utils.schedulers import learning_rate_at_step


log = logging.getLogger(__name__)

FG_SMARTS: dict[str, str] = {
    "hydroxyl": "[OX2H]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "amide": "[NX3][CX3](=[OX1])",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "aromatic_ring": "c1ccccc1",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "sulfonyl": "[#16X4](=[OX1])(=[OX1])",
    "phosphate": "[PX4](=[OX1])([OX2])",
    "halide": "[F,Cl,Br,I]",
    "ether": "[OD2]([#6])[#6]",
    "thiol": "[#16X2H]",
    "nitrile": "[NX1]#[CX2]",
}

REGRESSION_TARGET_KEYS = (
    "mol_weight",
    "logp",
    "num_heavy_atoms",
    "num_rings",
)

_CACHE_VERSION = 1
_CACHE_FILENAME = f"MassSpecGym_probe_targets_v{_CACHE_VERSION}.npz"
_MSG_PROBE_SELF_ATTENTION_BLOCKS = 3
_MSG_PROBE_MLP_RATIO = 4


@dataclass(slots=True)
class MsgProbeTargets:
    smiles: np.ndarray
    mol_props: dict[str, np.ndarray]
    fg_counts: dict[str, np.ndarray]
    valid_mol_mask: np.ndarray
    index_by_smiles: dict[str, int]


@dataclass(slots=True)
class MsgProbeTaskSpec:
    regression_tasks: tuple[str, ...]
    classification_tasks: tuple[str, ...]
    regression_means: dict[str, float]
    regression_stds: dict[str, float]


class MsgProbeSelfAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_attention_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.attn_norm = torch.nn.RMSNorm(input_dim)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        mlp_hidden_dim = mlp_ratio * input_dim
        self.mlp_norm = torch.nn.RMSNorm(input_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, mlp_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(mlp_hidden_dim, input_dim),
        )

    def forward(
        self,
        feature_tokens: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.attn_norm(feature_tokens)
        attended, _ = self.attn(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            key_padding_mask=~feature_mask,
            need_weights=False,
        )
        feature_tokens = feature_tokens + attended
        feature_tokens = feature_tokens + self.mlp(self.mlp_norm(feature_tokens))
        return feature_tokens


def _prepare_attention_inputs(
    feature_tokens: torch.Tensor,
    feature_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_tokens = feature_tokens * feature_mask.unsqueeze(-1)
    safe_mask = feature_mask.clone()
    empty_rows = ~safe_mask.any(dim=1)
    if empty_rows.any():
        safe_mask[empty_rows, 0] = True
    return safe_tokens, safe_mask


class MsgAttentiveProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_attention_heads: int,
        task_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            MsgProbeSelfAttentionBlock(
                input_dim=input_dim,
                num_attention_heads=num_attention_heads,
                mlp_ratio=_MSG_PROBE_MLP_RATIO,
            )
            for _ in range(_MSG_PROBE_SELF_ATTENTION_BLOCKS)
        ])
        self.query = torch.nn.Parameter(torch.empty(1, input_dim))
        torch.nn.init.xavier_normal_(self.query)
        self.readout = torch.nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.RMSNorm(hidden_dim),
            torch.nn.SiLU(),
        )
        self.heads = torch.nn.ModuleDict({
            name: torch.nn.Linear(hidden_dim, 1)
            for name in task_names
        })

    def forward(
        self,
        feature_tokens: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        token_states, safe_mask = _prepare_attention_inputs(feature_tokens, feature_mask)
        for block in self.blocks:
            token_states = block(token_states, safe_mask)
        query = self.query.unsqueeze(0).expand(token_states.shape[0], -1, -1)
        pooled, _ = self.readout(
            query=query,
            key=token_states,
            value=token_states,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        state = self.trunk(pooled[:, 0, :])
        return {name: head(state) for name, head in self.heads.items()}


def should_run_msg_probe(global_step: int, every_n_steps: int) -> bool:
    return every_n_steps > 0 and global_step % every_n_steps == 0


def iter_massspec_probe(
    probe_data: MassSpecProbeData,
    split: str,
    *,
    seed: int,
    peak_ordering: str,
    drop_remainder: bool,
):
    dataset = probe_data.build_dataset(
        split,
        seed=seed,
        peak_ordering=peak_ordering,
        shuffle=(split == "massspec_train"),
        drop_remainder=drop_remainder,
    )
    size_key = {
        "massspec_train": "massspec_train_size",
        "massspec_val": "massspec_val_size",
        "massspec_test": "massspec_test_size",
    }[split]
    size = int(probe_data.info[size_key])
    seen = 0
    for batch in dataset.as_numpy_iterator():
        remaining = size - seen
        if remaining <= 0:
            break
        take = min(int(batch["peak_mz"].shape[0]), remaining)
        if take != batch["peak_mz"].shape[0]:
            batch = {key: value[:take] for key, value in batch.items()}
        seen += take
        yield numpy_batch_to_torch(batch)


def probe_steps_per_epoch(
    probe_data: MassSpecProbeData,
    *,
    split: str,
    drop_remainder: bool,
) -> int:
    size_key = {
        "massspec_train": "massspec_train_size",
        "massspec_val": "massspec_val_size",
        "massspec_test": "massspec_test_size",
    }[split]
    size = int(probe_data.info[size_key])
    batch_size = int(probe_data.batch_size)
    if drop_remainder:
        return size // batch_size
    return math.ceil(size / batch_size)


def compute_probe_targets_for_smiles(
    smiles: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    n = len(smiles)
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in FG_SMARTS.items()}
    fg_counts = {name: np.zeros(n, dtype=np.int16) for name in FG_SMARTS}
    valid_mol_mask = np.ones(n, dtype=bool)

    mol_props = {
        "mol_weight": np.zeros(n, dtype=np.float32),
        "logp": np.zeros(n, dtype=np.float32),
        "num_heavy_atoms": np.zeros(n, dtype=np.int16),
        "num_rings": np.zeros(n, dtype=np.int16),
    }

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mol_mask[i] = False
            continue
        for name, pattern in patterns.items():
            fg_counts[name][i] = len(mol.GetSubstructMatches(pattern))
        mol_props["mol_weight"][i] = Descriptors.ExactMolWt(mol)
        mol_props["logp"][i] = Descriptors.MolLogP(mol)
        mol_props["num_heavy_atoms"][i] = mol.GetNumHeavyAtoms()
        mol_props["num_rings"][i] = rdMolDescriptors.CalcNumRings(mol)

    return mol_props, fg_counts, valid_mol_mask


def _load_unique_smiles_from_tsv(tsv_path: Path) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            smiles = row["smiles"]
            if smiles in seen:
                continue
            seen.add(smiles)
            unique.append(smiles)
    return unique


def _pack_targets(
    *,
    smiles: np.ndarray,
    mol_props: dict[str, np.ndarray],
    fg_counts: dict[str, np.ndarray],
    valid_mol_mask: np.ndarray,
) -> MsgProbeTargets:
    index_by_smiles = {s: i for i, s in enumerate(smiles.tolist())}
    return MsgProbeTargets(
        smiles=smiles,
        mol_props=mol_props,
        fg_counts=fg_counts,
        valid_mol_mask=valid_mol_mask,
        index_by_smiles=index_by_smiles,
    )


def _load_targets_from_npz(data: np.lib.npyio.NpzFile) -> MsgProbeTargets:
    smiles = data["smiles"]
    mol_props = {key: data[key] for key in REGRESSION_TARGET_KEYS}
    fg_counts = {name: data[f"fg_{name}"] for name in FG_SMARTS}
    valid_mol_mask = data["valid_mol_mask"]
    return _pack_targets(
        smiles=smiles,
        mol_props=mol_props,
        fg_counts=fg_counts,
        valid_mol_mask=valid_mol_mask,
    )


def load_or_build_msg_probe_targets(
    *,
    tsv_path: Path,
    cache_dir: Path,
) -> MsgProbeTargets:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _CACHE_FILENAME
    tsv_stat = tsv_path.stat()

    if cache_path.exists():
        data = np.load(cache_path)
        cache_version = int(data["cache_version"])
        tsv_size = int(data["tsv_size"])
        tsv_mtime_ns = int(data["tsv_mtime_ns"])
        if (
            cache_version == _CACHE_VERSION
            and tsv_size == int(tsv_stat.st_size)
            and tsv_mtime_ns == int(tsv_stat.st_mtime_ns)
        ):
            log.info("Loading MSG probe target cache from %s", cache_path)
            return _load_targets_from_npz(data)

    log.info("Building MSG probe target cache from %s", tsv_path)
    unique_smiles = _load_unique_smiles_from_tsv(tsv_path)
    mol_props, fg_counts, valid_mol_mask = compute_probe_targets_for_smiles(unique_smiles)
    smiles = np.asarray(unique_smiles, dtype=np.str_)

    save_dict: dict[str, np.ndarray] = {
        "cache_version": np.asarray(_CACHE_VERSION, dtype=np.int64),
        "tsv_size": np.asarray(int(tsv_stat.st_size), dtype=np.int64),
        "tsv_mtime_ns": np.asarray(int(tsv_stat.st_mtime_ns), dtype=np.int64),
        "smiles": smiles,
        "valid_mol_mask": valid_mol_mask,
    }
    for key, values in mol_props.items():
        save_dict[key] = values
    for name, values in fg_counts.items():
        save_dict[f"fg_{name}"] = values
    np.savez_compressed(cache_path, **save_dict)
    log.info("Wrote MSG probe target cache to %s", cache_path)

    return _pack_targets(
        smiles=smiles,
        mol_props=mol_props,
        fg_counts=fg_counts,
        valid_mol_mask=valid_mol_mask,
    )


def _resolve_tsv_and_cache_dir(
    config: config_dict.ConfigDict,
    cache_dir_override: str | Path | None,
) -> tuple[Path, Path]:
    massspec_cache_dir = (
        Path(config.get("tfrecord_dir", "data/gems_peaklist_tfrecord"))
        .expanduser()
        .resolve()
        / "massspec_probe"
    )
    tsv_path = download_massspec_tsv(massspec_cache_dir)
    if cache_dir_override is None:
        cache_dir = tsv_path.parent
    else:
        cache_dir = Path(cache_dir_override).expanduser().resolve()
    return tsv_path, cache_dir


def _extract_probe_features(
    backbone: PeakSetSIGReg,
    batch: dict[str, torch.Tensor],
    *,
    feature_source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        embeddings = backbone.encoder(
            batch["peak_mz"],
            batch["peak_intensity"],
            valid_mask=batch["peak_valid_mask"],
        )
    if feature_source == "encoder":
        return embeddings, batch["peak_valid_mask"]
    raise ValueError(f"Unknown msg_probe_feature_source: {feature_source!r}")


def _collect_split_target_indices(
    *,
    probe_data: MassSpecProbeData,
    split: str,
    peak_ordering: str,
    seed: int,
    targets: MsgProbeTargets,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for batch in iter_massspec_probe(
        probe_data=probe_data,
        split=split,
        seed=seed,
        peak_ordering=peak_ordering,
        drop_remainder=False,
    ):
        idx = np.asarray([targets.index_by_smiles[s] for s in batch["smiles"]], dtype=np.int64)
        batches.append(idx)
    return np.concatenate(batches, axis=0)


def _build_task_spec(
    *,
    targets: MsgProbeTargets,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> MsgProbeTaskSpec:
    regression_means: dict[str, float] = {}
    regression_stds: dict[str, float] = {}

    train_valid_idx = train_idx[targets.valid_mol_mask[train_idx]]
    test_valid_idx = test_idx[targets.valid_mol_mask[test_idx]]

    for name in REGRESSION_TARGET_KEYS:
        values = targets.mol_props[name][train_valid_idx].astype(np.float32)
        regression_means[name] = float(values.mean())
        regression_stds[name] = float(np.clip(values.std(), 1e-8, None))

    classification_tasks: list[str] = []
    for name in FG_SMARTS:
        y_train = (targets.fg_counts[name][train_valid_idx] > 0).astype(np.int32)
        y_test = (targets.fg_counts[name][test_valid_idx] > 0).astype(np.int32)
        train_prevalence = float(y_train.mean())
        test_prevalence = float(y_test.mean())
        if train_prevalence < 0.01 or train_prevalence > 0.99:
            continue
        if test_prevalence == 0.0 or test_prevalence == 1.0:
            continue
        classification_tasks.append(name)

    return MsgProbeTaskSpec(
        regression_tasks=REGRESSION_TARGET_KEYS,
        classification_tasks=tuple(classification_tasks),
        regression_means=regression_means,
        regression_stds=regression_stds,
    )


def _probe_step(
    probe: MsgAttentiveProbe,
    backbone: PeakSetSIGReg,
    batch: dict[str, torch.Tensor],
    *,
    targets: MsgProbeTargets,
    task_spec: MsgProbeTaskSpec,
    feature_source: str,
    device: torch.device,
) -> dict[str, object] | None:
    feature_tokens, feature_mask = _extract_probe_features(
        backbone,
        batch,
        feature_source=feature_source,
    )
    batch_idx = np.asarray([targets.index_by_smiles[s] for s in batch["smiles"]], dtype=np.int64)
    valid_mask = torch.from_numpy(targets.valid_mol_mask[batch_idx]).to(device=device, dtype=torch.bool)
    if not bool(valid_mask.any()):
        return None

    feature_tokens = feature_tokens[valid_mask]
    feature_mask = feature_mask[valid_mask]
    logits = probe(feature_tokens, feature_mask)

    active_idx = batch_idx[valid_mask.detach().cpu().numpy()]
    losses: dict[str, torch.Tensor] = {}
    predictions: dict[str, torch.Tensor] = {}
    task_targets: dict[str, torch.Tensor] = {}

    for name in task_spec.regression_tasks:
        target = torch.from_numpy(targets.mol_props[name][active_idx].astype(np.float32)).to(device)
        mean = task_spec.regression_means[name]
        std = task_spec.regression_stds[name]
        normalized_target = (target - mean) / std
        pred = logits[name].squeeze(-1)
        losses[name] = F.mse_loss(pred, normalized_target)
        predictions[name] = pred.detach() * std + mean
        task_targets[name] = target

    for name in task_spec.classification_tasks:
        target = torch.from_numpy((targets.fg_counts[name][active_idx] > 0).astype(np.float32)).to(device)
        pred = logits[name].squeeze(-1)
        losses[name] = F.binary_cross_entropy_with_logits(pred, target)
        predictions[name] = torch.sigmoid(pred.detach())
        task_targets[name] = target

    loss_total = torch.stack(list(losses.values())).mean()
    return {
        "loss_total": loss_total,
        "losses": losses,
        "predictions": predictions,
        "targets": task_targets,
        "batch_size": int(feature_tokens.shape[0]),
    }


def _new_epoch_state(task_spec: MsgProbeTaskSpec) -> dict[str, object]:
    task_names = task_spec.regression_tasks + task_spec.classification_tasks
    return {
        "count": 0,
        "loss_total": 0.0,
        "losses": {name: 0.0 for name in task_names},
        "predictions": {name: [] for name in task_names},
        "targets": {name: [] for name in task_names},
    }


def _update_epoch_state(
    epoch_state: dict[str, object],
    result: dict[str, object],
    task_spec: MsgProbeTaskSpec,
) -> None:
    batch_size = int(result["batch_size"])
    epoch_state["count"] += batch_size
    epoch_state["loss_total"] += float(result["loss_total"].detach()) * batch_size
    losses = epoch_state["losses"]
    predictions = epoch_state["predictions"]
    targets = epoch_state["targets"]
    for name in task_spec.regression_tasks + task_spec.classification_tasks:
        losses[name] += float(result["losses"][name].detach()) * batch_size
        predictions[name].append(result["predictions"][name].detach().cpu().numpy())
        targets[name].append(result["targets"][name].detach().cpu().numpy())


def _score_epoch_state(
    *,
    prefix: str,
    epoch_state: dict[str, object],
    task_spec: MsgProbeTaskSpec,
) -> dict[str, float]:
    count = int(epoch_state["count"])
    metrics: dict[str, float] = {
        f"{prefix}/samples": float(count),
        f"{prefix}/loss_total": float(epoch_state["loss_total"]) / count,
    }

    regression_r2_values: list[float] = []
    classification_auc_values: list[float] = []
    losses = epoch_state["losses"]
    predictions = epoch_state["predictions"]
    targets = epoch_state["targets"]

    for name in task_spec.regression_tasks:
        pred = np.concatenate(predictions[name], axis=0)
        target = np.concatenate(targets[name], axis=0)
        metrics[f"{prefix}/loss_{name}"] = float(losses[name]) / count
        metrics[f"{prefix}/r2_{name}"] = float(r2_score(target, pred))
        regression_r2_values.append(metrics[f"{prefix}/r2_{name}"])

    for name in task_spec.classification_tasks:
        pred = np.concatenate(predictions[name], axis=0)
        target = np.concatenate(targets[name], axis=0)
        metrics[f"{prefix}/loss_fg_{name}"] = float(losses[name]) / count
        metrics[f"{prefix}/auc_fg_{name}"] = float(roc_auc_score(target, pred))
        classification_auc_values.append(metrics[f"{prefix}/auc_fg_{name}"])

    metrics[f"{prefix}/r2_mean"] = float(np.mean(regression_r2_values))
    metrics[f"{prefix}/auc_fg_mean"] = float(np.mean(classification_auc_values))
    return metrics


def run_msg_probe(
    *,
    config: config_dict.ConfigDict,
    datamodule: TfLightningDataModule,
    model: PeakSetSIGReg,
    device: torch.device,
    cache_dir_override: str | Path | None = None,
) -> dict[str, float]:
    del datamodule
    num_probe_epochs = int(config.get("msg_probe_num_epochs", 5))
    probe_lr = float(config.get("msg_probe_learning_rate", 1e-3))
    probe_weight_decay = float(config.get("msg_probe_weight_decay", 1e-2))
    probe_warmup_steps = int(config.get("msg_probe_warmup_steps", 100))
    probe_hidden_dim = int(config.get("msg_probe_hidden_dim", 512))
    feature_source = str(config.get("msg_probe_feature_source", "encoder"))
    peak_ordering = str(config.get("peak_ordering", "intensity"))
    probe_data = MassSpecProbeData.from_config(config)

    tsv_path, cache_dir = _resolve_tsv_and_cache_dir(config, cache_dir_override)
    targets = load_or_build_msg_probe_targets(tsv_path=tsv_path, cache_dir=cache_dir)

    train_seed_base = int(config.seed) + 1_100_000
    test_seed_base = int(config.seed) + 1_200_000
    train_idx = _collect_split_target_indices(
        probe_data=probe_data,
        split="massspec_train",
        peak_ordering=peak_ordering,
        seed=train_seed_base,
        targets=targets,
    )
    test_idx = _collect_split_target_indices(
        probe_data=probe_data,
        split="massspec_test",
        peak_ordering=peak_ordering,
        seed=test_seed_base,
        targets=targets,
    )
    task_spec = _build_task_spec(targets=targets, train_idx=train_idx, test_idx=test_idx)
    task_names = task_spec.regression_tasks + task_spec.classification_tasks

    was_training = model.training
    model.eval()
    probe = MsgAttentiveProbe(
        input_dim=int(config.model_dim),
        hidden_dim=probe_hidden_dim,
        num_attention_heads=int(config.num_heads),
        task_names=task_names,
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=probe_lr, weight_decay=probe_weight_decay)
    steps_per_epoch = probe_steps_per_epoch(
        probe_data,
        split="massspec_train",
        drop_remainder=False,
    )
    total_steps = num_probe_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_idx: learning_rate_at_step(
            step_idx + 1,
            base_lr=probe_lr,
            total_steps=total_steps,
            warmup_steps=probe_warmup_steps,
            schedule_type="cosine",
            min_learning_rate=None,
        ) / probe_lr,
    )

    def move_batch(batch: dict[str, object]) -> dict[str, object]:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    final_metrics: dict[str, float] = {}
    for epoch_idx in range(num_probe_epochs):
        probe.train()
        train_state = _new_epoch_state(task_spec)
        train_seed = train_seed_base + epoch_idx
        for batch in iter_massspec_probe(
            probe_data,
            "massspec_train",
            seed=train_seed,
            peak_ordering=peak_ordering,
            drop_remainder=False,
        ):
            batch = move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            result = _probe_step(
                probe,
                model,
                batch,
                targets=targets,
                task_spec=task_spec,
                feature_source=feature_source,
                device=device,
            )
            if result is None:
                continue
            result["loss_total"].backward()
            optimizer.step()
            scheduler.step()
            _update_epoch_state(train_state, result, task_spec)

        probe.eval()
        test_state = _new_epoch_state(task_spec)
        test_seed = test_seed_base + epoch_idx
        with torch.no_grad():
            for batch in iter_massspec_probe(
                probe_data,
                "massspec_test",
                seed=test_seed,
                peak_ordering=peak_ordering,
                drop_remainder=False,
            ):
                batch = move_batch(batch)
                result = _probe_step(
                    probe,
                    model,
                    batch,
                    targets=targets,
                    task_spec=task_spec,
                    feature_source=feature_source,
                    device=device,
                )
                if result is None:
                    continue
                _update_epoch_state(test_state, result, task_spec)

        epoch_metrics = {
            **_score_epoch_state(prefix="msg_probe/train", epoch_state=train_state, task_spec=task_spec),
            **_score_epoch_state(prefix="msg_probe/test", epoch_state=test_state, task_spec=task_spec),
            "msg_probe/num_fg_tasks": float(len(task_spec.classification_tasks)),
            "msg_probe_epoch": float(epoch_idx + 1),
        }
        log.info(
            "MSG probe epoch %d/%d test_r2_mean=%.4f test_auc_fg_mean=%.4f fg_tasks=%d",
            epoch_idx + 1,
            num_probe_epochs,
            epoch_metrics["msg_probe/test/r2_mean"],
            epoch_metrics["msg_probe/test/auc_fg_mean"],
            int(epoch_metrics["msg_probe/num_fg_tasks"]),
        )
        final_metrics = epoch_metrics

    if was_training:
        model.train()
    return final_metrics
