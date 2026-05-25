from typing import Any

import torch
from ml_collections import config_dict

from spectra_learning.models.pooling import CovariancePool
from spectra_learning.models.transformer import CrossAttention
from spectra_learning.probes.massspec.msg_settings import (
    NUM_RINGS_TASK,
    MsgProbeTaskSpec,
    build_msg_probe_inputs,
)


def _config_get(config: config_dict.ConfigDict, key: str, default: Any) -> Any:
    return config.get(key, default)


class MsgLinearProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        task_names: tuple[str, ...],
        task_output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(
                    input_dim,
                    1 if task_output_dims is None else task_output_dims.get(name, 1),
                )
                for name in task_names
            }
        )

    def forward(
        self,
        probe_inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {name: head(probe_inputs) for name, head in self.heads.items()}


class MsgProbeHeads(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        task_names: tuple[str, ...],
        task_output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleDict(
            {
                name: _build_mlp_head(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=(
                        1 if task_output_dims is None else task_output_dims.get(name, 1)
                    ),
                    num_layers=num_layers,
                )
                for name in task_names
            }
        )

    def forward(
        self,
        probe_inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {name: head(probe_inputs) for name, head in self.heads.items()}


class MsgMeanPool(torch.nn.Module):
    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return build_msg_probe_inputs(peak_embeddings, valid_mask)


class MsgCovariancePool(CovariancePool):
    pass


class FrozenPooler(torch.nn.Module):
    _pooler: torch.nn.Module

    def __init__(self, pooler: torch.nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_pooler", pooler)

    @property
    def pooler(self) -> torch.nn.Module:
        return self._pooler

    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.pooler(peak_embeddings, valid_mask)


class MsgPmaPool(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_seeds: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.seed_vectors = torch.nn.Parameter(torch.empty(num_seeds, input_dim))
        torch.nn.init.trunc_normal_(self.seed_vectors, std=0.02)
        self.cross_attention = CrossAttention(
            dim=input_dim,
            n_heads=num_heads,
        )

    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        seed_vectors = self.seed_vectors.unsqueeze(0).expand(
            peak_embeddings.shape[0],
            -1,
            -1,
        )
        pooled = self.cross_attention(
            seed_vectors.to(dtype=peak_embeddings.dtype),
            peak_embeddings,
            memory_mask=valid_mask,
        )
        return pooled.mean(dim=1)


class MsgSequenceProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        pooler: torch.nn.Module,
        pooled_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        task_names: tuple[str, ...],
        task_output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.pooler = pooler
        self.heads = MsgProbeHeads(
            input_dim=pooled_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            task_names=task_names,
            task_output_dims=task_output_dims,
        )

    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.heads(self.pooler(peak_embeddings, valid_mask))


def build_msg_sequence_probe(
    variant: str,
    *,
    config: config_dict.ConfigDict,
    task_spec: MsgProbeTaskSpec,
    covariance_pooler: CovariancePool | None = None,
) -> MsgSequenceProbe:
    model_dim = int(config.model_dim)
    hidden_dim = int(_config_get(config, "msg_probe_mlp_hidden_dim", model_dim))
    num_layers = int(_config_get(config, "msg_probe_mlp_num_layers", 2))
    task_names = _probe_task_names(task_spec)
    task_output_dims = _probe_task_output_dims(task_spec)
    pooler, pooled_dim = _build_pooler(
        variant,
        config=config,
        model_dim=model_dim,
        covariance_pooler=covariance_pooler,
    )
    return MsgSequenceProbe(
        pooler=pooler,
        pooled_dim=pooled_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        task_names=task_names,
        task_output_dims=task_output_dims,
    )


def _build_mlp_head(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> torch.nn.Module:
    if num_layers == 1:
        return torch.nn.Linear(input_dim, output_dim)
    layers: list[torch.nn.Module] = [
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.SiLU(),
    ]
    for _ in range(num_layers - 2):
        layers.extend(
            [
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
            ]
        )
    layers.append(torch.nn.Linear(hidden_dim, output_dim))
    return torch.nn.Sequential(*layers)


def _probe_task_names(task_spec: MsgProbeTaskSpec) -> tuple[str, ...]:
    task_names = task_spec.regression_tasks
    if task_spec.num_rings_classes:
        task_names += (NUM_RINGS_TASK,)
    if task_spec.maccs_bits > 0:
        task_names += (task_spec.fingerprint_task,)
    return task_names


def _probe_task_output_dims(task_spec: MsgProbeTaskSpec) -> dict[str, int]:
    output_dims: dict[str, int] = {}
    if task_spec.num_rings_classes:
        output_dims[NUM_RINGS_TASK] = len(task_spec.num_rings_classes)
    if task_spec.maccs_bits > 0:
        output_dims[task_spec.fingerprint_task] = task_spec.maccs_bits
    return output_dims


def _build_pooler(
    variant: str,
    *,
    config: config_dict.ConfigDict,
    model_dim: int,
    covariance_pooler: CovariancePool | None,
) -> tuple[torch.nn.Module, int]:
    if variant == "mean":
        return MsgMeanPool(), model_dim
    if variant == "covariance":
        return _build_covariance_pooler(config, model_dim, covariance_pooler)
    if variant == "pma":
        return (
            MsgPmaPool(
                input_dim=model_dim,
                num_seeds=int(_config_get(config, "msg_probe_pma_num_seeds", 4)),
                num_heads=int(
                    _config_get(
                        config,
                        "msg_probe_pma_num_heads",
                        _config_get(config, "encoder_num_heads", 8),
                    )
                ),
            ),
            model_dim,
        )
    raise ValueError(f"Unsupported MSG probe variant: {variant!r}")


def _build_covariance_pooler(
    config: config_dict.ConfigDict,
    model_dim: int,
    covariance_pooler: CovariancePool | None,
) -> tuple[torch.nn.Module, int]:
    if covariance_pooler is None:
        compressed_dim = int(_config_get(config, "covariance_pooling_dim", 32))
        return (
            MsgCovariancePool(input_dim=model_dim, compressed_dim=compressed_dim),
            compressed_dim * compressed_dim,
        )
    compressed_dim = covariance_pooler.left_proj.out_features
    return FrozenPooler(covariance_pooler), compressed_dim * compressed_dim
