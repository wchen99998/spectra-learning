import torch
from ml_collections import config_dict

from spectra_learning.models.temporal import CovariancePool, CrossAttention
from spectra_learning.probes.massspec.msg_settings import (
    NUM_RINGS_TASK,
    MsgProbeTaskSpec,
    build_msg_probe_inputs,
)


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
        task_names: tuple[str, ...],
        task_output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(
                        hidden_dim,
                        1 if task_output_dims is None else task_output_dims.get(name, 1),
                    ),
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
        qk_norm: bool = False,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.seed_vectors = torch.nn.Parameter(torch.empty(num_seeds, input_dim))
        torch.nn.init.trunc_normal_(self.seed_vectors, std=0.02)
        self.cross_attention = CrossAttention(
            dim=input_dim,
            n_heads=num_heads,
            qk_norm=qk_norm,
            norm_type=norm_type,
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
        task_names: tuple[str, ...],
        task_output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.pooler = pooler
        self.heads = MsgProbeHeads(
            input_dim=pooled_dim,
            hidden_dim=hidden_dim,
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
    hidden_dim = int(config.get("msg_probe_mlp_hidden_dim", model_dim))
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
        task_names=task_names,
        task_output_dims=task_output_dims,
    )


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
                num_seeds=int(config.get("msg_probe_pma_num_seeds", 4)),
                num_heads=int(
                    config.get("msg_probe_pma_num_heads", config.get("encoder_num_heads", 8))
                ),
                qk_norm=bool(config.get("encoder_qk_norm", False)),
                norm_type=str(config.get("norm_type", "layernorm")),
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
        compressed_dim = int(config.get("msg_probe_covariance_dim", 32))
        return (
            MsgCovariancePool(input_dim=model_dim, compressed_dim=compressed_dim),
            compressed_dim * compressed_dim,
        )
    compressed_dim = int(covariance_pooler.left_proj.out_features)
    return FrozenPooler(covariance_pooler), compressed_dim * compressed_dim
