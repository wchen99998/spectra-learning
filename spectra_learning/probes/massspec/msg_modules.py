from typing import Any

import torch
from ml_collections import config_dict

from spectra_learning.models.pooling import CovariancePool, SinglePairCovariancePool
from spectra_learning.models.transformer import (
    CrossAttention,
    FeedForward,
    TransformerBlock,
    _build_norm,
)
from spectra_learning.probes.massspec.msg_settings import (
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
        for head in self.heads.values():
            _init_probe_output(head)

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


class MsgSinglePairCovariancePool(SinglePairCovariancePool):
    pass


class FrozenPooler(torch.nn.Module):
    _pooler: torch.nn.Module

    def __init__(self, pooler: torch.nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_pooler", pooler)

    @property
    def pooler(self) -> torch.nn.Module:
        return self._pooler

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.pooler(*args)


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
        peak_embeddings = peak_embeddings[:, : valid_mask.shape[1]]
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


class MsgPmaSetBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.latent_norm = _build_norm(dim, eps=norm_eps)
        self.memory_norm = _build_norm(dim, eps=norm_eps)
        self.cross_attention = CrossAttention(
            dim=dim,
            n_heads=num_heads,
        )
        self.self_attention = TransformerBlock(
            dim=dim,
            n_heads=num_heads,
            n_kv_heads=None,
            norm_eps=norm_eps,
            hidden_dim=hidden_dim,
        )
        self.ffn_norm = _build_norm(dim, eps=norm_eps)
        self.feed_forward = FeedForward(dim, hidden_dim=hidden_dim)

    def forward(
        self,
        latents: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        latents = latents + self.cross_attention(
            self.latent_norm(latents),
            self.memory_norm(memory),
            memory_mask=memory_mask,
        )
        latents = self.self_attention(latents)
        return latents + self.feed_forward(self.ffn_norm(latents))


class MsgPmaSetEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        num_tokens: int,
        num_heads: int,
        num_blocks: int,
        hidden_dim: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.input_proj = (
            torch.nn.Identity()
            if input_dim == latent_dim
            else torch.nn.Linear(input_dim, latent_dim)
        )
        self.seed_vectors = torch.nn.Parameter(torch.empty(num_tokens, latent_dim))
        torch.nn.init.trunc_normal_(self.seed_vectors, std=0.02)
        self.blocks = torch.nn.ModuleList(
            [
                MsgPmaSetBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = _build_norm(latent_dim, eps=norm_eps)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.input_proj(memory)
        latents = self.seed_vectors.unsqueeze(0).expand(memory.shape[0], -1, -1)
        latents = latents.to(dtype=memory.dtype)
        for block in self.blocks:
            latents = block(latents, memory, memory_mask)
        return self.final_norm(latents)


class MsgSinglePairPmaPool(torch.nn.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        latent_dim: int,
        num_tokens: int,
        num_heads: int,
        num_blocks: int,
        hidden_dim: int,
        norm_eps: float,
        include_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.include_cls_token = include_cls_token
        self.single_encoder = MsgPmaSetEncoder(
            input_dim=single_dim,
            latent_dim=latent_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            norm_eps=norm_eps,
        )
        self.pair_encoder = MsgPmaSetEncoder(
            input_dim=pair_dim,
            latent_dim=latent_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            norm_eps=norm_eps,
        )

    @property
    def output_dim(self) -> int:
        return 2 * self.num_tokens * self.latent_dim

    def _token_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        if not self.include_cls_token:
            return valid_mask
        cls_mask = torch.ones(
            valid_mask.shape[0],
            1,
            device=valid_mask.device,
            dtype=torch.bool,
        )
        return torch.cat([valid_mask, cls_mask], dim=1)

    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        pair_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        token_mask = self._token_mask(valid_mask)
        num_tokens = token_mask.shape[1]
        dtype = self.single_encoder.seed_vectors.dtype
        peak_embeddings = peak_embeddings[:, :num_tokens].to(dtype=dtype)
        pair_embeddings = pair_embeddings[:, :num_tokens, :num_tokens].to(dtype=dtype)
        batch_size, _, _, pair_dim = pair_embeddings.shape
        pair_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
        pair_memory = pair_embeddings.reshape(
            batch_size,
            num_tokens * num_tokens,
            pair_dim,
        )
        pair_memory_mask = pair_mask.reshape(batch_size, num_tokens * num_tokens)
        tokens = torch.cat(
            [
                self.single_encoder(peak_embeddings, token_mask),
                self.pair_encoder(pair_memory, pair_memory_mask),
            ],
            dim=1,
        )
        return tokens.flatten(start_dim=1)


class MsgSinglePairClsPool(torch.nn.Module):
    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        pair_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        cls_idx = valid_mask.shape[1]
        return torch.cat(
            [
                peak_embeddings[:, cls_idx],
                pair_embeddings[:, cls_idx, cls_idx],
            ],
            dim=-1,
        )


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
        pair_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.heads(self.pooler(peak_embeddings, valid_mask))


class MsgSinglePairLinearProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        pooler: MsgSinglePairPmaPool,
        pooled_dim: int,
        task_names: tuple[str, ...],
        task_output_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.pooler = pooler
        self.heads = MsgLinearProbe(
            input_dim=pooled_dim,
            task_names=task_names,
            task_output_dims=task_output_dims,
        )

    def forward(
        self,
        peak_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        pair_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert pair_embeddings is not None
        return self.heads(self.pooler(peak_embeddings, valid_mask, pair_embeddings))


class MsgSinglePairCovarianceProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        pooler: MsgSinglePairCovariancePool,
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
        pair_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert pair_embeddings is not None
        return self.heads(self.pooler(peak_embeddings, valid_mask, pair_embeddings))


class MsgSinglePairClsProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        pooler: MsgSinglePairClsPool,
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
        pair_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert pair_embeddings is not None
        return self.heads(self.pooler(peak_embeddings, valid_mask, pair_embeddings))


def build_msg_sequence_probe(
    variant: str,
    *,
    config: config_dict.ConfigDict,
    task_spec: MsgProbeTaskSpec,
    covariance_pooler: torch.nn.Module | None = None,
) -> torch.nn.Module:
    model_dim = int(config.model_dim)
    hidden_dim = int(_config_get(config, "msg_probe_mlp_hidden_dim", model_dim))
    num_layers = int(_config_get(config, "msg_probe_mlp_num_layers", 2))
    task_names = _probe_task_names(task_spec)
    task_output_dims = _probe_task_output_dims(task_spec)
    if variant == "cls":
        pair_dim = int(_config_get(config, "pairmixer_pair_dim", model_dim))
        return MsgSinglePairClsProbe(
            pooler=MsgSinglePairClsPool(),
            pooled_dim=model_dim + pair_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            task_names=task_names,
            task_output_dims=task_output_dims,
        )
    if _is_single_pair_covariance_variant(variant):
        if covariance_pooler is None:
            compressed_dim = int(_config_get(config, "covariance_pooling_dim", 32))
            pooler = MsgSinglePairCovariancePool(
                single_dim=model_dim,
                pair_dim=int(_config_get(config, "pairmixer_pair_dim", model_dim)),
                compressed_dim=compressed_dim,
                include_diagonal=bool(
                    _config_get(
                        config,
                        "msg_probe_single_pair_covariance_include_diagonal",
                        False,
                    )
                ),
            )
            pooled_dim = compressed_dim * compressed_dim
        else:
            pooler = (
                FrozenPooler(covariance_pooler)
                if bool(_config_get(config, "msg_probe_freeze_supplied_pooler", True))
                else covariance_pooler
            )
            pooled_dim = covariance_pooler.output_dim
        return MsgSinglePairCovarianceProbe(
            pooler=pooler,
            pooled_dim=pooled_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            task_names=task_names,
            task_output_dims=task_output_dims,
        )
    if _is_single_pair_pma_variant(variant):
        pair_dim = int(_config_get(config, "pairmixer_pair_dim", model_dim))
        num_tokens = int(_config_get(config, "msg_probe_single_pair_pma_num_tokens", 8))
        num_heads = int(
            _config_get(
                config,
                "msg_probe_single_pair_pma_num_heads",
                _config_get(
                    config,
                    "msg_probe_pma_num_heads",
                    _config_get(config, "encoder_num_heads", 8),
                ),
            )
        )
        num_blocks = _single_pair_pma_num_blocks(variant, config)
        pooler = MsgSinglePairPmaPool(
            single_dim=model_dim,
            pair_dim=pair_dim,
            latent_dim=model_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            norm_eps=float(_config_get(config, "norm_eps", 1e-5)),
            include_cls_token=bool(
                _config_get(config, "msg_probe_single_pair_pma_include_cls_token", False)
            ),
        )
        return MsgSinglePairLinearProbe(
            pooler=pooler,
            pooled_dim=pooler.output_dim,
            task_names=task_names,
            task_output_dims=task_output_dims,
        )
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
        linear = torch.nn.Linear(input_dim, output_dim)
        _init_probe_output(linear)
        return linear
    layers: list[torch.nn.Module] = [
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.SiLU(),
    ]
    _init_probe_hidden(layers[0])
    for _ in range(num_layers - 2):
        hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        _init_probe_hidden(hidden)
        layers.extend([hidden, torch.nn.SiLU()])
    output = torch.nn.Linear(hidden_dim, output_dim)
    _init_probe_output(output)
    layers.append(output)
    return torch.nn.Sequential(*layers)


def _init_probe_hidden(linear: torch.nn.Module) -> None:
    assert isinstance(linear, torch.nn.Linear)
    torch.nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)


def _init_probe_output(linear: torch.nn.Module) -> None:
    assert isinstance(linear, torch.nn.Linear)
    torch.nn.init.normal_(linear.weight, mean=0.0, std=1e-3)
    if linear.bias is not None:
        torch.nn.init.zeros_(linear.bias)


def _probe_task_names(task_spec: MsgProbeTaskSpec) -> tuple[str, ...]:
    if task_spec.maccs_bits > 0:
        return (task_spec.fingerprint_task,)
    return task_spec.regression_tasks


def _probe_prediction_names(task_spec: MsgProbeTaskSpec) -> tuple[str, ...]:
    task_names = task_spec.regression_tasks
    if task_spec.maccs_bits > 0:
        task_names += (task_spec.fingerprint_task,)
    return task_names


def _probe_task_output_dims(task_spec: MsgProbeTaskSpec) -> dict[str, int]:
    output_dims: dict[str, int] = {}
    if task_spec.maccs_bits > 0:
        output_dims[task_spec.fingerprint_task] = (
            len(task_spec.regression_tasks) + task_spec.maccs_bits
        )
    return output_dims


def _is_single_pair_pma_variant(variant: str) -> bool:
    return variant == "single_pair_pma" or variant.startswith("single_pair_pma_")


def _is_single_pair_covariance_variant(variant: str) -> bool:
    return variant in ("single_pair_covariance", "pair_covariance")


def _uses_pair_features(variant: str) -> bool:
    return (
        variant == "cls"
        or _is_single_pair_pma_variant(variant)
        or _is_single_pair_covariance_variant(variant)
    )


def _single_pair_pma_num_blocks(
    variant: str,
    config: config_dict.ConfigDict,
) -> int:
    prefix = "single_pair_pma_"
    if variant.startswith(prefix):
        suffix = variant[len(prefix):].removesuffix("blocks").removesuffix("block")
        return int(suffix)
    return int(_config_get(config, "msg_probe_single_pair_pma_num_blocks", 2))


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
    pooler = (
        FrozenPooler(covariance_pooler)
        if bool(_config_get(config, "msg_probe_freeze_supplied_pooler", True))
        else covariance_pooler
    )
    return pooler, compressed_dim * compressed_dim
