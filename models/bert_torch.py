"""PyTorch port of the masked language model BERT used in the JAX code."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from networks import transformer_torch


def _resolve_attention_heads(
    dim: int,
    num_heads: int,
    num_kv_heads: int | None,
) -> tuple[int, int]:
    heads = int(num_heads)
    kv_heads = heads if num_kv_heads is None else int(num_kv_heads)
    return heads, kv_heads


def _build_transformer_blocks(
    *,
    dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int | None,
    attention_mlp_multiple: float,
    norm_eps: float = 1e-5,
) -> nn.ModuleList:
    heads, kv_heads = _resolve_attention_heads(dim, num_heads, num_kv_heads)
    hidden_dim = int(math.ceil(dim * attention_mlp_multiple))
    blocks: list[transformer_torch.TransformerBlock] = []
    for _ in range(num_layers):
        blocks.append(
            transformer_torch.TransformerBlock(
                dim=dim,
                n_heads=heads,
                n_kv_heads=kv_heads,
                causal=False,
                norm_eps=norm_eps,
                mlp_type="swish",
                multiple_of=4,
                hidden_dim=hidden_dim,
                w_init_scale=1.0,
                use_rotary_embeddings=True,
            )
        )
    return nn.ModuleList(blocks)


class TransformerStack(nn.Module):
    """Stack of transformer blocks with rotary positional encoding."""

    def __init__(
        self,
        *,
        dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None,
        attention_mlp_multiple: float,
        rope_theta: float,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        heads, kv_heads = _resolve_attention_heads(dim, num_heads, num_kv_heads)
        self.dim = int(dim)
        self.num_heads = int(heads)
        self.rope_theta = float(rope_theta)
        self.dtype = dtype
        self.blocks = _build_transformer_blocks(
            dim=self.dim,
            num_layers=num_layers,
            num_heads=self.num_heads,
            num_kv_heads=kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
        )
        self.norm = nn.RMSNorm(self.dim, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        *,
        train: bool,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del train  # Transformer blocks here are deterministic.
        seq_len = x.shape[1]
        head_dim = self.dim // self.num_heads
        freqs_cos, freqs_sin = transformer_torch.precompute_freqs_cis(
            head_dim,
            seq_len,
            theta=self.rope_theta,
            device=x.device,
            dtype=self.dtype,
        )
        freqs_cos = freqs_cos[:seq_len].to(dtype=x.dtype)
        freqs_sin = freqs_sin[:seq_len].to(dtype=x.dtype)
        for block in self.blocks:
            x = block(
                x,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                attention_bias=attention_bias,
            )
        return self.norm(x)


class BERTTorch(nn.Module):
    """BERT-style masked language model."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_length: int,
        precursor_bins: int,
        precursor_offset: int,
        model_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0,
        num_segments: int = 2,
        mask_ratio: float = 0.15,
        mask_token_id: int = 103,
        pad_token_id: int = 0,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        rope_theta: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.precursor_bins = int(precursor_bins)
        self.precursor_offset = int(precursor_offset)
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_kv_heads = None if num_kv_heads is None else int(num_kv_heads)
        self.attention_mlp_multiple = float(attention_mlp_multiple)
        self.num_segments = int(num_segments)
        self.mask_ratio = float(mask_ratio)
        self.mask_token_id = int(mask_token_id)
        self.pad_token_id = int(pad_token_id)
        self.cls_token_id = int(cls_token_id)
        self.sep_token_id = int(sep_token_id)
        self.rope_theta = float(rope_theta)
        self.dtype = dtype

        self.token_embed = nn.Embedding(self.vocab_size, self.model_dim)
        self.segment_embed = nn.Embedding(self.num_segments, self.model_dim)
        self.embed_norm = nn.LayerNorm(self.model_dim)

        embed_std = 1.0 / math.sqrt(self.model_dim)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=embed_std)
        nn.init.normal_(self.segment_embed.weight, mean=0.0, std=embed_std)

        self.encoder = TransformerStack(
            dim=self.model_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            attention_mlp_multiple=self.attention_mlp_multiple,
            rope_theta=self.rope_theta,
            dtype=dtype,
        )

        self.lm_head = nn.Linear(self.model_dim, self.vocab_size)
        self.precursor_head = nn.Linear(self.model_dim, 2 * self.precursor_bins)
        self.retention_head = nn.Linear(self.model_dim, 2)

        nn.init.zeros_(self.lm_head.weight)
        nn.init.zeros_(self.precursor_head.weight)
        nn.init.zeros_(self.retention_head.weight)
        nn.init.zeros_(self.lm_head.bias)
        nn.init.zeros_(self.precursor_head.bias)
        nn.init.zeros_(self.retention_head.bias)

    def _make_attention_bias(self, allow: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        neg = torch.tensor(-1e9, dtype=dtype, device=allow.device)
        zero = torch.tensor(0.0, dtype=dtype, device=allow.device)
        bias = torch.where(allow, zero, neg)
        return bias[:, None, None, :]

    def _mask_tokens(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        maskable = attention_mask & (token_ids != self.cls_token_id)
        seq_len = token_ids.shape[1]
        mask_count = int(self.mask_ratio * seq_len)
        scores = torch.rand(token_ids.shape, device=token_ids.device, dtype=torch.float32)
        scores = torch.where(maskable, scores, torch.full_like(scores, -1.0))
        _, mask_idx = torch.topk(scores, k=mask_count, dim=1)
        mask = torch.zeros_like(maskable)
        mask.scatter_(1, mask_idx, True)
        masked_tokens = torch.where(mask, torch.full_like(token_ids, self.mask_token_id), token_ids)
        return masked_tokens, mask, mask_idx

    def _embed_inputs(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        tok = self.token_embed(token_ids)
        seg = self.segment_embed(segment_ids)
        return self.embed_norm(tok + seg)

    def _masked_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        mask_f = mask.to(per_token.dtype)
        return (per_token * mask_f).sum() / mask_f.sum()

    def _masked_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pred = logits.argmax(dim=-1)
        correct = (pred == labels).to(torch.float32)
        mask_f = mask.to(torch.float32)
        return (correct * mask_f).sum() / mask_f.sum()

    def _precursor_metrics(
        self,
        cls_state: torch.Tensor,
        precursor_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.precursor_head(cls_state).view(cls_state.shape[0], 2, self.precursor_bins)
        labels = (precursor_tokens - self.precursor_offset).to(torch.long)
        per_token = F.cross_entropy(
            logits.reshape(-1, self.precursor_bins),
            labels.reshape(-1),
            reduction="none",
        )
        loss = per_token.mean()
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).to(torch.float32).mean()
        return loss, acc

    def _retention_metrics(
        self,
        cls_state: torch.Tensor,
        rt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.retention_head(cls_state)
        labels = torch.where(rt[:, 0] < rt[:, 1], 0, 1).to(torch.long)
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).to(torch.float32).mean()
        return loss, acc

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = True,
        apply_mask: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if apply_mask is None:
            apply_mask = train

        token_ids = batch["token_ids"].to(torch.long)
        segment_ids = batch["segment_ids"].to(torch.long)
        attention_mask = token_ids != self.pad_token_id

        if apply_mask:
            masked_tokens = batch["masked_token_ids"].to(torch.long)
            mask = batch["mlm_mask"].to(torch.bool)
        else:
            masked_tokens = token_ids
            mask = torch.zeros_like(attention_mask, dtype=torch.bool)

        x = self._embed_inputs(masked_tokens, segment_ids)
        attention_bias = self._make_attention_bias(attention_mask, dtype=x.dtype)
        encoded = self.encoder(x, train=train, attention_bias=attention_bias)
        logits = self.lm_head(encoded)

        if apply_mask:
            mlm_loss = self._masked_cross_entropy(logits, token_ids, mask)
            token_accuracy = self._masked_accuracy(logits, token_ids, mask)
        else:
            mlm_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
            token_accuracy = torch.zeros((), dtype=torch.float32, device=logits.device)

        mask_ratio_actual = mask.to(torch.float32).mean()
        cls_state = encoded[:, 0, :]
        precursor_tokens = batch["precursor_mz"].to(torch.long)
        rt = batch["rt"]
        precursor_loss, precursor_accuracy = self._precursor_metrics(cls_state, precursor_tokens)
        retention_loss, retention_accuracy = self._retention_metrics(cls_state, rt)
        loss = mlm_loss + precursor_loss + retention_loss

        return {
            "loss": loss,
            "token_accuracy": token_accuracy,
            "mask_ratio_actual": mask_ratio_actual,
            "precursor_loss": precursor_loss,
            "precursor_accuracy": precursor_accuracy,
            "retention_loss": retention_loss,
            "retention_accuracy": retention_accuracy,
        }

    def encode(self, batch: dict[str, torch.Tensor], *, train: bool = False) -> torch.Tensor:
        token_ids = batch["token_ids"].to(torch.long)
        segment_ids = batch["segment_ids"].to(torch.long)
        attention_mask = token_ids != self.pad_token_id

        x = self._embed_inputs(token_ids, segment_ids)
        attention_bias = self._make_attention_bias(attention_mask, dtype=x.dtype)
        encoded = self.encoder(x, train=train, attention_bias=attention_bias)
        return encoded[:, 0, :]

    def compute_loss(self, batch: dict[str, torch.Tensor], *, train: bool = False):
        metrics = self(batch, train=train, apply_mask=train)
        return metrics["loss"], metrics


def build_lightning_module(
    model: BERTTorch,
    *,
    learning_rate: float,
    weight_decay: float,
    b2: float = 0.999,
) -> Any:
    """Wrap the model in a LightningModule without importing Lightning at import time."""
    import lightning.pytorch as pl

    class BERTLightningModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.b2 = b2

        def forward(self, batch: dict[str, torch.Tensor], *, train: bool, apply_mask: bool | None = None):
            return self.model(batch, train=train, apply_mask=apply_mask)

        def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            del batch_idx
            metrics = self.model(batch, train=True, apply_mask=True)
            self.log_dict({f"train/{k}": v for k, v in metrics.items()}, prog_bar=True)
            return metrics["loss"]

        def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            del batch_idx
            metrics = self.model(batch, train=False, apply_mask=False)
            self.log_dict({f"val/{k}": v for k, v in metrics.items()}, prog_bar=True)
            return metrics["loss"]

        def configure_optimizers(self):
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, float(self.b2)),
                weight_decay=self.weight_decay,
            )

    return BERTLightningModule()
