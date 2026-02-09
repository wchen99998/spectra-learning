"""PyTorch autoregressive transformer used in pretraining and downstream tasks."""

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
                causal=True,
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
        max_length: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None,
        attention_mlp_multiple: float,
        rope_theta: float,
        cache_rope_frequencies: bool,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        heads, kv_heads = _resolve_attention_heads(dim, num_heads, num_kv_heads)
        self.dim = int(dim)
        self.max_length = int(max_length)
        self.num_heads = int(heads)
        self.rope_theta = float(rope_theta)
        self.cache_rope_frequencies = bool(cache_rope_frequencies)
        self.dtype = dtype
        self.head_dim = self.dim // self.num_heads
        self.blocks = _build_transformer_blocks(
            dim=self.dim,
            num_layers=num_layers,
            num_heads=self.num_heads,
            num_kv_heads=kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
        )
        self.norm = nn.RMSNorm(self.dim, eps=1e-5)
        if self.cache_rope_frequencies:
            freqs_cos, freqs_sin = transformer_torch.precompute_freqs_cis(
                self.head_dim,
                self.max_length,
                theta=self.rope_theta,
                dtype=torch.float32,
            )
            rotary_cos = freqs_cos[None, :, None, :].repeat_interleave(2, dim=-1)
            rotary_sin = freqs_sin[None, :, None, :].repeat_interleave(2, dim=-1)
            self.register_buffer("_rotary_cos_cache", rotary_cos, persistent=False)
            self.register_buffer("_rotary_sin_cache", rotary_sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        train: bool,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del train  # Transformer blocks here are deterministic.
        seq_len = x.shape[1]
        if self.cache_rope_frequencies:
            rotary_cos = self._rotary_cos_cache[:, :seq_len].to(dtype=x.dtype)
            rotary_sin = self._rotary_sin_cache[:, :seq_len].to(dtype=x.dtype)
        else:
            freqs_cos, freqs_sin = transformer_torch.precompute_freqs_cis(
                self.head_dim,
                seq_len,
                theta=self.rope_theta,
                device=x.device,
                dtype=x.dtype,
            )
            rotary_cos = freqs_cos[None, :, None, :].repeat_interleave(2, dim=-1)
            rotary_sin = freqs_sin[None, :, None, :].repeat_interleave(2, dim=-1)
        for block in self.blocks:
            x = block(
                x,
                freqs_cos=rotary_cos,
                freqs_sin=rotary_sin,
                attention_mask=attention_mask,
            )
        return self.norm(x)


class BERTTorch(nn.Module):
    """Transformer model with causal next-token objective."""

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
        pad_token_id: int = 0,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        rope_theta: float = 10000.0,
        cache_rope_frequencies: bool = True,
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
        self.pad_token_id = int(pad_token_id)
        self.cls_token_id = int(cls_token_id)
        self.sep_token_id = int(sep_token_id)
        self.rope_theta = float(rope_theta)
        self.dtype = dtype

        self.register_buffer(
            "_pad_token_id_tensor",
            torch.tensor(self.pad_token_id, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_precursor_offset_tensor",
            torch.tensor(self.precursor_offset, dtype=torch.long),
            persistent=False,
        )

        self.token_embed = nn.Embedding(self.vocab_size, self.model_dim)
        self.segment_embed = nn.Embedding(self.num_segments, self.model_dim)
        self.embed_norm = nn.LayerNorm(self.model_dim)

        embed_std = 1.0 / math.sqrt(self.model_dim)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=embed_std)
        nn.init.normal_(self.segment_embed.weight, mean=0.0, std=embed_std)

        self.encoder = TransformerStack(
            dim=self.model_dim,
            max_length=self.max_length,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            attention_mlp_multiple=self.attention_mlp_multiple,
            rope_theta=self.rope_theta,
            cache_rope_frequencies=cache_rope_frequencies,
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

    def _embed_inputs(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        tok = self.token_embed(token_ids)
        seg = self.segment_embed(segment_ids)
        return self.embed_norm(tok + seg)

    def _next_token_metrics(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_logits = logits[:, :-1, :]
        next_labels = token_ids[:, 1:]
        valid = next_labels != self._pad_token_id_tensor
        valid_logits = next_logits[valid]
        valid_labels = next_labels[valid]
        token_loss = F.cross_entropy(valid_logits, valid_labels)
        pred = valid_logits.argmax(dim=-1)
        token_accuracy = (pred == valid_labels).to(torch.float32).mean()
        return token_loss, token_accuracy

    def _eos_state(self, encoded: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        sep_idx = (token_ids == self.sep_token_id).to(torch.int64).argmax(dim=1)
        batch_idx = torch.arange(token_ids.shape[0], device=token_ids.device)
        return encoded[batch_idx, sep_idx, :]

    def _precursor_metrics(
        self,
        cls_state: torch.Tensor,
        precursor_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_full = self.precursor_head(cls_state)
        labels = (precursor_tokens - self._precursor_offset_tensor).to(torch.long)
        num_bins = logits_full.shape[-1] // 2
        logits = logits_full[:, :num_bins]
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).to(torch.float32).mean()
        return loss, acc

    def _retention_metrics(
        self,
        cls_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss = torch.zeros((), dtype=cls_state.dtype, device=cls_state.device)
        acc = torch.zeros((), dtype=torch.float32, device=cls_state.device)
        return loss, acc

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train: bool = True,
    ) -> dict[str, torch.Tensor]:
        del train
        token_ids = batch["token_ids"].to(torch.long)
        segment_ids = batch["segment_ids"].to(torch.long)

        x = self._embed_inputs(token_ids, segment_ids)
        encoded = self.encoder(x, train=False, attention_mask=None)
        logits = self.lm_head(encoded)
        token_loss, token_accuracy = self._next_token_metrics(logits, token_ids)
        eos_state = self._eos_state(encoded, token_ids)
        precursor_tokens = batch["precursor_mz"].to(torch.long)
        precursor_loss, precursor_accuracy = self._precursor_metrics(eos_state, precursor_tokens)
        retention_loss, retention_accuracy = self._retention_metrics(eos_state)
        loss = token_loss + precursor_loss + retention_loss

        return {
            "loss": loss,
            "token_loss": token_loss,
            "token_accuracy": token_accuracy,
            "precursor_loss": precursor_loss,
            "precursor_accuracy": precursor_accuracy,
            "retention_loss": retention_loss,
            "retention_accuracy": retention_accuracy,
        }

    def encode(self, batch: dict[str, torch.Tensor], *, train: bool = False) -> torch.Tensor:
        del train
        token_ids = batch["token_ids"].to(torch.long)
        segment_ids = batch["segment_ids"].to(torch.long)

        x = self._embed_inputs(token_ids, segment_ids)
        encoded = self.encoder(x, train=False, attention_mask=None)
        return self._eos_state(encoded, token_ids)

    def compute_loss(self, batch: dict[str, torch.Tensor], *, train: bool = False):
        metrics = self(batch, train=train)
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

        def forward(self, batch: dict[str, torch.Tensor], *, train: bool):
            return self.model(batch, train=train)

        def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            del batch_idx
            metrics = self.model(batch, train=True)
            self.log_dict({f"train/{k}": v for k, v in metrics.items()}, prog_bar=True)
            return metrics["loss"]

        def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            del batch_idx
            metrics = self.model(batch, train=False)
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
