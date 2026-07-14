from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from ml_collections import config_dict
from torch import nn

from spectra_learning.data.ar_spectra import (
    SPECTRA_AR_TARGET_KINDS,
    SpectraARTokenKind,
    SpectraARTokenizer,
)


@dataclass(frozen=True)
class SpectraARTransformerConfig:
    vocab_size: int
    num_token_kinds: int
    max_sequence_length: int
    pad_token_id: int
    model_dim: int = 384
    num_layers: int = 6
    num_heads: int = 8
    mlp_multiple: float = 4.0
    dropout: float = 0.1
    rope_base: float = 10_000.0

    @classmethod
    def from_config(
        cls,
        config: config_dict.ConfigDict,
        tokenizer: SpectraARTokenizer,
    ) -> "SpectraARTransformerConfig":
        return cls(
            vocab_size=tokenizer.vocab_size,
            num_token_kinds=tokenizer.num_token_kinds,
            max_sequence_length=tokenizer.sequence_length,
            pad_token_id=tokenizer.pad_token_id,
            model_dim=int(config.get("ar_model_dim", 384)),
            num_layers=int(config.get("ar_num_layers", 6)),
            num_heads=int(config.get("ar_num_heads", 8)),
            mlp_multiple=float(config.get("ar_mlp_multiple", 4.0)),
            dropout=float(config.get("ar_dropout", 0.1)),
            rope_base=float(config.get("ar_rope_base", 10_000.0)),
        )


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_sequence_length: int,
        *,
        base: float,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / float(head_dim)
            )
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        seq_len = tensor.shape[-2]
        cos = self.cos[:seq_len].to(device=tensor.device, dtype=tensor.dtype)
        sin = self.sin[:seq_len].to(device=tensor.device, dtype=tensor.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.empty_like(tensor)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = even * sin + odd * cos
        return rotated


class SpectraARCausalSelfAttention(nn.Module):
    def __init__(self, config: SpectraARTransformerConfig) -> None:
        super().__init__()
        assert config.model_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        assert self.head_dim % 2 == 0
        self.qkv = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False)
        self.out_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(
            self.head_dim,
            config.max_sequence_length,
            base=config.rope_base,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = hidden.shape
        qkv = self.qkv(hidden).view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query = self.rope(query)
        key = self.rope(key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            model_dim,
        )
        return self.out_proj(attended)


class SpectraARTransformerBlock(nn.Module):
    def __init__(self, config: SpectraARTransformerConfig) -> None:
        super().__init__()
        hidden_dim = int(config.model_dim * config.mlp_multiple)
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.attention = SpectraARCausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.model_dim),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.residual_dropout(self.attention(self.attention_norm(hidden)))
        hidden = hidden + self.residual_dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden


class SpectraARTransformer(nn.Module):
    def __init__(self, config: SpectraARTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.pad_token_id = config.pad_token_id
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.kind_embedding = nn.Embedding(config.num_token_kinds, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            SpectraARTransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def hidden_states(
        self,
        input_ids: torch.Tensor,
        token_kinds: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(dtype=torch.long)
        token_kinds = token_kinds.to(dtype=torch.long)
        hidden = self.dropout(
            self.token_embedding(input_ids) + self.kind_embedding(token_kinds)
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.hidden_states(
            batch["input_token_ids"],
            batch["input_token_kinds"],
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.encode_batch(batch)
        logits = self.lm_head(hidden)
        output = {"logits": logits}
        if "target_token_ids" in batch:
            output.update(self._loss_metrics(logits, batch))
        return output

    def _loss_metrics(
        self,
        logits: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        labels = batch["target_token_ids"].to(dtype=torch.long)
        target_kinds = batch["target_token_kinds"].to(dtype=torch.long)
        loss_mask = batch["target_loss_mask"].to(dtype=torch.bool)
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        denominator = loss_mask.sum().clamp_min(1)
        loss = (token_loss * loss_mask).sum() / denominator
        predictions = logits.argmax(dim=-1)
        accuracy = ((predictions == labels) & loss_mask).sum() / denominator
        metrics = {
            "loss": loss,
            "token_accuracy": accuracy,
            "target_tokens": denominator.to(dtype=torch.float32),
        }
        for kind in SPECTRA_AR_TARGET_KINDS:
            kind_mask = loss_mask & target_kinds.eq(int(kind))
            kind_count = kind_mask.sum()
            kind_denominator = kind_count.clamp_min(1)
            kind_name = kind.name.lower()
            metrics[f"loss/{kind.name.lower()}"] = (
                token_loss * kind_mask
            ).sum() / kind_denominator
            metrics[f"token_accuracy/{kind_name}"] = (
                (predictions == labels) & kind_mask
            ).sum() / kind_denominator
            metrics[f"target_tokens/{kind_name}"] = kind_count.to(dtype=torch.float32)
        return metrics


def build_spectra_ar_model_from_config(
    config: config_dict.ConfigDict,
    tokenizer: SpectraARTokenizer,
) -> SpectraARTransformer:
    return SpectraARTransformer(SpectraARTransformerConfig.from_config(config, tokenizer))
