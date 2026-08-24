import math

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


def create_visible_attention_mask(
    visible_mask: Bool[Tensor, "batch tokens"],
) -> Bool[Tensor, "batch 1 1 tokens"]:
    # We mask keys only. Hidden query rows are dropped or ignored by callers,
    # and allowing them to attend to visible keys avoids empty-row SDPA masks.
    return visible_mask[:, None, None, :]


def _build_norm(
    dim: int,
    eps: float | None,
    *,
    affine: bool = True,
) -> nn.Module:
    eps = 1e-5 if eps is None else eps
    return nn.LayerNorm(dim, eps=eps, elementwise_affine=affine)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim={self.dim} must be divisible by n_heads={self.n_heads}"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} must be divisible by n_kv_heads={self.n_kv_heads}"
            )
        self.head_dim = self.dim // self.n_heads
        self.q_size = self.n_heads * self.head_dim
        self.kv_size = self.n_kv_heads * self.head_dim

        self.wqkv = nn.Linear(self.dim, self.q_size + 2 * self.kv_size, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        nn.init.xavier_normal_(self.wqkv.weight[: self.q_size])
        nn.init.xavier_normal_(self.wqkv.weight[self.q_size : self.q_size + self.kv_size])
        nn.init.xavier_normal_(self.wqkv.weight[self.q_size + self.kv_size :])
        nn.init.xavier_normal_(self.wo.weight)

    def forward(
        self,
        x: Float[Tensor, "batch tokens dim"],
        *,
        attn_mask: (
            Bool[Tensor, "batch #heads #query_tokens #key_tokens"]
            | Float[Tensor, "batch #heads #query_tokens #key_tokens"]
            | None
        ) = None,
    ) -> Float[Tensor, "batch tokens dim"]:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wqkv(x).split(
            (self.q_size, self.kv_size, self.kv_size),
            dim=-1,
        )
        # xq/xk/xv: [B, T, H, Dh] before transposing to SDPA's [B, H, T, Dh].
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq = xq.to(dtype=xv.dtype)
        xk = xk.to(dtype=xv.dtype)

        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(attn)


class CrossAttention(nn.Module):
    """Cross-attention with queries from x and keys/values from memory."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
        rope_max_sequence_length: int | None = None,
        rope_base: float = 10_000.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = self.dim // self.n_heads
        assert self.head_dim % 2 == 0
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)
        self.q_norm = nn.RMSNorm(
            self.head_dim,
            eps=norm_eps,
            elementwise_affine=False,
        )
        self.k_norm = nn.RMSNorm(
            self.head_dim,
            eps=norm_eps,
            elementwise_affine=False,
        )
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wkv.weight)
        nn.init.xavier_normal_(self.wo.weight)
        if rope_max_sequence_length is not None:
            inv_freq = 1.0 / (
                rope_base
                ** (
                    torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                    / self.head_dim
                )
            )
            frequencies = torch.outer(
                torch.arange(rope_max_sequence_length, dtype=torch.float32),
                inv_freq,
            )
            self.register_buffer("rope_cos", frequencies.cos(), persistent=False)
            self.register_buffer("rope_sin", frequencies.sin(), persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

    def _apply_rope(
        self,
        tensor: Float[Tensor, "batch heads tokens head_dim"],
        positions: Int[Tensor, "batch tokens"],
    ) -> Float[Tensor, "batch heads tokens head_dim"]:
        cos = self.rope_cos[positions].unsqueeze(1).to(dtype=tensor.dtype)
        sin = self.rope_sin[positions].unsqueeze(1).to(dtype=tensor.dtype)
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        ).flatten(-2)

    def forward(
        self,
        x: Float[Tensor, "batch tokens dim"],
        memory: Float[Tensor, "batch memory dim"],
        *,
        memory_mask: Bool[Tensor, "batch memory"] | None = None,
        query_positions: Int[Tensor, "batch tokens"] | None = None,
        memory_positions: Int[Tensor, "batch memory"] | None = None,
    ) -> Float[Tensor, "batch tokens dim"]:
        bsz, tgt_len, _ = x.shape
        mem_len = memory.shape[1]
        xq = self.wq(x).view(bsz, tgt_len, self.n_heads, self.head_dim)
        kv = self.wkv(memory)
        xk, xv = kv.split(
            [self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
            dim=-1,
        )
        xk = xk.view(bsz, mem_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, mem_len, self.n_kv_heads, self.head_dim)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if query_positions is not None:
            q = self._apply_rope(q, query_positions)
            k = self._apply_rope(k, memory_positions)
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        attn_mask = (
            None if memory_mask is None else memory_mask[:, None, None, :]
        )
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(bsz, tgt_len, self.dim)
        return self.wo(attn)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int | None = None,
    ):
        super().__init__()

        hidden_dim = hidden_dim or int((4 * dim) * 2 / 3)
        hidden_dim = 4 * math.ceil(hidden_dim / 4)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        nn.init.trunc_normal_(self.w1.weight, std=1.0 / math.sqrt(dim))
        nn.init.trunc_normal_(self.w2.weight, std=1.0 / math.sqrt(hidden_dim))

    def forward(self, x: Float[Tensor, "*batch dim"]) -> Float[Tensor, "*batch dim"]:
        return self.w2(F.silu(self.w1(x)))


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int | None = None,
    ):
        super().__init__()

        hidden_dim = hidden_dim or 4 * dim
        hidden_dim = 4 * math.ceil(hidden_dim / 4)

        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, dim, bias=False)

        nn.init.trunc_normal_(self.fc1.weight, std=1.0 / math.sqrt(dim))
        nn.init.trunc_normal_(self.fc2.weight, std=1.0 / math.sqrt(dim))
        nn.init.zeros_(self.fc3.weight)

    def forward(self, x: Float[Tensor, "*batch dim"]) -> Float[Tensor, "*batch dim"]:
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        norm_eps: float,
        hidden_dim: int,
        max_sequence_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = CrossAttention(
            dim,
            n_heads,
            rope_max_sequence_length=max_sequence_length,
            norm_eps=norm_eps,
        )
        self.feed_forward = SwiGLUFeedForward(dim, hidden_dim=hidden_dim)
        self.query_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.memory_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_post_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_post_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        query: Float[Tensor, "batch targets dim"],
        memory: Float[Tensor, "batch memory dim"],
        query_positions: Int[Tensor, "batch targets"],
        memory_positions: Int[Tensor, "batch memory"],
        query_mask: Bool[Tensor, "batch targets"],
        memory_mask: Bool[Tensor, "batch memory"],
    ) -> Float[Tensor, "batch targets dim"]:
        attention_update = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            memory_mask=memory_mask,
            query_positions=query_positions,
            memory_positions=memory_positions,
        )
        query = query + self.drop(self.attention_post_norm(attention_update))
        ffn_update = self.feed_forward(self.ffn_norm(query))
        query = query + self.drop(self.ffn_post_norm(ffn_update))
        return query * query_mask.unsqueeze(-1).to(query.dtype)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None,
        norm_eps: float,
        hidden_dim: int | None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = Attention(
            dim,
            n_heads,
            n_kv_heads=n_kv_heads,
        )
        self.feed_forward = FeedForward(
            dim,
            hidden_dim=hidden_dim,
        )
        self.attention_norm = _build_norm(dim, eps=norm_eps)
        self.ffn_norm = _build_norm(dim, eps=norm_eps)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: Float[Tensor, "batch tokens dim"],
        *,
        attn_mask: (
            Bool[Tensor, "batch #heads #query_tokens #key_tokens"]
            | Float[Tensor, "batch #heads #query_tokens #key_tokens"]
            | None
        ) = None,
    ) -> Float[Tensor, "batch tokens dim"]:
        h = x + self.drop(
            self.attention(
                self.attention_norm(x),
                attn_mask=attn_mask,
            )
        )
        return h + self.drop(self.feed_forward(self.ffn_norm(h)))
