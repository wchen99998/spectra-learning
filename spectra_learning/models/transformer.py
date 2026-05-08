import math

import torch
import torch.nn.functional as F
from torch import nn


def create_visible_attention_mask(visible_mask: torch.Tensor) -> torch.Tensor:
    # We mask keys only. Hidden query rows are dropped or ignored by callers,
    # and allowing them to attend to visible keys avoids empty-row SDPA masks.
    return visible_mask[:, None, None, :]


def combine_attention_mask_and_bias(
    attn_mask: torch.Tensor | None,
    attn_bias: torch.Tensor | None,
    *,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if attn_bias is None:
        return attn_mask

    bias = attn_bias.to(dtype=dtype)

    if attn_mask is None:
        return bias

    if attn_mask.dtype == torch.bool:
        return bias.masked_fill(~attn_mask, torch.finfo(dtype).min)

    return bias + attn_mask.to(dtype=dtype)


def _build_norm(
    dim: int,
    eps: float | None,
    norm_type: str,
    *,
    affine: bool = True,
) -> nn.Module:
    kind = str(norm_type).lower()
    eps = 1e-5 if eps is None else eps
    if kind == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps, elementwise_affine=affine)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=affine)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
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
        self.qk_norm = qk_norm
        self.q_size = self.n_heads * self.head_dim
        self.kv_size = self.n_kv_heads * self.head_dim

        self.wqkv = nn.Linear(self.dim, self.q_size + 2 * self.kv_size, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        if qk_norm:
            self.q_norm = _build_norm(self.head_dim, eps=norm_eps, norm_type=norm_type)
            self.k_norm = _build_norm(self.head_dim, eps=norm_eps, norm_type=norm_type)

        nn.init.xavier_normal_(self.wqkv.weight[: self.q_size])
        nn.init.xavier_normal_(self.wqkv.weight[self.q_size : self.q_size + self.kv_size])
        nn.init.xavier_normal_(self.wqkv.weight[self.q_size + self.kv_size :])
        nn.init.xavier_normal_(self.wo.weight)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        q_key = prefix + "wq.weight"
        k_key = prefix + "wk.weight"
        v_key = prefix + "wv.weight"
        qkv_key = prefix + "wqkv.weight"
        if qkv_key not in state_dict and q_key in state_dict:
            state_dict[qkv_key] = torch.cat(
                [state_dict.pop(q_key), state_dict.pop(k_key), state_dict.pop(v_key)],
                dim=0,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wqkv(x).split(
            (self.q_size, self.kv_size, self.kv_size),
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq = xq.to(dtype=xv.dtype)
        xk = xk.to(dtype=xv.dtype)

        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        sdpa_mask = combine_attention_mask_and_bias(
            attn_mask,
            attn_bias,
            dtype=q.dtype,
        )
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None,
        norm_eps: float,
        hidden_dim: int | None,
        qk_norm: bool = False,
        norm_type: str = "rmsnorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = Attention(
            dim,
            n_heads,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            norm_type=norm_type,
            norm_eps=norm_eps,
        )
        self.feed_forward = FeedForward(
            dim,
            hidden_dim=hidden_dim,
        )
        self.attention_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.ffn_norm = _build_norm(dim, eps=norm_eps, norm_type=norm_type)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x + self.drop(
            self.attention(
                self.attention_norm(x),
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )
        )
        return h + self.drop(self.feed_forward(self.ffn_norm(h)))
