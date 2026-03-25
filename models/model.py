import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from models.losses import SIGReg


def create_visible_block_mask(visible_mask: torch.Tensor) -> BlockMask:
    batch_size, seq_len = visible_mask.shape

    def mask_mod(batch_idx, _head_idx, q_idx, kv_idx):
        return visible_mask[batch_idx, q_idx] & visible_mask[batch_idx, kv_idx]

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=visible_mask.device,
    )


def create_cross_block_mask(
    context_mask: torch.Tensor,
    target_len: int,
) -> BlockMask:
    batch_size, context_len = context_mask.shape

    def mask_mod(batch_idx, _head_idx, _q_idx, kv_idx):
        return context_mask[batch_idx, kv_idx]

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=target_len,
        KV_LEN=context_len,
        device=context_mask.device,
    )


def _repeat_kv_for_gqa(
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_kv_heads == num_heads:
        return k, v
    repeat_factor = num_heads // num_kv_heads
    return (
        k.repeat_interleave(repeat_factor, dim=1),
        v.repeat_interleave(repeat_factor, dim=1),
    )


def masked_attention(xq, xk, xv, vis_mask, block_n=0):
    """Masked self-attention using flex_attention.

    Args:
        xq, xk, xv: [B, N, H, D] in (batch, seq, head, dim) layout
        vis_mask: [B, N] bool — True for visible positions
        block_n: unused (kept for API compat)
    """
    q = xq.transpose(1, 2)  # [B, H, N, D]
    k = xk.transpose(1, 2)
    v = xv.transpose(1, 2)
    block_mask = create_visible_block_mask(vis_mask)
    out = flex_attention(q, k, v, block_mask=block_mask)
    return out.transpose(1, 2)  # [B, N, H, D]


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    rotated = torch.empty_like(x)
    rotated[..., ::2] = -x[..., 1::2]
    rotated[..., 1::2] = x[..., ::2]
    return rotated


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor,
    freqs_cos: torch.Tensor, freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = _rotate_half(xq)
    k_rot = _rotate_half(xk)
    return (xq * freqs_cos) + (q_rot * freqs_sin), (xk * freqs_cos) + (k_rot * freqs_sin)


def _build_norm(dim: int, eps: float | None, norm_type: str) -> nn.Module:
    kind = str(norm_type).lower()
    if kind == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _normalize_norm_position(norm_position: str) -> str:
    kind = str(norm_position).lower()
    if kind not in ("prenorm", "postnorm"):
        raise ValueError(f"Unsupported norm_position: {norm_position}")
    return kind


def _precompute_rope_freqs(
    seq_len: int, inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
    angles = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    return angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, *, n_kv_heads: int | None = None,
                 qk_norm: bool = False, norm_type: str = "rmsnorm"):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = self.dim // self.n_heads
        self.qk_norm = qk_norm
        out_features = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.wqkv = nn.Linear(self.dim, out_features, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)
        if qk_norm:
            self.q_norm = _build_norm(self.head_dim, eps=None, norm_type=norm_type)
            self.k_norm = _build_norm(self.head_dim, eps=None, norm_type=norm_type)
        nn.init.xavier_normal_(self.wqkv.weight)
        nn.init.xavier_normal_(self.wo.weight)

    def forward(self, x: torch.Tensor, *, freqs_cos: torch.Tensor | None = None,
                freqs_sin: torch.Tensor | None = None,
                vis_mask: torch.Tensor | None = None, pad_to: int = 0) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.wqkv(x)
        xq, xk, xv = qkv.split(
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim,
             self.n_kv_heads * self.head_dim], dim=-1)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        if freqs_cos is not None and freqs_sin is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xq = xq.to(dtype=xv.dtype)
        xk = xk.to(dtype=xv.dtype)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        k, v = _repeat_kv_for_gqa(k, v, self.n_heads, self.n_kv_heads)
        block_mask = create_visible_block_mask(vis_mask) if vis_mask is not None else None
        attn = flex_attention(q, k, v, block_mask=block_mask)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(attn)


class FeedForward(nn.Module):
    def __init__(self, dim: int, *, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or int((4 * dim) * 2 / 3)
        hidden_dim = 4 * math.ceil(hidden_dim / 4)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.trunc_normal_(self.w1.weight, std=1.0 / math.sqrt(dim))
        nn.init.trunc_normal_(self.w2.weight, std=1.0 / math.sqrt(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))


class CrossAttention(nn.Module):
    """Cross-attention: Q from target queries, KV from context embeddings."""

    def __init__(self, dim: int, n_heads: int, *, n_kv_heads: int | None = None,
                 qk_norm: bool = False, norm_type: str = "rmsnorm"):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = self.dim // self.n_heads
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = _build_norm(self.head_dim, eps=None, norm_type=norm_type)
            self.k_norm = _build_norm(self.head_dim, eps=None, norm_type=norm_type)
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wkv.weight)
        nn.init.xavier_normal_(self.wo.weight)

    def forward(self, x: torch.Tensor, context: torch.Tensor, *,
                freqs_cos: torch.Tensor | None = None,
                freqs_sin: torch.Tensor | None = None,
                context_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, tgt_len, _ = x.shape
        ctx_len = context.shape[1]
        xq = self.wq(x).view(bsz, tgt_len, self.n_heads, self.head_dim)
        kv = self.wkv(context)
        xk, xv = kv.split(
            [self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim], dim=-1)
        xk = xk.view(bsz, ctx_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, ctx_len, self.n_kv_heads, self.head_dim)
        # RoPE on Q only (target positions); no RoPE on K
        if freqs_cos is not None and freqs_sin is not None:
            q_rot = _rotate_half(xq)
            xq = (xq * freqs_cos) + (q_rot * freqs_sin)
        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        xq = xq.to(dtype=xv.dtype)
        xk = xk.to(dtype=xv.dtype)
        q = xq.transpose(1, 2)  # [B, H, T, D]
        k = xk.transpose(1, 2)  # [B, H, S, D]
        v = xv.transpose(1, 2)
        k, v = _repeat_kv_for_gqa(k, v, self.n_heads, self.n_kv_heads)
        block_mask = (
            create_cross_block_mask(context_mask, tgt_len)
            if context_mask is not None
            else None
        )
        attn = flex_attention(q, k, v, block_mask=block_mask)
        attn = attn.transpose(1, 2).contiguous().view(bsz, tgt_len, self.dim)
        return self.wo(attn)


class TemporalDecoderBlock(nn.Module):
    """Decoder block: self-attention + cross-attention + FFN."""

    def __init__(self, *, dim: int, n_heads: int, n_kv_heads: int | None,
                 norm_eps: float, hidden_dim: int | None,
                 qk_norm: bool = False, norm_type: str = "rmsnorm",
                 norm_position: str = "prenorm"):
        super().__init__()
        self.norm_position = _normalize_norm_position(norm_position)
        self.attention = Attention(dim, n_heads, n_kv_heads=n_kv_heads,
                                   qk_norm=qk_norm, norm_type=norm_type)
        self.cross_attn = CrossAttention(dim, n_heads, n_kv_heads=n_kv_heads,
                                          qk_norm=qk_norm, norm_type=norm_type)
        self.feed_forward = FeedForward(dim, hidden_dim=hidden_dim)
        self.attention_norm = _build_norm(dim, eps=None, norm_type=norm_type)
        self.cross_attn_norm = _build_norm(dim, eps=None, norm_type=norm_type)
        self.ffn_norm = _build_norm(dim, eps=None, norm_type=norm_type)

    def forward(self, x: torch.Tensor, context: torch.Tensor, *,
                freqs_cos: torch.Tensor | None,
                freqs_sin: torch.Tensor | None,
                context_mask: torch.Tensor | None = None,
                vis_mask: torch.Tensor | None = None,
                pad_to: int = 0) -> torch.Tensor:
        if self.norm_position == "postnorm":
            h = self.attention_norm(
                x + self.attention(
                    x,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    vis_mask=vis_mask,
                    pad_to=pad_to,
                )
            )
            h = self.cross_attn_norm(
                h + self.cross_attn(
                    h,
                    context,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    context_mask=context_mask,
                )
            )
            return self.ffn_norm(h + self.feed_forward(h))
        h = x + self.attention(self.attention_norm(x), freqs_cos=freqs_cos,
                               freqs_sin=freqs_sin, vis_mask=vis_mask, pad_to=pad_to)
        h = h + self.cross_attn(self.cross_attn_norm(h), context,
                                freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                                context_mask=context_mask)
        return h + self.feed_forward(self.ffn_norm(h))


def _apply_depth_scaled_init(blocks: nn.ModuleList, num_layers: int) -> None:
    """Scale residual output projections by 1/sqrt(2*num_layers) (GPT-2 style).

    In pre-norm transformers each residual addition contributes ~unit variance,
    so after 2*L sub-layers the activation norm grows by sqrt(2*L).  Scaling
    the output projections (wo in attention, w2 in FFN) keeps the total
    variance growth O(1) regardless of depth.
    """
    if num_layers <= 0:
        return
    scale = 1.0 / math.sqrt(2.0 * num_layers)
    for block in blocks:
        if hasattr(block, "attention"):
            block.attention.wo.weight.data.mul_(scale)
        if hasattr(block, "cross_attn"):
            block.cross_attn.wo.weight.data.mul_(scale)
        if hasattr(block, "feed_forward"):
            block.feed_forward.w2.weight.data.mul_(scale)


def _build_temporal_decoder_blocks(*, dim: int, num_layers: int, num_heads: int,
                                    num_kv_heads: int | None, attention_mlp_multiple: float,
                                    norm_eps: float = 1e-5, qk_norm: bool = False,
                                    norm_type: str = "rmsnorm",
                                    norm_position: str = "prenorm") -> nn.ModuleList:
    block_kwargs = dict(
        dim=dim, n_heads=int(num_heads),
        n_kv_heads=int(num_heads) if num_kv_heads is None else int(num_kv_heads),
        norm_eps=norm_eps, hidden_dim=int(math.ceil(dim * attention_mlp_multiple)),
        qk_norm=qk_norm, norm_type=norm_type, norm_position=norm_position,
    )
    blocks = nn.ModuleList([TemporalDecoderBlock(**block_kwargs) for _ in range(num_layers)])
    _apply_depth_scaled_init(blocks, num_layers)
    return blocks


class TransformerBlock(nn.Module):
    def __init__(self, *, dim: int, n_heads: int, n_kv_heads: int | None,
                 norm_eps: float, hidden_dim: int | None,
                 qk_norm: bool = False, norm_type: str = "rmsnorm",
                 norm_position: str = "prenorm"):
        super().__init__()
        self.norm_position = _normalize_norm_position(norm_position)
        self.attention = Attention(dim, n_heads, n_kv_heads=n_kv_heads,
                                   qk_norm=qk_norm, norm_type=norm_type)
        self.feed_forward = FeedForward(dim, hidden_dim=hidden_dim)
        self.attention_norm = _build_norm(dim, eps=None, norm_type=norm_type)
        self.ffn_norm = _build_norm(dim, eps=None, norm_type=norm_type)

    def forward(self, x: torch.Tensor, *, freqs_cos: torch.Tensor | None,
                freqs_sin: torch.Tensor | None,
                vis_mask: torch.Tensor | None = None, pad_to: int = 0) -> torch.Tensor:
        if self.norm_position == "postnorm":
            h = self.attention_norm(
                x + self.attention(
                    x,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    vis_mask=vis_mask,
                    pad_to=pad_to,
                )
            )
            return self.ffn_norm(h + self.feed_forward(h))
        h = x + self.attention(self.attention_norm(x), freqs_cos=freqs_cos,
                               freqs_sin=freqs_sin, vis_mask=vis_mask, pad_to=pad_to)
        return h + self.feed_forward(self.ffn_norm(h))


def _build_non_causal_blocks(*, dim: int, num_layers: int, num_heads: int,
                              num_kv_heads: int | None, attention_mlp_multiple: float,
                              norm_eps: float = 1e-5, qk_norm: bool = False,
                              norm_type: str = "rmsnorm",
                              norm_position: str = "prenorm") -> nn.ModuleList:
    block_kwargs = dict(
        dim=dim, n_heads=int(num_heads),
        n_kv_heads=int(num_heads) if num_kv_heads is None else int(num_kv_heads),
        norm_eps=norm_eps, hidden_dim=int(math.ceil(dim * attention_mlp_multiple)),
        qk_norm=qk_norm, norm_type=norm_type, norm_position=norm_position,
    )
    blocks = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(num_layers)])
    _apply_depth_scaled_init(blocks, num_layers)
    return blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_rope_freqs(
    use_rope: bool, seq_len: int, inv_freq: torch.Tensor,
    device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not use_rope:
        return None, None
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    angles = positions.unsqueeze(-1) * inv_freq.to(device=device).view(1, 1, -1)
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    return angles.cos().to(dtype=dtype).unsqueeze(2), angles.sin().to(dtype=dtype).unsqueeze(2)


def _masked_embedding_stats(
    emb: torch.Tensor, valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    flat = emb.float().reshape(-1, emb.shape[-1])
    weights = valid_mask.reshape(-1).float()
    count = weights.sum().clamp_min(1.0)
    weights_col = weights.unsqueeze(-1)
    centered = flat - (flat * weights_col).sum(0) / count
    cov = centered.transpose(0, 1) @ (centered * weights_col) / count
    var = cov.diagonal()
    vs = var.clamp_min(1e-12)
    corr = cov / torch.sqrt(vs.unsqueeze(0) * vs.unsqueeze(1))
    d = cov.shape[0] * (cov.shape[0] - 1)
    return {
        "emb_std": torch.sqrt(var + 1e-6).mean(),
        "emb_norm": (flat.norm(dim=-1) * weights).sum() / count,
        "emb_var_mean": var.mean(),
        "emb_var_floor": var.amin(),
        "emb_cov_offdiag_abs_mean": (cov.abs().sum() - cov.diagonal().abs().sum()) / d,
        "emb_corr_offdiag_abs_mean": (corr.abs().sum() - corr.diagonal().abs().sum()) / d,
    }


def _masked_mean_token_norm(
    emb: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    norms = emb.float().norm(dim=-1)
    weights = valid_mask.float()
    count = weights.sum().clamp_min(1.0)
    return (norms * weights).sum() / count


# ---------------------------------------------------------------------------
# PeakSetEncoder with sparse packing + precomputed RoPE
# ---------------------------------------------------------------------------

class PeakSetEncoder(nn.Module):
    def __init__(self, *, model_dim: int, num_layers: int, num_heads: int,
                 num_kv_heads: int | None = None, attention_mlp_multiple: float = 4.0,
                 feature_mlp_hidden_dim: int = 128, use_rope: bool = False,
                 qk_norm: bool = False, norm_type: str = "rmsnorm",
                 norm_position: str = "prenorm", seq_len: int = 64):
        super().__init__()
        self.use_rope = bool(use_rope)
        norm_type = str(norm_type).lower()
        norm_position = _normalize_norm_position(norm_position)
        self.embedder = nn.Sequential(
            nn.Linear(3, feature_mlp_hidden_dim), nn.SiLU(),
            nn.Linear(feature_mlp_hidden_dim, model_dim),
        )
        for layer in self.embedder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        _h = model_dim // int(num_heads) // 2
        self.register_buffer(
            "rope_inv_freq",
            1.0 / (10000.0 ** (torch.arange(_h, dtype=torch.float32) / _h)),
            persistent=False,
        )
        if self.use_rope:
            rope_cos, rope_sin = _precompute_rope_freqs(seq_len, self.rope_inv_freq)
            self.register_buffer("rope_cos", rope_cos, persistent=False)
            self.register_buffer("rope_sin", rope_sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None
        self.blocks = _build_non_causal_blocks(
            dim=model_dim, num_layers=num_layers, num_heads=num_heads,
            num_kv_heads=num_kv_heads, attention_mlp_multiple=attention_mlp_multiple,
            qk_norm=qk_norm, norm_type=norm_type, norm_position=norm_position,
        )
        self.final_norm = _build_norm(model_dim, eps=None, norm_type=norm_type)

    def forward(self, peak_mz: torch.Tensor, peak_intensity: torch.Tensor,
                valid_mask: torch.Tensor | None = None,
                visible_mask: torch.Tensor | None = None,
                pack_n: int = 0, prefix_pack: bool = False,
                pad_to: int = 0) -> torch.Tensor:
        if visible_mask is not None and valid_mask is not None:
            vis = visible_mask & valid_mask
        else:
            vis = visible_mask if visible_mask is not None else valid_mask

        # Sparse packing path: gather only visible tokens
        if vis is not None and pack_n > 0:
            B, N = peak_mz.shape
            PACK_N = min(pack_n, N)
            if prefix_pack:
                packed_mz = peak_mz[:, :PACK_N]
                packed_intensity = peak_intensity[:, :PACK_N]
                vis = vis[:, :PACK_N]
                log_intensity = torch.log1p(packed_intensity.clamp(min=0.0))
                x = self.embedder(torch.stack([packed_mz, packed_intensity, log_intensity], dim=-1))
                D = x.shape[-1]
                if self.rope_cos is not None:
                    freqs_cos = self.rope_cos.to(dtype=x.dtype)[:, :PACK_N]
                    freqs_sin = self.rope_sin.to(dtype=x.dtype)[:, :PACK_N]
                else:
                    freqs_cos = freqs_sin = None
                for block in self.blocks:
                    x = block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                              vis_mask=vis, pad_to=pad_to)
                x = self.final_norm(x)
                return F.pad(x, (0, 0, 0, N - PACK_N))
            else:
                sort_idx = vis.to(dtype=torch.int8).argsort(dim=1, descending=True, stable=True)
                pack_idx = sort_idx[:, :PACK_N]
                packed_mz = peak_mz.gather(1, pack_idx)
                packed_intensity = peak_intensity.gather(1, pack_idx)
                vis = vis.gather(1, pack_idx)
                log_intensity = torch.log1p(packed_intensity.clamp(min=0.0))
                x = self.embedder(torch.stack([packed_mz, packed_intensity, log_intensity], dim=-1))
                D = x.shape[-1]
                if self.rope_cos is not None:
                    rc = self.rope_cos.to(dtype=x.dtype)[0, :N, 0, :]  # [N, head_dim]
                    rs = self.rope_sin.to(dtype=x.dtype)[0, :N, 0, :]
                    freqs_cos = rc[pack_idx].unsqueeze(2)
                    freqs_sin = rs[pack_idx].unsqueeze(2)
                else:
                    freqs_cos = freqs_sin = None
                for block in self.blocks:
                    x = block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                              vis_mask=vis, pad_to=pad_to)
                x = self.final_norm(x)
                idx_expand = pack_idx.unsqueeze(-1).expand(-1, -1, D)
                return torch.zeros(B, N, D, device=x.device, dtype=x.dtype).scatter(1, idx_expand, x)

        # Full sequence path (no packing)
        log_intensity = torch.log1p(peak_intensity.clamp(min=0.0))
        x = self.embedder(torch.stack([peak_mz, peak_intensity, log_intensity], dim=-1))
        seq_len = peak_mz.shape[1]
        if self.rope_cos is not None:
            freqs_cos = self.rope_cos.to(dtype=x.dtype)[:, :seq_len]
            freqs_sin = self.rope_sin.to(dtype=x.dtype)[:, :seq_len]
        else:
            freqs_cos, freqs_sin = _compute_rope_freqs(
                self.use_rope, peak_mz.shape[1], self.rope_inv_freq, peak_mz.device, x.dtype)
        for block in self.blocks:
            x = block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin, vis_mask=vis)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# PeakSetSIGReg — full JEPA model
# ---------------------------------------------------------------------------

class PeakSetSIGReg(nn.Module):
    def __init__(
        self, *, model_dim: int = 768, encoder_num_layers: int = 20,
        encoder_num_heads: int = 12, encoder_num_kv_heads: int | None = None,
        attention_mlp_multiple: float = 4.0, feature_mlp_hidden_dim: int = 128,
        encoder_use_rope: bool = False, masked_token_loss_weight: float = 0.0,
        representation_regularizer: str = "sigreg",
        masked_latent_predictor_num_layers: int = 2, sigreg_num_slices: int = 256,
        sigreg_lambda: float = 0.1,
        jepa_num_target_blocks: int = 2,
        encoder_qk_norm: bool = False, norm_type: str = "rmsnorm",
        norm_position: str = "prenorm",
        use_precursor_token: bool = False,
        num_peaks: int = 64, jepa_context_fraction: float = 0.3,
        jepa_target_fraction: float = 0.25,
        temporal_predictor_num_layers: int = 0,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.use_precursor_token = bool(use_precursor_token)
        self.jepa_num_target_blocks = int(jepa_num_target_blocks)
        self.sigreg_lambda = float(sigreg_lambda)
        self.representation_regularizer = str(representation_regularizer).lower()
        if self.representation_regularizer not in ("sigreg", "none", ""):
            raise ValueError(f"Unsupported regularizer: {self.representation_regularizer!r}")
        self.masked_token_loss_weight = float(masked_token_loss_weight)
        self.norm_type = str(norm_type).lower()
        self.norm_position = _normalize_norm_position(norm_position)
        if self.jepa_num_target_blocks < 1:
            raise ValueError("jepa_num_target_blocks must be >= 1")

        # Packing parameters (computed from config)
        N = int(num_peaks)
        self._context_pack_n = max(1, int(math.ceil(N * jepa_context_fraction)))
        self._target_pack_n = max(1, int(math.ceil(N * jepa_target_fraction)))
        self._predictor_pack_n = self._target_pack_n
        self._full_pack_n = N
        self._context_pad_to = 1 << max(1, math.ceil(math.log2(max(1, self._context_pack_n))))
        self._predictor_pad_to = 1 << max(1, math.ceil(math.log2(max(1, self._predictor_pack_n))))
        self._full_pad_to = 1 << max(1, math.ceil(math.log2(N)))

        self.encoder = PeakSetEncoder(
            model_dim=model_dim, num_layers=encoder_num_layers,
            num_heads=encoder_num_heads, num_kv_heads=encoder_num_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            feature_mlp_hidden_dim=feature_mlp_hidden_dim,
            use_rope=encoder_use_rope, qk_norm=encoder_qk_norm,
            norm_type=self.norm_type, norm_position=self.norm_position, seq_len=N,
        )
        self._encoder_forward = self.encoder.forward
        self.latent_mask_token = nn.Parameter(torch.empty(self.model_dim))
        nn.init.normal_(self.latent_mask_token, std=0.02)
        pred_heads = max(1, min(int(encoder_num_heads), self.model_dim // 16))
        while self.model_dim % pred_heads != 0:
            pred_heads -= 1
        _ph = self.model_dim // pred_heads // 2
        predictor_inv_freq = 1.0 / (10000.0 ** (torch.arange(_ph, dtype=torch.float32) / _ph))
        self.register_buffer("predictor_rope_inv_freq", predictor_inv_freq, persistent=False)
        pred_rope_cos, pred_rope_sin = _precompute_rope_freqs(N, predictor_inv_freq)
        self.register_buffer("predictor_rope_cos", pred_rope_cos, persistent=False)
        self.register_buffer("predictor_rope_sin", pred_rope_sin, persistent=False)
        self.masked_latent_predictor = _build_temporal_decoder_blocks(
            dim=self.model_dim, num_layers=int(masked_latent_predictor_num_layers),
            num_heads=pred_heads, num_kv_heads=None,
            attention_mlp_multiple=attention_mlp_multiple,
            qk_norm=encoder_qk_norm, norm_type=self.norm_type,
            norm_position=self.norm_position,
        )
        self.predictor_final_norm = _build_norm(self.model_dim, eps=None, norm_type=self.norm_type)
        self.sigreg = SIGReg(num_slices=int(sigreg_num_slices))

        # Temporal predictor (cross-attention decoder for next-spectrum prediction)
        self.temporal_predictor_num_layers = int(temporal_predictor_num_layers)
        if self.temporal_predictor_num_layers > 0:
            self.temporal_predictor = _build_temporal_decoder_blocks(
                dim=model_dim, num_layers=self.temporal_predictor_num_layers,
                num_heads=pred_heads, num_kv_heads=None,
                attention_mlp_multiple=attention_mlp_multiple,
                qk_norm=encoder_qk_norm, norm_type=self.norm_type,
                norm_position=self.norm_position,
            )
            self.temporal_final_norm = _build_norm(model_dim, eps=None, norm_type=self.norm_type)
            self.temporal_rt_proj = nn.Sequential(
                nn.Linear(1, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim),
            )
            # Init: second linear small so rt_emb starts near zero
            nn.init.trunc_normal_(self.temporal_rt_proj[2].weight, std=1.0 / math.sqrt(model_dim))
            nn.init.zeros_(self.temporal_rt_proj[2].bias)
            self.temporal_target_tokens = nn.Parameter(torch.empty(N, model_dim))
            nn.init.trunc_normal_(self.temporal_target_tokens, std=0.02)
            tp_rope_cos, tp_rope_sin = _precompute_rope_freqs(N, predictor_inv_freq)
            self.register_buffer("temporal_predictor_rope_cos", tp_rope_cos, persistent=False)
            self.register_buffer("temporal_predictor_rope_sin", tp_rope_sin, persistent=False)

    def _apply(self, fn):
        result = super()._apply(fn)
        self._encoder_forward = self.encoder.forward
        return result

    def predict_masked_latents(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        pack_n: int = 0,
    ) -> torch.Tensor:
        BK, N, D = x.shape
        if len(self.masked_latent_predictor) == 0:
            return x * target_mask.unsqueeze(-1).to(dtype=x.dtype)
        if pack_n > 0:
            PACK_N = min(pack_n, N)
            sort_idx = target_mask.to(dtype=torch.int8).argsort(dim=1, descending=True, stable=True)
            pack_idx = sort_idx[:, :PACK_N]
            packed_x = x.gather(1, pack_idx.unsqueeze(-1).expand(-1, -1, D))
            packed_target = target_mask.gather(1, pack_idx)
            context_pack_n = min(self._context_pack_n, context.shape[1])
            context_sort_idx = context_mask.to(dtype=torch.int8).argsort(dim=1, descending=True, stable=True)
            context_idx = context_sort_idx[:, :context_pack_n]
            packed_context = context.gather(1, context_idx.unsqueeze(-1).expand(-1, -1, D))
            packed_context_mask = context_mask.gather(1, context_idx)
            rc = self.predictor_rope_cos.to(dtype=x.dtype)[0, :N, 0, :]  # [N, head_dim]
            rs = self.predictor_rope_sin.to(dtype=x.dtype)[0, :N, 0, :]
            freqs_cos = rc[pack_idx].unsqueeze(2)
            freqs_sin = rs[pack_idx].unsqueeze(2)
            pad_to = self._predictor_pad_to
            for block in self.masked_latent_predictor:
                packed_x = block(
                    packed_x,
                    packed_context,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    vis_mask=packed_target,
                    pad_to=pad_to,
                    context_mask=packed_context_mask,
                )
            packed_x = self.predictor_final_norm(packed_x)
            packed_x = packed_x * packed_target.unsqueeze(-1).to(dtype=packed_x.dtype)
            idx_expand = pack_idx.unsqueeze(-1).expand(-1, -1, D)
            return torch.zeros(BK, N, D, device=packed_x.device, dtype=packed_x.dtype).scatter(
                1, idx_expand, packed_x)
        # Full sequence path
        freqs_cos, freqs_sin = _compute_rope_freqs(
            True, N, self.predictor_rope_inv_freq, x.device, x.dtype)
        for block in self.masked_latent_predictor:
            x = block(
                x,
                context,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                vis_mask=target_mask,
                context_mask=context_mask,
            )
        x = self.predictor_final_norm(x)
        return x * target_mask.unsqueeze(-1).to(dtype=x.dtype)

    def pool(self, embeddings: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).float()
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def prepend_precursor_token(
        peak_mz: torch.Tensor, peak_intensity: torch.Tensor,
        peak_valid_mask: torch.Tensor, precursor_mz: torch.Tensor,
        context_mask: torch.Tensor | None = None, target_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, N = peak_mz.shape
        device = peak_mz.device
        pre_int = torch.full((B, 1), -1.0, device=device, dtype=peak_mz.dtype)
        pre_valid = torch.ones(B, 1, device=device, dtype=torch.bool)
        result: dict[str, torch.Tensor] = {
            "peak_mz": torch.cat([precursor_mz.unsqueeze(1), peak_mz], dim=1),
            "peak_intensity": torch.cat([pre_int, peak_intensity], dim=1),
            "peak_valid_mask": torch.cat([pre_valid, peak_valid_mask], dim=1),
        }
        if context_mask is not None:
            pre_ctx = torch.ones(B, 1, device=device, dtype=torch.bool)
            result["context_mask"] = torch.cat([pre_ctx, context_mask], dim=1)
        if target_masks is not None:
            K = target_masks.shape[1]
            pre_tgt = torch.zeros(B, K, 1, device=device, dtype=torch.bool)
            result["target_masks"] = torch.cat([pre_tgt, target_masks], dim=2)
        return result

    def forward_augmented(
        self, augmented_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        peak_mz = augmented_batch["peak_mz"]
        peak_intensity = augmented_batch["peak_intensity"]
        peak_valid_mask = augmented_batch["peak_valid_mask"]
        context_mask = augmented_batch["context_mask"] & peak_valid_mask
        target_masks = augmented_batch["target_masks"] & peak_valid_mask.unsqueeze(1)
        B, N = peak_mz.shape
        K = self.jepa_num_target_blocks

        # Context encoder: context view with sparse packing
        context_emb = self._encoder_forward(
            peak_mz, peak_intensity,
            valid_mask=peak_valid_mask, visible_mask=context_mask,
            pack_n=self._context_pack_n, prefix_pack=False, pad_to=self._context_pad_to,
        )

        # Full encoder: all valid peaks (shared encoder, gradients flow)
        full_emb = self._encoder_forward(
            peak_mz, peak_intensity,
            valid_mask=peak_valid_mask, visible_mask=peak_valid_mask,
            pack_n=self._full_pack_n, prefix_pack=True, pad_to=self._full_pad_to,
        )
        target_token_target = full_emb.unsqueeze(1).expand(-1, K, -1, -1)

        # Predictor
        ctx_mask_v = context_mask.unsqueeze(1)
        context_emb_by_view = context_emb.unsqueeze(1).expand(-1, K, -1, -1)
        predictor_queries = torch.zeros_like(context_emb_by_view)
        predictor_queries = torch.where(
            target_masks.unsqueeze(-1),
            self.latent_mask_token.view(1, 1, 1, -1).to(context_emb),
            predictor_queries,
        )
        predictor_output = self.predict_masked_latents(
            predictor_queries.reshape(B * K, N, -1),
            context_emb_by_view.reshape(B * K, N, -1),
            target_mask=target_masks.reshape(B * K, N),
            context_mask=ctx_mask_v.expand(-1, K, -1).reshape(B * K, N),
            pack_n=self._predictor_pack_n,
        ).reshape(B, K, N, -1)

        # MSE loss between predictor output and encoder targets
        target_mask_flat = target_masks.reshape(B * K, N)
        loss_pred = predictor_output.reshape(B * K, N, -1)
        loss_target = target_token_target.reshape(B * K, N, -1)
        per_token_mse = (loss_pred - loss_target).square().mean(dim=-1)
        target_mask_float = target_mask_flat.float()
        local_global_loss = (per_token_mse * target_mask_float).sum() / target_mask_float.sum().clamp_min(1.0)
        jepa_term = self.masked_token_loss_weight * local_global_loss
        with torch.no_grad():
            activation_norm_metrics = {
                "context_encoder_output_norm": _masked_mean_token_norm(
                    context_emb, context_mask
                ).to(dtype=context_emb.dtype),
                "full_encoder_output_norm": _masked_mean_token_norm(
                    full_emb, peak_valid_mask
                ).to(dtype=context_emb.dtype),
                "predictor_output_norm": _masked_mean_token_norm(
                    predictor_output.reshape(B * K, N, -1), target_mask_flat
                ).to(dtype=context_emb.dtype),
            }

        # Regularizer: plain sigreg_lambda * sigreg_loss
        use_sigreg = self.representation_regularizer == "sigreg" and self.sigreg_lambda > 0

        if not use_sigreg:
            return {
                "loss": jepa_term,
                "local_global_loss": local_global_loss,
                "jepa_term": jepa_term,
                **activation_norm_metrics,
            }

        # SIGReg on encoder outputs: context + full (2 views)
        branch_emb = torch.stack([context_emb, full_emb], dim=1)  # [B, 2, N, D]
        branch_visible = torch.stack([context_mask, peak_valid_mask], dim=1)  # [B, 2, N]
        V = 2
        fused_emb = branch_emb.reshape(V * B, N, -1)
        fused_visible = branch_visible.reshape(V * B, N)
        token_sigreg_loss = self.sigreg(fused_emb, valid_mask=fused_visible)
        sigreg_term = self.sigreg_lambda * token_sigreg_loss
        with torch.no_grad():
            emb_f = fused_emb.float().reshape(B, V, N, -1)
            mask_f = fused_visible.reshape(B, V, N)
            collapse_metrics: dict[str, torch.Tensor] = {}
            for prefix, e, m in [("context", emb_f[:, 0], mask_f[:, 0]),
                                  ("full", emb_f[:, 1], mask_f[:, 1])]:
                for k, v in _masked_embedding_stats(e, m).items():
                    collapse_metrics[f"{prefix}_{k}"] = v
            reg_stats = _masked_embedding_stats(fused_emb, fused_visible)
        valid_peak_count = peak_valid_mask.float().sum().clamp_min(1.0)
        return {
            "loss": jepa_term + sigreg_term,
            "token_sigreg_loss": token_sigreg_loss,
            "local_global_loss": local_global_loss,
            "sigreg_term": sigreg_term, "jepa_term": jepa_term,
            "target_sigreg_term_over_jepa_term": sigreg_term / jepa_term.clamp_min(1e-8),
            "context_fraction": context_mask.float().sum() / valid_peak_count,
            "masked_fraction": target_masks.float().sum() / valid_peak_count,
            **activation_norm_metrics,
            **{f"encoder_{k}": v.to(context_emb.dtype) for k, v in reg_stats.items()},
            **collapse_metrics,
        }

    def forward_temporal(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Temporal prediction: predict target spectrum tokens from context spectrum."""
        context_mz = batch["context_peak_mz"]
        context_int = batch["context_peak_intensity"]
        context_valid = batch["context_peak_valid_mask"]
        target_mz = batch["target_peak_mz"]
        target_int = batch["target_peak_intensity"]
        target_valid = batch["target_peak_valid_mask"]
        context_rt = batch["context_rt"]
        target_rt = batch["target_rt"]
        B = context_mz.shape[0]

        # 1. Encode context spectrum (shared encoder, full sequence)
        context_emb = self._encoder_forward(
            context_mz, context_int,
            valid_mask=context_valid, visible_mask=context_valid,
            pack_n=self._full_pack_n, prefix_pack=True, pad_to=self._full_pad_to,
        )  # [B, N, D]

        context_emb = context_emb.float()

        # 2. RT conditioning (normalize to minutes for numerical stability)
        delta_rt = ((target_rt - context_rt) / 60.0).unsqueeze(-1)  # [B, 1]
        with torch.autocast(device_type=context_emb.device.type, enabled=False):
            rt_emb = self.temporal_rt_proj(delta_rt.float())  # [B, D]

            # 3. Initialize target queries
            queries = self.temporal_target_tokens.float().unsqueeze(0).expand(B, -1, -1) + rt_emb.unsqueeze(1)

            # 4. Run through temporal decoder blocks
            N_tgt = queries.shape[1]
            freqs_cos = self.temporal_predictor_rope_cos.float()[:, :N_tgt]
            freqs_sin = self.temporal_predictor_rope_sin.float()[:, :N_tgt]
            for block in self.temporal_predictor:
                queries = block(queries, context_emb,
                                freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                                context_mask=context_valid)
            predicted_target = self.temporal_final_norm(queries)  # [B, N, D]

        # 5. Target encoding (shared encoder, gradients flow)
        target_emb = self._encoder_forward(
            target_mz, target_int,
            valid_mask=target_valid, visible_mask=target_valid,
            pack_n=self._full_pack_n, prefix_pack=True, pad_to=self._full_pad_to,
        ).float()
        target_emb = target_emb.nan_to_num(0.0)

        # 6. MSE loss over valid target positions
        predicted_target = predicted_target.nan_to_num(0.0)
        per_token = (predicted_target - target_emb).square().mean(dim=-1)

        sample_has_context = context_valid.any(dim=-1, keepdim=True)  # [B, 1]
        effective_mask = target_valid & sample_has_context
        mask_float = effective_mask.float()
        loss = (per_token * mask_float).sum() / mask_float.sum().clamp_min(1.0)
        with torch.no_grad():
            temporal_norm_metrics = {
                "context_encoder_output_norm": _masked_mean_token_norm(
                    context_emb, context_valid
                ).to(dtype=predicted_target.dtype),
                "target_encoder_output_norm": _masked_mean_token_norm(
                    target_emb, target_valid
                ).to(dtype=predicted_target.dtype),
                "predictor_output_norm": _masked_mean_token_norm(
                    predicted_target, effective_mask
                ).to(dtype=predicted_target.dtype),
            }

        return {
            "loss": loss,
            "temporal_pred_loss": loss.detach(),
            **temporal_norm_metrics,
        }

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mz, intensity, valid = (
            batch["peak_mz"], batch["peak_intensity"], batch["peak_valid_mask"])
        return self.pool(self.encoder(mz, intensity, valid_mask=valid), valid)


PeakSetJEPA = PeakSetSIGReg
