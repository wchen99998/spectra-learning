from __future__ import annotations

import math
from typing import NamedTuple

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spectra_learning.models.pairmixer import PairMixerBlock
from spectra_learning.models.transformer import _build_norm


class InducedPairState(NamedTuple):
    inducing: Float[Tensor, "*batch inducing single"]
    pair: Float[Tensor, "*batch inducing inducing pair"]
    assignment: Float[Tensor, "*batch tokens inducing"]


class TokenInducingAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wk.weight)

    def forward(
        self,
        token: Float[Tensor, "batch tokens dim"],
        inducing: Float[Tensor, "batch inducing dim"],
    ) -> Float[Tensor, "batch tokens inducing"]:
        q = self.wq(token)
        k = self.wk(inducing)
        scores = torch.einsum("bid,bad->bia", q, k) * (1.0 / math.sqrt(self.dim))
        return torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)


class CrossAttentionWithWeights(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wkv = nn.Linear(dim, 2 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.xavier_normal_(self.wkv.weight)
        nn.init.xavier_normal_(self.wo.weight)

    def forward(
        self,
        query: Float[Tensor, "batch query dim"],
        memory: Float[Tensor, "batch memory dim"],
        *,
        memory_mask: Bool[Tensor, "batch memory"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch query dim"],
        Float[Tensor, "batch query memory"],
    ]:
        batch_size, query_len, _ = query.shape
        memory_len = memory.shape[1]
        q = self.wq(query).view(
            batch_size,
            query_len,
            self.num_heads,
            self.head_dim,
        )
        k, v = self.wkv(memory).view(
            batch_size,
            memory_len,
            2,
            self.num_heads,
            self.head_dim,
        ).unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * (
            1.0 / math.sqrt(self.head_dim)
        )
        if memory_mask is not None:
            scores = scores.masked_fill(~memory_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, self.dim)
        return self.wo(out), attn.mean(dim=1)


class InducedPairBlock(nn.Module):
    def __init__(
        self,
        *,
        single_dim: int,
        pair_dim: int,
        num_heads: int,
        attention_mlp_multiple: float,
        norm_eps: float,
        dropout: float,
        use_pair_bias_attention: bool = False,
    ) -> None:
        super().__init__()
        self.inducing_aggregate_norm = _build_norm(single_dim, eps=norm_eps)
        self.single_aggregate_norm = _build_norm(single_dim, eps=norm_eps)
        self.inducing_aggregate = CrossAttentionWithWeights(single_dim, num_heads)
        self.latent_pair_mixer = PairMixerBlock(
            single_dim=single_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            norm_eps=norm_eps,
            dropout=dropout,
            use_pair_bias_attention=use_pair_bias_attention,
        )
        self.single_update_norm = _build_norm(single_dim, eps=norm_eps)
        self.inducing_update_norm = _build_norm(single_dim, eps=norm_eps)
        self.single_update = CrossAttentionWithWeights(single_dim, num_heads)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        single: Float[Tensor, "batch tokens dim"],
        state: InducedPairState,
        peak_mask: Bool[Tensor, "batch tokens"],
        token_mask: Bool[Tensor, "batch tokens"],
    ) -> tuple[
        Float[Tensor, "batch tokens dim"],
        InducedPairState,
    ]:
        del peak_mask
        inducing_update, _ = self.inducing_aggregate(
            self.inducing_aggregate_norm(state.inducing),
            self.single_aggregate_norm(single),
            memory_mask=token_mask,
        )
        inducing = state.inducing + self.drop(inducing_update)
        inducing_mask = torch.ones(
            inducing.shape[:2],
            dtype=torch.bool,
            device=inducing.device,
        )
        inducing, pair = self.latent_pair_mixer(
            inducing,
            state.pair,
            inducing_mask,
            inducing_mask,
        )
        single_update, assignment = self.single_update(
            self.single_update_norm(single),
            self.inducing_update_norm(inducing),
        )
        single = single + self.drop(single_update)
        assignment = assignment * token_mask.unsqueeze(-1).to(dtype=assignment.dtype)
        return single, InducedPairState(inducing, pair, assignment)


def mask_induced_pair_assignment(
    state: InducedPairState,
    token_mask: Bool[Tensor, "*batch tokens"],
) -> InducedPairState:
    return InducedPairState(
        state.inducing,
        state.pair,
        state.assignment * token_mask.unsqueeze(-1).to(dtype=state.assignment.dtype),
    )


def induced_pair_batch_slice(
    state: InducedPairState,
    index,
) -> InducedPairState:
    return InducedPairState(
        state.inducing[index],
        state.pair[index],
        state.assignment[index],
    )


def induced_pair_slice_tokens(
    state: InducedPairState,
    token_count: int,
) -> InducedPairState:
    return InducedPairState(
        state.inducing,
        state.pair,
        state.assignment[..., :token_count, :],
    )


def induced_pair_expand_views(
    state: InducedPairState,
    num_views: int,
) -> InducedPairState:
    return InducedPairState(
        state.inducing[:, None].expand(-1, num_views, -1, -1),
        state.pair[:, None].expand(-1, num_views, -1, -1, -1),
        state.assignment[:, None].expand(-1, num_views, -1, -1),
    )


def induced_pair_flatten_views(state: InducedPairState) -> InducedPairState:
    batch_size, num_views = state.inducing.shape[:2]
    return InducedPairState(
        state.inducing.reshape(batch_size * num_views, *state.inducing.shape[2:]),
        state.pair.reshape(batch_size * num_views, *state.pair.shape[2:]),
        state.assignment.reshape(batch_size * num_views, *state.assignment.shape[2:]),
    )


def induced_pair_unflatten_views(
    state: InducedPairState,
    batch_size: int,
    num_views: int,
) -> InducedPairState:
    return InducedPairState(
        state.inducing.reshape(batch_size, num_views, *state.inducing.shape[1:]),
        state.pair.reshape(batch_size, num_views, *state.pair.shape[1:]),
        state.assignment.reshape(batch_size, num_views, *state.assignment.shape[1:]),
    )


def induced_pair_to_dense_pair(state: InducedPairState) -> Tensor:
    left = torch.einsum("...ia,...abd->...ibd", state.assignment, state.pair)
    return torch.einsum("...ibd,...jb->...ijd", left, state.assignment)


def induced_pair_distogram_logits(
    state: InducedPairState,
    head: nn.Linear,
) -> Tensor:
    sym_pair = state.pair + state.pair.transpose(-3, -2)
    latent_logits = head(sym_pair)
    left = torch.einsum("...ia,...abk->...ibk", state.assignment, latent_logits)
    return torch.einsum("...ibk,...jb->...ijk", left, state.assignment)
