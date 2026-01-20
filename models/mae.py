"""Masked autoencoder for GeMS peak-list spectra with per-peak masking."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from clu import metrics as clu_metrics
from flax import nnx

from networks import transformer

import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
from matplotlib.figure import Figure  # noqa: E402  pylint: disable=wrong-import-position


class FourierFrequencies(nnx.Variable):
    pass


def mask_active_peaks(
    key,
    intensity,
    mask_ratio: float,
    *,
    top_k: int | None = None,
    min_visible: int = 0,
    always_keep_top_n: int = 0,
):
    """
    Mask a fraction of active peaks per example.

    Masked positions are True.

    Fixes:
      - Uses floor instead of round (less likely to mask everything for small counts)
      - Enforces at least `min_visible` active peaks remain unmasked when possible
      - Supports always_keep_top_n "anchor" peaks that will never be masked
    """
    intensity = jnp.asarray(intensity)
    active = intensity > 0
    bsz, n = intensity.shape

    def _topn_mask(x: jnp.ndarray, n_keep: int) -> jnp.ndarray:
        # True for the top-n by intensity (only among active)
        order = jnp.argsort(x, axis=1)              # ascending
        rank = jnp.argsort(order, axis=1)          # 0..n-1
        desc_rank = (x.shape[1] - 1) - rank        # 0 is largest
        return active & (desc_rank < n_keep)

    anchors = jnp.zeros_like(active)
    if always_keep_top_n > 0:
        anchors = _topn_mask(intensity, int(always_keep_top_n))

    # candidates exclude anchors (anchors must remain visible)
    base_candidates = active & ~anchors

    def sample_mask(scores: jnp.ndarray, candidates: jnp.ndarray, num_mask: jnp.ndarray) -> jnp.ndarray:
        # Select num_mask highest-score candidates per row
        scores = jnp.where(candidates, scores, -1.0)
        order = jnp.argsort(scores, axis=1)           # ascending
        rank = jnp.argsort(order, axis=1)             # rank of each position
        desc_rank = (scores.shape[1] - 1) - rank
        take = desc_rank < num_mask[:, None]
        return take & candidates

    def clamp_num_mask(num_mask: jnp.ndarray, candidates: jnp.ndarray) -> jnp.ndarray:
        cand_counts = jnp.sum(candidates, axis=1).astype(jnp.int32)
        active_counts = jnp.sum(active, axis=1).astype(jnp.int32)

        # ensure at least min_visible active peaks remain
        max_mask_allowed = jnp.maximum(0, active_counts - int(min_visible))

        num_mask = jnp.minimum(num_mask, cand_counts)
        num_mask = jnp.minimum(num_mask, max_mask_allowed)
        num_mask = jnp.maximum(num_mask, 0)
        return num_mask

    if top_k is None:
        cand_counts = jnp.sum(base_candidates, axis=1).astype(jnp.int32)
        num_mask = jnp.floor(cand_counts.astype(jnp.float32) * mask_ratio).astype(jnp.int32)
        num_mask = clamp_num_mask(num_mask, base_candidates)

        scores = jax.random.uniform(key, intensity.shape)
        mask = sample_mask(scores, base_candidates, num_mask)
        mask = mask & ~anchors
        return mask, active

    # top_k masking split (then enforce min_visible globally)
    k = int(top_k)

    order = jnp.argsort(intensity, axis=1)
    rank = jnp.argsort(order, axis=1)
    desc_rank = (n - 1) - rank

    topk_candidates = base_candidates & (desc_rank < k)
    rest_candidates = base_candidates & ~topk_candidates

    topk_counts = jnp.sum(topk_candidates, axis=1).astype(jnp.int32)
    rest_counts = jnp.sum(rest_candidates, axis=1).astype(jnp.int32)

    topk_num = jnp.floor(topk_counts.astype(jnp.float32) * mask_ratio).astype(jnp.int32)
    rest_num = jnp.floor(rest_counts.astype(jnp.float32) * mask_ratio).astype(jnp.int32)

    topk_num = jnp.minimum(topk_num, topk_counts)
    rest_num = jnp.minimum(rest_num, rest_counts)
    topk_num = jnp.maximum(topk_num, 0)
    rest_num = jnp.maximum(rest_num, 0)

    key_topk, key_rest, key_fix = jax.random.split(key, 3)
    mask_topk = sample_mask(jax.random.uniform(key_topk, intensity.shape), topk_candidates, topk_num)
    mask_rest = sample_mask(jax.random.uniform(key_rest, intensity.shape), rest_candidates, rest_num)

    mask = (mask_topk | mask_rest) & ~anchors

    # Enforce global min_visible: unmask some currently masked peaks if needed
    if min_visible > 0:
        visible = jnp.sum(active & ~mask, axis=1).astype(jnp.int32)
        need = jnp.maximum(0, int(min_visible) - visible)  # (B,)

        candidates_to_unmask = mask & ~anchors
        to_unmask = sample_mask(jax.random.uniform(key_fix, intensity.shape), candidates_to_unmask, need)
        mask = mask & ~to_unmask

    return mask, active


def mask_mz_window(
    key,
    mz: jnp.ndarray,
    valid_mask: jnp.ndarray,
    *,
    mz_min: float,
    mz_max: float,
    width_range: tuple[float, float] = (50.0, 200.0),
    per_example_prob: float = 0.5,
) -> jnp.ndarray:
    """
    Structured mask: masks peaks inside a random m/z window per example.
    Returns boolean mask (True = masked).
    """
    bsz, _ = mz.shape
    key_c, key_w, key_p = jax.random.split(key, 3)

    center = jax.random.uniform(key_c, (bsz, 1), minval=mz_min, maxval=mz_max)
    width = jax.random.uniform(key_w, (bsz, 1), minval=width_range[0], maxval=width_range[1])
    half = 0.5 * width

    in_window = (mz >= (center - half)) & (mz <= (center + half))
    mask = valid_mask & in_window

    if per_example_prob < 1.0:
        do = jax.random.bernoulli(key_p, per_example_prob, (bsz, 1))
        mask = mask & do

    return mask


def intensity_global_log_jitter(
    key,
    intensity: jnp.ndarray,
    valid_mask: jnp.ndarray,
    *,
    log_sigma: float = 0.0,
) -> jnp.ndarray:
    """
    Multiply intensities by exp(N(0, log_sigma)) per example.
    Useful to enforce scale invariance even if upstream normalization isn't perfect.
    """
    if log_sigma <= 0.0:
        return intensity
    bsz = intensity.shape[0]
    scale = jnp.exp(log_sigma * jax.random.normal(key, (bsz, 1)).astype(intensity.dtype))
    out = intensity * scale
    return jnp.where(valid_mask, out, 0.0)


def _resolve_attention_heads(
    dim: int,
    requested_heads: Optional[int],
    requested_kv_heads: Optional[int],
) -> tuple[int, int]:
    heads = 1 if requested_heads is None else int(requested_heads)
    kv_heads = heads if requested_kv_heads is None else int(requested_kv_heads)
    return heads, kv_heads


def _normalize_num_transformer_blocks(
    num_transformer_blocks: int | Sequence[int],
) -> tuple[int, int]:
    if isinstance(num_transformer_blocks, (tuple, list)):
        encoder_blocks = int(num_transformer_blocks[0])
        decoder_blocks = int(num_transformer_blocks[1])
    else:
        encoder_blocks = decoder_blocks = int(num_transformer_blocks)
    return encoder_blocks, decoder_blocks


def _normalize_heads_config(
    value: int | Sequence[int] | None,
) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _build_transformer_blocks(
    rngs: nnx.Rngs,
    *,
    dim: int,
    num_layers: int,
    attention_heads: Optional[int],
    attention_kv_heads: Optional[int],
    attention_mlp_multiple: float,
    dtype: jnp.dtype,
    param_dtype: jnp.dtype,
) -> nnx.List[transformer.TransformerBlock]:
    heads, kv_heads = _resolve_attention_heads(dim, attention_heads, attention_kv_heads)
    hidden_dim = int(math.ceil(dim * attention_mlp_multiple))
    blocks = []
    for _ in range(num_layers):
        blocks.append(
            transformer.TransformerBlock(
                rngs=rngs,
                dim=dim,
                n_heads=heads,
                n_kv_heads=kv_heads,
                causal=False,
                dtype=dtype,
                param_dtype=param_dtype,
                norm_type="layernorm",
                norm_eps=1e-5,
                mlp_type="swish",
                multiple_of=4,
                hidden_dim=hidden_dim,
                w_init_scale=1.0,
                use_cross_attention=False,
                cross_attention_dim=None,
                use_rotary_embeddings=False,
            )
        )
    return nnx.List(blocks)


class TransformerStack(nnx.Module):
    """Stack of transformer blocks without rotary positional encoding."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        *,
        dim: int,
        num_layers: int,
        attention_heads: Optional[int],
        attention_kv_heads: Optional[int],
        attention_mlp_multiple: float,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
    ):
        self.blocks = _build_transformer_blocks(
            rngs=rngs,
            dim=dim,
            num_layers=num_layers,
            attention_heads=attention_heads,
            attention_kv_heads=attention_kv_heads,
            attention_mlp_multiple=attention_mlp_multiple,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.dtype = dtype
        heads, kv_heads = _resolve_attention_heads(dim, attention_heads, attention_kv_heads)

        # NEW: validate heads config
        if dim % heads != 0:
            raise ValueError(f"Transformer dim={dim} must be divisible by n_heads={heads}")
        if kv_heads > heads or (heads % kv_heads) != 0:
            raise ValueError(f"n_kv_heads={kv_heads} must be <= n_heads={heads} and divide it")


        self.head_dim = dim // heads
        self.norm = nnx.LayerNorm(
            num_features=dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        train: bool,
        attention_bias: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        for block in self.blocks:
            x = block(
                x,
                None,
                None,
                train=train,
                attention_bias=attention_bias,
            )
        return self.norm(x)


class MAE(nnx.Module):
    """Masked autoencoder with metadata prediction."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        *,
        spectrum_length: int,
        # masking / views
        mask_ratio: float = 0.4,
        mask_top_k_peaks: int | None = None,
        min_visible_peaks: int = 12,
        always_keep_top_n: int = 3,
        mz_window_prob: float = 0.5,
        mz_window_width: tuple[float, float] = (50.0, 200.0),
        intensity_log_jitter: float = 0.0,
        precursor_mask_prob: float = 0.5,
        rt_mask_prob: float = 0.5,

        # backbone dims
        encoder_dim: int = 256,
        decoder_dim: int | None = None,
        num_transformer_blocks: Sequence[int] | int = (4, 2),
        num_heads: int | Sequence[int] | None = None,
        num_kv_heads: int | Sequence[int] | None = None,
        attention_mlp_multiple: float = 4.0,

        # fourier / mz normalization
        mz_min: float = 0.0,
        mz_max: float = 1500.0,
        num_fourier_features: int = 32,
        fourier_max_freq: float = 16.0,
        peak_mlp_hidden_dim: int | None = None,

        # losses
        mz_loss_weight: float = 0.0,
        intensity_loss_weight: float = 1.0,
        recon_loss_weight: float = 0.1,
        aux_loss_weight: float = 1.0,

        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        self.num_peaks = int(spectrum_length)

        self.mask_ratio = float(mask_ratio)
        self.mask_top_k_peaks = None if mask_top_k_peaks is None else int(mask_top_k_peaks)
        self.min_visible_peaks = int(min_visible_peaks)
        self.always_keep_top_n = int(always_keep_top_n)
        self.mz_window_prob = float(mz_window_prob)
        self.mz_window_width = (float(mz_window_width[0]), float(mz_window_width[1]))
        self.intensity_log_jitter = float(intensity_log_jitter)
        self.precursor_mask_prob = float(precursor_mask_prob)
        self.rt_mask_prob = float(rt_mask_prob)

        self.encoder_dim = int(encoder_dim)
        self.decoder_dim = int(decoder_dim) if decoder_dim is not None else int(encoder_dim)
        self.num_transformer_blocks = _normalize_num_transformer_blocks(num_transformer_blocks)
        self.attention_mlp_multiple = float(attention_mlp_multiple)

        self.dtype = dtype
        self.param_dtype = param_dtype

        self.mz_min = float(mz_min)
        self.mz_max = float(mz_max)
        self.mz_range = float(mz_max - mz_min)

        self.num_fourier_features = int(num_fourier_features)
        self.fourier_max_freq = float(fourier_max_freq)

        self.mz_loss_weight = float(mz_loss_weight)
        self.intensity_loss_weight = float(intensity_loss_weight)
        self.recon_loss_weight = float(recon_loss_weight)
        self.aux_loss_weight = float(aux_loss_weight)

        encoder_heads, decoder_heads = _normalize_heads_config(num_heads)
        encoder_kv_heads, decoder_kv_heads = _normalize_heads_config(num_kv_heads)

        peak_feature_dim = 2 * self.num_fourier_features + 1
        precursor_feature_dim = 2 * self.num_fourier_features
        peak_hidden_dim = int(peak_mlp_hidden_dim) if peak_mlp_hidden_dim is not None else self.encoder_dim

        peak_proj_init = nnx.with_partitioning(
            nnx.initializers.lecun_normal(),
            ("input_embed", "hidden"),
        )

        self.peak_embed = nnx.Sequential(
            nnx.Linear(peak_feature_dim, peak_hidden_dim, dtype=dtype, param_dtype=param_dtype, kernel_init=peak_proj_init, rngs=rngs),
            nnx.swish,
            nnx.Linear(peak_hidden_dim, self.encoder_dim, dtype=dtype, param_dtype=param_dtype, kernel_init=peak_proj_init, rngs=rngs),
        )
        self.precursor_embed = nnx.Sequential(
            nnx.Linear(precursor_feature_dim, peak_hidden_dim, dtype=dtype, param_dtype=param_dtype, kernel_init=peak_proj_init, rngs=rngs),
            nnx.swish,
            nnx.Linear(peak_hidden_dim, self.encoder_dim, dtype=dtype, param_dtype=param_dtype, kernel_init=peak_proj_init, rngs=rngs),
        )
        self.retention_embed = nnx.Sequential(
            nnx.Linear(precursor_feature_dim, peak_hidden_dim, dtype=dtype, param_dtype=param_dtype, kernel_init=peak_proj_init, rngs=rngs),
            nnx.swish,
            nnx.Linear(peak_hidden_dim, self.encoder_dim, dtype=dtype, param_dtype=param_dtype, kernel_init=peak_proj_init, rngs=rngs),
        )

        self.encoder = TransformerStack(
            rngs=rngs,
            dim=self.encoder_dim,
            num_layers=self.num_transformer_blocks[0],
            attention_heads=encoder_heads,
            attention_kv_heads=encoder_kv_heads,
            attention_mlp_multiple=self.attention_mlp_multiple,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        encoder_to_decoder_init = nnx.with_partitioning(
            nnx.initializers.lecun_normal(),
            ("hidden", "hidden"),
        )
        self.encoder_to_decoder = nnx.Linear(
            in_features=self.encoder_dim,
            out_features=self.decoder_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=encoder_to_decoder_init,
            rngs=rngs,
        )

        # mask token for decoder
        self.mask_token = nnx.Param(jnp.zeros((self.decoder_dim,), dtype=self.dtype))
        self.cls_token = nnx.Param(jnp.zeros((self.encoder_dim,), dtype=self.dtype))
        self.metadata_mask_token = nnx.Param(jnp.zeros((self.encoder_dim,), dtype=self.dtype))

        self.decoder = TransformerStack(
            rngs=rngs,
            dim=self.decoder_dim,
            num_layers=self.num_transformer_blocks[1],
            attention_heads=decoder_heads,
            attention_kv_heads=decoder_kv_heads,
            attention_mlp_multiple=self.attention_mlp_multiple,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        decoder_proj_init = nnx.with_partitioning(
            nnx.initializers.lecun_normal(),
            ("hidden", "vocab"),
        )
        self.decoder_proj = nnx.Linear(
            in_features=self.decoder_dim,
            out_features=1,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=decoder_proj_init,
            rngs=rngs,
        )

        meta_pred_init = nnx.with_partitioning(nnx.initializers.lecun_normal(), ("hidden", "aux"))
        self.metadata_pred = nnx.Sequential(
            nnx.Linear(self.encoder_dim, self.encoder_dim, dtype=dtype, param_dtype=param_dtype, kernel_init=meta_pred_init, rngs=rngs),
            nnx.swish,
            nnx.Linear(self.encoder_dim, 2, dtype=dtype, param_dtype=param_dtype, kernel_init=meta_pred_init, rngs=rngs),
        )

        # always have a mask rng stream available
        self.mask_rngs = rngs["mask"].fork()

        freq_logs = jnp.linspace(
            0.0,
            jnp.log(self.fourier_max_freq),
            self.num_fourier_features,
            dtype=jnp.float32,
        )
        self.fourier_frequencies = FourierFrequencies(jnp.exp(freq_logs).astype(self.dtype))

    # ----------------------------
    # embeddings / helpers
    # ----------------------------
    def _prepare_peaks(self, batch: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        mz = jnp.asarray(batch["mz"], dtype=self.dtype)
        intensity = jnp.asarray(batch["intensity"], dtype=self.dtype)

        cur = mz.shape[1]
        if cur > self.num_peaks:
            mz = mz[:, : self.num_peaks]
            intensity = intensity[:, : self.num_peaks]
        elif cur < self.num_peaks:
            pad = self.num_peaks - cur
            mz = jnp.pad(mz, ((0, 0), (0, pad)))
            intensity = jnp.pad(intensity, ((0, 0), (0, pad)))

        return mz, intensity

    def _normalize_mz(self, mz: jnp.ndarray) -> jnp.ndarray:
        mz_norm = (mz - self.mz_min) / self.mz_range
        return jnp.clip(mz_norm, 0.0, 1.0)

    def _normalize_rt(self, rt: jnp.ndarray) -> jnp.ndarray:
        return rt

    def _fourier_encode(self, mz_norm: jnp.ndarray) -> jnp.ndarray:
        freqs = self.fourier_frequencies.value
        angles = 2.0 * jnp.pi * mz_norm[..., None] * freqs[None, None, :]
        return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

    def _embed_peaks(self, mz: jnp.ndarray, intensity: jnp.ndarray) -> jnp.ndarray:
        mz_norm = self._normalize_mz(mz)
        fourier = self._fourier_encode(mz_norm)
        features = jnp.concatenate([fourier, intensity[..., None]], axis=-1)
        return self.peak_embed(features)

    def _embed_precursor(self, precursor_mz: jnp.ndarray) -> jnp.ndarray:
        mz_norm = self._normalize_mz(precursor_mz)
        fourier = self._fourier_encode(mz_norm)
        return self.precursor_embed(fourier)

    def _embed_retention(self, retention_time: jnp.ndarray) -> jnp.ndarray:
        rt_norm = self._normalize_rt(retention_time)
        fourier = self._fourier_encode(rt_norm)
        return self.retention_embed(fourier)

    def _make_attention_bias(self, allow: jnp.ndarray, *, dtype: jnp.dtype) -> jnp.ndarray:
        # NEW: more stable than float32.min (works better in fp16/bf16 too)
        neg = jnp.asarray(-1e9, dtype=dtype)
        zero = jnp.asarray(0.0, dtype=dtype)
        bias = jnp.where(allow, zero, neg)
        return bias[:, None, None, :]

    def _masked_mse_per_example(self, pred: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        # NEW: per-example normalization to avoid bias toward dense spectra
        m = mask.astype(pred.dtype)
        se = jnp.square(pred - target) * m
        se_sum = jnp.sum(se, axis=1)
        denom = jnp.maximum(1.0, jnp.sum(m, axis=1))
        return jnp.mean(se_sum / denom)

    def _cosine_similarity(self, target: jnp.ndarray, recon: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        m = mask.astype(target.dtype)
        t = (target * m).reshape(target.shape[0], -1)
        r = (recon * m).reshape(recon.shape[0], -1)
        dot = jnp.sum(t * r, axis=-1)
        tn = jnp.linalg.norm(t, axis=-1)
        rn = jnp.linalg.norm(r, axis=-1)
        eps = jnp.asarray(1e-9, dtype=target.dtype)
        return jnp.mean(dot / jnp.maximum(tn * rn, eps))

    # ----------------------------
    # view sampling
    # ----------------------------
    def _sample_metadata_masks(self, key, batch_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        key_prec, key_rt = jax.random.split(key, 2)
        precursor_mask = jax.random.bernoulli(key_prec, self.precursor_mask_prob, (batch_size, 1))
        rt_mask = jax.random.bernoulli(key_rt, self.rt_mask_prob, (batch_size, 1))
        return precursor_mask, rt_mask

    def _sample_view(
        self, key, mz: jnp.ndarray, intensity: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns (intensity_view, mask, valid_mask, precursor_mask, rt_mask).
        """
        valid_mask = intensity > 0

        key_jit, key_rand, key_win, key_meta = jax.random.split(key, 4)

        intensity_view = intensity_global_log_jitter(
            key_jit,
            intensity,
            valid_mask,
            log_sigma=self.intensity_log_jitter,
        )

        # random peak dropout mask
        mask_rand, _ = mask_active_peaks(
            key_rand,
            intensity,
            self.mask_ratio,
            top_k=self.mask_top_k_peaks,
            min_visible=self.min_visible_peaks,
            always_keep_top_n=self.always_keep_top_n,
        )

        # optional structured window mask (then enforce anchors/min-visible via the mask_active_peaks call + below clamp)
        mask_win = mask_mz_window(
            key_win,
            mz,
            valid_mask,
            mz_min=self.mz_min,
            mz_max=self.mz_max,
            width_range=self.mz_window_width,
            per_example_prob=self.mz_window_prob,
        )

        mask = (mask_rand | mask_win) & valid_mask

        # final clamp: ensure anchors are always visible
        if self.always_keep_top_n > 0:
            # recompute anchors and unmask them
            order = jnp.argsort(intensity, axis=1)
            rank = jnp.argsort(order, axis=1)
            desc_rank = (intensity.shape[1] - 1) - rank
            anchors = valid_mask & (desc_rank < self.always_keep_top_n)
            mask = mask & ~anchors

        # enforce min_visible if window masking made it too aggressive
        if self.min_visible_peaks > 0:
            active = valid_mask
            visible = jnp.sum(active & ~mask, axis=1).astype(jnp.int32)
            need = jnp.maximum(0, self.min_visible_peaks - visible)

            # unmask random masked peaks if needed
            key_fix = jax.random.fold_in(key, 12345)
            scores = jax.random.uniform(key_fix, intensity.shape)
            candidates = mask  # only unmask from currently masked
            # reuse selection logic: select `need` masked peaks to unmask
            # (same logic as in mask_active_peaks)
            scores = jnp.where(candidates, scores, -1.0)
            order2 = jnp.argsort(scores, axis=1)
            rank2 = jnp.argsort(order2, axis=1)
            desc_rank2 = (scores.shape[1] - 1) - rank2
            to_unmask = (desc_rank2 < need[:, None]) & candidates
            mask = mask & ~to_unmask

        precursor_mask, rt_mask = self._sample_metadata_masks(key_meta, intensity.shape[0])
        return intensity_view, mask, valid_mask, precursor_mask, rt_mask

    # ----------------------------
    # single-view forward (stable positions)
    # ----------------------------
    def _forward_view(
        self,
        mz: jnp.ndarray,
        intensity: jnp.ndarray,
        precursor_mz: jnp.ndarray,
        retention_time: jnp.ndarray,
        *,
        mask: jnp.ndarray,
        valid_mask: jnp.ndarray,
        precursor_mask: jnp.ndarray,
        rt_mask: jnp.ndarray,
        train: bool,
        return_reconstruction: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Returns (embedding_h, recon_loss, aux_loss, metrics).
        """
        # Masks: valid_mask marks real peaks; mask marks which valid peaks have intensity hidden.
        mask = mask & valid_mask

        # Encoder input: keep m/z for all valid peaks, mask intensity with -1.0
        mz_enc = jnp.where(valid_mask, mz, 0.0)
        int_enc = jnp.where(valid_mask, intensity, 0.0)
        int_enc = jnp.where(mask, jnp.asarray(-1.0, dtype=int_enc.dtype), int_enc)

        # Tokenization: CLS + precursor + retention time + per-peak tokens.
        peak_tokens = self._embed_peaks(mz_enc, int_enc)                 # (B, N, enc_dim)
        precursor_token = self._embed_precursor(precursor_mz)            # (B, 1, enc_dim)
        retention_token = self._embed_retention(retention_time)          # (B, 1, enc_dim)
        meta_mask_token = self.metadata_mask_token[None, None, :]
        precursor_token = jnp.where(precursor_mask[..., None], meta_mask_token, precursor_token)
        retention_token = jnp.where(rt_mask[..., None], meta_mask_token, retention_token)

        batch_size = mz.shape[0]
        cls_token = jnp.broadcast_to(self.cls_token[None, None, :], (batch_size, 1, self.encoder_dim))
        tokens = jnp.concatenate([cls_token, precursor_token, retention_token, peak_tokens], axis=1)

        meta_allow = jnp.ones((batch_size, 3), dtype=jnp.bool_)
        encoder_allow = jnp.concatenate([meta_allow, valid_mask], axis=1)
        encoder_bias = self._make_attention_bias(encoder_allow, dtype=tokens.dtype)

        # Encoder outputs contextualized tokens; CLS token is the representation.
        encoded = self.encoder(tokens, train=train, attention_bias=encoder_bias)
        h = encoded[:, 0, :]

        meta_pred = self.metadata_pred(h)
        pred_precursor = meta_pred[:, :1]
        pred_rt = meta_pred[:, 1:2]
        target_precursor = self._normalize_mz(precursor_mz)
        target_rt = self._normalize_rt(retention_time)

        precursor_mse = self._masked_mse_per_example(pred_precursor, target_precursor, precursor_mask)
        rt_mse = self._masked_mse_per_example(pred_rt, target_rt, rt_mask)
        aux_loss = precursor_mse + rt_mse

        # Decoder path (optional recon objective)
        enc_to_dec = self.encoder_to_decoder(encoded)
        enc_meta = enc_to_dec[:, :3, :]
        enc_peaks = enc_to_dec[:, 3:, :]

        # Decoder tokens: keep encoder states for all valid peaks, add a mask token offset
        # for masked intensities so m/z context stays available.
        dec_peaks = jnp.where(valid_mask[..., None], enc_peaks, 0.0)
        dec_peaks = dec_peaks + (mask[..., None] * self.mask_token[None, None, :])
        dec_tokens = jnp.concatenate([enc_meta, dec_peaks], axis=1)

        decoder_allow = jnp.concatenate([meta_allow, valid_mask], axis=1)
        decoder_bias = self._make_attention_bias(decoder_allow, dtype=dec_tokens.dtype)

        decoded = self.decoder(dec_tokens, train=train, attention_bias=decoder_bias)
        decoded = self.decoder_proj(decoded)      # (B, 3+N, 1)
        decoded_peaks = decoded[:, 3:, :]

        pred_intensity = nnx.softplus(decoded_peaks[..., 0])

        target_intensity = intensity

        mse_int_masked = self._masked_mse_per_example(pred_intensity, target_intensity, mask)
        recon_loss = self.intensity_loss_weight * mse_int_masked

        # some useful metrics (per-example averaged)
        mask_count = jnp.sum(mask.astype(jnp.float32), axis=1)
        cand_count = jnp.sum(valid_mask.astype(jnp.float32), axis=1)
        mask_ratio_actual = jnp.mean(mask_count / jnp.maximum(1.0, cand_count))

        metrics = {
            "reconstruction_loss": recon_loss,
            "metadata_loss": aux_loss,
            "precursor_mse_masked": precursor_mse,
            "rt_mse_masked": rt_mse,
            "precursor_mask_ratio": jnp.mean(precursor_mask.astype(jnp.float32)),
            "rt_mask_ratio": jnp.mean(rt_mask.astype(jnp.float32)),
            "mask_ratio_actual_active": mask_ratio_actual,
            "cosine_similarity_intensity_masked": self._cosine_similarity(target_intensity, pred_intensity, mask),
        }

        if return_reconstruction:
            restored_mz = mz_enc
            restored_int = jnp.where(mask, pred_intensity, int_enc)
            metrics["reconstruction"] = jnp.stack([restored_mz, restored_int], axis=-1)
            metrics["decoder_output"] = pred_intensity[..., None]
            metrics["masked_input"] = jnp.stack([mz_enc, int_enc], axis=-1)

        return h, recon_loss, aux_loss, metrics

    # ----------------------------
    # public API
    # ----------------------------
    def encode(self, batch: dict[str, jnp.ndarray], *, train: bool = False) -> jnp.ndarray:
        """
        Deterministic spectrum embedding for downstream tasks (no masking).
        """
        mz, intensity = self._prepare_peaks(batch)
        valid_mask = intensity > 0
        precursor_mz = jnp.asarray(batch["precursor_mz"], dtype=self.dtype)
        retention_time = jnp.asarray(batch["rt"], dtype=self.dtype).reshape(-1, 1)

        # no masking
        mask = jnp.zeros_like(valid_mask, dtype=jnp.bool_)
        batch_size = mz.shape[0]
        precursor_mask = jnp.zeros((batch_size, 1), dtype=jnp.bool_)
        rt_mask = jnp.zeros((batch_size, 1), dtype=jnp.bool_)

        h, _, _, _ = self._forward_view(
            mz,
            intensity,
            precursor_mz,
            retention_time,
            mask=mask,
            valid_mask=valid_mask,
            precursor_mask=precursor_mask,
            rt_mask=rt_mask,
            train=train,
            return_reconstruction=False,
        )
        return h

    def __call__(
        self,
        batch: dict[str, jnp.ndarray],
        *,
        train: bool = True,
        apply_mask: bool | None = None,
        return_reconstruction: bool = False,
    ) -> dict[str, jnp.ndarray]:
        """
        Single-view forward (masking off by default when train=False).
        """
        if apply_mask is None:
            apply_mask = train

        mz, intensity = self._prepare_peaks(batch)
        precursor_mz = jnp.asarray(batch["precursor_mz"], dtype=self.dtype)
        retention_time = jnp.asarray(batch["rt"], dtype=self.dtype).reshape(-1, 1)
        valid_mask = intensity > 0

        if apply_mask:
            key = self.mask_rngs()
            intensity_view, mask, valid_mask, precursor_mask, rt_mask = self._sample_view(
                key, mz, intensity
            )
        else:
            intensity_view = intensity
            mask = jnp.zeros_like(valid_mask, dtype=jnp.bool_)
            batch_size = mz.shape[0]
            precursor_mask = jnp.zeros((batch_size, 1), dtype=jnp.bool_)
            rt_mask = jnp.zeros((batch_size, 1), dtype=jnp.bool_)

        _, recon_loss, aux_loss, m = self._forward_view(
            mz,
            intensity_view,
            precursor_mz,
            retention_time,
            mask=mask,
            valid_mask=valid_mask,
            precursor_mask=precursor_mask,
            rt_mask=rt_mask,
            train=train,
            return_reconstruction=return_reconstruction,
        )

        total = (
            self.recon_loss_weight * recon_loss
            + self.aux_loss_weight * aux_loss
        )
        metrics = {
            "loss": total,
            **m,
        }
        return metrics

    def compute_loss(self, batch: dict[str, jax.Array], *, train: bool = False):
        metrics = self(batch, train=train, apply_mask=train, return_reconstruction=False)
        loss = metrics["loss"]

        # keep only scalars for logging
        for k in ["reconstruction", "decoder_output"]:
            metrics.pop(k, None)

        return loss, metrics



def create_train_metrics_class_from_keys(metric_keys):
    """Create train metrics collection class from dictionary."""
    average_keys = [
        "loss",
        "reconstruction_loss",
        "metadata_loss",
        "precursor_mse_masked",
        "rt_mse_masked",
        "precursor_mask_ratio",
        "rt_mask_ratio",
        "mask_ratio_actual",
        "mask_ratio_actual_active",
        "mz_mse_active",
        "intensity_mse_active",
        "cosine_similarity_intensity",
        "cosine_similarity_intensity_masked",
    ]
    stats = dict(
        (k, clu_metrics.Average.from_output(k))
        if k in average_keys
        else (k, clu_metrics.LastValue.from_output(k))
        for k in metric_keys
    )
    return clu_metrics.Collection.create(**stats)


def create_train_metrics_class():
    metric_keys = sorted(
        [
            "loss",
            "reconstruction_loss",
            "metadata_loss",
            "precursor_mse_masked",
            "rt_mse_masked",
            "precursor_mask_ratio",
            "rt_mask_ratio",
            "mask_ratio_actual_active",
            "cosine_similarity_intensity_masked",
        ]
    )
    return create_train_metrics_class_from_keys(metric_keys)



TrainMetrics = create_train_metrics_class()


def get_train_metrics_class():
    return TrainMetrics


def create_train_metrics():
    """Create CLU-based train metrics collection instance."""
    return TrainMetrics.empty()


def create_eval_metrics_class():
    metric_keys = sorted(
        [
            "loss",
            "reconstruction_loss",
            "metadata_loss",
            "precursor_mse_masked",
            "rt_mse_masked",
            "precursor_mask_ratio",
            "rt_mask_ratio",
            "mask_ratio_actual_active",
            "cosine_similarity_intensity_masked",
        ]
    )
    return create_train_metrics_class_from_keys(metric_keys)


EvalMetrics = create_eval_metrics_class()


def create_eval_metrics():
    """Create CLU-based eval metrics collection instance."""
    return EvalMetrics.empty()


def create_reconstruction_comparison_figures(
    reconstruction: jnp.ndarray | np.ndarray,
    batch: dict[str, jnp.ndarray | np.ndarray],
    *,
    masked_input: jnp.ndarray | np.ndarray | None = None,
    max_examples: int = 4,
    figsize: tuple[float, float] = (8.0, 4.5),
) -> list[Figure]:
    """Return matplotlib figures comparing original, masked-input, and reconstruction peaks."""
    recon_np = np.asarray(reconstruction)
    mz = np.asarray(batch["mz"])
    intensity = np.asarray(batch["intensity"])
    masked_np = None if masked_input is None else np.asarray(masked_input)

    num_examples = min(int(recon_np.shape[0]), int(max_examples))
    figures: list[Figure] = []

    for idx in range(num_examples):
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle(f"Peak reconstruction #{idx}")

        active_mask = intensity[idx] > 0
        masked_positions = None
        if masked_np is not None:
            masked_int = masked_np[idx, :, 1]
            masked_positions = (masked_int < 0) & active_mask

        axes[0].vlines(
            mz[idx][active_mask],
            0.0,
            intensity[idx][active_mask],
            color="tab:blue",
            alpha=0.7,
            linewidth=1.5,
            label="original",
        )
        if masked_positions is not None and masked_positions.any():
            axes[0].vlines(
                mz[idx][masked_positions],
                0.0,
                intensity[idx][masked_positions],
                color="tab:red",
                alpha=0.9,
                linewidth=2.0,
                label="masked peaks",
            )
        axes[0].set_ylabel("intensity")
        axes[0].set_title("Original peaks")
        axes[0].legend(loc="upper right")

        if masked_np is not None:
            masked_mz = masked_np[idx, :, 0]
            masked_int = masked_np[idx, :, 1]
            masked_active = masked_int > 0
            axes[1].vlines(
                masked_mz[masked_active],
                0.0,
                masked_int[masked_active],
                color="tab:purple",
                alpha=0.7,
                linewidth=1.5,
                label="masked input",
            )
            if masked_positions is not None and masked_positions.any():
                axes[1].vlines(
                    mz[idx][masked_positions],
                    0.0,
                    intensity[idx][masked_positions],
                    color="tab:red",
                    alpha=0.6,
                    linewidth=2.0,
                    label="masked peaks (hidden)",
                )
        axes[1].set_ylabel("intensity")
        axes[1].set_title("Masked input peaks")
        axes[1].legend(loc="upper right")

        recon_mz = recon_np[idx, :, 0]
        recon_int = recon_np[idx, :, 1]
        axes[2].vlines(
            recon_mz,
            0.0,
            recon_int,
            color="tab:orange",
            alpha=0.7,
            linewidth=1.5,
            label="reconstruction",
        )
        if masked_positions is not None and masked_positions.any():
            axes[2].vlines(
                recon_mz[masked_positions],
                0.0,
                recon_int[masked_positions],
                color="tab:red",
                alpha=0.9,
                linewidth=2.0,
                label="reconstruction (masked)",
            )
        axes[2].set_xlabel("m/z")
        axes[2].set_ylabel("intensity")
        axes[2].set_title("Reconstructed peaks")
        axes[2].legend(loc="upper right")

        if active_mask.any():
            x_min = min(float(mz[idx][active_mask].min()), float(recon_mz.min()))
            x_max = max(float(mz[idx][active_mask].max()), float(recon_mz.max()))
            axes[2].set_xlim(x_min, x_max)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        figures.append(fig)

    return figures


def figures_to_image_array(figures: Sequence[Figure] | Figure) -> np.ndarray:
    """Convert matplotlib figures to a numpy image array.

    Args:
        figures: A single figure or a sequence of figures to convert.

    Returns:
        A numpy array of shape [N, H, W, C] if multiple figures are provided,
        or [H, W, C] for a single figure.
    """
    if isinstance(figures, Figure):
        figure_list = [figures]
    else:
        figure_list = list(figures)

    if not figure_list:
        raise ValueError("No figures provided for conversion to images.")

    images: list[np.ndarray] = []
    for fig in figure_list:
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = buffer[..., :3].copy()
        images.append(image)
        plt.close(fig)

    if len(images) == 1:
        return images[0]
    return np.stack(images, axis=0)
