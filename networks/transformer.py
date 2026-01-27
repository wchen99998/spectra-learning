"""JAX implementation of a LLaMA2-style Transformer using flax.nnx."""

import dataclasses
import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


activation_map = dict(
    swiglu=nnx.swish,
    geglu=nnx.gelu,
    glu=nnx.sigmoid,
    swish=nnx.swish,
)

GATED_MLP_TYPES = frozenset({"swiglu", "geglu", "glu"})


@dataclasses.dataclass(unsafe_hash=True)
class ModelArgs:
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = None
    output_channels: int = 1024
    hidden_dim: Optional[int] = None
    multiple_of: int = 32
    norm_eps: float = 1e-5
    w_init_scale: float = 1.0
    depth_scaled_init: bool = False
    mlp_type: str = "swiglu"
    cond_type: str = "adaln"
    embed_input: bool = False
    n_embed_classes: int = 1024
    causal: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    rope_theta: float = 10000.0
    use_rotary_embeddings: bool = True
    input_dim: Optional[int] = None  # Required when embed_input is False


def precompute_freqs_cis(dim, end, theta: float = 10000.0, dtype=jnp.float32):
    freqs = 1.0 / (
        theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim)
    )
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)
    return freqs_cos.astype(dtype), freqs_sin.astype(dtype)


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(shape)


def jax_unstack(x, axis=0):
    return [
        jax.lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])
    ]


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xq_r, xq_i = jax_unstack(xq.reshape(xq.shape[:-1] + (-1, 2)), -1)
    xk_r, xk_i = jax_unstack(xk.reshape(xk.shape[:-1] + (-1, 2)), -1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = jnp.stack([xq_out_r, xq_out_i], axis=-1).reshape(
        xq_out_r.shape[:3] + (-1,)
    )
    xk_out = jnp.stack([xk_out_r, xk_out_i], axis=-1).reshape(
        xk_out_r.shape[:3] + (-1,)
    )

    return xq_out, xk_out


def repeat_kv(x, n_rep):
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.tile(x[:, :, :, None, :], [1, 1, 1, n_rep, 1]).reshape(
        bs, slen, n_kv_heads * n_rep, head_dim
    )


def _make_norm(
    rngs: nnx.Rngs,
    *,
    num_features: int,
    dtype: jnp.dtype,
    param_dtype: jnp.dtype,
    epsilon: float,
):
    return nnx.RMSNorm(
        num_features=num_features,
        epsilon=epsilon,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )


class Attention(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        causal: bool = False,
        qkv_bias: bool = False,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self._n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert self.n_heads % self._n_kv_heads == 0
        self.n_rep = self.n_heads // self._n_kv_heads
        self.head_dim = self.dim // self.n_heads
        self.causal = causal
        self.dtype = dtype
        self.param_dtype = param_dtype

        init_qkv = nnx.with_partitioning(
            nnx.initializers.xavier_normal(), ("hidden", "attn_qkv")
        )
        init_o = nnx.with_partitioning(
            nnx.initializers.xavier_normal(), ("attn_o", "hidden")
        )

        self.wqkv = nnx.Linear(
            in_features=self.dim,
            out_features=(self.n_heads + 2 * self._n_kv_heads) * self.head_dim,
            use_bias=qkv_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init_qkv,
            rngs=rngs,
        )
        self.wo = nnx.Linear(
            in_features=self.dim,
            out_features=self.dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init_o,
            rngs=rngs,
        )

    def __call__(
        self,
        x,
        freqs_cos=None,
        freqs_sin=None,
        attention_bias=None,
        train: bool = False,
    ):
        del train  # Attention is deterministic in this model.
        bsz, seqlen, _ = x.shape

        qkv = self.wqkv(x)
        q_size = self.n_heads * self.head_dim
        kv_size = self._n_kv_heads * self.head_dim
        xq, xk, xv = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)

        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self._n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self._n_kv_heads, self.head_dim)

        if freqs_cos is not None and freqs_sin is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        attention_bias = None
        output = jax.nn.dot_product_attention(
            xq,
            xk,
            xv,
            bias=attention_bias,
            is_causal=self.causal,
            implementation="cudnn",
        )
        output = output.reshape(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        dim: int,
        multiple_of: int,
        hidden_dim: int | None = None,
        w_init_scale: float = 1.0,
        mlp_type: str = "swiglu",
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        self.dim = dim
        self.dtype = dtype
        self.mlp_type = mlp_type
        if mlp_type not in activation_map:
            raise ValueError(f"Unsupported mlp_type: {mlp_type}")

        hidden_dim = hidden_dim or int((4 * dim) * 2 / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim
        w_init = nnx.initializers.variance_scaling(
            w_init_scale, "fan_in", "truncated_normal"
        )
        init_mlp = nnx.with_partitioning(w_init, ("hidden", "ff_mlp"))
        init_out = nnx.with_partitioning(w_init, ("ff_mlp", "hidden"))

        self._uses_gating = mlp_type in GATED_MLP_TYPES

        if self._uses_gating:
            self.w12 = nnx.Linear(
                in_features=dim,
                out_features=2 * hidden_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=init_mlp,
                rngs=rngs,
            )
        else:
            self.w1 = nnx.Linear(
                in_features=dim,
                out_features=hidden_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=init_mlp,
                rngs=rngs,
            )
        self.w2 = nnx.Linear(
            in_features=hidden_dim,
            out_features=dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=init_out,
            rngs=rngs,
        )
        self.w3 = None

    def __call__(self, x, train: bool = False):
        del train
        act = activation_map[self.mlp_type]
        if self._uses_gating:
            w1, w3 = jnp.split(self.w12(x), 2, axis=-1)
            hidden = act(w1) * w3
        else:
            hidden = act(self.w1(x))
        y = self.w2(hidden)
        return y.astype(self.dtype)


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        *,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        causal: bool,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        norm_eps: float,
        mlp_type: str,
        multiple_of: int,
        hidden_dim: Optional[int],
        w_init_scale: float,
        use_rotary_embeddings: bool,
    ):
        self.dim = dim
        self.use_rotary_embeddings = use_rotary_embeddings
        self.attention = Attention(
            rngs,
            dim,
            n_heads,
            n_kv_heads=n_kv_heads,
            causal=causal,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.feed_forward = FeedForward(
            rngs,
            dim=dim,
            multiple_of=multiple_of,
            hidden_dim=hidden_dim,
            w_init_scale=w_init_scale,
            mlp_type=mlp_type,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.attention_norm = _make_norm(
            rngs,
            num_features=dim,
            dtype=dtype,
            param_dtype=param_dtype,
            epsilon=norm_eps,
        )
        self.ffn_norm = _make_norm(
            rngs,
            num_features=dim,
            dtype=dtype,
            param_dtype=param_dtype,
            epsilon=norm_eps,
        )

    def __call__(
        self,
        x,
        freqs_cos=None,
        freqs_sin=None,
        attention_bias=None,
        train: bool = False,
    ):
        if not self.use_rotary_embeddings:
            freqs_cos = None
            freqs_sin = None
        elif freqs_cos is None or freqs_sin is None:
            raise ValueError(
                "freqs_cos and freqs_sin must be provided when "
                "use_rotary_embeddings is True."
            )

        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos,
            freqs_sin,
            attention_bias=attention_bias,
            train=train,
        )

        return h + self.feed_forward(self.ffn_norm(h), train=train)


class Transformer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, args: ModelArgs):
        self.args = args

        if args.embed_input:
            embedding_init = nnx.with_partitioning(
                nnx.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                ("embed_vocab", "hidden"),
            )
            self.input_layer = nnx.Embed(
                num_embeddings=args.n_embed_classes,
                features=args.dim,
                dtype=args.dtype,
                param_dtype=args.param_dtype,
                embedding_init=embedding_init,
                rngs=rngs,
            )
            self._input_is_embedding = True
        else:
            if args.input_dim is None:
                raise ValueError(
                    "ModelArgs.input_dim must be set when embed_input is False."
                )
            proj_init = nnx.with_partitioning(
                nnx.initializers.lecun_normal(), ("input_embed", "hidden")
            )
            self.input_layer = nnx.Linear(
                in_features=args.input_dim,
                out_features=args.dim,
                dtype=args.dtype,
                param_dtype=args.param_dtype,
                kernel_init=proj_init,
                rngs=rngs,
            )
            self._input_is_embedding = False

        w_init_scale = (
            2.0 / args.n_layers if args.depth_scaled_init else args.w_init_scale
        )
        blocks = []
        for _ in range(args.n_layers):
            blocks.append(
                TransformerBlock(
                    rngs=rngs,
                    dim=args.dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    causal=args.causal,
                    dtype=args.dtype,
                    param_dtype=args.param_dtype,
                    norm_eps=args.norm_eps,
                    mlp_type=args.mlp_type,
                    multiple_of=args.multiple_of,
                    hidden_dim=args.hidden_dim,
                    w_init_scale=w_init_scale,
                    use_rotary_embeddings=args.use_rotary_embeddings,
                )
            )
        self.blocks = nnx.List(blocks)
        self.output_norm = _make_norm(
            rngs,
            num_features=args.dim,
            dtype=args.dtype,
            param_dtype=args.param_dtype,
            epsilon=args.norm_eps,
        )
        out_init = nnx.with_partitioning(nnx.initializers.zeros, ("hidden", "vocab"))
        self.output_proj = nnx.Linear(
            in_features=args.dim,
            out_features=args.output_channels,
            use_bias=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=out_init,
            rngs=rngs,
        )

    def __call__(
        self,
        x,
        train: bool = False,
        output_channels: Optional[int] = None,
        attention_bias=None,
    ):
        args = self.args
        if output_channels is None:
            output_channels = args.output_channels

        if self._input_is_embedding:
            h = self.input_layer(x)
        else:
            h = self.input_layer(x)

        freqs_cos = None
        freqs_sin = None
        if args.use_rotary_embeddings:
            seqlen = x.shape[1]
            freqs_cos, freqs_sin = precompute_freqs_cis(
                args.dim // args.n_heads,
                seqlen,
                theta=args.rope_theta,
                dtype=args.dtype,
            )
            freqs_cos = freqs_cos[:seqlen].astype(h.dtype)
            freqs_sin = freqs_sin[:seqlen].astype(h.dtype)

        for block in self.blocks:
            h = block(
                h,
                freqs_cos,
                freqs_sin,
                attention_bias=attention_bias,
                train=train,
            )

        h = self.output_norm(h)

        if output_channels != args.output_channels:
            raise ValueError(
                "Overriding output_channels is not supported by the nnx Transformer."
            )

        return self.output_proj(h)
