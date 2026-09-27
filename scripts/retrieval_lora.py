"""Attention-only LoRA for retrieval, installed after restoring the JEPA student."""

from flax import nnx
import jax
import jax.numpy as jnp


class LoRALinear(nnx.Module):
    def __init__(self, base, *, rank: int, alpha: float, key):
        out_dim, in_dim = base.weight.shape
        if not 0 < rank <= min(in_dim, out_dim):
            raise ValueError("LoRA rank must fit the projection dimensions")
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nnx.Param(jax.random.normal(key, (in_dim, rank)) / in_dim**.5)
        self.lora_b = nnx.Param(jnp.zeros((rank, out_dim), jnp.float32))

    def __call__(self, x):
        original = self.base(x)
        delta = (x.astype(jnp.float32) @ self.lora_a[...] @ self.lora_b[...]) * self.scale
        return (original.astype(jnp.float32) + delta).astype(original.dtype)


def install_attention_lora(encoder, *, rank: int, alpha: float, seed: int):
    """Adapt fused Q/K/V and attention output; leave MLPs, gates and conditioning frozen."""
    keys = jax.random.split(jax.random.PRNGKey(seed), len(encoder.blocks) * 2)
    for index, block in enumerate(encoder.blocks):
        attention = block.single_attention
        for offset, name in enumerate(("qkv", "o")):
            setattr(attention, name, LoRALinear(
                getattr(attention, name), rank=rank, alpha=alpha, key=keys[2 * index + offset]))
