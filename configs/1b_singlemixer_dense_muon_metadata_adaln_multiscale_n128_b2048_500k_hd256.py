"""500k multiscale run with the head dimension matched to the v6e MXU.

Same 1.0004B model as the parent config; the only change is 6 heads instead of
12, which makes `head_dim = model_dim / num_heads = 1536 / 6 = 256`. Trillium's
MXU is 256x256 (v4/v5e/v5p were 128x128), and `head_dim` was the one MaxText
dimension rule this config broke — `model_dim`, the SwiGLU hidden dim and both
3072-wide MLPs are already multiples of 256.

Measured on one v6e chip, forward+backward at micro-batch 32, once the
contiguous-RoPE fix is in place: 0.0380 s -> 0.0356 s, i.e. 38.0% -> 40.5% MFU at
identical FLOPs. Sweeping the whole range gave 17.3 / 16.8 / 17.5 / 18.0% for
head_dim 64 / 128 / 192 / 256 before the RoPE fix, so 256 is the best of the four
either way.

This is a fresh-run config: 6 wide heads is a different parameterisation from 12
narrow ones, so it cannot resume the parent run's checkpoint.

Two step-level flags are on here; together they measured 209 -> 147 ms on a
four-chip step, and neither touches the forward or backward pass.

`jax_grad_all_reduce_dtype = "bfloat16"` reduces the gradients over the wire in
bf16 rather than fp32 (they stay fp32 either side). A `psum` of a
parameter-shaped fp32 tree is 4 GB and measured 40 ms on four chips, so this is
worth 209 -> 185 ms on its own. It is the one change here that is not exact:
gradients are rounded before being summed across ranks.

`jax_shard_optimizer` is on here. Muon's Newton-Schulz is identical work on
every data-parallel rank and does not shrink as chips are added, so it was
120 ms of a 209 ms step at four chips; distributing whole tensors across ranks
and summing the updates back measured 209 -> 177 ms on its own, and gets better
with scale because the compute saving grows while the collective shrinks. The update is
mathematically identical to the replicated one - see
tests/test_muon_sharded_orthogonalization.py. It costs a ~15 minute first
compile.

One further knob is left alone deliberately: `muon_ns_steps = 3` is another
1.22x on the step at 64 chips and is a one-line change, but unlike the sharding
it changes the update rule - three Newton-Schulz steps leave the singular values
further from 1 than five do - so it wants a loss A/B before a 500k-step run.
"""

from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module(
        "configs."
        "1b_singlemixer_dense_muon_metadata_adaln_"
        "multiscale_n128_b2048_500k"
    ).get_config()

    cfg.encoder_num_heads = 6
    cfg.masked_latent_predictor_num_heads = 6

    cfg.jax_shard_optimizer = True
    cfg.jax_grad_all_reduce_dtype = "bfloat16"

    cfg.run_name_suffix = (
        cfg.run_name_suffix.replace("-h12-", "-h6-").removesuffix("-500k")
        + "-hd256-500k"
    )

    return cfg
