"""1.78B at micro-batch 96, the shape that buys parameters with the wasted step.

Same architecture as
`1b_singlemixer_dense_adamw_metadata_adaln_multiscale_n128_b2048_500k_hd256`
widened from `model_dim` 1536 to 2048, `num_heads` 6 to 8 so `head_dim` stays at
the 256 that matches Trillium's MXU, and the two `2*d` MLPs from 3072 to 4096.
That is 1.7816B parameters against 1.0004B, a 1.78x, at the same 25 encoder
layers and 5 predictor layers.

The point is not the extra parameters, it is that they are close to free. On a
v6e-16 slice, measured at the per-chip shape this config runs:

    1.0004B, micro-batch 32   91.8 ms   15.8% MFU   349 samples/chip/s
    1.7816B, micro-batch 96  275.0 ms   29.5% MFU   349 samples/chip/s

Identical token throughput for a 1.78x larger model. 61% of the old step was
gradient all-reduce and a replicated Adam update - work that scales with
parameters and not with batch - and at micro-batch 32 the chip spent more time
on that than on the forward and backward pass. Micro-batch 96 amortises it.
The forward and backward pass also got *more* efficient with width, 39.8% ->
44.6% MFU at matched micro-batch, so the wider model beat a linear-in-parameters
step-time projection by 12%.

What this costs is the global batch: 96 x 64 chips is 6144, three times the
2048 the 1B run uses. That is an optimisation change, not a free one.

`learning_rate = 6e-4` doubles the 1B run's 3e-4 against a 3x batch. Square-root
scaling would say 5.2e-4 and linear scaling 9e-4, so this sits between them and
nearer the square root. `warmup_steps` stays at 20000, which at this batch is
123M samples rather than 41M - three times longer in tokens, deliberately, since
the learning rate doubled.

`training_max_steps = 666_000` is 4.092e9 samples, matching the 2M-step 1B run's
4.096e9 to within 0.1%. Because samples per chip-second are the same, that is
also about the same wall clock.

`optimizer_state_dtype = "bf16"` is not optional here. fp32 Adam moments are
14.3 GB at this size and OOMed at micro-batch 32, let alone 96. bf16 moments
measured faster as well as smaller at every micro-batch tested. It does put
Adam's second moment in bf16.

ZeRO-1 is deliberately not used. It was implemented and measured, and it does
let micro-batch 128 fit where replicated moments OOM - but micro-batch 96 fits
without it and lands within 2% of that on useful FLOPs per chip-second, while
ZeRO-1 costs 8.9% on the step where the two are directly comparable.

One known wart: `predictor_mlp_multiple` is 8/3, so the predictor's SwiGLU
hidden dim is `ceil(2048 * 8/3) = 5462`, which is not a multiple of 256 - the
one place this config breaks the MXU rule the rest of it follows. It is 5 of 30
blocks and the measurements above include it. Rounding to 5376 or 5632 is worth
a benchmark before a long run, but it changes the parameter count, so it is not
folded in silently here.
"""

from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module(
        "configs."
        "1b_singlemixer_dense_adamw_metadata_adaln_"
        "multiscale_n128_b2048_500k_hd256"
    ).get_config()

    cfg.optimizer = "adam"
    cfg.weight_decay = 0.0

    cfg.model_dim = 2048
    cfg.predictor_dim = 2048
    cfg.encoder_num_heads = 8
    cfg.masked_latent_predictor_num_heads = 8
    cfg.encoder_fourier_mlp_hidden_dim = 4096
    cfg.feature_mlp_hidden_dim = 4096

    # 96 per chip on 64 chips, accumulation off: ga divides the micro-batch,
    # which is the one thing that costs.
    cfg.batch_size = 6144
    cfg.gradient_accumulation_steps = 1

    cfg.learning_rate = 6e-4
    cfg.min_learning_rate = cfg.learning_rate / 10

    cfg.optimizer_state_dtype = "bf16"

    cfg.training_max_steps = 666_000

    cfg.run_name_suffix = (
        cfg.run_name_suffix.replace("-1b-200m-", "-1p8b-200m-")
        .replace("-d1536-", "-d2048-")
        .replace("-h6-", "-h8-")
        .replace("-feat3072-", "-feat4096-")
        .replace("-fmlp3072-", "-fmlp4096-")
        .replace("-xattnrope1536-", "-xattnrope2048-")
        .replace("-bs2048-", "-bs6144-")
        .replace("-ga2-", "-ga1-")
        .replace("-fp32state-", "-bf16state-")
        .replace("-lr3e-4-", "-lr6e-4-")
        .removesuffix("-2m")
        + "-666k"
    )

    return cfg
