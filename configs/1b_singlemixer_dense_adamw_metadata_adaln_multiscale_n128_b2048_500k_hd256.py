"""The hd256 2M-step multiscale run with Adam in place of Muon.

Identical model to
`1b_singlemixer_dense_muon_metadata_adaln_multiscale_n128_b2048_500k_hd256`
- same 1.0004B parameters, same 6 heads at head_dim 256, same data and schedule.
Only the optimizer changes, so this is the config to run against that one when
the question is whether Muon earns its cost.

It does cost. On four v6e chips at micro-batch 32 the full step measured 147 ms
with Muon against 96 ms with Adam: Muon's Newton-Schulz is 3.4e13 FLOPs per
step, more than the entire forward and backward pass, and even after sharding
whole tensors across data-parallel ranks it stays the largest single item in the
step. Adam's update is ~31 ms and bandwidth bound. Muon has to be worth roughly
1.5x in convergence per step to break even here.

Settings that carry over, and one that does not:

- `learning_rate = 3e-4` is inherited unchanged. The parent config already
  applies it to Muon's Adam branch (`muon_adam_learning_rate` is unset, so it
  falls back to `learning_rate`), and 3e-4 with `b1=0.9`, `b2=0.95`,
  a 20k-step warmup is a conventional Adam recipe at
  this scale. It is still worth a short sweep: an LR tuned for Muon's
  orthogonalised updates is not automatically right for Adam's.
- `jax_grad_all_reduce_dtype = "bfloat16"` carries over. It reduces gradients
  over the wire in bf16 and has nothing to do with which optimizer runs, worth
  209 -> 185 ms on the Muon step and the same saving here.
- `jax_shard_optimizer` is explicitly off. It only shards Muon's Newton-Schulz;
  Adam-family state sharding is controlled separately by
  `jax_zero1_optimizer`.

Note Adam needs more optimizer state, not less: two moments at 4 GB each
against Muon's single momentum buffer, so 8 GB replicated per chip rather than
4 GB. `optimizer_state_dtype = "bf16"` halves that and measured 31.9 -> 25.3 ms
on the update, but it puts Adam's second moment in bf16, so it is left off here
rather than folded silently into a 2M-step run.
"""

from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module(
        "configs."
        "1b_singlemixer_dense_muon_metadata_adaln_"
        "multiscale_n128_b2048_500k_hd256"
    ).get_config()

    cfg.optimizer = "adam"
    cfg.b1 = 0.9
    cfg.weight_decay = 0.0
    cfg.training_max_steps = 2_000_000

    # Sharding the optimizer is a Muon-only path.
    cfg.jax_shard_optimizer = False

    cfg.run_name_suffix = (
        cfg.run_name_suffix.replace("-muon-", "-adam-")
        # Inherited from the Muon parent, which decays at 0.1.
        .replace("-wd1e-1-", "-wd0-")
        .removesuffix("-500k")
        + "-2m"
    )

    return cfg
