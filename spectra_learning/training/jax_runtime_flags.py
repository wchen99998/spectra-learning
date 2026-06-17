from __future__ import annotations

import os


JAX_TPU_XLA_FLAGS = (
    "--xla_tpu_scoped_vmem_limit_kib=98304",
    "--xla_tpu_enable_async_collective_fusion=true",
    "--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true",
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
    "--xla_enable_async_all_reduce=true",
    "--xla_tpu_use_minor_sharding_for_major_trivial_input=true",
    "--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1",
    "--xla_tpu_assign_all_reduce_scatter_layout=true",
)


def jax_tpu_xla_flags_string() -> str:
    return " ".join(JAX_TPU_XLA_FLAGS)


def configure_jax_tpu_xla_flags() -> None:
    os.environ["LIBTPU_INIT_ARGS"] = jax_tpu_xla_flags_string()
