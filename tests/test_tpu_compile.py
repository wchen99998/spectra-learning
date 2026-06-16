import os

from ml_collections import config_dict
import numpy as np

from spectra_learning.training import tpu_compile
from spectra_learning.training.jax_runtime_flags import (
    JAX_TPU_XLA_FLAGS,
    jax_tpu_xla_flags_string,
)


def test_ct6e_standard_8t_target_is_single_host_v6e_8():
    target = tpu_compile.resolve_tpu_compile_target("ct6e-standard-8t")

    assert target.topology_name == "v6e:2x4"
    assert target.chips_per_host_bounds == (2, 4, 1)
    assert target.devices_per_slice == 8
    assert target.vm_count == 1
    assert target.process_count == 1


def test_v6e_8_alias_prefers_single_host_8t_layout():
    target = tpu_compile.resolve_tpu_compile_target("v6e-8")

    assert target.name == "ct6e-standard-8t"
    assert target.chips_per_host_bounds == (2, 4, 1)
    assert target.process_count == 1


def test_v6e_8_multihost_alias_keeps_four_chip_vm_layout():
    target = tpu_compile.resolve_tpu_compile_target("v6e-8-multihost")

    assert target.name == "ct6e-standard-4t-v6e-8"
    assert target.chips_per_host_bounds == (2, 2, 1)
    assert target.process_count == 2


def test_target_microbatch_size_uses_single_host_process_count():
    cfg = config_dict.ConfigDict()
    cfg.batch_size = 512
    cfg.gradient_accumulation_steps = 4
    target = tpu_compile.resolve_tpu_compile_target("ct6e-standard-8t")

    assert tpu_compile.target_microbatch_size(cfg, target) == 128


def test_compile_variants_selects_default_pack_only_for_fast_iteration():
    cfg = config_dict.ConfigDict()
    cfg.mae_context_encoder_pack_tokens = 20
    cfg.mae_context_encoder_pack_token_choices = (20, 24, 28)

    variants = tpu_compile.compile_variants(cfg, "default")

    assert variants == (
        tpu_compile.CompileVariant("pack20", 20, False),
    )


def test_compile_variants_all_includes_pack_choices_and_full_fallback():
    cfg = config_dict.ConfigDict()
    cfg.mae_context_encoder_pack_tokens = 20
    cfg.mae_context_encoder_pack_token_choices = (20, 24, 28)

    variants = tpu_compile.compile_variants(cfg, "all")

    assert [variant.name for variant in variants] == [
        "pack20",
        "pack24",
        "pack28",
        "full",
    ]


def test_runtime_precompile_variants_match_aot_default_policy():
    from spectra_learning.training import pretrain_jax

    assert pretrain_jax._jax_train_step_compile_variants(
        (20, 24, 28),
        selector="default",
        has_default_train_step=False,
        has_full_fallback=True,
    ) == (("pack20", 20, False),)


def test_runtime_precompile_variants_all_includes_full_fallback():
    from spectra_learning.training import pretrain_jax

    assert pretrain_jax._jax_train_step_compile_variants(
        (20, 24),
        selector="all",
        has_default_train_step=False,
        has_full_fallback=True,
    ) == (
        ("pack20", 20, False),
        ("pack24", 24, False),
        ("full", 0, True),
    )


def test_runtime_precompile_uses_selected_default_pack_variant():
    from spectra_learning.training import pretrain_jax

    calls = []

    class FakeDataModule:
        def train_loader_for_epoch(self, epoch):
            del epoch
            batch = {
                "peak_valid_mask": np.ones((2, 31), dtype=bool),
                "context_mask": np.ones((2, 31), dtype=bool),
            }
            return iter([batch])

    def fake_step(name):
        def step(params, static_state, opt_state, batch):
            del static_state, batch
            calls.append(name)
            return params, opt_state, {"loss": pretrain_jax.jnp.asarray(0.0)}

        return step

    cfg = config_dict.ConfigDict()
    cfg.jax_precompile_train_steps = True
    cfg.jax_precompile_repetitions = 1
    cfg.jax_precompile_variant = "default"

    metrics = pretrain_jax._precompile_jax_training_steps(
        config=cfg,
        datamodule=FakeDataModule(),
        grad_accum_steps=1,
        use_sharded_step=False,
        data_mesh=None,
        pure_trainable_params={"p": pretrain_jax.jnp.asarray(0.0)},
        pure_opt_state={"o": pretrain_jax.jnp.asarray(0.0)},
        pure_static_state={},
        pure_train_step=fake_step("default"),
        pure_pack_train_steps=[
            (20, {}, fake_step("pack20")),
            (24, {}, fake_step("pack24")),
        ],
        pure_full_static_state={},
        pure_full_train_step=fake_step("full"),
    )

    assert calls == ["pack20"]
    assert metrics["run/precompile_train_steps"] == 1.0
    assert metrics["run/precompile_pack_variants"] == 1.0
    assert metrics["run/precompile_full_fallback"] == 0.0


def test_configure_jax_runtime_sets_persistent_cache_controls(monkeypatch):
    from spectra_learning.training import pretrain_jax

    calls = []

    def fake_update(name, value):
        calls.append((name, value))

    monkeypatch.setattr(pretrain_jax.jax.config, "update", fake_update)
    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    cfg = config_dict.ConfigDict()
    cfg.jax_compilation_cache_dir = "artifacts/jax_compile_cache/v6e8-alpha"
    cfg.jax_enable_compilation_cache = True
    cfg.jax_persistent_cache_min_compile_time_secs = 0.0
    cfg.jax_persistent_cache_min_entry_size_bytes = 0

    pretrain_jax.configure_jax_runtime(cfg)

    assert calls == [
        ("jax_compilation_cache_dir", "artifacts/jax_compile_cache/v6e8-alpha"),
        ("jax_enable_compilation_cache", True),
        ("jax_persistent_cache_min_compile_time_secs", 0.0),
        ("jax_persistent_cache_min_entry_size_bytes", 0),
    ]


def test_configure_jax_runtime_sets_libtpu_init_args(monkeypatch):
    from spectra_learning.training import pretrain_jax

    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    cfg = config_dict.ConfigDict()

    pretrain_jax.configure_jax_runtime(cfg)

    assert os.environ["LIBTPU_INIT_ARGS"] == jax_tpu_xla_flags_string()


def test_jax_tpu_xla_flags_are_global_defaults():
    assert JAX_TPU_XLA_FLAGS == (
        "--xla_tpu_scoped_vmem_limit_kib=98304",
        "--xla_tpu_enable_async_collective_fusion=true",
        "--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true",
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
        "--xla_tpu_overlap_compute_collective_tc=true",
        "--xla_enable_async_all_reduce=true",
        "--xla_tpu_use_minor_sharding_for_major_trivial_input=true",
        "--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1",
        "--xla_tpu_assign_all_reduce_scatter_layout=true",
    )
    assert jax_tpu_xla_flags_string() == " ".join(JAX_TPU_XLA_FLAGS)


def test_cost_analysis_summary_keeps_only_high_level_totals():
    assert tpu_compile._cost_analysis_summary(
        {
            "flops": 10.0,
            "bytes accessed": 20.0,
            "transcendentals": 3.0,
            "optimal_seconds": 0.5,
            "utilization0{}": 100.0,
        }
    ) == {
        "flops": 10.0,
        "bytes accessed": 20.0,
        "transcendentals": 3.0,
        "optimal_seconds": 0.5,
    }


def test_parse_compiler_options_accepts_libtpu_flag_string():
    assert tpu_compile.parse_compiler_options(
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_spmd_rng_bit_generator_unsafe=true"
    ) == {
        "xla_tpu_enable_async_collective_fusion": "true",
        "xla_tpu_spmd_rng_bit_generator_unsafe": "true",
    }
