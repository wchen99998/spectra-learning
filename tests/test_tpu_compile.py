import os
from types import SimpleNamespace

import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np
import pytest

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


def test_default_compiler_options_do_not_import_libtpu_runtime_flags():
    options = tpu_compile.build_compiler_options(None)

    assert options == {}
    assert "xla_enable_async_all_reduce" not in options
    assert "xla_tpu_overlap_compute_collective_tc" not in options


def test_explicit_compiler_options_are_preserved():
    assert tpu_compile.build_compiler_options(
        {"xla_tpu_num_sparse_cores_for_gather_offloading": "1"}
    ) == {"xla_tpu_num_sparse_cores_for_gather_offloading": "1"}


def test_serializable_compiled_object_unwraps_nnx_compiled_wrapper():
    inner = object()
    wrapped = SimpleNamespace(compiled=inner)

    assert tpu_compile.serializable_compiled_object(wrapped) is inner
    assert tpu_compile.serializable_compiled_object(inner) is inner


def test_compile_lowered_reuses_offline_topology_cache_hit():
    class FakeLowered:
        def compile(self, *, compiler_options):
            assert compiler_options == {"x": "1"}
            raise RuntimeError(tpu_compile.OFFLINE_TOPOLOGY_CACHE_HIT_ERROR)

    tpu_compile.jax.config.update("jax_raise_persistent_cache_errors", False)

    result = tpu_compile.compile_lowered_or_reuse_persistent_cache(
        FakeLowered(),
        {"x": "1"},
        label="pack20",
    )

    assert result.compiled is None
    assert result.reused_persistent_cache is True
    assert result.compile_seconds >= 0.0
    assert tpu_compile.jax.config.jax_raise_persistent_cache_errors is False


def test_compile_lowered_returns_compiled_object_and_restores_existing_cache_error_mode():
    compiled = object()

    class FakeLowered:
        def compile(self, *, compiler_options):
            assert compiler_options == {}
            return compiled

    tpu_compile.jax.config.update("jax_raise_persistent_cache_errors", True)
    try:
        result = tpu_compile.compile_lowered_or_reuse_persistent_cache(
            FakeLowered(),
            {},
            label="pack20",
        )

        assert result.compiled is compiled
        assert result.reused_persistent_cache is False
        assert tpu_compile.jax.config.jax_raise_persistent_cache_errors is True
    finally:
        tpu_compile.jax.config.update("jax_raise_persistent_cache_errors", False)


def test_compile_lowered_reraises_other_persistent_cache_errors():
    class FakeLowered:
        def compile(self, *, compiler_options):
            del compiler_options
            raise RuntimeError("cache entry failed checksum")

    tpu_compile.jax.config.update("jax_raise_persistent_cache_errors", False)

    with pytest.raises(RuntimeError, match="failed checksum"):
        tpu_compile.compile_lowered_or_reuse_persistent_cache(
            FakeLowered(),
            {},
            label="pack20",
        )

    assert tpu_compile.jax.config.jax_raise_persistent_cache_errors is False


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


def test_dynamic_v6e_2x4_multihost_target_matches_default_launcher():
    target = tpu_compile.resolve_tpu_compile_target("v6e-2x4-multihost")

    assert target.name == "ct6e-standard-4t-v6e-2x4"
    assert target.topology_name == "v6e:2x4"
    assert target.chips_per_host_bounds == (2, 2, 1)
    assert target.devices_per_slice == 8
    assert target.vm_count == 2
    assert target.process_count == 2


def test_dynamic_v6e_4x4_multihost_target_uses_four_hosts():
    target = tpu_compile.resolve_tpu_compile_target("v6e-4x4-multihost")

    assert target.name == "ct6e-standard-4t-v6e-4x4"
    assert target.topology_name == "v6e:4x4"
    assert target.chips_per_host_bounds == (2, 2, 1)
    assert target.devices_per_slice == 16
    assert target.vm_count == 4
    assert target.process_count == 4


def test_dynamic_v6e_8x8_multihost_target_uses_sixteen_hosts():
    target = tpu_compile.resolve_tpu_compile_target("v6e-8x8-multihost")

    assert target.name == "ct6e-standard-4t-v6e-8x8"
    assert target.topology_name == "v6e:8x8"
    assert target.chips_per_host_bounds == (2, 2, 1)
    assert target.devices_per_slice == 64
    assert target.vm_count == 16
    assert target.process_count == 16


def test_abstract_long_run_batches_match_4x4_topology():
    cfg = config_dict.ConfigDict()
    cfg.batch_size = 4096
    cfg.gradient_accumulation_steps = 4
    cfg.num_peaks = 31
    cfg.jepa_num_target_blocks = 1
    target = tpu_compile.resolve_tpu_compile_target("v6e-4x4-multihost")
    mesh = tpu_compile.build_tpu_compile_mesh(target)

    train_batch = tpu_compile.abstract_train_batch(cfg, target=target, data_mesh=mesh)
    eval_batch = tpu_compile.abstract_eval_batch(cfg, target=target, data_mesh=mesh)

    assert train_batch["peak_mz"].shape == (4, 256, 31)
    assert train_batch["target_masks"].shape == (4, 256, 1, 31)
    assert eval_batch["peak_mz"].shape == (256, 31)
    assert eval_batch["target_masks"].shape == (256, 1, 31)


def test_abstract_msg_probe_batch_uses_global_probe_batch_per_process():
    cfg = config_dict.ConfigDict()
    cfg.batch_size = 4096
    cfg.msg_probe_batch_size = 512
    cfg.num_peaks = 31
    cfg.model_dim = 640
    cfg.pairmixer_pair_dim = 256
    target = tpu_compile.resolve_tpu_compile_target("v6e-4x4-multihost")

    batch = tpu_compile.abstract_msg_probe_batch(cfg, target=target)
    task_spec = tpu_compile.abstract_msg_probe_task_spec(cfg)
    step_batch = tpu_compile.abstract_msg_probe_step_batch(
        cfg,
        task_spec,
        target=target,
    )
    features = tpu_compile.abstract_msg_probe_features(
        cfg,
        probe_batch_size=128,
        use_pair_features=True,
    )

    assert batch["peak_mz"].shape == (128, 31)
    assert batch["probe_maccs"].shape == (128, 166)
    assert batch["probe_maccs"].dtype == jnp.int32
    assert set(step_batch) == {
        "peak_valid_mask",
        "probe_valid_mol",
        "probe_maccs",
    }
    single, pair = features
    assert single.shape == (128, 32, 640)
    assert pair.shape == (128, 32, 32, 256)


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
        def train_loader_for_precompile(self):
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
