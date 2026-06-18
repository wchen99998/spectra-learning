import math

import pytest

import train_sky
from spectra_learning.config import load_config


MUON_CONFIG = "configs/medium_pairmixer_100m_20m_mae_beta_isoflops_muon.py"
MUON_WORKDIR = "gs://metal-repeater-411410-spectra-checkpoints/skypilot/test-run"


def test_launcher_requires_explicit_config_and_workdir():
    with pytest.raises(SystemExit):
        train_sky.parse_args([])


def test_launcher_shape_comes_from_explicit_muon_long_run_config():
    args, sky_args = train_sky.parse_args(
        [
            "--config",
            MUON_CONFIG,
            "--workdir",
            MUON_WORKDIR,
        ]
    )
    defaults = train_sky.load_config_defaults(args.config)

    assert sky_args == []
    assert args.config == MUON_CONFIG
    assert args.workdir == MUON_WORKDIR
    assert args.topology == "4x4"
    assert args.training_max_steps is None
    assert defaults.training_max_steps == 250_000
    assert defaults.batch_size == 4096
    assert defaults.gradient_accumulation_steps == 4
    assert defaults.msg_probe_every_n_steps == 100_000
    assert defaults.val_every_n_steps == 25_000
    assert defaults.val_num_steps == 1_000
    assert defaults.dataloader_num_workers == 12
    assert defaults.aot_variant == "all"


def test_dryrun_alias_maps_to_dry_run_flag():
    args, _sky_args = train_sky.parse_args(
        [
            "--config",
            MUON_CONFIG,
            "--workdir",
            MUON_WORKDIR,
            "--dryrun",
        ]
    )

    assert args.dry_run is True


def test_muon_long_run_config_scales_lr_and_probe_schedule():
    cfg = load_config(MUON_CONFIG)

    assert cfg.training_max_steps == 250_000
    assert cfg.batch_size == 4096
    assert cfg.jax_mesh_devices == "16"
    assert cfg.learning_rate == pytest.approx(3e-4 * math.sqrt(2.0))
    assert cfg.min_learning_rate == pytest.approx(3e-5 * math.sqrt(2.0))
    assert cfg.muon_adam_learning_rate == cfg.learning_rate
    assert cfg.muon_adam_min_learning_rate == cfg.min_learning_rate
    assert cfg.msg_probe_every_n_steps == 100_000
    assert cfg.msg_probe_at_final_step is True
    assert cfg.val_every_n_steps == 25_000
    assert cfg.val_num_steps == 1_000
    assert cfg.jax_precompile_eval_steps is True
    assert cfg.jax_precompile_msg_probe is True
    assert cfg.jax_precompile_variant == "all"


def test_resolve_topology_4x4_maps_to_16_chip_pool():
    topology = train_sky.resolve_topology("4x4")

    assert topology.topology == "4x4"
    assert topology.total_chips == 16
    assert topology.num_nodes == 4
    assert topology.chips_per_node == 4
    assert topology.node_pool == "skypilot-v6e-16-flex"
    assert topology.jax_mesh_devices == "16"
    assert topology.aot_target == "v6e-4x4-multihost"


def test_resolve_topology_rejects_unknown_pool_without_override():
    with pytest.raises(ValueError, match="no default node pool"):
        train_sky.resolve_topology("4x8")


def test_resolve_topology_accepts_explicit_unknown_pool():
    topology = train_sky.resolve_topology("4x8", node_pool="custom-v6e-32-flex")

    assert topology.total_chips == 32
    assert topology.num_nodes == 8
    assert topology.node_pool == "custom-v6e-32-flex"
    assert topology.aot_target == "v6e-4x8-multihost"


def test_build_task_constructs_topology_resources_and_env():
    task = train_sky.build_task(
        topology=train_sky.resolve_topology("4x4"),
        envs={"SPECTRA_RUN_ID": "new", "SPECTRA_TRAINING_MAX_STEPS": "100"},
        infra="k8s/skypilot-training",
    )

    assert task["name"] == "spectra-100m-muon-v6e-kueue"
    assert task["workdir"] == "."
    assert task["num_nodes"] == 4
    assert task["resources"]["infra"] == "k8s/skypilot-training"
    assert task["resources"]["image_id"] == "docker:python:3.12-bookworm"
    assert task["resources"]["accelerators"] == "tpu-v6e-4"
    assert task["resources"]["accelerator_args"]["tpu_vm"] is False
    assert task["resources"]["cpus"] == 64
    assert task["resources"]["memory"] == 256
    assert task["envs"]["SPECTRA_RUN_ID"] == "new"
    assert task["envs"]["SPECTRA_TRAINING_MAX_STEPS"] == "100"
    assert "SPECTRA_TRAIN_OVERRIDES_JSON must be set" in task["run"]
    assert '"jax_mesh_devices": "16"' not in task["run"]
    node_selector = task["config"]["kubernetes"]["pod_config"]["spec"]["nodeSelector"]
    assert node_selector["cloud.google.com/gke-nodepool"] == "skypilot-v6e-16-flex"
    assert node_selector["cloud.google.com/gke-tpu-topology"] == "4x4"
    tolerations = task["config"]["kubernetes"]["pod_config"]["spec"]["tolerations"]
    assert {
        "key": "cloud.google.com/gke-queued",
        "operator": "Equal",
        "value": "true",
        "effect": "NoSchedule",
    } in tolerations


def test_default_aot_cache_gcs_uses_explicit_workdir_bucket():
    assert (
        train_sky.default_aot_cache_gcs(
            "gs://checkpoint-bucket/skypilot/run-1",
            "cache-key",
        )
        == "gs://checkpoint-bucket/skypilot-aot-cache/cache-key"
    )


def test_default_aot_cache_gcs_requires_gcs_workdir():
    with pytest.raises(ValueError, match="non-GCS --workdir"):
        train_sky.default_aot_cache_gcs("/tmp/run-1", "cache-key")


def test_token_readers_use_environment_first():
    assert train_sky.read_hf_token({"HF_TOKEN": " hf-token "}) == "hf-token"
    assert train_sky.read_wandb_api_key({"WANDB_API_KEY": " wandb-token "}) == "wandb-token"


def test_dryrun_prints_generated_assets_without_token_lookup(
    tmp_path,
    monkeypatch,
    capsys,
):
    def fail_token_lookup(_env):
        raise AssertionError("dryrun should not read token secrets")

    monkeypatch.setattr(train_sky, "read_hf_token", fail_token_lookup)
    monkeypatch.setattr(train_sky, "read_wandb_api_key", fail_token_lookup)
    run_id = "dryrun-assets"

    train_sky.main(
        [
            "--dryrun",
            "--run-id",
            run_id,
            "--config",
            MUON_CONFIG,
            "--workdir",
            f"{MUON_WORKDIR}-{run_id}",
            "--task-output-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    task_path = tmp_path / f"{run_id}.yaml"
    assert task_path.is_file()
    assert "===== SkyPilot Task Path =====" in output
    assert str(task_path) in output
    assert "===== SkyPilot Task YAML =====" in output
    assert "name: spectra-100m-muon-v6e-kueue" in output
    assert f"SPECTRA_CONFIG: {MUON_CONFIG}" in output
    assert "SPECTRA_WORKDIR:" in output
    assert "SPECTRA_TRAIN_OVERRIDES_JSON:" in output
    assert "LIBTPU_INIT_ARGS:" in output
    assert "xla_enable_async_all_reduce" in output
    assert "===== AOT Overrides JSON =====" in output
    assert '"jax_mesh_devices":"16"' in output
    assert "===== SkyPilot Command =====" in output
    assert "sky launch" in output
