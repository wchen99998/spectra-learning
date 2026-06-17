from pathlib import Path
import math

import pytest
import yaml

import train_sky
from spectra_learning.config import load_config


def test_default_launcher_shape_comes_from_muon_long_run_config():
    args, sky_args = train_sky.parse_args([])
    defaults = train_sky.load_config_defaults(args.config)

    assert sky_args == []
    assert args.topology == "4x4"
    assert args.training_max_steps is None
    assert defaults.training_max_steps == 250_000
    assert defaults.batch_size == 4096
    assert defaults.gradient_accumulation_steps == 4
    assert defaults.msg_probe_every_n_steps == 100_000
    assert defaults.val_every_n_steps == 25_000
    assert defaults.val_num_steps == 1_000
    assert defaults.dataloader_num_workers == 12


def test_muon_long_run_config_scales_lr_and_probe_schedule():
    cfg = load_config(train_sky.DEFAULT_CONFIG)

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


def test_build_task_overwrites_topology_resources_and_env(tmp_path: Path):
    template = tmp_path / "task.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "num_nodes": 2,
                "resources": {
                    "infra": "k8s/old",
                    "accelerators": "tpu-v6e-4",
                },
                "envs": {
                    "SPECTRA_RUN_ID": "old",
                },
                "config": {
                    "kubernetes": {
                        "pod_config": {
                            "spec": {
                                "nodeSelector": {
                                    "cloud.google.com/gke-nodepool": "old-pool",
                                    "cloud.google.com/gke-tpu-topology": "2x4",
                                }
                            }
                        }
                    }
                },
            }
        )
    )

    task = train_sky.build_task(
        template_path=template,
        topology=train_sky.resolve_topology("4x4"),
        envs={"SPECTRA_RUN_ID": "new", "SPECTRA_TRAINING_MAX_STEPS": "100"},
        infra="k8s/skypilot-training",
    )

    assert task["num_nodes"] == 4
    assert task["resources"]["infra"] == "k8s/skypilot-training"
    assert task["resources"]["accelerators"] == "tpu-v6e-4"
    assert task["resources"]["accelerator_args"]["tpu_vm"] is False
    assert task["envs"]["SPECTRA_RUN_ID"] == "new"
    assert task["envs"]["SPECTRA_TRAINING_MAX_STEPS"] == "100"
    node_selector = task["config"]["kubernetes"]["pod_config"]["spec"]["nodeSelector"]
    assert node_selector["cloud.google.com/gke-nodepool"] == "skypilot-v6e-16-flex"
    assert node_selector["cloud.google.com/gke-tpu-topology"] == "4x4"


def test_token_readers_use_environment_first():
    assert train_sky.read_hf_token({"HF_TOKEN": " hf-token "}) == "hf-token"
    assert train_sky.read_wandb_api_key({"WANDB_API_KEY": " wandb-token "}) == "wandb-token"
