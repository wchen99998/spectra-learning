import json
import os
import subprocess
from pathlib import Path

import pytest

import train_sky
from spectra_learning.config import load_config


TRAIN_CONFIG = "configs/100m_pairmixer_dense_adamw.py"
DENSE_ADAMW_CONFIG = "configs/300m_pairmixer_dense_adamw.py"
TRAIN_WORKDIR = "gs://metal-repeater-411410-spectra-checkpoints/skypilot/test-run"


def test_launcher_requires_explicit_config_and_workdir():
    with pytest.raises(SystemExit):
        train_sky.parse_args([])


def test_launcher_shape_comes_from_explicit_current_run_config():
    args, sky_args = train_sky.parse_args(
        [
            "--config",
            TRAIN_CONFIG,
            "--workdir",
            TRAIN_WORKDIR,
        ]
    )
    config = load_config(args.config)

    assert sky_args == []
    assert args.config == TRAIN_CONFIG
    assert args.workdir == TRAIN_WORKDIR
    assert args.topology == ""
    assert args.chips is None
    assert args.region == "us-central1"
    assert args.infra == "gcp/us-central1"
    assert args.sky_bin == train_sky.DEFAULT_SKY_BIN
    assert args.cpus == ""
    assert args.memory == ""
    assert args.stream_logs is True
    assert args.dws_run_duration_seconds == train_sky.DEFAULT_DWS_RUN_DURATION_SECONDS
    assert args.provision_timeout_seconds == train_sky.DEFAULT_PROVISION_TIMEOUT_SECONDS
    assert config.training_max_steps == 1_000_000
    assert config.batch_size == 2048
    assert config.gradient_accumulation_steps == 4
    assert config.msg_probe_every_n_steps == -1
    assert config.val_every_n_steps == 10_000
    assert config.val_num_steps == 500
    assert config.dataloader_num_workers == 32
    assert config.msg_probe_at_final_step is False


def test_region_flag_sets_gcp_infra():
    args, sky_args = train_sky.parse_args(
        [
            "--config",
            TRAIN_CONFIG,
            "--workdir",
            TRAIN_WORKDIR,
            "--region",
            "asia-northeast1",
        ]
    )

    assert sky_args == []
    assert args.region == "asia-northeast1"
    assert args.infra == "gcp/asia-northeast1"


def test_region_flag_accepts_gcp_prefix():
    args, _sky_args = train_sky.parse_args(
        [
            "--config",
            TRAIN_CONFIG,
            "--workdir",
            TRAIN_WORKDIR,
            "--region",
            "gcp/us-south1",
        ]
    )

    assert args.region == "us-south1"
    assert args.infra == "gcp/us-south1"


def test_infra_flag_sets_region_from_infra():
    args, _sky_args = train_sky.parse_args(
        [
            "--config",
            TRAIN_CONFIG,
            "--workdir",
            TRAIN_WORKDIR,
            "--infra",
            "gcp/us-east5",
        ]
    )

    assert args.region == "us-east5"
    assert args.infra == "gcp/us-east5"


def test_region_rejects_conflicting_infra():
    with pytest.raises(SystemExit):
        train_sky.parse_args(
            [
                "--config",
                TRAIN_CONFIG,
                "--workdir",
                TRAIN_WORKDIR,
                "--region",
                "us-south1",
                "--infra",
                "gcp/us-east5",
            ]
        )


def test_dryrun_alias_maps_to_dry_run_flag():
    args, _sky_args = train_sky.parse_args(
        [
            "--config",
            TRAIN_CONFIG,
            "--workdir",
            TRAIN_WORKDIR,
            "--dryrun",
        ]
    )

    assert args.dry_run is True


def test_sky_launcher_rejects_non_jax_training_config(tmp_path):
    with pytest.raises(ValueError, match="requires device_backend='jax'"):
        train_sky.main(
            [
                "--dryrun",
                "--config",
                "configs/pretrain.py",
                "--workdir",
                f"{TRAIN_WORKDIR}-torch",
                "--task-output-dir",
                str(tmp_path),
            ]
        )


def test_sky_launcher_rejects_unsupported_jax_task(tmp_path):
    with pytest.raises(ValueError, match="does not support device_backend='jax'"):
        train_sky.main(
            [
                "--dryrun",
                "--config",
                TRAIN_CONFIG,
                "--workdir",
                f"{TRAIN_WORKDIR}-contrastive",
                "--task-output-dir",
                str(tmp_path),
                "--override",
                'training_task="contrastive"',
            ]
        )


def test_detach_run_submits_without_log_streaming():
    args, sky_args = train_sky.parse_args(
        [
            "--config",
            TRAIN_CONFIG,
            "--workdir",
            TRAIN_WORKDIR,
            "--detach-run",
        ]
    )

    assert sky_args == []
    assert args.stream_logs is False


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("600", 600),
        ("600s", 600),
        ("10m", 600),
        ("6h", 21600),
        ("7d", 604800),
    ],
)
def test_dws_run_duration_parses_seconds_and_suffixes(value, seconds):
    args, _sky_args = train_sky.parse_args(
        [
            "--config",
            TRAIN_CONFIG,
            "--workdir",
            TRAIN_WORKDIR,
            "--dws-max-run-duration",
            value,
        ]
    )

    assert args.dws_run_duration_seconds == seconds


def test_dws_provision_timeout_defaults_to_effectively_unbounded_wait():
    args, _sky_args = train_sky.parse_args(
        [
            "--config",
            TRAIN_CONFIG,
            "--workdir",
            TRAIN_WORKDIR,
            "--dws-max-run-duration",
            "2d",
        ]
    )

    assert args.dws_run_duration_seconds == 172800
    assert args.provision_timeout_seconds == train_sky.DEFAULT_PROVISION_TIMEOUT_SECONDS
    assert args.provision_timeout_seconds == 2_147_483_647


def test_dws_provision_timeout_can_be_overridden():
    args, _sky_args = train_sky.parse_args(
        [
            "--config",
            TRAIN_CONFIG,
            "--workdir",
            TRAIN_WORKDIR,
            "--dws-max-run-duration",
            "2d",
            "--dws-provision-timeout",
            "6h",
        ]
    )

    assert args.dws_run_duration_seconds == 172800
    assert args.provision_timeout_seconds == 21600


def test_current_run_config_training_shape_and_probe_schedule():
    cfg = load_config(TRAIN_CONFIG)

    assert cfg.training_max_steps == 1_000_000
    assert cfg.num_epochs == 98
    assert cfg.batch_size == 2048
    assert cfg.jax_mesh_devices == "32"
    assert cfg.learning_rate == pytest.approx(6e-4)
    assert cfg.min_learning_rate == pytest.approx(6e-6)
    assert cfg.msg_probe_every_n_steps == -1
    assert cfg.msg_probe_at_final_step is False
    assert cfg.val_every_n_steps == 10_000
    assert cfg.val_num_steps == 500


def test_dense_adamw_config_uses_32_chips_and_disables_online_probe():
    cfg = load_config(DENSE_ADAMW_CONFIG)

    assert cfg.jax_mesh_devices == "32"
    assert cfg.msg_probe_every_n_steps == -1
    assert cfg.msg_probe_at_final_step is False


@pytest.mark.parametrize(
    ("chips", "topology_name", "num_nodes"),
    [
        (4, "2x2x1", 1),
        (8, "2x2x2", 2),
        (16, "2x2x4", 4),
        (32, "2x4x4", 8),
        (64, "4x4x4", 16),
        (128, "4x4x8", 32),
        (256, "4x8x8", 64),
        (512, "8x8x8", 128),
        (1024, "8x8x16", 256),
        (2048, "8x16x16", 512),
    ],
)
def test_resolve_topology_maps_supported_chip_counts(chips, topology_name, num_nodes):
    topology = train_sky.resolve_topology(chips=chips)

    assert topology.topology == topology_name
    assert topology.total_chips == chips
    assert topology.num_nodes == num_nodes
    assert topology.chips_per_node == 4
    assert topology.instance_type == "tpu7x-standard-4t"
    assert topology.jax_mesh_devices == str(2 * chips)


def test_resolve_topology_accepts_explicit_topology_and_chip_count_shorthand():
    assert train_sky.resolve_topology("4x4x4").total_chips == 64
    assert train_sky.resolve_topology("64").topology == "4x4x4"
    assert train_sky.resolve_topology("v7x:64").topology == "4x4x4"
    assert train_sky.resolve_topology("tpu7x:64").topology == "4x4x4"


def test_resolve_topology_rejects_conflicting_topology_and_chips():
    with pytest.raises(ValueError, match="maps to topology"):
        train_sky.resolve_topology("2x4x4", chips=64)


def test_resolve_topology_rejects_unsupported_direct_tpu_size():
    with pytest.raises(ValueError, match="supported TPU7x topologies"):
        train_sky.resolve_topology("1x4x4")
    with pytest.raises(ValueError, match="supported TPU7x MIG sizes"):
        train_sky.resolve_topology(chips=12)


def test_build_task_constructs_direct_gcp_dws_resources_and_env():
    task = train_sky.build_task(
        topology=train_sky.resolve_topology("2x2x4"),
        envs={"SPECTRA_RUN_ID": "new", "SPECTRA_CONFIG_JSON": "{}"},
        infra="gcp/us-central1",
    )

    assert task["name"] == "spectra-tpu7x-mig-dws"
    assert task["workdir"] == "."
    assert task["num_nodes"] == 4
    assert task["resources"]["infra"] == "gcp/us-central1"
    assert "image_id" not in task["resources"]
    assert task["resources"]["instance_type"] == "tpu7x-standard-4t"
    assert "accelerators" not in task["resources"]
    assert "accelerator_args" not in task["resources"]
    assert "cpus" not in task["resources"]
    assert "memory" not in task["resources"]
    assert "remote_identity" not in task["config"]["gcp"]
    assert task["config"]["gcp"]["managed_instance_group"] == {
        "run_duration": 172800,
        "provision_timeout": train_sky.DEFAULT_PROVISION_TIMEOUT_SECONDS,
        "accelerator_topology": "2x2x4",
        "accelerator_topology_mode": "AUTO_CONNECT",
    }
    assert task["envs"]["SPECTRA_RUN_ID"] == "new"
    assert task["envs"]["SPECTRA_CONFIG_JSON"] == "{}"
    assert "for attempt in {1..180}" in task["setup"]
    assert "-o DPkg::Lock::Timeout=300 -o Acquire::Retries=5" in task["setup"]
    assert "apt_get update" in task["setup"]
    assert "apt_get install -y" in task["setup"]
    subprocess.run(["bash", "-n"], input=task["setup"], text=True, check=True)
    assert "libgomp1" in task["setup"]
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" in task["setup"]
    assert f"uv python install {train_sky.DEFAULT_PYTHON_VERSION}" in task["setup"]
    assert (
        f"uv sync --python {train_sky.DEFAULT_PYTHON_VERSION} "
        "--frozen --no-dev --extra tpu"
    ) in task["setup"]
    assert "python -m pip" not in task["setup"]
    assert "SPECTRA_CONFIG_JSON must be set" in task["run"]
    assert 'export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}"' in task["run"]
    assert 'echo "JAX compilation cache ${JAX_COMPILATION_CACHE_DIR}"' in task["run"]
    assert 'config["jax_compilation_cache_dir"] = os.environ["JAX_CACHE_DIR"]' in task["run"]
    assert ".venv/bin/python train.py" in task["run"]
    assert "SPECTRA_AOT" not in task["run"]
    assert "precompile" not in task["run"].lower()


def test_build_task_sets_dws_run_duration():
    task = train_sky.build_task(
        topology=train_sky.resolve_topology("2x2x4"),
        envs={"SPECTRA_RUN_ID": "new", "SPECTRA_CONFIG_JSON": "{}"},
        infra="gcp",
        dws_run_duration_seconds=21600,
    )

    assert task["resources"]["instance_type"] == "tpu7x-standard-4t"
    assert "remote_identity" not in task["config"]["gcp"]
    assert task["config"]["gcp"]["managed_instance_group"] == {
        "run_duration": 21600,
        "provision_timeout": train_sky.DEFAULT_PROVISION_TIMEOUT_SECONDS,
        "accelerator_topology": "2x2x4",
        "accelerator_topology_mode": "AUTO_CONNECT",
    }


def test_build_task_adds_optional_direct_vm_resource_constraints():
    task = train_sky.build_task(
        topology=train_sky.resolve_topology("2x2x4"),
        envs={"SPECTRA_RUN_ID": "new", "SPECTRA_CONFIG_JSON": "{}"},
        infra="gcp",
        cpus="64+",
        memory="256+",
    )

    assert task["resources"]["cpus"] == "64+"
    assert task["resources"]["memory"] == "256+"


def test_default_job_name_uses_run_id():
    assert (
        train_sky.default_job_name("16-pair-48-peaks-test-run")
        == "spectra-16-pair-48-peaks-test-run"
    )


def test_default_job_name_bounds_long_run_id():
    job_name = train_sky.default_job_name(
        "100m-muon-v6e4x8-b4096-accum4-extra-long-experiment-name-20260618-155229"
    )

    assert len(job_name) <= train_sky.MAX_SKY_JOB_NAME_LENGTH
    assert job_name.startswith("spectra-100m-muon-v6e4x8")


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
            TRAIN_CONFIG,
            "--workdir",
            f"{TRAIN_WORKDIR}-{run_id}",
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
    assert "name: spectra-tpu7x-mig-dws" in output
    assert "num_nodes: 2" in output
    assert "infra: gcp/us-central1" in output
    assert "instance_type: tpu7x-standard-4t" in output
    assert "image_id:" not in output
    assert "docker:" not in output
    assert f"uv python install {train_sky.DEFAULT_PYTHON_VERSION}" in output
    assert (
        f"uv sync --python {train_sky.DEFAULT_PYTHON_VERSION} "
        "--frozen --no-dev --extra tpu"
    ) in output
    assert "accelerators:" not in output
    assert "accelerator_args:" not in output
    assert "runtime_version:" not in output
    assert "gcp_queued_resource:" not in output
    assert "cpus:" not in output
    assert "memory:" not in output
    assert "managed_instance_group:" in output
    assert "remote_identity: SERVICE_ACCOUNT" not in output
    assert "--config gcp.remote_identity=SERVICE_ACCOUNT" in output
    assert "run_duration: 172800" in output
    assert f"provision_timeout: {train_sky.DEFAULT_PROVISION_TIMEOUT_SECONDS}" in output
    assert "accelerator_topology: 2x2x2" in output
    assert "accelerator_topology_mode: AUTO_CONNECT" in output
    assert "kubernetes:" not in output
    assert "kueue" not in output.lower()
    assert "gke" not in output.lower()
    assert f"SPECTRA_CONFIG: {TRAIN_CONFIG}" in output
    assert "SPECTRA_WORKDIR:" in output
    assert "SPECTRA_CONFIG_JSON:" in output
    assert "SPECTRA_AOT" not in output
    assert "LIBTPU_INIT_ARGS:" not in output
    assert '"jax_mesh_devices":"16"' in output
    assert '"msg_probe_at_final_step":false' in output
    assert '"jax_checkpoint_max_to_keep":5' in output
    assert f'"id":"{run_id}"' in output
    assert '"resume":"allow"' in output
    assert '"wandb_resume_from_env":false' in output
    assert "precompile" not in output.lower()
    assert "===== SkyPilot Launch Command =====" in output
    assert "sky jobs launch" in output
    assert "--detach-run --name spectra-dryrun-assets" in output
    assert "--name spectra-dryrun-assets" in output
    assert "===== SkyPilot Logs Command =====" in output
    assert "sky jobs logs -n spectra-dryrun-assets" in output


def test_dryrun_allows_64_chip_count(
    tmp_path,
    monkeypatch,
    capsys,
):
    def fail_token_lookup(_env):
        raise AssertionError("dryrun should not read token secrets")

    monkeypatch.setattr(train_sky, "read_hf_token", fail_token_lookup)
    monkeypatch.setattr(train_sky, "read_wandb_api_key", fail_token_lookup)
    run_id = "dryrun-64-chip-assets"

    train_sky.main(
        [
            "--dryrun",
            "--chips",
            "64",
            "--run-id",
            run_id,
            "--config",
            DENSE_ADAMW_CONFIG,
            "--workdir",
            f"{TRAIN_WORKDIR}-{run_id}",
            "--task-output-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert "instance_type: tpu7x-standard-4t" in output
    assert "num_nodes: 16" in output
    assert "accelerator_topology: 4x4x4" in output
    assert '"jax_mesh_devices":"128"' in output


def test_dryrun_preserves_dense_adamw_disabled_probe(
    tmp_path,
    monkeypatch,
    capsys,
):
    def fail_token_lookup(_env):
        raise AssertionError("dryrun should not read token secrets")

    monkeypatch.setattr(train_sky, "read_hf_token", fail_token_lookup)
    monkeypatch.setattr(train_sky, "read_wandb_api_key", fail_token_lookup)
    run_id = "dense-dryrun-assets"

    train_sky.main(
        [
            "--dryrun",
            "--run-id",
            run_id,
            "--config",
            DENSE_ADAMW_CONFIG,
            "--workdir",
            f"{TRAIN_WORKDIR}-{run_id}",
            "--task-output-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert '"jax_mesh_devices":"16"' in output
    assert '"msg_probe_every_n_steps":-1.0' in output
    assert '"msg_probe_at_final_step":false' in output


def test_dryrun_includes_explicit_json_overrides(
    tmp_path,
    monkeypatch,
    capsys,
):
    def fail_token_lookup(_env):
        raise AssertionError("dryrun should not read token secrets")

    monkeypatch.setattr(train_sky, "read_hf_token", fail_token_lookup)
    monkeypatch.setattr(train_sky, "read_wandb_api_key", fail_token_lookup)

    train_sky.main(
        [
            "--dryrun",
            "--run-id",
            "override-dryrun",
            "--config",
            "configs/ar_spectra_coarse_to_fine.py",
            "--workdir",
            f"{TRAIN_WORKDIR}-override-dryrun",
            "--task-output-dir",
            str(tmp_path),
            "--chips",
            "4",
            "--override",
            'ar_attention_kernel="xla"',
            "--override",
            "ar_attention_block_size=64",
        ]
    )

    output = capsys.readouterr().out
    assert "accelerator_topology: 2x2x1" in output
    assert '"ar_attention_kernel":"xla"' in output
    assert '"ar_attention_block_size":64' in output


def test_launch_failure_preserves_exit_code_without_wrapper_down(tmp_path, monkeypatch):
    command_log = _install_fake_sky(tmp_path, monkeypatch, jobs_launch_returncode=17)
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-token")

    with pytest.raises(SystemExit) as exc:
        train_sky.main(
            [
                "--run-id",
                "fake-fail",
                "--config",
                TRAIN_CONFIG,
                "--workdir",
                f"{TRAIN_WORKDIR}-fake-fail",
                "--sky-bin",
                "sky",
                "--task-output-dir",
                str(tmp_path / "tasks"),
            ]
        )

    commands = _read_fake_sky_commands(command_log)
    assert exc.value.code == 17
    assert commands == [
        [
            "jobs",
            "launch",
            "--detach-run",
            "--name",
            "spectra-fake-fail",
            "--config",
            "gcp.remote_identity=SERVICE_ACCOUNT",
            "--secret",
            "HF_TOKEN",
            "--secret",
            "HUGGING_FACE_HUB_TOKEN",
            "--secret",
            "WANDB_API_KEY",
            "--yes",
            str(tmp_path / "tasks" / "fake-fail.yaml"),
        ],
    ]


def test_successful_submit_streams_managed_job_logs(tmp_path, monkeypatch):
    command_log = _install_fake_sky(tmp_path, monkeypatch, jobs_launch_returncode=0)
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-token")

    train_sky.main(
        [
            "--run-id",
            "fake-stream",
            "--config",
            TRAIN_CONFIG,
            "--workdir",
            f"{TRAIN_WORKDIR}-fake-stream",
            "--sky-bin",
            "sky",
            "--task-output-dir",
            str(tmp_path / "tasks"),
        ]
    )

    commands = _read_fake_sky_commands(command_log)
    assert commands == [
        [
            "jobs",
            "launch",
            "--detach-run",
            "--name",
            "spectra-fake-stream",
            "--config",
            "gcp.remote_identity=SERVICE_ACCOUNT",
            "--secret",
            "HF_TOKEN",
            "--secret",
            "HUGGING_FACE_HUB_TOKEN",
            "--secret",
            "WANDB_API_KEY",
            "--yes",
            str(tmp_path / "tasks" / "fake-stream.yaml"),
        ],
        ["jobs", "logs", "-n", "spectra-fake-stream"],
    ]


def test_skypilot_managed_job_flags_pass_through_without_log_streaming(
    tmp_path,
    monkeypatch,
):
    command_log = _install_fake_sky(tmp_path, monkeypatch, jobs_launch_returncode=0)
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-token")

    train_sky.main(
        [
            "--run-id",
            "fake-managed",
            "--config",
            TRAIN_CONFIG,
            "--workdir",
            f"{TRAIN_WORKDIR}-fake-managed",
            "--sky-bin",
            "sky",
            "--task-output-dir",
            str(tmp_path / "tasks"),
            "--detach-run",
            "--job-recovery",
            "none",
        ]
    )

    commands = _read_fake_sky_commands(command_log)
    assert len(commands) == 1
    assert commands[0][:4] == ["jobs", "launch", "--detach-run", "--name"]
    assert commands[0][-3:] == [
        "--job-recovery",
        "none",
        str(tmp_path / "tasks" / "fake-managed.yaml"),
    ]


def _install_fake_sky(
    tmp_path,
    monkeypatch,
    *,
    jobs_launch_returncode: int,
    jobs_logs_returncode: int = 0,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    command_log = tmp_path / "sky-commands.jsonl"
    sky_path = bin_dir / "sky"
    sky_path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["FAKE_SKY_COMMAND_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")

if len(sys.argv) > 2 and sys.argv[1:3] == ["jobs", "launch"]:
    raise SystemExit(int(os.environ["FAKE_SKY_JOBS_LAUNCH_RETURNCODE"]))
if len(sys.argv) > 2 and sys.argv[1:3] == ["jobs", "logs"]:
    raise SystemExit(int(os.environ["FAKE_SKY_JOBS_LOGS_RETURNCODE"]))
raise SystemExit(2)
"""
    )
    sky_path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_SKY_COMMAND_LOG", str(command_log))
    monkeypatch.setenv("FAKE_SKY_JOBS_LAUNCH_RETURNCODE", str(jobs_launch_returncode))
    monkeypatch.setenv("FAKE_SKY_JOBS_LOGS_RETURNCODE", str(jobs_logs_returncode))
    return command_log


def _read_fake_sky_commands(command_log: Path):
    return [json.loads(line) for line in command_log.read_text().splitlines()]
