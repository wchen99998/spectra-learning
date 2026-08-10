from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import logging
import netrc
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from spectra_learning.config import config_to_dict, load_config
from spectra_learning.training.routing import resolve_training_route


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = "metal-repeater-411410"
DEFAULT_REGION = "us-central1"
DEFAULT_INFRA = f"gcp/{DEFAULT_REGION}"
DEFAULT_TASK_NAME = "spectra-tpu7x-mig-dws"
V6E_TASK_NAME = "spectra-v6e-mig-dws"
V6E_VM_IMAGE_ID = (
    "projects/ubuntu-os-accelerator-images/global/images/"
    "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-v20260623"
)
DEFAULT_PYTHON_VERSION = "3.12.11"
DEFAULT_SKY_BIN = "/home/wuhao/skypilot/.venv/bin/sky"
TPU_V6E_TOPOLOGY_BY_CHIPS = {
    4: "2x2",
    8: "2x4",
    16: "4x4",
    32: "4x8",
    64: "8x8",
    128: "8x16",
    256: "16x16",
}
TPU_V6E_MACHINE_LAYOUT_BY_CHIPS = {
    4: ("ct6e-standard-4t", 1),
    8: ("ct6e-standard-8t", 1),
    16: ("ct6e-standard-4t", 4),
    32: ("ct6e-standard-4t", 8),
    64: ("ct6e-standard-4t", 16),
    128: ("ct6e-standard-4t", 32),
    256: ("ct6e-standard-4t", 64),
}
TPU7X_TOPOLOGY_BY_CHIPS = {
    4: "2x2x1",
    8: "2x2x2",
    16: "2x2x4",
    32: "2x4x4",
    64: "4x4x4",
    128: "4x4x8",
    256: "4x8x8",
    512: "8x8x8",
    1024: "8x8x16",
    2048: "8x16x16",
}
SUPPORTED_TPU_V6E_CHIPS = set(TPU_V6E_TOPOLOGY_BY_CHIPS)
SUPPORTED_TPU7X_CHIPS = set(TPU7X_TOPOLOGY_BY_CHIPS)
DEFAULT_CHIPS = 8
DEFAULT_TOPOLOGY = TPU7X_TOPOLOGY_BY_CHIPS[DEFAULT_CHIPS]
DEFAULT_CHIPS_PER_NODE = 4
DEFAULT_INSTANCE_TYPE = "tpu7x-standard-4t"
DEFAULT_DWS_RUN_DURATION_SECONDS = 172800
MIN_DWS_RUN_DURATION_SECONDS = 600
MAX_DWS_RUN_DURATION_SECONDS = 604800
DEFAULT_PROVISION_TIMEOUT_SECONDS = 2_147_483_647
MAX_SKY_JOB_NAME_LENGTH = 63
TASK_SETUP = """\
set -euo pipefail
apt_get() {
  local attempt
  for attempt in {1..180}; do
    if sudo env DEBIAN_FRONTEND=noninteractive apt-get \\
      -o DPkg::Lock::Timeout=300 -o Acquire::Retries=5 "$@"; then
      return
    fi
    echo "apt-get failed (attempt ${attempt}/180); retrying in 10 seconds" >&2
    sleep 10
  done
  return 1
}
apt_get update
apt_get install -y \\
  build-essential \\
  ca-certificates \\
  curl \\
  git \\
  libgomp1 \\
  libsm6 \\
  libxext6 \\
  libxrender1 \\
  pkg-config \\
  xz-utils
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version
uv python install __SPECTRA_PYTHON_VERSION__
uv sync --python __SPECTRA_PYTHON_VERSION__ --frozen --no-dev --extra tpu
.venv/bin/python --version
.venv/bin/python - <<'PY'
import importlib.metadata as md

print("jax", md.version("jax"))
print("jaxlib", md.version("jaxlib"))
print("libtpu", md.version("libtpu"))
print("wandb", md.version("wandb"))
PY
""".replace("__SPECTRA_PYTHON_VERSION__", DEFAULT_PYTHON_VERSION)
TASK_RUN = """\
set -euo pipefail
: "${SPECTRA_CONFIG:?SPECTRA_CONFIG must be set}"
: "${SPECTRA_WORKDIR:?SPECTRA_WORKDIR must be set}"
: "${SPECTRA_RUN_ID:?SPECTRA_RUN_ID must be set}"
: "${SPECTRA_JAX_CACHE_DIR:?SPECTRA_JAX_CACHE_DIR must be set}"
: "${SPECTRA_CONFIG_JSON:?SPECTRA_CONFIG_JSON must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set via --secret}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set via --secret}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"

rm -f "${HOME}/.config/gcloud/application_default_credentials.json"
unset GOOGLE_APPLICATION_CREDENTIALS
unset CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE
JAX_CACHE_DIR="$(.venv/bin/python - <<'PY'
import os
from pathlib import Path

print(Path(os.environ["SPECTRA_JAX_CACHE_DIR"]).expanduser().resolve())
PY
)"
export JAX_CACHE_DIR
export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "${HF_HOME}" "${WANDB_DIR}" "${JAX_CACHE_DIR}"

COORDINATOR_IP="$(printf '%s\\n' "${SKYPILOT_NODE_IPS}" | sed -n '1p')"
TPU_WORKER_HOSTNAMES="$(printf '%s\\n' "${SKYPILOT_NODE_IPS}" | paste -sd, -)"
TPU_PROCESS_ADDRESSES="$(printf '%s\\n' "${SKYPILOT_NODE_IPS}" | sed 's/$/:8471/' | paste -sd, -)"
export JAX_DISTRIBUTED_INITIALIZE=1
export JAX_COORDINATOR_ADDRESS="${COORDINATOR_IP}:12345"
export JAX_NUM_PROCESSES="${SKYPILOT_NUM_NODES}"
export JAX_PROCESS_ID="${SKYPILOT_NODE_RANK}"
export JAX_PLATFORMS=tpu
export TPU_WORKER_ID="${SKYPILOT_NODE_RANK}"
export TPU_WORKER_HOSTNAMES
export TPU_PROCESS_ADDRESSES
export TPU_PROCESS_PORT=8471
export TF_CPP_MIN_LOG_LEVEL=0

METRICS_JSON="${SPECTRA_WORKDIR%/}/${SPECTRA_METRICS_JSON#/}"
CONFIG_JSON="$(.venv/bin/python - <<'PY'
import json
import os

config = json.loads(os.environ["SPECTRA_CONFIG_JSON"])
config["jax_compilation_cache_dir"] = os.environ["JAX_CACHE_DIR"]
print(json.dumps(config, sort_keys=True, separators=(",", ":")))
PY
)"

echo "SkyPilot node rank ${SKYPILOT_NODE_RANK}/${SKYPILOT_NUM_NODES}"
echo "Coordinator ${JAX_COORDINATOR_ADDRESS}"
echo "TPU worker ${TPU_WORKER_ID}: ${TPU_WORKER_HOSTNAMES}"
echo "Workdir ${SPECTRA_WORKDIR}"
echo "JAX cache ${JAX_CACHE_DIR}"
echo "JAX compilation cache ${JAX_COMPILATION_CACHE_DIR}"
.venv/bin/python train.py \\
  --config "${SPECTRA_CONFIG}" \\
  --workdir "${SPECTRA_WORKDIR}" \\
  --overrides-json "${CONFIG_JSON}" \\
  --metrics-json "${METRICS_JSON}"
"""


class LiteralString(str):
    pass


def _literal_string_representer(
    dumper: yaml.SafeDumper,
    data: LiteralString,
) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.SafeDumper.add_representer(LiteralString, _literal_string_representer)


@dataclass(frozen=True)
class TopologySpec:
    generation: str
    topology: str
    total_chips: int
    num_nodes: int
    chips_per_node: int
    instance_type: str

    @property
    def jax_mesh_devices(self) -> str:
        devices = self.total_chips if self.generation == "v6e" else 2 * self.total_chips
        return str(devices)

    @property
    def slug(self) -> str:
        return f"{self.generation}{self.topology}"


def parse_duration_seconds(value: str) -> int:
    text = value.strip().lower()
    unit = text[-1] if text[-1].isalpha() else "s"
    amount = text[:-1] if unit != "s" else text.removesuffix("s")
    if unit not in {"s", "m", "h", "d"}:
        raise argparse.ArgumentTypeError("duration unit must be one of s, m, h, d")
    return int(amount) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def parse_dws_run_duration_seconds(value: str) -> int:
    seconds = parse_duration_seconds(value)
    if not MIN_DWS_RUN_DURATION_SECONDS <= seconds <= MAX_DWS_RUN_DURATION_SECONDS:
        raise argparse.ArgumentTypeError(
            "DWS run duration must be between 600 seconds and 7 days"
        )
    return seconds


def parse_provision_timeout_seconds(value: str) -> int:
    seconds = parse_duration_seconds(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("provision timeout must be positive")
    return seconds


def topology_for_chips(chips: int, generation: str = "v7x") -> str:
    topologies = (
        TPU_V6E_TOPOLOGY_BY_CHIPS if generation == "v6e" else TPU7X_TOPOLOGY_BY_CHIPS
    )
    generation_name = "TPU v6e" if generation == "v6e" else "TPU7x"
    try:
        return topologies[int(chips)]
    except KeyError:
        raise ValueError(
            f"chips={chips} is unsupported; supported {generation_name} MIG sizes are "
            f"{sorted(topologies)}"
        ) from None


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Submit Spectra JAX training to SkyPilot GCP DWS TPU VMs."
    )
    parser.add_argument("--config", required=True, help="Training config path.")
    parser.add_argument(
        "--workdir",
        required=True,
        help="Training output directory, usually a gs:// checkpoint URI.",
    )
    parser.add_argument(
        "--topology",
        default="",
        help=(
            "TPU topology, such as v6e 4x8 or v7x 2x4x4, or a chip-count "
            "shorthand such as 32. Defaults from --chips."
        ),
    )
    parser.add_argument(
        "--tpu-generation",
        choices=("v6e", "v7x"),
        default="v7x",
    )
    parser.add_argument(
        "--chips",
        "--chip-count",
        dest="chips",
        type=int,
        choices=sorted(SUPPORTED_TPU_V6E_CHIPS | SUPPORTED_TPU7X_CHIPS),
        default=None,
        help="TPU chip count. Maps to a topology for --tpu-generation.",
    )
    parser.add_argument(
        "--job-name",
        default="",
        help="SkyPilot managed job name. Defaults to a run-specific name.",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Top-level config override as KEY=JSON_VALUE. May be repeated.",
    )
    parser.add_argument("--jax-cache-dir", default="")
    parser.add_argument("--yes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", "--dryrun", dest="dry_run", action="store_true")
    parser.add_argument(
        "--stream-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream managed-job logs after submitting the SkyPilot job.",
    )
    parser.add_argument(
        "--detach-run",
        dest="stream_logs",
        action="store_false",
        help="Submit the managed job and return without streaming logs.",
    )
    parser.add_argument("--task-output-dir", default="tmp/skypilot_tasks")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--sky-bin", default=DEFAULT_SKY_BIN)
    parser.add_argument("--cpus", default="")
    parser.add_argument("--memory", default="")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--region",
        default="",
        help=(
            f"GCP region for SkyPilot resources. Defaults to {DEFAULT_REGION}. "
            "Equivalent to --infra gcp/<region>."
        ),
    )
    parser.add_argument("--infra", default="")
    parser.add_argument("--metrics-json", default="metrics/final.json")
    parser.add_argument("--queue-tag", default="flex-start")
    parser.add_argument(
        "--provision-timeout-seconds",
        "--dws-provision-timeout",
        "--flex-start-provision-timeout",
        dest="provision_timeout_seconds",
        type=parse_provision_timeout_seconds,
        default=DEFAULT_PROVISION_TIMEOUT_SECONDS,
        help=(
            "SkyPilot GCP DWS provisioning wait. Defaults to an effectively "
            "unbounded wait; accepts seconds or s/m/h/d suffixes."
        ),
    )
    parser.add_argument(
        "--dws-run-duration",
        "--flex-start-run-duration",
        "--flex-start-max-run-duration",
        "--dws-max-run-duration",
        dest="dws_run_duration_seconds",
        type=parse_dws_run_duration_seconds,
        default=DEFAULT_DWS_RUN_DURATION_SECONDS,
        help=(
            "DWS Flex-start VM runtime. Accepts seconds or s/m/h/d suffixes; "
            "GCP requires 600-604800 seconds."
        ),
    )
    args, sky_args = parser.parse_known_args(argv)
    if not args.config.strip():
        parser.error("--config cannot be empty")
    if not args.workdir.strip():
        parser.error("--workdir cannot be empty")
    args.region, args.infra = resolve_region_and_infra(
        region=args.region,
        infra=args.infra,
        parser=parser,
    )
    if sky_args and sky_args[0] == "--":
        sky_args = sky_args[1:]
    return args, sky_args


def resolve_region_and_infra(
    *,
    region: str,
    infra: str,
    parser: argparse.ArgumentParser,
) -> tuple[str, str]:
    raw_region = region.strip()
    region = normalize_gcp_region(raw_region) if raw_region else ""
    if raw_region and not region:
        parser.error("--region cannot be empty")
    infra = infra.strip()
    if not region and not infra:
        return DEFAULT_REGION, DEFAULT_INFRA
    if region and not infra:
        return region, f"gcp/{region}"
    if not region:
        return gcp_region_from_infra(infra), infra
    expected_infra = f"gcp/{region}"
    if infra != expected_infra:
        parser.error(f"--region {region} conflicts with --infra {infra}")
    return region, infra


def normalize_gcp_region(region: str) -> str:
    region = region.strip()
    prefix = "gcp/"
    if region.startswith(prefix):
        region = region[len(prefix) :].strip()
    return region


def resolve_topology(
    topology: str = "",
    *,
    chips: int | None = None,
    generation: str = "v7x",
) -> TopologySpec:
    topologies = (
        TPU_V6E_TOPOLOGY_BY_CHIPS if generation == "v6e" else TPU7X_TOPOLOGY_BY_CHIPS
    )
    prefixes = (
        ("tpu-v6e:", "tpuv6e:", "v6e:")
        if generation == "v6e"
        else ("tpu7x:", "tpu-v7x:", "v7x:")
    )
    normalized = topology.lower().strip()
    for prefix in prefixes:
        normalized = normalized.removeprefix(prefix)
    if normalized:
        if normalized.isdigit():
            chip_count = int(normalized)
            normalized = topology_for_chips(chip_count, generation)
        elif chips is not None:
            expected = topology_for_chips(chips, generation)
            if normalized != expected:
                raise ValueError(
                    f"--chips={chips} maps to topology {expected}, but "
                    f"--topology={topology!r} was also provided"
                )
    else:
        normalized = topology_for_chips(
            DEFAULT_CHIPS if chips is None else chips,
            generation,
        )
    parts = normalized.split("x")
    dimensions = 2 if generation == "v6e" else 3
    example = "4x8" if generation == "v6e" else "2x2x4"
    if len(parts) != dimensions or not all(part.isdigit() for part in parts):
        raise ValueError(f"topology must look like {example}; got {topology!r}")
    dims = tuple(int(part) for part in parts)
    total_chips = 1
    for dimension in dims:
        total_chips *= dimension
    if total_chips <= 0:
        raise ValueError(f"topology must contain positive dimensions; got {topology!r}")
    expected_topology = topologies.get(total_chips)
    if expected_topology != normalized:
        generation_name = "TPU v6e" if generation == "v6e" else "TPU7x"
        raise ValueError(
            f"topology {normalized!r} is unsupported; supported {generation_name} "
            f"topologies are {list(topologies.values())}"
        )
    if chips is not None and total_chips != chips:
        raise ValueError(
            f"--chips={chips} conflicts with topology {normalized!r}, which "
            f"has {total_chips} chips"
        )
    if generation == "v6e":
        instance_type, num_nodes = TPU_V6E_MACHINE_LAYOUT_BY_CHIPS[total_chips]
    else:
        instance_type = DEFAULT_INSTANCE_TYPE
        num_nodes = total_chips // DEFAULT_CHIPS_PER_NODE
    return TopologySpec(
        generation=generation,
        topology=normalized,
        total_chips=total_chips,
        num_nodes=num_nodes,
        chips_per_node=total_chips // num_nodes,
        instance_type=instance_type,
    )


def read_hf_token(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    path = Path.home() / ".cache" / "huggingface" / "token"
    if path.is_file():
        return path.read_text().strip()
    raise SystemExit("HF_TOKEN is missing; set it or run huggingface-cli login first.")


def read_wandb_api_key(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    token = env.get("WANDB_API_KEY")
    if token:
        return token.strip()
    for machine in ("api.wandb.ai", "wandb.ai"):
        try:
            auth = netrc.netrc().authenticators(machine)
        except (FileNotFoundError, netrc.NetrcParseError):
            auth = None
        if auth and auth[2]:
            return auth[2].strip()
    settings = Path.home() / ".config" / "wandb" / "settings"
    if settings.is_file():
        parser = configparser.ConfigParser()
        try:
            parser.read(settings)
            for section in parser.sections():
                if parser.has_option(section, "api_key"):
                    return parser.get(section, "api_key").strip()
        except configparser.Error:
            pass
        for line in settings.read_text().splitlines():
            if line.strip().startswith("api_key") and "=" in line:
                return line.split("=", 1)[1].strip()
    raise SystemExit("WANDB_API_KEY is missing; set it or run wandb login first.")


def json_compact(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_config_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        key, value = item.split("=", 1)
        overrides[key] = json.loads(value)
    return overrides


def config_slug(config_path: str) -> str:
    return name_slug(Path(config_path).stem)


def build_train_overrides(
    *,
    run_id: str,
    jax_mesh_devices: str,
    jax_cache_dir: str,
    queue_tag: str,
    experiment_tag: str,
    tpu_generation: str = "v7x",
) -> dict[str, Any]:
    return {
        "jax_distributed_initialize": True,
        "jax_mesh_devices": str(jax_mesh_devices),
        "jax_compilation_cache_dir": jax_cache_dir,
        "jax_enable_compilation_cache": True,
        "jax_persistent_cache_min_compile_time_secs": 0.0,
        "jax_persistent_cache_min_entry_size_bytes": 0,
        "jax_enable_async_checkpointing": False,
        "wandb_resume_from_env": False,
        "wandb_kwargs": {
            "id": run_id,
            "name": run_id,
            "resume": "allow",
            "tags": [
                "skypilot",
                "gcp",
                "dws",
                queue_tag,
                f"tpu-{tpu_generation}",
                experiment_tag,
            ],
            "notes": (
                f"SkyPilot GCP DWS TPU {tpu_generation} run launched by train_sky.py."
            ),
        },
    }


def build_task(
    *,
    topology: TopologySpec,
    envs: dict[str, str],
    infra: str,
    task_name: str = DEFAULT_TASK_NAME,
    cpus: str = "",
    memory: str = "",
    dws_run_duration_seconds: int = DEFAULT_DWS_RUN_DURATION_SECONDS,
    provision_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    if provision_timeout_seconds is None:
        provision_timeout_seconds = DEFAULT_PROVISION_TIMEOUT_SECONDS
    resources: dict[str, Any] = {
        "infra": infra,
        "instance_type": topology.instance_type,
    }
    if topology.generation == "v6e":
        resources["image_id"] = {
            gcp_region_from_infra(infra): V6E_VM_IMAGE_ID,
        }
    if cpus:
        resources["cpus"] = cpus
    if memory:
        resources["memory"] = memory
    task: dict[str, Any] = {
        "name": task_name,
        "workdir": ".",
        "num_nodes": topology.num_nodes,
        "resources": resources,
        "envs": dict(envs),
        "setup": LiteralString(TASK_SETUP),
        "run": LiteralString(TASK_RUN),
        "config": {
            "gcp": {
                "managed_instance_group": {
                    "run_duration": int(dws_run_duration_seconds),
                    "provision_timeout": int(provision_timeout_seconds),
                    "accelerator_topology": topology.topology,
                    "accelerator_topology_mode": "AUTO_CONNECT",
                },
            },
        },
    }
    return task


def gcp_region_from_infra(infra: str) -> str:
    prefix = "gcp/"
    if infra.startswith(prefix) and infra[len(prefix) :].strip():
        return infra[len(prefix) :].strip()
    return DEFAULT_REGION


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    logging.info("Running: %s", shlex.join(cmd))
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, check=False)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def render_task_yaml(task: dict[str, Any]) -> str:
    return yaml.safe_dump(task, sort_keys=False)


def write_task_file(task: dict[str, Any], output_dir: Path, run_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_id}.yaml"
    path.write_text(render_task_yaml(task))
    return path


def print_dry_run_assets(
    *,
    task_path: Path,
    task: dict[str, Any],
    config_json: str,
    launch_command: list[str],
    logs_command: list[str] | None,
) -> None:
    print("===== SkyPilot Task Path =====")
    print(task_path)
    print()
    print("===== SkyPilot Task YAML =====")
    print(render_task_yaml(task).rstrip())
    print()
    print("===== Resolved Config JSON =====")
    print(config_json)
    print()
    print("===== SkyPilot Launch Command =====")
    print(shlex.join(launch_command))
    if logs_command is not None:
        print()
        print("===== SkyPilot Logs Command =====")
        print(shlex.join(logs_command))


def name_slug(value: str) -> str:
    chars = []
    previous_dash = False
    for char in value.lower():
        is_ascii_alnum = ("a" <= char <= "z") or ("0" <= char <= "9")
        if is_ascii_alnum:
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-")


def default_job_name(run_id: str) -> str:
    prefix = "spectra"
    suffix = name_slug(run_id)
    job_name = f"{prefix}-{suffix}"
    if len(job_name) <= MAX_SKY_JOB_NAME_LENGTH:
        return job_name

    digest = hashlib.sha1(suffix.encode()).hexdigest()[:8]
    suffix_length = MAX_SKY_JOB_NAME_LENGTH - len(prefix) - len(digest) - 2
    shortened_suffix = suffix[:suffix_length].rstrip("-")
    return f"{prefix}-{shortened_suffix}-{digest}"


def build_sky_jobs_launch_command(
    *,
    sky_bin: str,
    job_name: str,
    task_path: Path,
    yes: bool,
    sky_args: list[str],
) -> list[str]:
    cmd = [sky_bin, "jobs", "launch", "--detach-run", "--name", job_name]
    cmd.extend(["--config", "gcp.remote_identity=SERVICE_ACCOUNT"])
    for secret in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY"):
        cmd.extend(["--secret", secret])
    if yes:
        cmd.append("--yes")
    cmd.extend(sky_args)
    cmd.append(str(task_path))
    return cmd


def build_sky_jobs_logs_command(*, sky_bin: str, job_name: str) -> list[str]:
    return [sky_bin, "jobs", "logs", "-n", job_name]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args, sky_args = parse_args(argv)
    topology = resolve_topology(
        args.topology,
        chips=args.chips,
        generation=args.tpu_generation,
    )
    user_overrides = parse_config_overrides(args.override)
    config = load_config(args.config, user_overrides)
    _task, backend = resolve_training_route(config)
    if backend != "jax":
        raise ValueError("train_sky.py requires device_backend='jax'")
    batch_size = int(config.batch_size)
    grad_accum = int(config.gradient_accumulation_steps)
    experiment_slug = config_slug(args.config)
    experiment_tag = experiment_slug.replace("-", "_")
    run_id = args.run_id or (
        f"{experiment_slug}-{topology.slug}-b{batch_size}-accum{grad_accum}-"
        f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}")
    workdir = args.workdir.rstrip("/")
    job_name = args.job_name or default_job_name(run_id)
    cache_key = f"{experiment_tag}_{topology.slug}_b{batch_size}_accum{grad_accum}"
    jax_cache_dir = args.jax_cache_dir or f"/tmp/spectra-jax-cache/{cache_key}"
    overrides = build_train_overrides(
        run_id=run_id,
        jax_mesh_devices=topology.jax_mesh_devices,
        jax_cache_dir=jax_cache_dir,
        queue_tag=args.queue_tag,
        experiment_tag=experiment_tag,
        tpu_generation=topology.generation,
    )
    overrides.update(user_overrides)
    config = load_config(args.config, overrides)
    config_json = json_compact(config_to_dict(config))

    logging.info(
        "Topology: topology=%s hosts=%d chips=%d instance_type=%s",
        topology.topology,
        topology.num_nodes,
        topology.total_chips,
        topology.instance_type,
    )
    logging.warning(
        "GCP DWS uses SkyPilot gcp.managed_instance_group with Flex-start "
        "MIGs. Use the vendored SkyPilot build at %s; it contains the TPU "
        "Compute Engine workload-policy support required by this launcher.",
        args.sky_bin,
    )
    logging.info(
        "Training shape: mesh=%s batch=%d grad_accum=%d steps=%d",
        config.jax_mesh_devices,
        config.batch_size,
        config.gradient_accumulation_steps,
        config.training_max_steps,
    )
    logging.info("Run ID: %s", run_id)
    logging.info("Workdir: %s", workdir)

    launch_env = dict(os.environ)
    if args.dry_run:
        logging.info("Dry run requested; skipping HF_TOKEN/WANDB_API_KEY lookup.")
    else:
        hf_token = read_hf_token(launch_env)
        wandb_key = read_wandb_api_key(launch_env)
        launch_env["HF_TOKEN"] = hf_token
        launch_env["HUGGING_FACE_HUB_TOKEN"] = launch_env.get(
            "HUGGING_FACE_HUB_TOKEN",
            hf_token,
        )
        launch_env["WANDB_API_KEY"] = wandb_key
        logging.info("Loaded HF_TOKEN and WANDB_API_KEY for SkyPilot secrets.")

    task_envs = {
        "SPECTRA_CONFIG": args.config,
        "SPECTRA_RUN_ID": run_id,
        "SPECTRA_WORKDIR": workdir,
        "SPECTRA_METRICS_JSON": args.metrics_json,
        "SPECTRA_JAX_CACHE_DIR": jax_cache_dir,
        "SPECTRA_CONFIG_JSON": config_json,
        "JAX_INITIALIZATION_TIMEOUT": "3600",
        "HF_HOME": "/tmp/huggingface",
        "WANDB_DIR": "/tmp/wandb",
        "UV_CACHE_DIR": "/tmp/uv-cache",
        "UV_LINK_MODE": "copy",
        "GOOGLE_CLOUD_PROJECT": args.project,
        "GCLOUD_PROJECT": args.project,
        "CLOUDSDK_CORE_PROJECT": args.project,
    }
    task = build_task(
        topology=topology,
        envs=task_envs,
        infra=args.infra,
        task_name=(
            V6E_TASK_NAME
            if topology.generation == "v6e" and args.task_name == DEFAULT_TASK_NAME
            else args.task_name
        ),
        cpus=args.cpus,
        memory=args.memory,
        dws_run_duration_seconds=args.dws_run_duration_seconds,
        provision_timeout_seconds=args.provision_timeout_seconds,
    )
    task_path = write_task_file(task, REPO_ROOT / args.task_output_dir, run_id)
    logging.info("Wrote SkyPilot task: %s", task_path)

    sky_bin = args.sky_bin
    if not args.dry_run:
        if os.sep in sky_bin:
            if not Path(sky_bin).is_file():
                raise SystemExit(f"SkyPilot executable not found: {sky_bin}")
        elif shutil.which(sky_bin) is None:
            raise SystemExit(f"SkyPilot executable not found on PATH: {sky_bin}")
    launch_cmd = build_sky_jobs_launch_command(
        sky_bin=sky_bin,
        job_name=job_name,
        task_path=task_path,
        yes=args.yes,
        sky_args=sky_args,
    )
    logs_cmd = (
        build_sky_jobs_logs_command(sky_bin=sky_bin, job_name=job_name)
        if args.stream_logs
        else None
    )
    logging.info("SkyPilot launch command: %s", shlex.join(launch_cmd))
    if logs_cmd is not None:
        logging.info("SkyPilot logs command: %s", shlex.join(logs_cmd))
    if args.dry_run:
        print_dry_run_assets(
            task_path=task_path,
            task=task,
            config_json=config_json,
            launch_command=launch_cmd,
            logs_command=logs_cmd,
        )
        logging.info("Dry run requested; not launching SkyPilot.")
        return
    run_command(launch_cmd, cwd=REPO_ROOT, env=launch_env)
    if logs_cmd is not None:
        run_command(logs_cmd, cwd=REPO_ROOT, env=launch_env)


if __name__ == "__main__":
    main()
