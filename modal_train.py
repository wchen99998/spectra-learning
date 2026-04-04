"""Train on Modal with persistent storage for data and checkpoints.

Usage:
    # Single run
    modal run modal_train.py
    modal run modal_train.py --config configs/gems_small.py --gpu H100

    # Parallel sweep (launches all experiments concurrently)
    modal run modal_train.py --sweep sweep_optim

Setup:
    1. modal setup
    2. modal secret create wandb-secret WANDB_API_KEY=<your-key>
    3. modal run modal_train.py
"""

import json
from pathlib import Path

import modal

MINUTES = 60
HOURS = 60 * MINUTES
DEFAULT_GPU = "H100"
PROJECT_ROOT = "/root/spectra-learning"

# ---------------------------------------------------------------------------
# Persistent volume — data + experiments survive across runs
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("spectra-volume", create_if_missing=True)
volume_path = Path("/vol")

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.11.0",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .uv_pip_install(
        "tensorflow-cpu==2.19.0",
        "lightning==2.5.5",
        "ml-collections>=1.1.0",
        "rdkit>=2025.3.3",
        "scikit-learn>=1.8.0",
        "huggingface-hub>=0.33.2",
        "h5py>=3.11.0",
        "tqdm>=4.66.4",
        "wandb==0.23.1",
        "numpy",
        "matplotlib",
    )
    .run_commands(
        "pip install --no-build-isolation gram-newton-schulz@git+https://github.com/Dao-AILab/gram-newton-schulz"
    )
    .env(
        {
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "TF_ENABLE_ONEDNN_OPTS": "0",
        }
    )
)

# Add only the source directories needed for training
local = Path(__file__).parent
image = (
    base_image
    .add_local_file(local / "train.py", remote_path=f"{PROJECT_ROOT}/train.py")
    .add_local_file(local / "input_pipeline.py", remote_path=f"{PROJECT_ROOT}/input_pipeline.py")
    .add_local_dir(local / "configs", remote_path=f"{PROJECT_ROOT}/configs")
    .add_local_dir(local / "models", remote_path=f"{PROJECT_ROOT}/models")
    .add_local_dir(local / "networks", remote_path=f"{PROJECT_ROOT}/networks")
    .add_local_dir(local / "utils", remote_path=f"{PROJECT_ROOT}/utils")
    .add_local_dir(local / "kernels", remote_path=f"{PROJECT_ROOT}/kernels")
    .add_local_dir(local / "optimizers", remote_path=f"{PROJECT_ROOT}/optimizers")
)

app = modal.App("spectra-training", image=image)


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------
SWEEPS: dict[str, list[dict]] = {
    # Optimizer sweep: downstream probe perf degrades during training.
    # Hypotheses: LR too aggressive, WD too low, EMA teacher tracks student
    # too closely (collapse), min_lr floor keeps perturbing late.
    "sweep_optim": [
        # 0) baseline — current settings (ema=0.995, update_every=2)
        {},
        # 1) lower peak LR — less aggressive updates preserve representations
        {"learning_rate": 2e-4},
        # 2) higher weight decay — stronger regularization against collapse
        {"weight_decay": 0.1},
        # 3) slower EMA teacher — teacher lags more, provides stabler targets
        {"teacher_ema_decay": 0.999, "teacher_ema_decay_start": 0.996},
        # 4) much slower EMA + update every step — maximally stable teacher
        {
            "teacher_ema_decay": 0.9996,
            "teacher_ema_decay_start": 0.999,
            "teacher_ema_update_every": 1,
        },
        # 5) lower LR floor + more warmup — less perturbation early and late
        {"min_learning_rate": 1e-5, "warmup_steps": 40_000},
        # 6) lower LR + higher WD — combined conservative regularization
        {"learning_rate": 2e-4, "weight_decay": 0.1},
        # 7) full stability: slow EMA + lower LR + lower floor + more warmup
        {
            "learning_rate": 2e-4,
            "teacher_ema_decay": 0.9996,
            "teacher_ema_decay_start": 0.999,
            "teacher_ema_update_every": 1,
            "min_learning_rate": 1e-5,
            "warmup_steps": 40_000,
        },
    ],
}


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={volume_path: volume},
    gpu=DEFAULT_GPU,
    timeout=3 * HOURS,
    secrets=[modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])],
)
def train(
    config_path: str = "configs/gems_small.py",
    overrides_json: str = "{}",
):
    import logging
    import os
    import sys

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    logging.basicConfig(level=logging.INFO)

    from train import train_and_evaluate
    from utils.training import auto_run_name, load_config

    config = load_config(config_path)

    # Apply experiment overrides
    overrides = json.loads(overrides_json)
    config.update(overrides)

    # Point data at the persistent volume
    config.tfrecord_dir = str(volume_path / "data" / "gems_peaklist_tfrecord_alpha")

    # Muon NS kernels require SM90+ (H100/B200); disable on older GPUs
    import torch
    sm_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    if sm_major < 9:
        config.muon_ns_use_kernels = False

    run_name = auto_run_name(config)
    workdir = volume_path / "experiments" / run_name
    workdir.mkdir(parents=True, exist_ok=True)

    logging.info("Run: %s", run_name)
    if overrides:
        logging.info("Overrides: %s", overrides)

    results = train_and_evaluate(config, workdir=workdir)

    volume.commit()
    logging.info("Training complete. Results: %s", results)
    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    config: str = "configs/gems_small.py",
    sweep: str = "",
    overrides: str = "{}",
):
    if sweep:
        experiments = SWEEPS[sweep]
        print(f"Launching {len(experiments)} experiments in parallel ({sweep}):")
        for i, exp in enumerate(experiments):
            print(f"  [{i}] {exp or '(baseline)'}")
        handles = []
        for exp in experiments:
            merged = {**json.loads(overrides), **exp}
            handles.append(train.spawn(config_path=config, overrides_json=json.dumps(merged)))
        for i, handle in enumerate(handles):
            result = handle.get()
            print(f"[{i}] done: {result}")
    else:
        train.remote(config_path=config, overrides_json=overrides)
