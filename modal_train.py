"""Train on Modal with persistent storage for data and checkpoints.

Usage:
    # Single run
    modal run modal_train.py
    modal run modal_train.py --config configs/gems_small.py --gpu H100

    # Parallel sweep (launches all experiments concurrently)
    modal run modal_train.py --sweep sweep_optim
    modal run modal_train.py --sweep sweep_optim_refine
    modal run modal_train.py --sweep sweep_sigreg_compare
    modal run modal_train.py --sweep sweep_10m_sigreg_ema_ablation
    modal run modal_train.py --sweep sweep_10m_noema_sigreg_log_lambda
    modal run modal_train.py --sweep sweep_10m_noema_sigreg_high_lambda
    modal run modal_train.py --sweep sweep_10m_ema_stopgrad
    modal run modal_train.py --sweep sweep_10m_ema_stopgrad_warmup_update

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
MAX_SWEEP_CONCURRENCY = 10

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
        "pandas>=2.0.0",
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
BEST_SWEEP_OPTIM = {
    "jepa_target_normalization": "zscore",
    "teacher_ema_decay": 0.9996,
    "teacher_ema_decay_start": 0.999,
    "teacher_ema_update_every": 1,
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "representation_regularizer": "none",
    "sigreg_lambda": 0.02,
}

TEN_M_BACKBONE = {
    "model_dim": 256,
    "encoder_num_layers": 8,
    "encoder_num_heads": 8,
    "encoder_num_kv_heads": 8,
    "feature_mlp_hidden_dim": 512,
    "predictor_dim": 128,
    "masked_latent_predictor_num_layers": 4,
    "masked_latent_predictor_num_heads": 8,
    "jepa_target_layers": [1, 3, 5, 8],
    "msg_probe_pma_num_heads": 8,
}

TEN_M_BEST_SWEEP_OPTIM = {
    **BEST_SWEEP_OPTIM,
    **TEN_M_BACKBONE,
}

NOEMA_SIGREG_LOG_LAMBDAS = [10.0 ** exp for exp in (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0)]
NOEMA_SIGREG_HIGH_LAMBDAS = [
    ("2e-02", 0.02),
    ("2e-01", 0.2),
    ("2e00", 2.0),
    ("1e01", 10.0),
    ("5e01", 50.0),
]
SIGREG_SAMPLE_SCALE_TAG = "sigcfscale"
EMA_STOPGRAD_SWEEP_TAG = "emastop"
EMA_STOPGRAD_RECIPES = [
    (
        "d995-s990-w500k-u2",
        {
            "teacher_ema_decay": 0.995,
            "teacher_ema_decay_start": 0.99,
            "teacher_ema_decay_warmup_steps": 500_000,
            "teacher_ema_update_every": 2,
        },
    ),
    (
        "d999-s996-w100k-u1",
        {
            "teacher_ema_decay": 0.999,
            "teacher_ema_decay_start": 0.996,
            "teacher_ema_decay_warmup_steps": 100_000,
            "teacher_ema_update_every": 1,
        },
    ),
    (
        "d999-s996-w100k-u2",
        {
            "teacher_ema_decay": 0.999,
            "teacher_ema_decay_start": 0.996,
            "teacher_ema_decay_warmup_steps": 100_000,
            "teacher_ema_update_every": 2,
        },
    ),
    (
        "d999-s999-w0-u1",
        {
            "teacher_ema_decay": 0.999,
            "teacher_ema_decay_start": 0.999,
            "teacher_ema_decay_warmup_steps": 0,
            "teacher_ema_update_every": 1,
        },
    ),
    (
        "d9996-s999-w500k-u1",
        {
            "teacher_ema_decay": 0.9996,
            "teacher_ema_decay_start": 0.999,
            "teacher_ema_decay_warmup_steps": 500_000,
            "teacher_ema_update_every": 1,
        },
    ),
    (
        "d9996-s999-w100k-u1",
        {
            "teacher_ema_decay": 0.9996,
            "teacher_ema_decay_start": 0.999,
            "teacher_ema_decay_warmup_steps": 100_000,
            "teacher_ema_update_every": 1,
        },
    ),
    (
        "d9996-s999-w500k-u2",
        {
            "teacher_ema_decay": 0.9996,
            "teacher_ema_decay_start": 0.999,
            "teacher_ema_decay_warmup_steps": 500_000,
            "teacher_ema_update_every": 2,
        },
    ),
    (
        "d9996-s9996-w0-u1",
        {
            "teacher_ema_decay": 0.9996,
            "teacher_ema_decay_start": 0.9996,
            "teacher_ema_decay_warmup_steps": 0,
            "teacher_ema_update_every": 1,
        },
    ),
    (
        "d9998-s999-w100k-u1",
        {
            "teacher_ema_decay": 0.9998,
            "teacher_ema_decay_start": 0.999,
            "teacher_ema_decay_warmup_steps": 100_000,
            "teacher_ema_update_every": 1,
        },
    ),
    (
        "d9998-s999-w100k-u2",
        {
            "teacher_ema_decay": 0.9998,
            "teacher_ema_decay_start": 0.999,
            "teacher_ema_decay_warmup_steps": 100_000,
            "teacher_ema_update_every": 2,
        },
    ),
    (
        "d9998-s9998-w0-u1",
        {
            "teacher_ema_decay": 0.9998,
            "teacher_ema_decay_start": 0.9998,
            "teacher_ema_decay_warmup_steps": 0,
            "teacher_ema_update_every": 1,
        },
    ),
    (
        "d9999-s9996-w100k-u1",
        {
            "teacher_ema_decay": 0.9999,
            "teacher_ema_decay_start": 0.9996,
            "teacher_ema_decay_warmup_steps": 100_000,
            "teacher_ema_update_every": 1,
        },
    ),
]
EMA_STOPGRAD_FIXED = {
    "teacher_ema_decay": 0.999,
    "teacher_ema_decay_start": 0.996,
}
EMA_STOPGRAD_WARMUP_STEPS = [5_000, 15_000]
EMA_STOPGRAD_UPDATE_EVERY_VALUES = [1, 2, 5, 10, 20]


SWEEPS: dict[str, list[dict]] = {
    # Anti-collapse sweep: downstream probe perf degrades during training.
    # Three axes: (A) slower EMA teacher, (B) zscore target norm, (C) LR/WD.
    # Goal: isolate which mechanism prevents representation collapse.
    "sweep_optim": [
        # 0) baseline — current settings (ema=0.995, update_every=2, norm=none)
        {},
        # -- Axis A: slower EMA teacher --
        # 1) moderate slowdown — teacher lags more, stabler targets
        {"teacher_ema_decay": 0.999, "teacher_ema_decay_start": 0.996},
        # 2) very slow EMA + every-step update — maximally stable teacher
        {
            "teacher_ema_decay": 0.9996,
            "teacher_ema_decay_start": 0.999,
            "teacher_ema_update_every": 1,
        },
        # -- Axis B: zscore target normalization --
        # 3) zscore alone — normalizes per-layer teacher targets, prevents
        #    variance collapse without changing EMA dynamics
        {"jepa_target_normalization": "zscore"},
        # 4) zscore + slower EMA — both stabilization mechanisms together
        {
            "jepa_target_normalization": "zscore",
            "teacher_ema_decay": 0.999,
            "teacher_ema_decay_start": 0.996,
        },
        # 5) zscore + very slow EMA — maximal anti-collapse
        {
            "jepa_target_normalization": "zscore",
            "teacher_ema_decay": 0.9996,
            "teacher_ema_decay_start": 0.999,
            "teacher_ema_update_every": 1,
        },
        # -- Axis C: conservative LR/WD to reduce student drift --
        # 6) lower LR + higher WD — student changes less per step
        {"learning_rate": 2e-4, "weight_decay": 0.1},
        # 7) kitchen sink: zscore + slow EMA + conservative LR/WD
        dict(BEST_SWEEP_OPTIM),
    ],
    # Refine around the winning anti-collapse run from sweep_optim:
    # zscore targets + every-step slow EMA + conservative optimizer.
    #
    # This sweep is intentionally local. It keeps the stabilization recipe fixed
    # and probes the remaining uncertainty in two places:
    #   (A) LR/WD neighbourhood around the best 2e-4 / 0.1 corner
    #   (B) Slightly faster/slower teacher lag around 0.9996 / 0.999
    #
    # If the centre point still wins after this sweep, we have much stronger
    # evidence that the original best run is not a fluke from a broad search.
    "sweep_optim_refine": [
        # 0) anchor — current winning setting
        dict(BEST_SWEEP_OPTIM),
        # -- Local LR / WD neighbourhood with teacher recipe fixed --
        {**BEST_SWEEP_OPTIM, "learning_rate": 1.5e-4, "weight_decay": 0.075},
        {**BEST_SWEEP_OPTIM, "learning_rate": 1.5e-4, "weight_decay": 0.10},
        {**BEST_SWEEP_OPTIM, "learning_rate": 1.5e-4, "weight_decay": 0.15},
        {**BEST_SWEEP_OPTIM, "learning_rate": 2.0e-4, "weight_decay": 0.075},
        {**BEST_SWEEP_OPTIM, "learning_rate": 2.0e-4, "weight_decay": 0.15},
        {**BEST_SWEEP_OPTIM, "learning_rate": 3.0e-4, "weight_decay": 0.075},
        {**BEST_SWEEP_OPTIM, "learning_rate": 3.0e-4, "weight_decay": 0.10},
        {**BEST_SWEEP_OPTIM, "learning_rate": 3.0e-4, "weight_decay": 0.15},
        # -- Teacher lag sensitivity at the winning optimizer point --
        {
            **BEST_SWEEP_OPTIM,
            "teacher_ema_decay": 0.9993,
            "teacher_ema_decay_start": 0.9985,
        },
        {
            **BEST_SWEEP_OPTIM,
            "teacher_ema_decay": 0.9998,
            "teacher_ema_decay_start": 0.9993,
        },
    ],
    # Direct A/B on the current best gems_small recipe.
    #
    # This keeps the winning JEPA stabilization settings fixed and changes only
    # the representation regularizer so the comparison is attributable.
    "sweep_sigreg_compare": [
        {
            **BEST_SWEEP_OPTIM,
            "run_name_suffix": "sigcmp-none",
        },
        {
            **BEST_SWEEP_OPTIM,
            "representation_regularizer": "sigreg",
            "run_name_suffix": "sigcmp-sigreg",
        },
    ],
    # Controlled 10M-scale ablation.
    #
    # Backbone: 256d / 8L / 8H with 128d predictor and 4 predictor layers.
    # This is ~10.5M trainable params without an EMA teacher.
    #
    # Runs:
    #   1) EMA on,  SIGREG off  -> current JEPA baseline at this scale
    #   2) EMA on,  SIGREG on   -> does SIGREG help even when EMA is present?
    #   3) EMA off, SIGREG off  -> collapse-prone same-backbone control
    #   4) EMA off, SIGREG on   -> can SIGREG replace teacher EMA as stabilizer?
    "sweep_10m_sigreg_ema_ablation": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "use_ema_teacher_target": True,
            "representation_regularizer": "none",
            "run_name_suffix": "10m-ema-none",
        },
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "use_ema_teacher_target": True,
            "representation_regularizer": "sigreg",
            "run_name_suffix": "10m-ema-sigreg",
        },
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "use_ema_teacher_target": False,
            "representation_regularizer": "none",
            "run_name_suffix": "10m-noema-none",
        },
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "use_ema_teacher_target": False,
            "representation_regularizer": "sigreg",
            "run_name_suffix": "10m-noema-sigreg",
        },
    ],
    # Follow-up sweep from the 10M ablation:
    # keep the best EMA baseline as anchor, then sweep no-EMA + SIGREG with
    # target/student gradients enabled through the same backbone.
    "sweep_10m_noema_sigreg_log_lambda": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "use_ema_teacher_target": True,
            "representation_regularizer": "none",
            "run_name_suffix": f"10m-ema-none-anchor-{SIGREG_SAMPLE_SCALE_TAG}",
        },
        *[
            {
                **TEN_M_BEST_SWEEP_OPTIM,
                "use_ema_teacher_target": False,
                "representation_regularizer": "sigreg",
                "sigreg_lambda": sigreg_lambda,
                "run_name_suffix": (
                    f"10m-noema-sigreg-gradtgt-lam{sigreg_lambda:.0e}-"
                    f"{SIGREG_SAMPLE_SCALE_TAG}"
                ),
            }
            for sigreg_lambda in NOEMA_SIGREG_LOG_LAMBDAS
        ],
    ],
    # Follow-up on the failed low-lambda no-EMA sweep:
    # keep the EMA baseline anchor and test much stronger SIGREG weights.
    "sweep_10m_noema_sigreg_high_lambda": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "use_ema_teacher_target": True,
            "representation_regularizer": "none",
            "run_name_suffix": f"10m-ema-none-anchor-hi-{SIGREG_SAMPLE_SCALE_TAG}",
        },
        *[
            {
                **TEN_M_BEST_SWEEP_OPTIM,
                "use_ema_teacher_target": False,
                "representation_regularizer": "sigreg",
                "sigreg_lambda": sigreg_lambda,
                "run_name_suffix": (
                    f"10m-noema-sigreg-gradtgt-hi-lam{label}-"
                    f"{SIGREG_SAMPLE_SCALE_TAG}"
                ),
            }
            for label, sigreg_lambda in NOEMA_SIGREG_HIGH_LAMBDAS
        ],
    ],
    # Follow-up after the SIGREG sweep underperformed:
    # keep SIGREG off, keep stopgrad on via EMA teacher targets, and sweep the
    # EMA recipe itself at the 10M backbone scale.
    "sweep_10m_ema_stopgrad": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "use_ema_teacher_target": True,
            "representation_regularizer": "none",
            "run_name_suffix": f"10m-{EMA_STOPGRAD_SWEEP_TAG}-{label}",
            **ema_overrides,
        }
        for label, ema_overrides in EMA_STOPGRAD_RECIPES
    ],
    # Refine around the best same-step EMA recipe:
    # fix decay/start and sweep only warmup and update cadence.
    "sweep_10m_ema_stopgrad_warmup_update": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            **EMA_STOPGRAD_FIXED,
            "use_ema_teacher_target": True,
            "representation_regularizer": "none",
            "teacher_ema_decay_warmup_steps": warmup_steps,
            "teacher_ema_update_every": update_every,
            "run_name_suffix": (
                f"10m-{EMA_STOPGRAD_SWEEP_TAG}-d999-s996-"
                f"w{warmup_steps // 1000}k-u{update_every}"
            ),
        }
        for warmup_steps in EMA_STOPGRAD_WARMUP_STEPS
        for update_every in EMA_STOPGRAD_UPDATE_EVERY_VALUES
    ],
}


# ---------------------------------------------------------------------------
# Data preparation — run once to warm the volume before parallel sweeps
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={volume_path: volume},
    cpu=8.0,
    memory=32768,  # 32 GiB
    timeout=30 * MINUTES,
)
def prepare_data(
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

    from utils.training import load_config

    config = load_config(config_path)
    config.update(json.loads(overrides_json))
    config.tfrecord_dir = str(volume_path / "data" / "gems_peaklist_tfrecord_alpha")

    # 1) Download training data (GeMS TFRecords)
    logging.info("Preparing training data...")
    from input_pipeline import TfLightningDataModule

    datamodule = TfLightningDataModule(config, seed=int(config.seed))
    logging.info(
        "Training data ready: %d train steps, %d peaks",
        datamodule.train_steps,
        datamodule.info["num_peaks"],
    )

    # 2) Download + process probe data
    logging.info("Preparing probe data...")
    from utils.massspec_probe_data import MassSpecProbeData

    probe_data = MassSpecProbeData.from_config(config)
    logging.info(
        "Probe data ready: %d train / %d test samples",
        probe_data.info["massspec_train_size"],
        probe_data.info["massspec_test_size"],
    )

    volume.commit()
    logging.info("Volume committed — data is cached for all future runs.")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={volume_path: volume},
    cpu=8.0,
    memory=32768,  # 32 GiB
    gpu=DEFAULT_GPU,
    timeout=5 * HOURS,
    secrets=[modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])],
)
def train(
    config_path: str = "configs/gems_small.py",
    overrides_json: str = "{}",
    workdir_tag: str = "",
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
    workdir_root = volume_path / "experiments"
    if workdir_tag:
        workdir_root = workdir_root / workdir_tag
    workdir = workdir_root / run_name
    workdir.mkdir(parents=True, exist_ok=True)

    logging.info("Run: %s", run_name)
    logging.info("Workdir: %s", workdir)
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
    workdir_tag: str = "",
):
    if sweep:
        experiments = SWEEPS[sweep]
        prepare_payloads: list[str] = []
        base_overrides = json.loads(overrides)
        for exp in experiments:
            merged = {**base_overrides, **exp}
            prepare_payloads.append(json.dumps(merged, sort_keys=True))
        print("Preparing data on volume...")
        for payload in dict.fromkeys(prepare_payloads):
            prepare_data.remote(config_path=config, overrides_json=payload)
        print("Data ready.\n")
        print(
            f"Launching {len(experiments)} experiments "
            f"with up to {MAX_SWEEP_CONCURRENCY} concurrent runs ({sweep}):"
        )
        for i, exp in enumerate(experiments):
            print(f"  [{i}] {exp or '(baseline)'}")
        for batch_start in range(0, len(experiments), MAX_SWEEP_CONCURRENCY):
            batch = experiments[batch_start: batch_start + MAX_SWEEP_CONCURRENCY]
            print(
                f"Starting batch {batch_start // MAX_SWEEP_CONCURRENCY + 1}: "
                f"experiments {batch_start}-{batch_start + len(batch) - 1}"
            )
            handles = []
            for exp in batch:
                merged = {**base_overrides, **exp}
                handles.append(
                    train.spawn(
                        config_path=config,
                        overrides_json=json.dumps(merged),
                        workdir_tag=workdir_tag,
                    )
                )
            for offset, handle in enumerate(handles):
                result = handle.get()
                print(f"[{batch_start + offset}] done: {result}")
    else:
        print("Preparing data on volume...")
        prepare_data.remote(config_path=config, overrides_json=overrides)
        print("Data ready.\n")
        train.remote(
            config_path=config,
            overrides_json=overrides,
            workdir_tag=workdir_tag,
        )
