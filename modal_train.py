"""Train on Modal with persistent storage for data and checkpoints.

Usage:
    # Single run
    modal run modal_train.py
    modal run modal_train.py --config configs/gems_small.py --gpu H100
    modal run modal_train.py --config configs/gems_small.py --workdir my_run
    modal run modal_train.py --config configs/gems_small.py --async-probes

    # Parallel sweep (launches all experiments concurrently)
    modal run modal_train.py --sweep sweep_optim
    modal run modal_train.py --sweep sweep_optim_refine
    modal run modal_train.py --sweep sweep_sigreg_compare
    modal run modal_train.py --sweep sweep_10m_sigreg_ablation
    modal run modal_train.py --sweep sweep_10m_sigreg_log_lambda
    modal run modal_train.py --sweep sweep_10m_sigreg_high_lambda
    modal run modal_train.py --sweep sweep_10m_masking
    modal run modal_train.py --sweep sweep_10m_deep_supervision
    modal run modal_train.py --sweep sweep_10m_batch_size_flops_matched
    modal run modal_train.py --config configs/gems_small.py --sweep sweep_gems_small_peak_filtering
    modal run modal_train.py --config configs/gems_small.py --sweep sweep_gems_small_predictor_scale
    modal run modal_train.py --config configs/gems_small.py --sweep sweep_gems_small_predictor_scale_depth
    modal run modal_train.py --config configs/gems_small.py --sweep sweep_gems_small_scale_100m_300m --detach
    modal run modal_train.py --config configs/gems_small_norm.py --sweep sweep_sigreg_lambda_wide --detach
    modal run modal_train.py --config configs/gems_small_norm.py --sweep sweep_gems_small_norm_sigreg_lambda --detach

Setup:
    1. modal setup
    2. modal secret create wandb-secret WANDB_API_KEY=<your-key>
    3. modal secret create huggingface-secret HF_TOKEN=<your-token>
    4. modal run modal_train.py
"""

import json
import time
from pathlib import Path

import modal

MINUTES = 60
HOURS = 60 * MINUTES
DEFAULT_GPU = "H100"
PROBE_GPU = "L4"
PROJECT_ROOT = "/root/spectra-learning"
MAX_SWEEP_CONCURRENCY = 10
TRAIN_TIMEOUT_HOURS = 24
PROBE_TIMEOUT_HOURS = 6
GEMS_SMALL_SWEEP_RUNTIME_HOURS = 12.0

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
        "pyarrow>=16.0.0",
        "modal>=1.4.1",
    )
    .run_commands(
        "pip install --no-build-isolation gram-newton-schulz@git+https://github.com/Dao-AILab/gram-newton-schulz"
    )
)

# Add only the source directories needed for training
local = Path(__file__).parent
image = (
    base_image
    .add_local_file(local / "train.py", remote_path=f"{PROJECT_ROOT}/train.py")
    .add_local_file(
        local / "modal_train.py",
        remote_path=f"{PROJECT_ROOT}/modal_train.py",
    )
    .add_local_dir(local / "spectra_learning", remote_path=f"{PROJECT_ROOT}/spectra_learning")
    .add_local_dir(local / "configs", remote_path=f"{PROJECT_ROOT}/configs")
    .add_local_dir(local / "scripts", remote_path=f"{PROJECT_ROOT}/scripts")
)

app = modal.App("spectra-training", image=image)
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret",
    required_keys=["HF_TOKEN"],
)
wandb_secret = modal.Secret.from_name(
    "wandb-secret",
    required_keys=["WANDB_API_KEY"],
)


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------
BEST_SWEEP_OPTIM = {
    "jepa_target_normalization": "zscore",
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
    "encoder_fourier_mlp_hidden_dim": 1024,
    "encoder_fourier_mlp_num_layers": 4,
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
SIGREG_LAMBDA_WIDE_VALUES = (5e-06, 1e-06,)
SIGREG_SAMPLE_SCALE_TAG = "sigcfscale"
JEPA_MASKING_SWEEP_TAG = "mask"
JEPA_DEEP_SUPERVISION_SWEEP_TAG = "dsup"
BATCH_SIZE_SWEEP_TAG = "bsflops"
PEAK_FILTER_SWEEP_TAG = "peakfilt"
GEMS_SMALL_PREDICTOR_SCALE_SWEEP_TAG = "predscale"
JEPA_MASKING_RECIPES = [
    (
        "b2-c35-t20",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.35,
            "jepa_target_fraction": 0.20,
        },
    ),
    (
        "rag-c35-t20",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.35,
            "jepa_target_fraction": 0.20,
            "jepa_mask_strategy": "ragged",
        },
    ),
    (
        "all-c35-t20",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.35,
            "jepa_target_fraction": 0.20,
            "jepa_mask_strategy": "all",
        },
    ),
    (
        "b2-c25-t20",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.25,
            "jepa_target_fraction": 0.20,
        },
    ),
    (
        "b2-c45-t20",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.45,
            "jepa_target_fraction": 0.20,
        },
    ),
    (
        "b2-c50-t20",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.50,
            "jepa_target_fraction": 0.20,
        },
    ),
    (
        "b2-c35-t10",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.35,
            "jepa_target_fraction": 0.10,
        },
    ),
    (
        "b2-c35-t15",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.35,
            "jepa_target_fraction": 0.15,
        },
    ),
    (
        "b2-c35-t25",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.35,
            "jepa_target_fraction": 0.25,
        },
    ),
    (
        "b2-c35-t30",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.35,
            "jepa_target_fraction": 0.30,
        },
    ),
    (
        "b1-c35-t20",
        {
            "jepa_num_target_blocks": 1,
            "jepa_context_fraction": 0.35,
            "jepa_target_fraction": 0.20,
        },
    ),
    (
        "b4-c35-t20",
        {
            "jepa_num_target_blocks": 4,
            "jepa_context_fraction": 0.35,
            "jepa_target_fraction": 0.20,
        },
    ),
]
JEPA_DEEP_SUPERVISION_RECIPES = [
    (
        "spread4-z",
        {
            "jepa_target_layers": [1, 3, 5, 8],
            "jepa_target_normalization": "zscore",
        },
    ),
    (
        "spread4-none",
        {
            "jepa_target_layers": [1, 3, 5, 8],
            "jepa_target_normalization": "none",
        },
    ),
    (
        "deep2-z",
        {
            "jepa_target_layers": [5, 8],
            "jepa_target_normalization": "zscore",
        },
    ),
    (
        "deep2-none",
        {
            "jepa_target_layers": [5, 8],
            "jepa_target_normalization": "none",
        },
    ),
]
BATCH_SIZE_FLOPS_MATCHED_RECIPES = [
    (
        "bs256",
        {
            "batch_size": 256,
        },
    ),
    (
        "bs2048",
        {
            "batch_size": 2048,
        },
    ),
]
PEAK_FILTER_RECIPES = [
    (
        "peakoff-windowoff",
        {
            "peak_drop_min_intensity": 1e-4,
            "precursor_peak_exclusion_window_da": 0.0,
        },
    ),
    (
        "peakon-windowoff",
        {
            "peak_drop_min_intensity": 0.01,
            "precursor_peak_exclusion_window_da": 0.0,
        },
    ),
    (
        "peakoff-windowon",
        {
            "peak_drop_min_intensity": 1e-4,
            "precursor_peak_exclusion_window_da": 5.0,
        },
    ),
    (
        "peakon-windowon",
        {
            "peak_drop_min_intensity": 0.01,
            "precursor_peak_exclusion_window_da": 5.0,
        },
    ),
]
GEMS_SMALL_SCALE_SWEEP_TAG = "scale"
GEMS_SMALL_100M = {
    "model_dim": 768,
    "encoder_num_layers": 12,
    "encoder_num_heads": 12,
    "encoder_num_kv_heads": 12,
    "feature_mlp_hidden_dim": 1024,
    "encoder_fourier_mlp_hidden_dim": 1024,
    "encoder_fourier_mlp_num_layers": 4,
    "predictor_dim": 384,
    "masked_latent_predictor_num_layers": 10,
    "masked_latent_predictor_num_heads": 16,
    "jepa_target_layers": [1, 4, 8, 12],
}
GEMS_SMALL_300M = {
    "model_dim": 1024,
    "encoder_num_layers": 20,
    "encoder_num_heads": 16,
    "encoder_num_kv_heads": 16,
    "feature_mlp_hidden_dim": 2048,
    "encoder_fourier_mlp_hidden_dim": 1024,
    "encoder_fourier_mlp_num_layers": 4,
    "predictor_dim": 512,
    "masked_latent_predictor_num_layers": 12,
    "masked_latent_predictor_num_heads": 16,
    "jepa_target_layers": [4, 8, 12, 16, 20],
}


SWEEPS: dict[str, list[dict]] = {
    # Anti-collapse sweep: downstream probe perf degrades during training.
    # Two axes: (A) target normalization, (B) LR/WD.
    # Goal: isolate which mechanism prevents representation collapse.
    "sweep_optim": [
        # 0) baseline — current settings
        {},
        # -- Axis A: zscore target normalization --
        # 1) zscore alone — normalizes per-layer targets, prevents variance collapse
        {"jepa_target_normalization": "zscore"},
        # -- Axis B: conservative LR/WD to reduce drift --
        # 2) lower LR + higher WD — student changes less per step
        {"learning_rate": 2e-4, "weight_decay": 0.1},
        # 3) kitchen sink: zscore + conservative LR/WD
        dict(BEST_SWEEP_OPTIM),
    ],
    # Refine around the winning anti-collapse run from sweep_optim:
    # zscore targets + conservative optimizer.
    #
    # This sweep is intentionally local. It keeps the stabilization recipe fixed
    # and probes the remaining LR/WD neighbourhood around the best 2e-4 / 0.1 corner.
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
            "representation_regularizer": "sigreg-proj",
            "run_name_suffix": "sigcmp-sigreg-proj",
        },
    ],
    # Controlled 10M-scale sigreg ablation on the shared-backbone target path.
    #
    # Backbone: 256d / 8L / 8H with 128d predictor and 4 predictor layers.
    "sweep_10m_sigreg_ablation": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "representation_regularizer": "none",
            "run_name_suffix": "10m-none",
        },
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "representation_regularizer": "sigreg-proj",
            "run_name_suffix": "10m-sigreg-proj",
        },
    ],
    # Shared-backbone SIGREG sweep over a broad log-scale lambda range.
    "sweep_10m_sigreg_log_lambda": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "representation_regularizer": "sigreg-proj",
            "sigreg_lambda": sigreg_lambda,
            "run_name_suffix": (
                f"10m-sigreg-proj-lam{sigreg_lambda:.0e}-{SIGREG_SAMPLE_SCALE_TAG}"
            ),
        }
        for sigreg_lambda in NOEMA_SIGREG_LOG_LAMBDAS
    ],
    # Follow-up sweep with much stronger SIGREG weights.
    "sweep_10m_sigreg_high_lambda": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "representation_regularizer": "sigreg-proj",
            "sigreg_lambda": sigreg_lambda,
            "run_name_suffix": (
                f"10m-sigreg-proj-hi-lam{label}-{SIGREG_SAMPLE_SCALE_TAG}"
            ),
        }
        for label, sigreg_lambda in NOEMA_SIGREG_HIGH_LAMBDAS
    ],
    # Sweep only the JEPA masking pattern around the current GeMS default.
    "sweep_10m_masking": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "representation_regularizer": "none",
            "run_name_suffix": f"10m-{JEPA_MASKING_SWEEP_TAG}-{label}",
            **masking_overrides,
        }
        for label, masking_overrides in JEPA_MASKING_RECIPES
    ],
    # Ablate the target stack and target normalization.
    #
    # This is a clean 2x2:
    #   - layer stack: current spread 4-layer targets [1,3,5,8] vs deeper 2-layer targets [5,8]
    #   - target normalization: per-layer zscore vs no normalization
    "sweep_10m_deep_supervision": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "representation_regularizer": "none",
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.35,
            "jepa_target_fraction": 0.20,
            "run_name_suffix": f"10m-{JEPA_DEEP_SUPERVISION_SWEEP_TAG}-{label}",
            **deep_supervision_overrides,
        }
        for label, deep_supervision_overrides in JEPA_DEEP_SUPERVISION_RECIPES
    ],
    # Compute-matched batch-size A/B.
    #
    # For a fixed model and dataset, keeping num_epochs fixed means each run
    # sees roughly the same total number of samples, so total training FLOPs
    # stay roughly matched across batch sizes. Keep probe batch size fixed so
    # online probe metrics remain directly comparable across runs.
    "sweep_10m_batch_size_flops_matched": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            "representation_regularizer": "none",
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.35,
            "jepa_target_fraction": 0.20,
            "msg_probe_batch_size": 256,
            "run_name_suffix": f"10m-{BATCH_SIZE_SWEEP_TAG}-{label}",
            **batch_size_overrides,
        }
        for label, batch_size_overrides in BATCH_SIZE_FLOPS_MATCHED_RECIPES
    ],
    # Full 2x2 on/off sweep:
    #   - peak filtering: base 1e-4 vs 0.01
    #   - precursor window filtering: off vs 5 Da
    "sweep_gems_small_peak_filtering": [
        {
            "msg_probe_batch_size": 256,
            "run_name_suffix": f"{PEAK_FILTER_SWEEP_TAG}-{label}",
            **peak_filter_overrides,
        }
        for label, peak_filter_overrides in PEAK_FILTER_RECIPES
    ],
    # Predictor-size A/B on the current gems_small encoder.
    #
    # These settings were checked against the actual trainable parameter counts
    # for configs/gems_small.py:
    #   - predictor_dim=256 -> 3,414,272 predictor params vs 9,190,144 encoder
    #     params (37.2%)
    #   - predictor_dim=360 -> 6,689,664 predictor params vs 9,190,144 encoder
    #     params (72.8%)
    "sweep_gems_small_predictor_scale": [
        {
            "msg_probe_batch_size": 256,
            "predictor_dim": 256,
            "run_name_suffix": f"{GEMS_SMALL_PREDICTOR_SCALE_SWEEP_TAG}-50pct",
        },
        {
            "msg_probe_batch_size": 256,
            "predictor_dim": 360,
            "run_name_suffix": f"{GEMS_SMALL_PREDICTOR_SCALE_SWEEP_TAG}-100pct",
        },
    ],
    # Single depth-matched near-100% predictor run.
    #
    # This is separated from the original width-scaled sweep so it launches
    # only the depth-based near-encoder-size predictor setting:
    #   - predictor_dim=256, masked_latent_predictor_num_layers=8 ->
    #     6,564,096 predictor params vs 9,190,144 encoder params (71.4%)
    "sweep_gems_small_predictor_scale_depth_2": [
        {
            "msg_probe_batch_size": 256,
            "predictor_dim": 256,
            "masked_latent_predictor_num_layers": 8,
            "run_name_suffix": (
                f"{GEMS_SMALL_PREDICTOR_SCALE_SWEEP_TAG}-100pct-depth"
            ),
        },
    ],
    "sweep_gems_small_scale_100m_300m": [
        {
            "max_duration_hours": GEMS_SMALL_SWEEP_RUNTIME_HOURS,
            "run_name_suffix": f"{GEMS_SMALL_SCALE_SWEEP_TAG}-100m-12h",
            **GEMS_SMALL_100M,
        },
        {
            "max_duration_hours": GEMS_SMALL_SWEEP_RUNTIME_HOURS,
            "run_name_suffix": f"{GEMS_SMALL_SCALE_SWEEP_TAG}-300m-12h",
            **GEMS_SMALL_300M,
        },
    ],
    "sweep_sigreg_lambda_wide": [
        {
            "sigreg_lambda": sigreg_lambda,
            "run_name_suffix": f"sigreg-lam{sigreg_lambda:.0e}",
        }
        for sigreg_lambda in SIGREG_LAMBDA_WIDE_VALUES
    ],
    "sweep_gems_small_norm_sigreg_lambda": [
        {
            "sigreg_lambda": sigreg_lambda,
            "run_name_suffix": f"norm-sigreg-lam{sigreg_lambda:.0e}",
        }
        for sigreg_lambda in (3e-4, 1e-3, 3e-3, 1e-2)
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
    secrets=[huggingface_secret],
)
def prepare_data(
    config_path: str = "configs/gems_small.py",
    overrides_json: str = "{}",
    prepare_probe: bool = True,
):
    import logging
    import os
    import sys

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    logging.basicConfig(level=logging.INFO)

    from spectra_learning.training.api import load_config

    config = load_config(config_path)
    config.update(json.loads(overrides_json))
    config.artifact_dir = str(volume_path / "data" / "gems_artifacts_alpha")

    # 1) Download training data (GeMS native shards)
    logging.info("Preparing training data...")
    from spectra_learning.data.gems.datamodule import GemsNativeDataModule

    datamodule = GemsNativeDataModule(config, seed=int(config.seed))
    logging.info(
        "Training data ready: %d train steps, %d peaks",
        datamodule.train_steps,
        datamodule.info["num_peaks"],
    )

    # 2) Download + process probe data
    if prepare_probe:
        logging.info("Preparing probe data...")
        from spectra_learning.probes.massspec.data import MassSpecProbeData

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
    timeout=TRAIN_TIMEOUT_HOURS * HOURS,
    secrets=[huggingface_secret, wandb_secret],
    single_use_containers=True,
)
def train(
    config_path: str = "configs/gems_small.py",
    overrides_json: str = "{}",
    workdir: str = "",
    workdir_tag: str = "",
):
    import logging
    import os
    import sys

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    logging.basicConfig(level=logging.INFO)

    from spectra_learning.training.pretrain import train_and_evaluate
    from spectra_learning.training.api import auto_run_name, load_config

    config = load_config(config_path)

    # Apply experiment overrides
    overrides = json.loads(overrides_json)
    config.update(overrides)

    # Point data at the persistent volume
    config.artifact_dir = str(volume_path / "data" / "gems_artifacts_alpha")

    # Muon NS kernels require SM90+ (H100/B200); disable on older GPUs
    import torch
    sm_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    if sm_major < 9:
        config.muon_ns_use_kernels = False

    run_name = auto_run_name(config)
    workdir_root = volume_path / "experiments"
    if workdir:
        workdir_path = workdir_root / workdir
    else:
        if workdir_tag:
            workdir_root = workdir_root / workdir_tag
        workdir_path = workdir_root / run_name
    workdir_path.mkdir(parents=True, exist_ok=True)

    logging.info("Run: %s", run_name)
    logging.info("Workdir: %s", workdir_path)
    if overrides:
        logging.info("Overrides: %s", overrides)

    results = train_and_evaluate(config, workdir=workdir_path)

    volume.commit()
    logging.info("Training complete. Results: %s", results)
    return results


@app.function(
    image=image,
    volumes={volume_path: volume},
    cpu=8.0,
    memory=32768,  # 32 GiB
    gpu=PROBE_GPU,
    timeout=PROBE_TIMEOUT_HOURS * HOURS,
    secrets=[huggingface_secret, wandb_secret],
    single_use_containers=True,
)
def run_probe_checkpoint(
    config_json: str,
    checkpoint_path: str,
    workdir: str,
    global_step: int,
):
    import logging
    import os
    import sys

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    logging.basicConfig(level=logging.INFO)
    volume.reload()

    from spectra_learning.probes.massspec.checkpoint_probe import run_checkpoint_msg_probe

    config_json = _modal_probe_config_json(config_json)
    logging.info(
        "Running Modal MSG probe on %s at global_step=%d",
        checkpoint_path,
        int(global_step),
    )
    metrics = run_checkpoint_msg_probe(
        config_json=config_json,
        checkpoint_path=checkpoint_path,
        workdir=workdir,
        global_step=int(global_step),
    )
    volume.commit()
    logging.info("Modal MSG probe complete: %s", metrics)
    return metrics


@app.function(
    image=image,
    volumes={volume_path: volume},
    cpu=8.0,
    memory=32768,
    gpu=PROBE_GPU,
    timeout=PROBE_TIMEOUT_HOURS * HOURS,
    secrets=[huggingface_secret],
    single_use_containers=True,
)
def run_fluorine_checkpoint(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    label: str,
    embedding_cache_dir: str,
):
    import logging
    import os
    import sys
    from argparse import Namespace
    from pathlib import Path

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    logging.basicConfig(level=logging.INFO)
    volume.reload()

    from scripts.train_fluorine_detection import HF_REPO_ID, HF_SUBDIR, run

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    args = Namespace(
        source="checkpoint",
        config=Path(config_path),
        checkpoint=Path(checkpoint_path),
        train_covariance_pooler=True,
        covariance_dim=64,
        embedding_cache_dir=Path(embedding_cache_dir),
        force_embedding_cache=False,
        embedding_dtype="float16",
        repo_id=HF_REPO_ID,
        revision="main",
        subdir=HF_SUBDIR,
        cache_dir=volume_path / "data" / "fluorine_detection_fine_tuned",
        num_shards=16,
        parquet_batch_size=50_000,
        output_json=output_dir_path / f"{label}.json",
        device="cuda",
        seed=42,
        batch_size=512,
        num_peaks=None,
        peak_ordering=None,
        epochs=20,
        patience=5,
        hidden_dims="256,512",
        learning_rates="0.001,0.0003",
        weight_decays="0.0001",
        dropouts="0.1",
        focal_alpha="auto",
        focal_gamma=2.0,
        select_metric="average_precision",
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
    )
    logging.info("Running fluorine checkpoint evaluation for %s", label)
    metrics = run(args)
    metrics["label"] = label
    metrics["checkpoint_path"] = checkpoint_path
    volume.commit()
    logging.info("Fluorine checkpoint evaluation complete for %s", label)
    return metrics


@app.function(
    image=image,
    volumes={volume_path: volume},
    cpu=16.0,
    memory=65536,  # 64 GiB
    gpu=f"{DEFAULT_GPU}:4",
    timeout=TRAIN_TIMEOUT_HOURS * HOURS,
    secrets=[huggingface_secret, wandb_secret],
    single_use_containers=True,
)
def train_4gpu(
    config_path: str = "configs/gems_small.py",
    overrides_json: str = "{}",
    workdir: str = "",
    workdir_tag: str = "",
):
    return _train_torchrun_4gpu(
        config_path=config_path,
        overrides_json=overrides_json,
        workdir=workdir,
        workdir_tag=workdir_tag,
        gpu_label=DEFAULT_GPU,
    )


@app.function(
    image=image,
    volumes={volume_path: volume},
    cpu=32.0,
    memory=131072,  # 128 GiB
    gpu=f"{DEFAULT_GPU}:8",
    timeout=TRAIN_TIMEOUT_HOURS * HOURS,
    secrets=[huggingface_secret, wandb_secret],
    single_use_containers=True,
)
def train_8gpu(
    config_path: str = "configs/gems_small.py",
    overrides_json: str = "{}",
    workdir: str = "",
    workdir_tag: str = "",
):
    return _train_torchrun_ngpu(
        config_path=config_path,
        overrides_json=overrides_json,
        workdir=workdir,
        workdir_tag=workdir_tag,
        gpu_label=DEFAULT_GPU,
        nproc_per_node=8,
    )


@app.function(
    image=image,
    volumes={volume_path: volume},
    cpu=16.0,
    memory=65536,  # 64 GiB
    gpu="L40S:4",
    timeout=TRAIN_TIMEOUT_HOURS * HOURS,
    secrets=[huggingface_secret, wandb_secret],
    single_use_containers=True,
)
def train_4gpu_l40s(
    config_path: str = "configs/gems_small.py",
    overrides_json: str = "{}",
    workdir: str = "",
    workdir_tag: str = "",
):
    return _train_torchrun_4gpu(
        config_path=config_path,
        overrides_json=overrides_json,
        workdir=workdir,
        workdir_tag=workdir_tag,
        gpu_label="L40S",
    )


@app.function(
    image=image,
    volumes={volume_path: volume},
    cpu=16.0,
    memory=65536,  # 64 GiB
    gpu="A10G:4",
    timeout=TRAIN_TIMEOUT_HOURS * HOURS,
    secrets=[huggingface_secret, wandb_secret],
    single_use_containers=True,
)
def train_4gpu_a10g(
    config_path: str = "configs/gems_small.py",
    overrides_json: str = "{}",
    workdir: str = "",
    workdir_tag: str = "",
):
    return _train_torchrun_4gpu(
        config_path=config_path,
        overrides_json=overrides_json,
        workdir=workdir,
        workdir_tag=workdir_tag,
        gpu_label="A10G",
    )


@app.function(
    image=image,
    volumes={volume_path: volume},
    cpu=16.0,
    memory=65536,  # 64 GiB
    gpu="L4:4",
    timeout=TRAIN_TIMEOUT_HOURS * HOURS,
    secrets=[huggingface_secret, wandb_secret],
    single_use_containers=True,
)
def train_4gpu_l4(
    config_path: str = "configs/gems_small.py",
    overrides_json: str = "{}",
    workdir: str = "",
    workdir_tag: str = "",
):
    return _train_torchrun_4gpu(
        config_path=config_path,
        overrides_json=overrides_json,
        workdir=workdir,
        workdir_tag=workdir_tag,
        gpu_label="L4",
    )


def _train_torchrun_4gpu(
    *,
    config_path: str,
    overrides_json: str,
    workdir: str,
    workdir_tag: str,
    gpu_label: str,
):
    return _train_torchrun_ngpu(
        config_path=config_path,
        overrides_json=overrides_json,
        workdir=workdir,
        workdir_tag=workdir_tag,
        gpu_label=gpu_label,
        nproc_per_node=4,
    )


def _train_torchrun_ngpu(
    *,
    config_path: str,
    overrides_json: str,
    workdir: str,
    workdir_tag: str,
    gpu_label: str,
    nproc_per_node: int,
):
    import logging
    import os
    import subprocess
    import sys

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    logging.basicConfig(level=logging.INFO)

    from spectra_learning.training.api import auto_run_name, load_config

    config = load_config(config_path)
    overrides = json.loads(overrides_json)
    config.update(overrides)
    config.artifact_dir = str(volume_path / "data" / "gems_artifacts_alpha")

    import torch

    sm_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    if sm_major < 9:
        config.muon_ns_use_kernels = False

    run_name = auto_run_name(config)
    workdir_root = volume_path / "experiments"
    if workdir:
        workdir_path = workdir_root / workdir
    else:
        if workdir_tag:
            workdir_root = workdir_root / workdir_tag
        workdir_path = workdir_root / run_name
    workdir_path.mkdir(parents=True, exist_ok=True)

    subprocess_overrides = dict(overrides)
    subprocess_overrides["artifact_dir"] = str(volume_path / "data" / "gems_artifacts_alpha")
    subprocess_overrides["muon_ns_use_kernels"] = bool(config.get("muon_ns_use_kernels", True))

    metrics_path = workdir_path / "metrics.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "train.py",
        "--config",
        config_path,
        "--workdir",
        str(workdir_path),
        "--overrides-json",
        json.dumps(subprocess_overrides, sort_keys=True),
        "--metrics-json",
        str(metrics_path),
    ]
    logging.info("Run: %s", run_name)
    logging.info("GPU: %dx %s", nproc_per_node, gpu_label)
    logging.info("Workdir: %s", workdir_path)
    logging.info("Launching torchrun: %s", " ".join(command))
    subprocess.run(command, check=True)

    volume.commit()
    results = json.loads(metrics_path.read_text())
    logging.info("%d-GPU training complete. Results: %s", nproc_per_node, results)
    return results


FOUR_GPU_TRAINERS = {
    "h100": train_4gpu,
    "l40s": train_4gpu_l40s,
    "a10": train_4gpu_a10g,
    "a10g": train_4gpu_a10g,
    "l4": train_4gpu_l4,
}


def _modal_probe_config_json(config_json: str) -> str:
    payload = json.loads(config_json)
    payload["artifact_dir"] = str(
        payload.get(
            "modal_probe_artifact_dir",
            volume_path / "data" / "gems_artifacts_alpha",
        )
    )
    return json.dumps(payload, sort_keys=True)


def _upload_probe_checkpoint(
    checkpoint_path: Path,
    workdir: Path,
    global_step: int,
) -> Path:
    from spectra_learning.training.checkpointing import covariance_pooler_checkpoint_path
    from spectra_learning.training.modal_probe import modal_probe_remote_label

    label = modal_probe_remote_label(workdir, global_step)
    remote_dir = Path("modal_probe_checkpoints") / label
    with volume.batch_upload(force=True) as batch:
        batch.put_file(
            str(checkpoint_path),
            f"/{remote_dir.as_posix()}/{checkpoint_path.name}",
        )
        pooler_path = covariance_pooler_checkpoint_path(checkpoint_path)
        if pooler_path.exists():
            batch.put_file(
                str(pooler_path),
                f"/{remote_dir.as_posix()}/{pooler_path.name}",
            )
    return volume_path / remote_dir / checkpoint_path.name


def _modal_probe_workdir(workdir: Path, global_step: int) -> Path:
    from spectra_learning.training.modal_probe import modal_probe_remote_label

    return volume_path / "modal_probe_runs" / modal_probe_remote_label(
        workdir,
        global_step,
    )


def _submit_probe_from_local(
    *,
    config_json_path: str,
    checkpoint_path: str,
    workdir: str,
    global_step: int,
    wait: bool,
) -> None:
    local_checkpoint = Path(checkpoint_path).expanduser().resolve()
    local_workdir = Path(workdir).expanduser().resolve()
    config_json = _modal_probe_config_json(
        Path(config_json_path).expanduser().resolve().read_text()
    )
    remote_checkpoint = _upload_probe_checkpoint(
        local_checkpoint,
        local_workdir,
        int(global_step),
    )
    remote_workdir = _modal_probe_workdir(local_workdir, int(global_step))
    handle = run_probe_checkpoint.spawn(
        config_json=config_json,
        checkpoint_path=str(remote_checkpoint),
        workdir=str(remote_workdir),
        global_step=int(global_step),
    )
    print(
        json.dumps(
            {
                "call_id": handle.object_id,
                "checkpoint_path": str(remote_checkpoint),
                "workdir": str(remote_workdir),
                "global_step": int(global_step),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if wait:
        print(json.dumps(handle.get(), indent=2, sort_keys=True))


def _submit_probe_sweep_from_local(
    *,
    config_path: str,
    sweep_json_path: str,
    checkpoint_path: str,
    workdir: str,
    global_step: int,
    wait_for_results: bool,
) -> None:
    from spectra_learning.training.api import load_config
    from spectra_learning.training.logging import _config_to_wandb_dict

    local_checkpoint = Path(checkpoint_path).expanduser().resolve()
    local_workdir = Path(workdir).expanduser().resolve()
    local_workdir.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(sweep_json_path).expanduser().resolve().read_text())
    remote_checkpoint = _upload_probe_checkpoint(
        local_checkpoint,
        local_workdir,
        int(global_step),
    )
    handles = []
    records = []
    for job in jobs:
        name = str(job["name"])
        overrides = dict(job["overrides"])
        job_dir = local_workdir / name
        job_dir.mkdir(parents=True, exist_ok=True)
        config = load_config(config_path)
        with config.ignore_type():
            for key, value in overrides.items():
                setattr(config, key, value)
        config_dict = _config_to_wandb_dict(config)
        (job_dir / "config.json").write_text(
            json.dumps(config_dict, indent=2, sort_keys=True)
        )
        remote_workdir = volume_path / "modal_probe_runs" / local_workdir.name / name
        handle = run_probe_checkpoint.spawn(
            config_json=_modal_probe_config_json(json.dumps(config_dict, sort_keys=True)),
            checkpoint_path=str(remote_checkpoint),
            workdir=str(remote_workdir),
            global_step=int(global_step),
        )
        record = {
            "name": name,
            "call_id": handle.object_id,
            "local_dir": str(job_dir),
            "remote_workdir": str(remote_workdir),
            "remote_checkpoint": str(remote_checkpoint),
            "overrides": overrides,
        }
        records.append(record)
        handles.append((handle, record))
        print(f"spawned {name}: {handle.object_id}")
    manifest_path = local_workdir / "manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True))
    print(f"manifest: {manifest_path}")
    if not wait_for_results:
        return

    results = []
    for handle, record in handles:
        metrics = handle.get()
        job_dir = Path(record["local_dir"])
        (job_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True)
        )
        summary = {
            **record,
            "best_epoch": metrics["msg_probe/covariance/epoch"],
            "val_auc": metrics["msg_probe/covariance/val/auc_maccs_mean"],
            "test_auc": metrics["msg_probe/covariance/test/auc_maccs_mean"],
            "test_ap": metrics[
                "msg_probe/covariance/test/average_precision_maccs_mean"
            ],
        }
        results.append(summary)
        results_path = local_workdir / "results.partial.json"
        results_path.write_text(
            json.dumps(
                sorted(results, key=lambda item: item["test_auc"], reverse=True),
                indent=2,
                sort_keys=True,
            )
        )
        print(
            f"completed {record['name']}: "
            f"val_auc={summary['val_auc']:.4f} "
            f"test_auc={summary['test_auc']:.4f} "
            f"epoch={summary['best_epoch']:.0f}"
        )
    results = sorted(results, key=lambda item: item["test_auc"], reverse=True)
    results_path = local_workdir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


def _submit_fluorine_sweep_from_local(
    *,
    config_path: str,
    checkpoint_dir: str,
    workdir: str,
    max_checkpoints: int,
    wait_for_results: bool,
) -> None:
    import re

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from spectra_learning.training.checkpointing import is_main_checkpoint_path

    local_workdir = Path(workdir).expanduser().resolve()
    local_workdir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    checkpoints = sorted(
        [
            path
            for path in checkpoint_root.glob("*.pt")
            if path.name.startswith("step-") and is_main_checkpoint_path(path)
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[: int(max_checkpoints)]
    handles = []
    records = []
    for order_idx, checkpoint_path in enumerate(checkpoints):
        match = re.search(r"step-(\d+)", checkpoint_path.name)
        global_step = int(match.group(1))
        label = f"{order_idx:02d}_{checkpoint_path.stem}"
        remote_checkpoint = _upload_probe_checkpoint(
            checkpoint_path,
            local_workdir / "checkpoint_uploads",
            global_step,
        )
        remote_output_dir = volume_path / "fluorine_checkpoint_sweeps" / local_workdir.name
        remote_embedding_cache_dir = (
            volume_path / "fluorine_checkpoint_embeddings" / local_workdir.name / label
        )
        handle = run_fluorine_checkpoint.spawn(
            config_path=config_path,
            checkpoint_path=str(remote_checkpoint),
            output_dir=str(remote_output_dir),
            label=label,
            embedding_cache_dir=str(remote_embedding_cache_dir),
        )
        record = {
            "order": order_idx,
            "label": label,
            "global_step": global_step,
            "checkpoint_path": str(checkpoint_path),
            "remote_checkpoint": str(remote_checkpoint),
            "remote_output_dir": str(remote_output_dir),
            "remote_embedding_cache_dir": str(remote_embedding_cache_dir),
            "call_id": handle.object_id,
        }
        records.append(record)
        handles.append((handle, record))
        print(f"spawned {label}: {handle.object_id}")
    manifest_path = local_workdir / "manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True))
    print(f"manifest: {manifest_path}")
    if not wait_for_results:
        return

    def write_fluorine_results(results: list[dict], final: bool) -> None:
        ordered_results = sorted(results, key=lambda value: value["order"])
        partial_path = local_workdir / "results.partial.json"
        partial_path.write_text(json.dumps(ordered_results, indent=2, sort_keys=True))

        plot_path = local_workdir / "fluorine_precision_recall_curves.png"
        pdf_path = local_workdir / "fluorine_precision_recall_curves.pdf"
        fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
        colors = plt.cm.viridis_r(
            [idx / max(1, len(ordered_results) - 1) for idx in range(len(ordered_results))]
        )
        for color, item in zip(colors, ordered_results):
            curve = item["test_pr_curve"]
            ax.plot(
                curve["recall"],
                curve["precision"],
                color=color,
                linewidth=2.0,
                label=f"step {item['global_step']} AP={item['average_precision']:.3f}",
            )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", fontsize=8)
        ax.set_title(
            f"Fluorine Detection Precision-Recall by Checkpoint "
            f"({len(ordered_results)}/{len(records)} ready)"
        )
        fig.tight_layout()
        fig.savefig(plot_path)
        fig.savefig(pdf_path)
        plt.close(fig)

        if final:
            results_path = local_workdir / "results.json"
            results_path.write_text(json.dumps(ordered_results, indent=2, sort_keys=True))
            print(f"plot: {plot_path}")
            print(json.dumps(ordered_results, indent=2, sort_keys=True))

    import time

    results_by_label = {}
    pending = dict(handles)
    while pending:
        made_progress = False
        for handle, record in list(pending.items()):
            try:
                metrics = handle.get(timeout=1)
            except TimeoutError:
                continue

            made_progress = True
            pending.pop(handle)
            output_path = local_workdir / f"{record['label']}.json"
            output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
            summary = {
                **record,
                "average_precision": metrics["test"]["test/average_precision"],
                "roc_auc": metrics["test"]["test/roc_auc"],
                "best_epoch": metrics["best_epoch"],
                "best_hparams": metrics["best_hparams"],
                "test_pr_curve": metrics["test_pr_curve"],
            }
            results_by_label[record["label"]] = summary
            write_fluorine_results(list(results_by_label.values()), final=False)
            print(
                f"completed {record['label']}: "
                f"AP={summary['average_precision']:.4f} "
                f"AUC={summary['roc_auc']:.4f}"
            )
        if pending and not made_progress:
            time.sleep(30.0)

    write_fluorine_results(list(results_by_label.values()), final=True)


def _with_async_probe_override(overrides: str, async_probes: bool) -> str:
    if not async_probes:
        return overrides
    payload = json.loads(overrides)
    payload["msg_probe_backend"] = "modal"
    return json.dumps(payload, sort_keys=True)


def _wait_for_modal_probe_calls(results: dict) -> None:
    call_ids = list(results.get("run/modal_probe_call_ids", []) or [])
    if not call_ids:
        return
    print(f"Waiting for {len(call_ids)} Modal probe job(s)...")
    calls = [modal.FunctionCall.from_id(call_id) for call_id in call_ids]
    modal.FunctionCall.gather(*calls)
    print("Modal probe jobs complete.")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    config: str = "configs/gems_small.py",
    workdir: str = "",
    sweep: str = "",
    overrides: str = "{}",
    workdir_tag: str = "",
    detach: bool = False,
    run_8xh100: bool = False,
    benchmark_multigpu: bool = False,
    benchmark_constant_local_batch: bool = False,
    benchmark_4gpu: str = "",
    benchmark_steps: int = 40,
    benchmark_warmup_steps: int = 10,
    async_probes: bool = False,
    submit_probe_config_json_path: str = "",
    submit_probe_checkpoint_path: str = "",
    submit_probe_workdir: str = "",
    submit_probe_global_step: int = 0,
    submit_probe_wait: bool = False,
    submit_probe_sweep_json_path: str = "",
    submit_fluorine_checkpoint_dir: str = "",
    submit_fluorine_workdir: str = "",
    submit_fluorine_max_checkpoints: int = 10,
):
    if submit_fluorine_checkpoint_dir:
        _submit_fluorine_sweep_from_local(
            config_path=config,
            checkpoint_dir=submit_fluorine_checkpoint_dir,
            workdir=submit_fluorine_workdir,
            max_checkpoints=int(submit_fluorine_max_checkpoints),
            wait_for_results=submit_probe_wait,
        )
        return

    if submit_probe_sweep_json_path:
        _submit_probe_sweep_from_local(
            config_path=config,
            sweep_json_path=submit_probe_sweep_json_path,
            checkpoint_path=submit_probe_checkpoint_path,
            workdir=submit_probe_workdir,
            global_step=int(submit_probe_global_step),
            wait_for_results=submit_probe_wait,
        )
        return

    if submit_probe_checkpoint_path:
        _submit_probe_from_local(
            config_json_path=submit_probe_config_json_path,
            checkpoint_path=submit_probe_checkpoint_path,
            workdir=submit_probe_workdir,
            global_step=int(submit_probe_global_step),
            wait=submit_probe_wait,
        )
        return

    overrides = _with_async_probe_override(overrides, async_probes)

    if benchmark_4gpu:
        gpu_key = benchmark_4gpu.lower()
        if gpu_key not in FOUR_GPU_TRAINERS:
            raise ValueError(
                "--benchmark-4gpu must be one of: "
                f"{', '.join(sorted(FOUR_GPU_TRAINERS))}"
            )
        benchmark_overrides = {
            **json.loads(overrides),
            "enable_wandb": False,
            "msg_probe_every_n_steps": 0,
            "log_every_n_steps": 0,
            "collapse_metrics_every_n_steps": 0,
            "checkpoint_every_steps": 1_000_000,
            "muon_ns_use_kernels": True,
            "training_max_steps": int(benchmark_steps),
            "throughput_warmup_steps": int(benchmark_warmup_steps),
        }
        benchmark_payload = json.dumps(benchmark_overrides, sort_keys=True)
        benchmark_tag = workdir or f"ddp_4gpu_{gpu_key}_bench_{int(time.time())}"
        print("Preparing data on volume...")
        prepare_data.remote(
            config_path=config,
            overrides_json=benchmark_payload,
            prepare_probe=False,
        )
        print("Data ready.\n")
        print(
            f"Running 4-GPU {benchmark_4gpu} benchmark: "
            f"{benchmark_steps} steps, {benchmark_warmup_steps} warmup steps."
        )
        result = FOUR_GPU_TRAINERS[gpu_key].remote(
            config_path=config,
            overrides_json=benchmark_payload,
            workdir=f"{benchmark_tag}/4gpu_{gpu_key}",
            workdir_tag=workdir_tag,
        )
        print("\n4-GPU benchmark:")
        print(json.dumps({
            "gpu": benchmark_4gpu,
            "global_batch_size": result["run/global_batch_size"],
            "local_batch_size": result["run/local_batch_size"],
            "samples_per_second": result["run/measured_samples_per_second"],
            "steps_per_second": result["run/measured_steps_per_second"],
            "metrics": result,
        }, indent=2, sort_keys=True))
        return

    if run_8xh100:
        payload = json.dumps(json.loads(overrides), sort_keys=True)
        print("Preparing data on volume...")
        prepare_data.remote(config_path=config, overrides_json=payload, prepare_probe=False)
        print("Data ready.\n")
        if detach:
            handle = train_8gpu.spawn(
                config_path=config,
                overrides_json=payload,
                workdir=workdir,
                workdir_tag="",
            )
            print(f"spawned: {handle.object_id}")
            return
        result = train_8gpu.remote(
            config_path=config,
            overrides_json=payload,
            workdir=workdir,
            workdir_tag="",
        )
        _wait_for_modal_probe_calls(result)
        return

    if benchmark_constant_local_batch:
        base_overrides = json.loads(overrides)
        base_batch_size = int(base_overrides.get("batch_size", 256))
        common_benchmark_overrides = {
            **base_overrides,
            "enable_wandb": False,
            "msg_probe_every_n_steps": 0,
            "log_every_n_steps": 0,
            "collapse_metrics_every_n_steps": 0,
            "checkpoint_every_steps": 1_000_000,
            "muon_ns_use_kernels": True,
            "training_max_steps": int(benchmark_steps),
            "throughput_warmup_steps": int(benchmark_warmup_steps),
        }
        one_gpu_overrides = {
            **common_benchmark_overrides,
            "batch_size": base_batch_size,
        }
        four_gpu_overrides = {
            **common_benchmark_overrides,
            "batch_size": base_batch_size * 4,
        }
        benchmark_tag = workdir or f"ddp_constant_local_batch_bench_{int(time.time())}"
        print("Preparing data on volume...")
        prepare_data.remote(
            config_path=config,
            overrides_json=json.dumps(one_gpu_overrides, sort_keys=True),
            prepare_probe=False,
        )
        print("Data ready.\n")
        print(
            "Running constant-local-batch benchmark: "
            f"{benchmark_steps} steps, {benchmark_warmup_steps} warmup steps, "
            f"local batch {base_batch_size}."
        )
        one_gpu = train.remote(
            config_path=config,
            overrides_json=json.dumps(one_gpu_overrides, sort_keys=True),
            workdir=f"{benchmark_tag}/1gpu_bs{base_batch_size}",
            workdir_tag=workdir_tag,
        )
        four_gpu = train_4gpu.remote(
            config_path=config,
            overrides_json=json.dumps(four_gpu_overrides, sort_keys=True),
            workdir=f"{benchmark_tag}/4gpu_bs{base_batch_size * 4}",
            workdir_tag=workdir_tag,
        )
        speedup = (
            four_gpu["run/measured_samples_per_second"]
            / one_gpu["run/measured_samples_per_second"]
        )
        print("\nConstant local batch benchmark:")
        print(json.dumps({
            "local_batch_size": base_batch_size,
            "single_gpu_global_batch_size": one_gpu["run/global_batch_size"],
            "four_gpu_global_batch_size": four_gpu["run/global_batch_size"],
            "single_gpu_samples_per_second": one_gpu["run/measured_samples_per_second"],
            "four_gpu_samples_per_second": four_gpu["run/measured_samples_per_second"],
            "speedup": speedup,
            "single_gpu_metrics": one_gpu,
            "four_gpu_metrics": four_gpu,
        }, indent=2, sort_keys=True))
        return

    if benchmark_multigpu:
        base_overrides = json.loads(overrides)
        benchmark_overrides = {
            **base_overrides,
            "enable_wandb": False,
            "msg_probe_every_n_steps": 0,
            "log_every_n_steps": 0,
            "collapse_metrics_every_n_steps": 0,
            "checkpoint_every_steps": 1_000_000,
            "muon_ns_use_kernels": True,
            "training_max_steps": int(benchmark_steps),
            "throughput_warmup_steps": int(benchmark_warmup_steps),
        }
        benchmark_payload = json.dumps(benchmark_overrides, sort_keys=True)
        benchmark_tag = workdir or f"ddp_fixed_batch_bench_{int(time.time())}"
        print("Preparing data on volume...")
        prepare_data.remote(
            config_path=config,
            overrides_json=benchmark_payload,
            prepare_probe=False,
        )
        print("Data ready.\n")
        print(
            "Running fixed-global-batch benchmark: "
            f"{benchmark_steps} steps, {benchmark_warmup_steps} warmup steps."
        )
        one_gpu = train.remote(
            config_path=config,
            overrides_json=benchmark_payload,
            workdir=f"{benchmark_tag}/1gpu",
            workdir_tag=workdir_tag,
        )
        four_gpu = train_4gpu.remote(
            config_path=config,
            overrides_json=benchmark_payload,
            workdir=f"{benchmark_tag}/4gpu",
            workdir_tag=workdir_tag,
        )
        speedup = (
            four_gpu["run/measured_samples_per_second"]
            / one_gpu["run/measured_samples_per_second"]
        )
        print("\nFixed batch benchmark:")
        print(json.dumps({
            "global_batch_size": one_gpu["run/global_batch_size"],
            "single_gpu_samples_per_second": one_gpu["run/measured_samples_per_second"],
            "four_gpu_samples_per_second": four_gpu["run/measured_samples_per_second"],
            "speedup": speedup,
            "single_gpu_metrics": one_gpu,
            "four_gpu_metrics": four_gpu,
        }, indent=2, sort_keys=True))
        return

    if sweep:
        if workdir:
            raise ValueError(
                "--workdir is only supported for single runs; use --workdir-tag for sweeps."
            )
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
        if detach:
            handles = []
            for exp in experiments:
                merged = {**base_overrides, **exp}
                handles.append(
                    train.spawn(
                        config_path=config,
                        overrides_json=json.dumps(merged),
                        workdir="",
                        workdir_tag=workdir_tag,
                    )
                )
            for i, handle in enumerate(handles):
                print(f"[{i}] spawned: {handle.object_id}")
            return
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
                        workdir="",
                        workdir_tag=workdir_tag,
                    )
                )
            for offset, handle in enumerate(handles):
                result = handle.get()
                _wait_for_modal_probe_calls(result)
                print(f"[{batch_start + offset}] done: {result}")
    else:
        print("Preparing data on volume...")
        prepare_data.remote(config_path=config, overrides_json=overrides)
        print("Data ready.\n")
        if detach:
            handle = train.spawn(
                config_path=config,
                overrides_json=overrides,
                workdir=workdir,
                workdir_tag=workdir_tag,
            )
            print(f"spawned: {handle.object_id}")
        else:
            result = train.remote(
                config_path=config,
                overrides_json=overrides,
                workdir=workdir,
                workdir_tag=workdir_tag,
            )
            _wait_for_modal_probe_calls(result)
