"""Train on Modal with persistent storage for data and checkpoints.

Usage:
    # Single run
    modal run modal_train.py
    modal run modal_train.py --config configs/gems_small.py --gpu H100
    modal run modal_train.py --config configs/gems_small.py --workdir my_run

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
PROJECT_ROOT = "/root/spectra-learning"
MAX_SWEEP_CONCURRENCY = 10
TRAIN_TIMEOUT_HOURS = 24
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
    .add_local_dir(local / "spectra_learning", remote_path=f"{PROJECT_ROOT}/spectra_learning")
    .add_local_dir(local / "configs", remote_path=f"{PROJECT_ROOT}/configs")
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
):
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
        train_8gpu.remote(
            config_path=config,
            overrides_json=payload,
            workdir=workdir,
            workdir_tag="",
        )
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
            train.remote(
                config_path=config,
                overrides_json=overrides,
                workdir=workdir,
                workdir_tag=workdir_tag,
            )
