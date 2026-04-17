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
    modal run modal_train.py --sweep sweep_10m_sigreg_ema_ablation
    modal run modal_train.py --sweep sweep_10m_noema_sigreg_log_lambda
    modal run modal_train.py --sweep sweep_10m_noema_sigreg_high_lambda
    modal run modal_train.py --sweep sweep_10m_ema_stopgrad
    modal run modal_train.py --sweep sweep_10m_ema_stopgrad_warmup_update
    modal run modal_train.py --sweep sweep_10m_ema_masking
    modal run modal_train.py --sweep sweep_10m_ema_deep_supervision
    modal run modal_train.py --sweep sweep_10m_ema_batch_size_flops_matched
    modal run modal_train.py --config configs/gems_small.py --sweep sweep_gems_small_peak_filtering
    modal run modal_train.py --config configs/gems_small.py --sweep sweep_gems_small_predictor_scale
    modal run modal_train.py --config configs/gems_small.py --sweep sweep_gems_small_predictor_scale_depth
    modal run modal_train.py --config configs/gems_small.py --sweep sweep_gems_small_scale_100m_300m --detach

Setup:
    1. modal setup
    2. modal secret create wandb-secret WANDB_API_KEY=<your-key>
    3. modal secret create huggingface-secret HF_TOKEN=<your-token>
    4. modal run modal_train.py
"""

import json
from pathlib import Path

import modal

MINUTES = 60
HOURS = 60 * MINUTES
DEFAULT_GPU = "H100"
PROJECT_ROOT = "/root/spectra-learning"
MAX_SWEEP_CONCURRENCY = 10
TRAIN_TIMEOUT_HOURS = 13
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
    "teacher_ema_decay": 0.9996,
    "teacher_ema_decay_start": 0.999,
    "teacher_ema_update_every": 1,
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "representation_regularizer": "none",
    "sigreg_lambda": 0.02,
    "vicreg_lambda": 0.02,
    "vicreg_inv_coeff": 0.0,
    "vicreg_var_coeff": 25.0,
    "vicreg_cov_coeff": 1.0,
    "vicreg_variance_target": 1.0,
    "vicreg_eps": 1e-4,
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
EMA_STOPGRAD_BEST_SAME_STEP = {
    **EMA_STOPGRAD_FIXED,
    "teacher_ema_decay_warmup_steps": 5_000,
    "teacher_ema_update_every": 5,
}
JEPA_MASKING_SWEEP_TAG = "emask"
JEPA_DEEP_SUPERVISION_SWEEP_TAG = "emadsup"
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
            "jepa_context_fraction_range": (0.35, 0.35),
            "jepa_target_fraction": 0.20,
            "jepa_target_fraction_range": (0.20, 0.20),
            "jepa_mask_strategy": "ragged",
        },
    ),
    (
        "rnd-c25-45-t15-25",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.35,
            "jepa_context_fraction_range": (0.25, 0.45),
            "jepa_target_fraction": 0.20,
            "jepa_target_fraction_range": (0.15, 0.25),
            "jepa_mask_strategy": "random",
        },
    ),
    (
        "all-c25-45-t15-25",
        {
            "jepa_num_target_blocks": 2,
            "jepa_context_fraction": 0.35,
            "jepa_context_fraction_range": (0.25, 0.45),
            "jepa_target_fraction": 0.20,
            "jepa_target_fraction_range": (0.15, 0.25),
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
    "predictor_dim": 512,
    "masked_latent_predictor_num_layers": 12,
    "masked_latent_predictor_num_heads": 16,
    "jepa_target_layers": [4, 8, 12, 16, 20],
}


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
    # Fix the winning same-step EMA recipe and sweep only the JEPA masking
    # pattern around the current GeMS default.
    "sweep_10m_ema_masking": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            **EMA_STOPGRAD_BEST_SAME_STEP,
            "use_ema_teacher_target": True,
            "representation_regularizer": "none",
            "run_name_suffix": f"10m-{JEPA_MASKING_SWEEP_TAG}-{label}",
            **masking_overrides,
        }
        for label, masking_overrides in JEPA_MASKING_RECIPES
    ],
    # Fix the current best recipe (EMA + masking)
    # and ablate the "bootleg deep supervision" target stack.
    #
    # This is a clean 2x2:
    #   - layer stack: current spread 4-layer targets [1,3,5,8] vs deeper 2-layer targets [5,8]
    #   - target normalization: per-layer zscore vs no normalization
    "sweep_10m_ema_deep_supervision": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            **EMA_STOPGRAD_BEST_SAME_STEP,
            "use_ema_teacher_target": True,
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
    "sweep_10m_ema_batch_size_flops_matched": [
        {
            **TEN_M_BEST_SWEEP_OPTIM,
            **EMA_STOPGRAD_BEST_SAME_STEP,
            "use_ema_teacher_target": True,
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
    #   - predictor_dim=256 -> 3,414,272 predictor params vs 6,779,392 encoder
    #     params (50.4%)
    #   - predictor_dim=360 -> 6,689,664 predictor params vs 6,779,392 encoder
    #     params (98.7%)
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
    #     6,564,096 predictor params vs 6,779,392 encoder params (96.8%)
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
):
    import logging
    import os
    import sys

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    logging.basicConfig(level=logging.INFO)

    from utils.training import load_config

    config = load_config(config_path)
    config.update(json.loads(overrides_json))
    config.artifact_dir = str(volume_path / "data" / "gems_artifacts_alpha")

    # 1) Download training data (GeMS native shards)
    logging.info("Preparing training data...")
    from input_pipeline import GemsNativeDataModule

    datamodule = GemsNativeDataModule(config, seed=int(config.seed))
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
    timeout=TRAIN_TIMEOUT_HOURS * HOURS,
    secrets=[huggingface_secret, wandb_secret],
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

    from train import train_and_evaluate
    from utils.training import auto_run_name, load_config

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
):
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
