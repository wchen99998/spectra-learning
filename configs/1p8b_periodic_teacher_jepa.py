"""Fresh 1.8B JEPA student with a teacher promoted every 50,000 steps."""

from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.1p8b_mae_teacher_jepa").get_config()
    # Start from the trained JEPA encoder, not the MAE teacher stored alongside
    # it. Only the teacher is restored; the student and optimizer start fresh.
    cfg.frozen_teacher_config_path = "configs/1p8b_mae_teacher_jepa.py"
    cfg.frozen_teacher_checkpoint_path = (
        "gs://metal-repeater-411410-spectra-checkpoints/skypilot/"
        "1p8b-ftjepa-t1p8b-s666k-perleaf-b1024-v6e32-20260905-115335/"
        "checkpoints/orbax/1190000"
    )
    cfg.frozen_teacher_update_every_steps = 50_000
    cfg.run_name_suffix = (
        "jax-periodic-teacher-jepa-1p8b-student-jepa-step1190000-"
        "promote50k-n128-cls-v6e4x8-bs1024-ga8-lr3e-4-random-"
        "ctx60-target25-target-only-adam-bf16state-zero1-perleaf-2m-ckpt500"
    )
    return cfg
