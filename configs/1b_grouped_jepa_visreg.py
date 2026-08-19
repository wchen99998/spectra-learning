from importlib import import_module

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    cfg = import_module("configs.1b_grouped_jepa").get_config()

    cfg.group_jepa_use_ema_teacher = False
    cfg.group_jepa_ema_momentum = None
    cfg.group_jepa_teacher_target_mode = "same_step"

    cfg.group_jepa_invariance_loss_weight = 0.9
    cfg.group_jepa_visreg_loss_weight = 0.1
    cfg.group_jepa_visreg_num_projections = 256
    cfg.group_jepa_visreg_center_weight = 1.0
    cfg.group_jepa_visreg_scale_weight = 1.0
    cfg.group_jepa_visreg_shape_weight = 1.0
    cfg.group_jepa_visreg_gather_embeddings = True
    cfg.training_max_steps = 300_000

    cfg.run_name_suffix = (
        "jax-grouped-jepa-visreg-massive-v2-1b-singlemixer-n64-g256-s8-t3-"
        "stopgrad-k256-muon-lr4e-5-300k"
    )
    return cfg
