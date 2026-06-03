from ml_collections import config_dict

from configs.medium_pairmixer_teacher_jepa_benchmark import (
    get_config as get_benchmark_config,
)


def get_config() -> config_dict.ConfigDict:
    cfg = get_benchmark_config()

    cfg.log_every_n_steps = 250
    cfg.checkpoint_every_steps = 25_000
    cfg.disable_tqdm = True
    cfg.enable_wandb = True
    cfg.dataloader_num_workers = 0
    cfg.dataloader_pin_memory = False
    cfg.dataloader_persistent_workers = False
    cfg.run_name_suffix = "torchax-pairmixer-teacher-jepa-medium-bs128"

    return cfg
