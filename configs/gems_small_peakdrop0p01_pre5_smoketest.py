from configs.gems_small import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.peak_drop_min_intensity = 0.01
    cfg.precursor_peak_exclusion_window_da = 5.0
    cfg.dataloader_num_workers = 8
    cfg.dataloader_persistent_workers = True
    cfg.dataloader_prefetch_factor = 2
    cfg.enable_wandb = False
    cfg.log_every_n_steps = 10
    cfg.collapse_metrics_every_n_steps = 10
    cfg.checkpoint_every_steps = 1_000_000
    cfg.msg_probe_every_n_steps = 0
    with cfg.ignore_type():
        cfg.num_epochs = 0.0006
    return cfg
