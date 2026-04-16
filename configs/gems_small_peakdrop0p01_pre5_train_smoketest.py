from configs.gems_small import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    with cfg.ignore_type():
        cfg.num_epochs = 0.005
    cfg.peak_drop_min_intensity = 0.01
    cfg.precursor_peak_exclusion_window_da = 5.0
    cfg.enable_wandb = False
    cfg.msg_probe_every_n_steps = 0
    cfg.log_every_n_steps = 20
    cfg.checkpoint_every_steps = 1000
    cfg.dataloader_num_workers = 8
    return cfg
