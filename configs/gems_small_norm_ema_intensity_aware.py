from configs.gems_small_norm_ema import get_config as get_base_config


aware_mixed = dict(
    tau=0.5,
    alpha=0.75,
    beta_context=0.85,
    beta_target=0.85,
    eps_context=0.04,
    eps_target=0.08,
    anchor_keep=0.60,
    min_eff_context=4.0,
    min_eff_target=1.8,
    tail_target_mix=0.10,
    local_gap_da=1.0,
    local_gap_probability=0.50,
    min_unused_mass=0.09,
)

inactive_count_mask_fields = (
    "jepa_block_min_len",
    "jepa_context_fraction",
    "jepa_context_fraction_range",
    "jepa_target_fraction",
    "jepa_target_fraction_range",
    "jepa_mask_lengths",
    "jepa_mask_round_from",
)


def get_config():
    cfg = get_base_config()
    cfg.jepa_mask_strategy = "intensity_aware"
    for key in inactive_count_mask_fields:
        if key in cfg:
            del cfg[key]
    for key, value in aware_mixed.items():
        setattr(cfg, f"jepa_intensity_aware_{key}", value)
    cfg.run_name_suffix = "ema-teacher-intensity-aware-mixed"
    return cfg
